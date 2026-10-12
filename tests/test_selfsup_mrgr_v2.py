"""RVTC data pairing, shared-model branch gradients, safety and selection boundary."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from pool_fixture import make_pool
from geophy_set3d.data.rich_measurement import RichMeasurementDataset, rich_measurement_collate
from geophy_set3d.training import selfsup_v2 as v2
from geophy_set3d.training.utils import load_config, permute_views


class RVTCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary,cls.root,cls.records,_=make_pool()
        cls.config=load_config('configs/train_selfsup_oari_mrgr_v2.yaml')
        cls.config['dataset']['root']=str(cls.root); cls.config['training']['device']='cpu'
        cls.config['loader'].update(num_workers=0,pin_memory=False,persistent_workers=False)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def batch(self):
        ds=RichMeasurementDataset(self.root,split='train')
        return rich_measurement_collate([ds[0],ds[1]])

    def test_rvtc_T_is_union_of_A_B(self):
        b=self.batch(); v2.assert_rich_union(b)
        for i in range(2):
            a,bb,t=[b[k]['input_station_indices'][i,b[k]['view_mask'][i]] for k in ('A','B','T')]
            self.assertEqual(t.tolist(),a.tolist()+bb.tolist())
            for k in b['T']:
                if k.startswith('input_'):
                    expected=torch.cat([b[x][k][i,b[x]['view_mask'][i]] for x in ('A','B')])
                    torch.testing.assert_close(b['T'][k][i,b['T']['view_mask'][i]],expected)

    def test_T_excludes_H(self):
        b=self.batch()
        for i in range(2):
            ids=lambda k:set(b[k]['input_station_indices'][i,b[k]['view_mask'][i]].tolist())
            self.assertFalse(ids('T')&ids('H'))
        bad=copy.deepcopy(b); bad['T']['input_station_indices'][0,0]=bad['H']['input_station_indices'][0,0]
        with self.assertRaises(ValueError):v2.assert_rich_union(bad)

    def test_A_B_H_pairwise_disjoint(self):
        ds=RichMeasurementDataset(self.root,split='train')
        for epoch in (0,1,39):
            ds.set_epoch(epoch)
            for i in range(len(ds)):
                v2.assert_rich_union(rich_measurement_collate([ds[i]]))

    def test_teacher_uses_same_model(self):
        b=self.batch(); calls=[]
        class Stub(torch.nn.Module):
            def forward(self,subset):
                calls.append(subset)
                return torch.ones(2,1,2,2,2)*(.1*len(calls))
        def project(c,t):return torch.ones_like(t['input_projection_raw'])*c.mean()
        with patch.object(v2,'project_input',side_effect=project):
            loss,parts,_,_=v2.objective(Stub(),b)
        self.assertEqual([id(x) for x in calls],[id(b[k]) for k in ('A','B','T')])
        self.assertNotIn('consistency',parts)
        torch.testing.assert_close(loss,parts['hold']+.5*parts['cross']+.1*parts['observed']+
                                   parts['hold_teacher']+.1*parts['observed_teacher']+.1*parts['teacher'])

    def branch_volumes(self):
        return [torch.full((2,1,2,2,2),value,requires_grad=True) for value in (.2,.4,.3)]

    def test_teacher_stop_gradient(self):
        a,b,t=self.branch_volumes()
        v2.teacher_discrepancy(a,b,t).backward()
        self.assertGreater(float(a.grad.abs().sum()),0); self.assertGreater(float(b.grad.abs().sum()),0)
        self.assertIsNone(t.grad)

    def test_student_teacher_loss_has_student_gradient(self):
        a,b,t=self.branch_volumes()
        grads=torch.autograd.grad(v2.teacher_discrepancy(a,b,t),(a,b))
        self.assertTrue(all(float(g.abs().sum())>0 for g in grads))

    def test_student_teacher_loss_has_no_teacher_gradient(self):
        a,b,t=self.branch_volumes()
        g,=torch.autograd.grad(v2.teacher_discrepancy(a,b,t),(t,),allow_unused=True)
        self.assertIsNone(g)

    def test_teacher_measurement_loss_has_gradient(self):
        b=rich_measurement_collate([RichMeasurementDataset(self.root,split='train')[0]])
        model=v2.initialize_fresh_model(self.config)
        _,parts,(_,_,ct),_=v2.objective(model,b)
        ct.retain_grad(); parts['hold_teacher'].backward()
        self.assertTrue(torch.isfinite(ct.grad).all())
        self.assertGreater(float(ct.grad.abs().sum()),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)

    def test_sample_relative_teacher_normalization(self):
        a=torch.tensor([1.,4.]).reshape(2,1,1,1,1)
        t=torch.tensor([2.,8.]).reshape_as(a)
        self.assertAlmostEqual(float(v2.sample_relative_l1(a,t)),.5,places=6)

    def test_teacher_view_order_permutation_safe(self):
        torch.manual_seed(17); model=v2.initialize_fresh_model(self.config).eval()
        b=self.batch()['T']
        with torch.no_grad():
            model.residual_adapter[-1].weight.normal_(0,.02)
            p=model(b); q=model(permute_views(b,torch.arange(b['view_mask'].shape[1]).flip(0)))
        self.assertLessEqual(float((p-q).abs().max()),1e-5)

    def test_teacher_padding_safe(self):
        model=v2.initialize_fresh_model(self.config).eval(); b=self.batch()['T']; pad=dict(b)
        for k,v in b.items():
            if (k.startswith('input_') or k=='view_mask') and isinstance(v,torch.Tensor):
                pad[k]=torch.cat((v,torch.zeros_like(v[:,:1])),1)
        pad['input_projection'][:,-1]=float('nan'); pad['input_projection_raw'][:,-1]=float('nan')
        with torch.no_grad():p=model(b); q=model(pad)
        self.assertLessEqual(float((p-q).abs().max()),1e-5)

    def test_all_invalid_safe(self):
        b=self.batch()
        for subset in b.values():
            subset['view_mask'].zero_(); subset['input_valid_ray'].zero_()
        model=v2.initialize_fresh_model(self.config)
        loss,_,volumes,_=v2.objective(model,b)
        self.assertTrue(torch.isfinite(loss) and all(torch.isfinite(c).all() for c in volumes))
        loss.backward()

    def test_selfsup_v2_no_gt(self):
        with tempfile.TemporaryDirectory() as own:
            guard=v2.V2ReadGuard(self.root,own)
            cfg=copy.deepcopy(self.config); cfg['loader'].update(num_workers=2,persistent_workers=False)
            with guard:
                batch=next(iter(v2.make_loader(RichMeasurementDataset(self.root,split='train'),cfg)))
                model=v2.initialize_fresh_model(self.config)
                loss,_,_,_=v2.objective(model,batch); loss.backward()
            self.assertTrue(v2.audit_is_clean(guard.summary()))
            self.assertGreater(guard.summary()['projection_open_count'],0)
        batch=self.batch(); batch['T']['gt_volume_64']=torch.ones(1)
        with self.assertRaises(ValueError):v2.objective(model,batch)

    def test_selfsup_v2_no_supervised_checkpoint(self):
        with patch('torch.load',side_effect=AssertionError('Forbidden checkpoint initialization')):
            model=v2.initialize_fresh_model(self.config)
        self.assertEqual(sum(p.numel() for p in model.parameters()),332070)
        with tempfile.TemporaryDirectory() as own:
            guard=v2.V2ReadGuard(self.root,own)
            with guard:
                for path in (Path('/tmp/supervised.pt'),Path('/tmp/selfsup_v1.pt')):
                    with self.assertRaises(PermissionError):path.open('rb')
            self.assertEqual(guard.summary()['external_checkpoint_attempts'],2)

    def test_runtime_forbidden_reads_counted(self):
        with tempfile.TemporaryDirectory() as own:
            guard=v2.V2ReadGuard(self.root,own)
            with guard:
                for path in (self.root/self.records[0]['gt_volume_64_path'],Path(self.records[0]['source_target_path']),
                             self.root/self.records[0]['ood_projection_path'],self.root/'geometry/ood_geometry.npz'):
                    with self.assertRaises(PermissionError):path.open('rb')
            self.assertEqual(guard.summary()['gt_open_attempts'],2)
            self.assertEqual(guard.summary()['Pool_B_access_attempts'],2)
            self.assertEqual(guard.summary()['forbidden_projection_attempts'],1)

    def test_selection_metric_sparse_heldout_only(self):
        validation=dict(measurement=dict(heldout=dict(relative_l1=.2),teacher_heldout=dict(relative_l1=-999)),
                        voxel_l1=-999,loss=dict(teacher=-999,total=-999))
        self.assertEqual(v2.selection_metric(validation),.2)
        with self.assertRaises(KeyError):v2.selection_metric(dict(measurement=dict(teacher_heldout=dict(relative_l1=.1))))

    def test_v2_protocol_rejects_weight_or_architecture_change(self):
        v2.validate_selfsup_config(self.config)
        for section,key,value in (('loss','teacher',.5),('model','lift_resolution',16)):
            config=copy.deepcopy(self.config); config[section][key]=value
            with self.assertRaises(ValueError):v2.validate_selfsup_config(config)
