import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from pool_fixture import make_pool
from geophy_set3d.models import GeoLiftOARIMRGRRes3D
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset, DualMeasurementDataset, dual_measurement_collate, assert_no_gt, DisjointABHSampler
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.selfsup import measurement_discrepancy, objective, initialize_fresh_model, selection_metric, make_loader, run_epoch
from geophy_set3d.training.utils import load_config, permute_views


class SelfsupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary,cls.root,cls.records,_=make_pool()
        cls.config=load_config('configs/train_selfsup_oari_mrgr_v1.yaml')
        cls.config['dataset']['root']=str(cls.root);cls.config['training']['device']='cpu'
        cls.config['loader'].update(num_workers=0,pin_memory=False,persistent_workers=False)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def dataset(self,**kwargs):
        return DualMeasurementDataset(self.root,split='train',**kwargs)

    def batch(self): return dual_measurement_collate([self.dataset()[0]])

    def test_selfsup_dataset_has_no_gt(self):
        sample=MeasurementOnlyPoolDataset(self.root,split='train',hold_views=0)[0]
        self.assertNotIn('gt_volume_64',sample);self.assertNotIn('gt_volume_128',sample)
        with self.assertRaises(ValueError): MeasurementOnlyPoolDataset(self.root,load_gt=True)
        with self.assertRaises(ValueError): MeasurementOnlyPoolDataset(self.root,load_gt128=True)

    def test_selfsup_batch_contains_no_gt(self):
        assert_no_gt(self.batch())
        for subset in self.batch().values(): self.assertFalse(any(k.startswith('gt_') for k in subset))

    def test_selfsup_loader_never_reads_gt(self):
        with tempfile.TemporaryDirectory() as checkpoints:
            guard=GTReadGuard(self.root,checkpoints)
            config=copy.deepcopy(self.config);config['loader'].update(num_workers=2,persistent_workers=False)
            with guard:
                rows=list(make_loader(self.dataset(),config))
                for b in rows: assert_no_gt(b)
            self.assertFalse(guard.summary()['gt_file_opened'])
            self.assertGreater(guard.summary()['projection_open_count'],0)
            self.assertEqual(guard.summary()['unique_pool_a_projection_files'],3)

    def test_selfsup_training_never_loads_gt64(self):
        real_load=np.load
        forbidden={str(self.root/r['gt_volume_64_path']) for r in self.records}
        def strict(path,*args,**kwargs):
            if str(path) in forbidden: raise AssertionError('GT64 was opened')
            return real_load(path,*args,**kwargs)
        with patch('numpy.load',side_effect=strict):
            model=initialize_fresh_model(self.config)
            b=self.batch();loss,_,_,_=objective(model,b)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_selfsup_training_never_loads_gt128(self):
        forbidden={r['source_target_path'] for r in self.records};real_load=np.load
        def strict(path,*args,**kwargs):
            if str(path) in forbidden: raise AssertionError('GT128 was opened')
            return real_load(path,*args,**kwargs)
        with patch('numpy.load',side_effect=strict): assert_no_gt(self.batch())

    def test_runtime_guard_denies_direct_gt_and_foreign_checkpoint(self):
        with tempfile.TemporaryDirectory() as own:
            guard=GTReadGuard(self.root,own)
            with guard:
                for path in (self.root/self.records[0]['gt_volume_64_path'],Path(self.records[0]['source_target_path'])):
                    with self.assertRaises(PermissionError): path.open('rb')
                with self.assertRaises(PermissionError): Path('/tmp/supervised.pt').open('rb')
            self.assertEqual(guard.summary()['gt_open_attempts'],2)
            self.assertEqual(guard.summary()['supervised_checkpoint_open_attempts'],1)

    def test_A_B_H_disjoint(self):
        sampler=DisjointABHSampler()
        for epoch in (0,1,31):
            for index in range(5):
                a,b,h=sampler.sample(12,epoch=epoch,sample_index=index,case_id=f'case-{index}')
                self.assertTrue(3<=len(a)<=5 and 3<=len(b)<=5 and len(h)==2)
                self.assertEqual(len(set(a)|set(b)|set(h)),len(a)+len(b)+len(h))

    def test_selfsup_sampler_worker_invariant(self):
        ds=self.dataset()
        def ids(workers):
            return [{k:b[k]['input_station_indices'].tolist() for k in b} for b in DataLoader(ds,batch_size=1,num_workers=workers,collate_fn=dual_measurement_collate)]
        self.assertEqual(ids(0),ids(2))
        np.random.seed(65); a=ds[0]; np.random.seed(96);b=ds[0]
        for key in a: self.assertTrue(torch.equal(a[key]['input_station_indices'],b[key]['input_station_indices']))
        ds.set_epoch(1);self.assertFalse(torch.equal(a['A']['input_station_indices'],ds[0]['A']['input_station_indices']))

    def test_measurement_loss_valid_ray_only(self):
        b=self.batch()['H']; valid=b['input_valid_ray'].bool()
        p=b['input_projection_raw'].clone().requires_grad_()
        b=copy.deepcopy(b);b['input_projection_raw']=torch.where(valid,b['input_projection_raw'],float('nan'))
        p_bad=torch.where(valid,p,torch.full_like(p,float('nan')))
        loss=measurement_discrepancy(p_bad,b)
        self.assertEqual(float(loss),0);loss.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_zero_measurement_valid_ray_preserved(self):
        b=self.batch()['H']; b['input_projection_raw']=torch.zeros_like(b['input_projection_raw'])
        p=torch.ones_like(b['input_projection_raw']).requires_grad_()
        loss=measurement_discrepancy(p,b);self.assertGreater(float(loss),0)
        loss.backward();valid=b['input_valid_ray'].bool()
        self.assertGreater(float(p.grad[valid].abs().sum()),0)
        self.assertTrue((p.grad[~valid]==0).all())

    def test_selfsup_loss_has_no_gt_dependency(self):
        b=self.batch();b['A']['gt_volume_64']=torch.ones(1,1,64,64,64)
        with self.assertRaises(ValueError): objective(GeoLiftOARIMRGRRes3D(),b)

    def test_cross_view_prediction_geometry_correct(self):
        b=self.batch();calls=[]
        class Stub(torch.nn.Module):
            def forward(self,subset):
                value=1. if torch.equal(subset['input_station_indices'],b['A']['input_station_indices']) else 2.
                return torch.full((1,1,64,64,64),value)
        def project(c,subset):
            calls.append((float(c.mean()),subset['input_station_indices'].tolist()))
            return torch.zeros_like(subset['input_projection_raw'])
        with patch('geophy_set3d.training.selfsup.project_input',side_effect=project):
            loss,parts,_,_=objective(Stub(),b)
        self.assertIn((1.,b['B']['input_station_indices'].tolist()),calls)
        self.assertIn((2.,b['A']['input_station_indices'].tolist()),calls)
        self.assertEqual(len(calls),6)
        torch.testing.assert_close(loss,parts['hold']+.5*parts['cross']+.1*parts['observed']+.05*parts['consistency'])

    def test_heldout_views_not_input(self):
        b=self.batch();seen=[]
        class Stub(torch.nn.Module):
            def forward(self,subset):
                seen.append(subset['input_station_indices'].tolist())
                return torch.ones(1,1,64,64,64)*.01
        with patch('geophy_set3d.training.selfsup.project_input',side_effect=lambda c,t:torch.zeros_like(t['input_projection_raw'])):
            objective(Stub(),b)
        self.assertEqual(seen,[b['A']['input_station_indices'].tolist(),b['B']['input_station_indices'].tolist()])
        self.assertNotIn(b['H']['input_station_indices'].tolist(),seen)

    def test_selfsup_permutation_invariance(self):
        torch.manual_seed(5);model=GeoLiftOARIMRGRRes3D().eval();b=self.batch()['A']
        with torch.no_grad():
            model.residual_adapter[-1].weight.normal_(0,.02)
            p=model(b);q=model(permute_views(b,torch.arange(b['view_mask'].shape[1]).flip(0)))
        torch.testing.assert_close(p,q,atol=3e-6,rtol=0)

    def test_selfsup_padding_invariance(self):
        model=GeoLiftOARIMRGRRes3D().eval();b=self.batch()['A'];pad=dict(b)
        for k,v in b.items():
            if (k.startswith('input_') or k=='view_mask') and isinstance(v,torch.Tensor):
                pad[k]=torch.cat((v,torch.zeros_like(v[:,:1])),1)
        pad['input_projection'][:,-1]=float('nan');pad['input_projection_raw'][:,-1]=float('nan')
        with torch.no_grad():
            model.residual_adapter[-1].weight.normal_(0,.02)
            p=model(b);q=model(pad)
        torch.testing.assert_close(p,q,atol=3e-6,rtol=0)

    def test_selfsup_all_invalid_safe(self):
        b=self.batch()
        for subset in b.values():
            subset['view_mask']=torch.zeros_like(subset['view_mask']);subset['input_valid_ray']=torch.zeros_like(subset['input_valid_ray'])
        model=GeoLiftOARIMRGRRes3D().eval()
        loss,_,volumes,_=objective(model,b)
        self.assertTrue(torch.isfinite(loss) and all(torch.isfinite(c).all() for c in volumes))
        loss.backward()

    def test_formal_training_loads_no_supervised_checkpoint(self):
        with patch('torch.load',side_effect=AssertionError('No checkpoint initialization')):
            model=initialize_fresh_model(self.config)
        self.assertEqual(sum(p.numel() for p in model.parameters()),332070)

    def test_checkpoint_selection_uses_measurement_only(self):
        validation=dict(measurement=dict(heldout=dict(relative_l1=.2)),voxel_l1=-999,loss=dict(total=-999))
        self.assertEqual(selection_metric(validation),.2)
        with self.assertRaises(KeyError):selection_metric(dict(voxel_l1=.1))

    def test_gt_files_can_be_absent(self):
        temporary,root,records,_=make_pool()
        try:
            for r in records:
                (root/r['gt_volume_64_path']).unlink();Path(r['source_target_path']).unlink()
            ds=DualMeasurementDataset(root,split='train')
            assert_no_gt(dual_measurement_collate([ds[0],ds[1]]))
        finally:temporary.cleanup()
