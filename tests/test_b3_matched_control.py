import copy
import unittest
from unittest.mock import patch
import torch
from pool_fixture import make_pool
from geophy_set3d.models import build_model
from geophy_set3d.training.utils import load_config
from geophy_set3d.training.losses import supervised_loss
from geophy_set3d.training.matched_control import (B2_CHECKPOINT, CANDIDATES, control_config,
    validate_control, verify_station_sampling, run_control_epoch)
from geophy_set3d.data.measurement_sampling import measurement_dataset
from geophy_set3d.evaluation.matched_control import choose_candidate, pair_rows, paired_statistics


class TinyControlModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.conv=torch.nn.Conv3d(1,1,1)

    def forward(self,batch):
        return self.conv(batch['gt_volume_64']).sigmoid()


class MatchedControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control=load_config('configs/train_b3_gate_control.yaml')
        cls.candidates=[load_config(f'configs/train_b3_gate_{n}.yaml') for n in CANDIDATES]
        cls.protocols=[dict(initial_checkpoint=str(B2_CHECKPOINT)) for _ in CANDIDATES]
        cls.tmp,cls.root,_,_=make_pool()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_control_uses_same_initial_checkpoint(self):
        self.assertEqual(validate_control(self.control,self.candidates,self.protocols),B2_CHECKPOINT)
        protocols=copy.deepcopy(self.protocols);protocols[1]['initial_checkpoint']='wrong.pt'
        with self.assertRaises(ValueError):validate_control(self.control,self.candidates,protocols)

    def datasets(self):
        configs=copy.deepcopy([self.control,*self.candidates])
        for config in configs:config['dataset']['root']=str(self.root)
        return [measurement_dataset(c) for c in configs]

    def test_control_uses_same_input_station_ids(self):
        datasets=self.datasets()
        for epoch in range(20):
            for d in datasets:d.set_epoch(epoch)
            for i in range(len(datasets[0])):
                expected=datasets[0][i]['input_station_indices']
                for d in datasets[1:]:torch.testing.assert_close(expected,d[i]['input_station_indices'],rtol=0,atol=0)

    def test_control_uses_same_hold_station_ids(self):
        datasets=self.datasets()
        for epoch in range(20):
            for d in datasets:d.set_epoch(epoch)
            for i in range(len(datasets[0])):
                expected=datasets[0][i]['hold_station_indices']
                self.assertEqual(len(expected),1)
                for d in datasets[1:]:torch.testing.assert_close(expected,d[i]['hold_station_indices'],rtol=0,atol=0)
        configs=copy.deepcopy([self.control,*self.candidates])
        for c in configs:c['dataset']['root']=str(self.root)
        self.assertTrue(verify_station_sampling(configs[0],configs[1:])['passed'])

    def test_control_has_zero_physics_weights(self):
        self.assertEqual(self.control['physics']['lambda_obs'],0)
        self.assertEqual(self.control['physics']['lambda_hold'],0)
        self.assertEqual(self.control['train']['hold_views'],1)
        self.assertEqual(self.control['validation']['hold_views'],1)
        for c in self.candidates:self.assertEqual(control_config(c),self.control)

    def test_control_architecture_matches_candidates(self):
        for candidate in self.candidates:self.assertEqual(self.control['model'],candidate['model'])
        model=build_model(self.control)
        self.assertEqual(sum(p.numel() for p in model.parameters()),319042)
        self.assertIs(type(model),type(build_model(self.candidates[0])))

    def test_control_optimizer_matches_candidates(self):
        for c in self.candidates:
            self.assertEqual(self.control['optimizer'],c['optimizer'])
            self.assertEqual(self.control['scheduler'],c['scheduler'])
        self.assertEqual(self.control['optimizer'],dict(name='adamw',lr=5e-5,weight_decay=1e-4))
        self.assertEqual(self.control['scheduler'],dict(name='constant'))
        bad=copy.deepcopy(self.control);bad['optimizer']['lr']=1e-4
        with self.assertRaises(ValueError):validate_control(bad,self.candidates,self.protocols)

    def test_control_epoch_budget_matches_candidates(self):
        self.assertEqual(self.control['training']['max_epochs'],20)
        for c in self.candidates:self.assertEqual(self.control['training'],c['training'])
        bad=copy.deepcopy(self.control);bad['training']['max_epochs']=21
        with self.assertRaises(ValueError):validate_control(bad,self.candidates,self.protocols)

    def test_control_skips_physics_calls_and_matches_supervised_update(self):
        config=copy.deepcopy(self.control);config['training']['device']='cpu'
        torch.manual_seed(10);model=TinyControlModel();expected=copy.deepcopy(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4)
        other=torch.optim.AdamW(expected.parameters(),lr=5e-5,weight_decay=1e-4)
        batch=dict(gt_volume_64=torch.full((1,1,4,4,4),.05),case_id=['case'])
        loss,_=supervised_loss(expected(batch),batch['gt_volume_64'],**config['loss'])
        loss.backward();torch.nn.utils.clip_grad_norm_(expected.parameters(),1.,error_if_nonfinite=True);other.step()
        with patch('geophy_set3d.training.measurement_engine.compute_observed_consistency',side_effect=AssertionError('Observed called')) as obs, \
             patch('geophy_set3d.training.measurement_engine.compute_heldout_consistency',side_effect=AssertionError('Hold called')) as hold:
            result=run_control_epoch(model,[batch],config,optimizer)
            obs.assert_not_called();hold.assert_not_called()
        self.assertEqual(result['loss']['total'],result['loss']['supervised'])
        for k in ('observed','heldout','observed_weighted','heldout_weighted'):self.assertEqual(result['loss'][k],0)
        for a,b in zip(model.parameters(),expected.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_candidate_selection_prefers_smaller_lambda_when_close(self):
        comparisons={n:dict(eligible=True,heldout_improvement_absolute=v) for n,v in zip(CANDIDATES,[.002,.0025,.0028])}
        self.assertEqual(choose_candidate(comparisons),'weak')
        comparisons['strong']['heldout_improvement_absolute']=.005
        self.assertEqual(choose_candidate(comparisons),'strong')
        for r in comparisons.values():r['eligible']=False
        self.assertIsNone(choose_candidate(comparisons))

    def test_paired_metrics_reject_station_mismatch(self):
        row=dict(case_id='case',input_station_ids=[0,1,2,3],hold_station_ids=[4],pool='train',draw=0,
                 relative_l1=.1,observed=dict(valid_rays=2),heldout=dict(valid_rays=2,relative_l1=.2))
        other=copy.deepcopy(row);other['relative_l1']=.09;other['heldout']['relative_l1']=.19
        paired=pair_rows([other],[row]);stats=paired_statistics(paired,'voxel_l1')
        self.assertEqual(stats['improved'],1);self.assertAlmostEqual(stats['mean_paired_change'],-.01)
        other['hold_station_ids']=[5]
        with self.assertRaises(AssertionError):pair_rows([other],[row])


if __name__=='__main__':unittest.main()
