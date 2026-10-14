import copy
from pathlib import Path
import sys
import unittest
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from geophy_set3d.training.selfsup import make_dataset,move_dual_batch
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.training.senseiver_consistency_probe import validate,objective,verify_resume
from geophy_set3d.training.senseiver_init_probe import TrainEightGuard
from diagnose_senseiver_consistency_probe import gradient_split,amplitude_probe

class ConsistencyProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.c=yaml.safe_load(Path('configs/ray_senseiver_consistency_probe_v1.yaml').read_text())

    def test_same_graph_parameter_output_equivalence_and_amplitude(self):
        ds=make_dataset(self.c)
        with TrainEightGuard(self.c['dataset']['root'],Path('/tmp/consistency_probe_test'),ds.records):
            b=move_dual_batch(dual_measurement_collate([ds[0]]),'cpu')
            class Tiny(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.encoder=torch.nn.Linear(1,1)
                    self.encoder.latent=torch.nn.Parameter(torch.tensor(.001))
                    self.decoder=torch.nn.Linear(1,1)
                def forward(self,x):
                    v=self.decoder(self.encoder(torch.ones(1,1))+self.encoder.latent).sigmoid()*.003
                    v=v+(1e-5 if x is b['B'] else 0.)
                    return v.expand(1,1,64,64,64)
            m=Tiny();on=objective(m,b,.05)
            split=gradient_split(m,on,b,True)
            self.assertTrue(split['equivalence']['loss_parts_bitwise_equal'])
            self.assertLess(split['only_difference']['gradient_remainder_l2'],1e-5)
            off=objective(m,b,0.)
            a=torch.autograd.grad(off[0],tuple(m.parameters()),retain_graph=True)
            z=torch.autograd.grad(off[4]['measurement_smooth'],tuple(m.parameters()))
            for x,y in zip(a,z):torch.testing.assert_close(x,y)
            self.assertGreater(float(off[4]['measurement_relative_raw'].detach()),0.)
            for row in amplitude_probe(on[2],b,[1.,.1,.01]):
                self.assertGreater(row['branches']['consistency_weighted']['dL_dalpha'],0.)
                torch.testing.assert_close(torch.tensor(row['branches']['consistency_weighted']['dL_dalpha']),torch.tensor(row['analytic_consistency_weighted_dL_dalpha']),rtol=1e-3,atol=2e-7)
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            with self.assertRaises(ValueError):objective(m,b,.01)

    def test_frozen_configuration_and_own_resume(self):
        validate(self.c)
        for section,key,value in [('loss','relative_l1',.1),('loss','consistency',0.),('intervention','control_consistency',.1),('training','seed',1)]:
            c=copy.deepcopy(self.c);c[section][key]=value
            with self.assertRaises(ValueError):validate(c)
        saved=dict(config=self.c,arm='treatment',relative_l1=0.,consistency_weight=0.,initial_hash='i',registration_sha256='r',source_manifest_sha256='s',epoch=25,updates=100,scheduler=dict(last_epoch=25),exit_state=dict(failure_streak=1))
        verify_resume(saved,self.c,'treatment','i','r','s')
        self.assertEqual(saved['exit_state']['failure_streak'],1)
        for key,val in [('relative_l1',.1),('consistency_weight',.05),('arm','control'),('updates',101),('exit_state',dict(reason='FAILED_TO_LEARN'))]:
            bad=copy.deepcopy(saved);bad[key]=val
            with self.assertRaises(ValueError):verify_resume(bad,self.c,'treatment','i','r','s')

if __name__=='__main__':unittest.main()
