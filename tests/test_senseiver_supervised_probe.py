import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
import yaml
from geophy_set3d.training.senseiver_supervised_probe import (
    SupervisedProbeGuard,normalized_mse,volume_metrics,init_scale,fresh,separate,
    FixedStations,exit_event,verify_resume)
from geophy_set3d.data.measurement_only import assert_no_gt
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.utils import rng_state,restore_rng


class SupervisedProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.c=yaml.safe_load(Path('configs/ray_senseiver_supervised_probe_v1.yaml').read_text())

    def test_loss_per_sample_normalization_gradient_and_zero_baseline(self):
        g=torch.tensor([[1.,2.],[.01,.02]]).reshape(2,1,1,1,2)
        p=(g*.5).detach().requires_grad_()
        self.assertAlmostEqual(float(normalized_mse(torch.zeros_like(g),g)),1.)
        self.assertAlmostEqual(float(normalized_mse(p,g).detach()),.25)
        normalized_mse(p,g).backward()
        expected=2*(p.detach()-g)/2/2/g.square().flatten(1).mean(1).reshape(2,1,1,1,1)
        torch.testing.assert_close(p.grad,expected)
        self.assertAlmostEqual(volume_metrics(p[0],g[0])['relative_l2'],.5)
        self.assertAlmostEqual(volume_metrics(p[0],g[0])['nmse'],.25)
        self.assertTrue(torch.isfinite(normalized_mse(torch.zeros_like(g),torch.zeros_like(g))))

    def test_scale_label_separation_original_model_and_xyz_complete_grid(self):
        scale=init_scale([torch.full((1,64,64,64),.002)])
        self.assertAlmostEqual(scale['gt_all_voxel_mean'],.002,places=8)
        m=fresh(scale);self.assertEqual(sum(p.numel() for p in m.parameters()),254913)
        self.assertFalse(any('latent_condition' in k for k in m.state_dict()))
        self.assertAlmostEqual(float(m.decoder.postproc.bias.detach()),scale['bias_logit'],places=5)
        x,g=separate(dict(gt_volume_64=torch.ones(1),case_id='x'))
        assert_no_gt(x)
        with self.assertRaises(ValueError):assert_no_gt(dict(gt_volume_64=g))
        coords=m.query_coordinates_m.reshape(64,64,64,3)
        torch.testing.assert_close(coords[0,63,0],torch.tensor([0.,.6,0.]))
        self.assertEqual(m.query_encoding.shape,(64**3,96))
        a,h=FixedStations().sample(64,epoch=999);self.assertEqual(a.tolist(),[26,27,45,42]);self.assertEqual(len(h),0)

    def test_independent_whitelist_and_historical_guard_still_denies_gt(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=[]
            for i in range(10):
                r=dict(split='train' if i<9 else 'val',train_projection_path=f'p{i}.npy',gt_volume_64_path=f'g{i}.npy',source_target_path=str(root/f'G{i}.npy'))
                rows.append(r)
                for n in [f'p{i}.npy',f'g{i}.npy',f'G{i}.npy']:(root/n).write_bytes(b'x')
            (root/'source_index.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            own=root/'own';own.mkdir();(own/'last.pt').write_bytes(b'x')
            with SupervisedProbeGuard(root,own,[0]) as guard:
                (root/'p0.npy').read_bytes();(root/'g0.npy').read_bytes();(own/'last.pt').read_bytes()
                for n in ['g1.npy','G0.npy','p8.npy','p9.npy','foreign.pt','ood_geometry.npz']:
                    with self.assertRaises(PermissionError):(root/n).read_bytes()
            self.assertEqual(len(guard.summary()['denied_attempts']),6)
            with GTReadGuard(root,own):
                with self.assertRaises(PermissionError):(root/'g0.npy').read_bytes()

    def test_pass_plateau_minimum_budget_and_b_dependency(self):
        lim=self.c['stages']['A'];bad=dict(mean=dict(nmse=1.,relative_l1=1.))
        s=exit_event('A',500,[1.]*500,bad,lim,{},0)
        self.assertFalse(s['plateau_condition'])
        s=exit_event('A',600,[1.]*600,bad,lim,s,0)
        self.assertIsNone(s['reason']);self.assertEqual(s['plateau_streak'],1)
        self.assertEqual(exit_event('A',700,[1.]*700,bad,lim,s,0)['reason'],'PLATEAU_FAILED_TO_LEARN')
        improving=[1.]*400+[.8]*200
        self.assertFalse(exit_event('A',600,improving,bad,lim,{},0)['plateau_condition'])
        good=dict(mean=dict(nmse=.08,relative_l1=.25))
        s=exit_event('A',100,[1.]*100,good,lim,{},0)
        self.assertEqual(exit_event('A',200,[1.]*200,good,lim,s,0)['reason'],'PASSED')
        self.assertEqual(exit_event('A',1000,[1.]*1000,bad,lim,{},0)['reason'],'BUDGET_EXHAUSTED_UNRESOLVED')
        self.assertEqual(exit_event('A',100,[1.]*100,bad,lim,{},900)['reason'],'BUDGET_EXHAUSTED_UNRESOLVED')
        lim=self.c['stages']['B'];good=dict(mean=dict(nmse=.1,relative_l1=.3),interventions=[dict(relative_MA=.002),dict(relative_MA=.002),dict(relative_MA=0.)])
        s=exit_event('B',1000,[.1]*1000,good,lim,{},0,.5)
        self.assertEqual(exit_event('B',1200,[.1]*1200,good,lim,s,0,.5)['reason'],'PASSED')
        baddep=copy.deepcopy(good);baddep['interventions'][1]['relative_MA']=0
        s=exit_event('B',1000,[.1]*1000,baddep,lim,{},0,.5)
        self.assertEqual(exit_event('B',1100,[.1]*1100,baddep,lim,s,0,.5)['reason'],'PLATEAU_FAILED_TO_LEARN')

    def test_resume_full_state_rng_and_scientific_stop(self):
        generator=torch.Generator().manual_seed(2);rs=rng_state(generator);a=torch.rand(3);restore_rng(rs,generator);torch.testing.assert_close(a,torch.rand(3),rtol=0,atol=0)
        saved=dict(stage='A',config=self.c,registration_sha256='r',exit_state={},updates=2,history=[{},{}],optimizer={},rng=rs,elapsed_seconds=1.,initial_hash='i')
        verify_resume(saved,'A',self.c,'r')
        for key,val in [('stage','B'),('updates',3),('exit_state',dict(reason='PASSED'))]:
            x=copy.deepcopy(saved);x[key]=val
            with self.assertRaises(ValueError):verify_resume(x,'A',self.c,'r')


if __name__=='__main__':unittest.main()
