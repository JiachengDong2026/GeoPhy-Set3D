"""Explicit small fixtures validate contrast mechanics; never substitute experiment data."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from pool_fixture import make_pool
from geophy_set3d.training import selfsup_v2 as v2
from geophy_set3d.training.rvtc_matched import (contrast_loss, state_hash, validate_pair,
    ContrastReadGuard, diagnostic_context, paired_effect, applied_gradient_geometry)
from geophy_set3d.training.utils import seed_everything, load_config, rng_state, restore_rng


class MatchedContrastTests(unittest.TestCase):
    def pair(self):
        return [yaml.safe_load(Path(f'configs/selfsup_rvtc_matched_{arm}.yaml').read_text())
                for arm in ('control','treatment')]

    def test_only_arm_weight_differs_and_finite_budget(self):
        pair=self.pair();validate_pair(pair)
        for key,value in [('epochs',300),('shuffle',True),('scheduler','cosine'),('seed',42),('optimizer_steps',321)]:
            bad=copy.deepcopy(pair)
            for cfg in bad:cfg[key]=value
            with self.assertRaises(ValueError):validate_pair(bad)
        bad=copy.deepcopy(pair);bad[1]['device']='cuda:3'
        with self.assertRaises(ValueError):validate_pair(bad)

    def test_exact_loss_control_has_no_consistency_dependency(self):
        parts={key:torch.tensor(float(i+1),requires_grad=True) for i,key in enumerate(('hold','cross','observed','hold_teacher','observed_teacher','teacher'))}
        frozen=copy.deepcopy(v2.FROZEN_LOSS)
        c,t=contrast_loss(parts,0),contrast_loss(parts,.1)
        measure=parts['hold']+.5*parts['cross']+.1*parts['observed']+parts['hold_teacher']+.1*parts['observed_teacher']
        torch.testing.assert_close(c,measure);torch.testing.assert_close(t,measure+.1*parts['teacher'])
        cg,=torch.autograd.grad(c,parts['teacher'],allow_unused=True,retain_graph=True)
        tg,=torch.autograd.grad(t,parts['teacher']);self.assertIsNone(cg);self.assertAlmostEqual(float(tg),.1)
        self.assertEqual(v2.FROZEN_LOSS,frozen)
        with self.assertRaises(ValueError):contrast_loss(parts,.2)

    def test_fresh_seed_tensor_identity_without_loading(self):
        cfg=load_config('configs/train_selfsup_oari_mrgr_v2.yaml');cfg['training']['device']='cpu'
        with patch('torch.load',side_effect=AssertionError('No external initialization')):
            seed_everything(20261011);a=v2.initialize_fresh_model(cfg)
            seed_everything(20261011);b=v2.initialize_fresh_model(cfg)
        self.assertEqual(sum(p.numel() for p in a.parameters()),332070)
        self.assertTrue(all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()))
        self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))

    def test_control_applied_zero_and_separate_hypothetical(self):
        vectors=dict(student_measurement=torch.tensor([1.,0.]),teacher_measurement=torch.tensor([1.,1.]),consistency=torch.tensor([-.2,0.]))
        c=applied_gradient_geometry(vectors,'Control');t=applied_gradient_geometry(vectors,'Treatment')
        self.assertEqual(c['applied']['consistency_norm'],0)
        self.assertIsNone(c['applied']['cosine_measurement_consistency'])
        self.assertGreater(c['hypothetical_original_weight_consistency']['consistency_norm'],0)
        self.assertLess(t['applied']['cosine_measurement_consistency'],0)

    def test_diagnostics_preserve_rng_state_and_mode(self):
        model=torch.nn.Linear(2,1).train();before=state_hash(model.state_dict());seed_everything(17)
        state=torch.get_rng_state().clone()
        with diagnostic_context(model):
            loss=model(torch.ones(2)).sum();torch.autograd.grad(loss,tuple(model.parameters()))
        self.assertTrue(model.training);self.assertEqual(before,state_hash(model.state_dict()))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        with self.assertRaises(AssertionError):
            with diagnostic_context(model):torch.rand(1)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_pair_bootstrap_decisions_and_case_unit(self):
        for offset,decision in [(-.1,'extra_measurement_benefit'),(0.,'inconclusive'),(.1,'measurement_harm')]:
            result=paired_effect([1.,2.,3.],np.array([1.,2.,3.])+offset)
            self.assertEqual(result['decision'],decision);self.assertEqual(result['case_count'],3)
        result=paired_effect([1.,1.,1.],[.9,1.,1.2])
        self.assertEqual(result['decision'],'inconclusive')
        self.assertEqual(result,paired_effect([1.,1.,1.],[.9,1.,1.2]))
        with self.assertRaises(ValueError):paired_effect([1.],[1.,2.])

    def test_own_checkpoint_only_guard_and_test_pool_denied(self):
        temp,root,records,_=make_pool()
        try:
            own=root.parent/'own';own.mkdir();c=own/'Control'/'last.pt';c.parent.mkdir();c.write_bytes(b'fixture')
            guard=ContrastReadGuard(root,own,[c])
            with guard:
                self.assertEqual(c.read_bytes(),b'fixture')
                c.with_suffix('.tmp').write_bytes(b'fixture')
                for p in [own/'Treatment'/'last.pt',root.parent/'supervised.pt',root/records[0]['gt_volume_64_path'],
                          root/records[0]['ood_projection_path'],root/'geometry/ood_geometry.npz',root/'test_projection.npy']:
                    with self.assertRaises(PermissionError):p.open('rb')
            self.assertEqual(guard.summary()['gt_open_attempts'],1)
            self.assertEqual(guard.summary()['Pool_B_access_attempts'],2)
        finally:temp.cleanup()

    def test_checkpoint_optimizer_rng_resume_equivalence(self):
        # Small optimizer fixture establishes exact restoration; not an experimental arm.
        seed_everything(9);m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4)
        gen=torch.Generator();x=torch.ones(2)
        def step(model,opt):
            (model(x)**2).sum().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
        step(m,o);saved=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),rng=rng_state(gen)))
        step(m,o);expected=state_hash(m.state_dict())
        n=torch.nn.Linear(2,1);q=torch.optim.AdamW(n.parameters(),lr=2e-4,weight_decay=1e-4)
        n.load_state_dict(saved['model']);q.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],gen);step(n,q)
        self.assertEqual(expected,state_hash(n.state_dict()))

    def test_accumulation_two_equals_pair_mean_and_four_updates(self):
        seed_everything(4);a=torch.nn.Linear(1,1);b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=2e-4);ob=torch.optim.AdamW(b.parameters(),lr=2e-4)
        data=torch.arange(8.).reshape(8,1);updates=0
        for i in range(0,8,2):
            for sample in data[i:i+2]:(a(sample).square().sum()/2).backward()
            b(data[i:i+2]).square().mean().backward()
            for m,o in [(a,oa),(b,ob)]:
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step();o.zero_grad(set_to_none=True)
            updates+=1
        self.assertEqual(updates,4)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)


if __name__=='__main__':unittest.main()
