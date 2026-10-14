import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
import yaml
from geophy_set3d.models.ray_senseiver_ss import RaySenseiverSS
from geophy_set3d.physics.projectors import mass_to_mole
from geophy_set3d.training.senseiver_init_probe import validate,fitted_scale,initialize_pair,exit_event,ObservedAdamW,TrainEightGuard
from geophy_set3d.training.ray_senseiver import validate_comparison_config
from geophy_set3d.training.utils import atomic_checkpoint,rng_state,restore_rng


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.c=yaml.safe_load(Path('configs/ray_senseiver_init_probe_v1.yaml').read_text())

    def test_scale_units_valid_zeros_and_no_fallback(self):
        l=torch.tensor([.2,.4,.8],dtype=torch.float64);chi=.002
        scale=fitted_scale(l,1e6*l*chi)
        self.assertAlmostEqual(scale['chi_init'],chi,places=15)
        self.assertAlmostEqual(float(mass_to_mole(scale['C_init'])),chi,places=15)
        self.assertFalse(scale['clamp_applied'])
        zero=fitted_scale(l,torch.tensor([0.,0.,1600.],dtype=torch.float64))
        self.assertAlmostEqual(zero['chi_init'],.8*1600/(1e6*float(l.square().sum())),places=15)
        self.assertEqual(zero['valid_rays'],3)
        for length,y in [([0.],[1.]),([1.],[0.]),([1.],[float('nan')]),([1.],[2e6])]:
            with self.assertRaises(ValueError):fitted_scale(length,y)
        self.assertTrue(fitted_scale([1.],[1e-6])['clamp_applied'])

    def test_only_final_layer_and_nonzero_weights(self):
        torch.manual_seed(20261011);base=RaySenseiverSS()
        scale=fitted_scale([.2,.4],[100.,200.]);a,b=initialize_pair(base,scale)
        for n,p in a.named_parameters():
            if not n.startswith('decoder.postproc.'):self.assertTrue(torch.equal(p,b.state_dict()[n]))
        torch.testing.assert_close(b.decoder.postproc.weight,a.decoder.postproc.weight*.01,rtol=0,atol=0)
        self.assertTrue(torch.count_nonzero(b.decoder.postproc.weight)>0)
        self.assertAlmostEqual(float(b.decoder.postproc.bias.detach()),scale['bias_logit'],places=5)
        self.assertTrue(torch.equal(base.decoder.postproc.weight,a.decoder.postproc.weight))

    def test_strict_schema_keeps_old_run_validator(self):
        validate(self.c)
        with self.assertRaises((ValueError,KeyError)):validate_comparison_config(self.c)
        old=yaml.safe_load(Path('configs/train_ray_senseiver_selfsup_v1.yaml').read_text())
        old['training']['max_epochs']=40
        with self.assertRaises(ValueError):validate_comparison_config(old)
        for section,key,v in [('training','max_epochs',41),('probe','weight_factor',.1),('scheduler','T_max',40),('loss','consistency',0),('train','split','val')]:
            c=copy.deepcopy(self.c);c[section][key]=v
            with self.assertRaises(ValueError):validate(c)

    def test_eligible_consecutive_boundaries_and_windows(self):
        def health(h=1.,mean=1e-8,log=.99):return dict(full_fixed_subset=dict(measurement=dict(heldout=dict(relative_l1=h)),output=dict(mean=mean)),logits=dict(fraction_below_minus12=log))
        history=[dict(eval=dict(measurement=dict(heldout=dict(relative_l1=1.)))) for _ in range(40)]
        state=exit_event(5,history[:5],health(),.001,{})
        self.assertEqual(state['collapse_streak'],0)
        state=exit_event(10,history[:10],health(),.001,state);self.assertIsNone(state['reason'])
        state=exit_event(15,history[:15],health(),.001,state);self.assertEqual(state['reason'],'NEAR_ZERO_COLLAPSE')
        state=exit_event(20,history[:20],health(mean=.001,log=0),.001,{})
        self.assertEqual(state['failure_streak'],0)
        state=exit_event(25,history[:25],health(mean=.001,log=0),.001,state);self.assertIsNone(state['reason'])
        state=exit_event(30,history[:30],health(mean=.001,log=0),.001,state);self.assertEqual(state['reason'],'FAILED_TO_LEARN')
        state=exit_event(35,history[:35],health(h=.8,mean=.001,log=0),.001,state)
        # Full healthy H does not override a flat >=.98 recent5: use actual improved windows.
        improved=[dict(eval=dict(measurement=dict(heldout=dict(relative_l1=.8)))) for _ in range(40)]
        state=exit_event(35,improved[:35],health(h=.8,mean=.001,log=0),.001,{})
        self.assertIsNone(state['reason'])
        self.assertEqual(exit_event(40,improved,health(h=.8,mean=.001,log=0),.001,state)['reason'],'BUDGET_COMPLETE')

    def test_full_own_last_resume_matches_next_update(self):
        class Small(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Linear(2,2)
                self.encoder.register_parameter('latent',torch.nn.Parameter(torch.randn(2)))
                self.decoder=torch.nn.Linear(2,1)
            def forward(self):return self.decoder(self.encoder(torch.ones(1,2))+self.encoder.latent).square().sum()
        torch.manual_seed(3);a=Small();opt=ObservedAdamW(a,lr=2e-4,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300);gen=torch.Generator().manual_seed(3)
        a().backward();opt.step();opt.zero_grad(set_to_none=True);sch.step()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'last.pt';atomic_checkpoint(p,dict(model=a.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=rng_state(gen),updates=opt.updates))
            saved=torch.load(p,weights_only=False);b=Small();other=ObservedAdamW(b,lr=2e-4,weight_decay=1e-4);s2=torch.optim.lr_scheduler.CosineAnnealingLR(other,T_max=300)
            b.load_state_dict(saved['model']);other.load_state_dict(saved['optimizer']);s2.load_state_dict(saved['scheduler']);other.updates=saved['updates'];restore_rng(saved['rng'],gen)
            for m,o,s in [(a,opt,sch),(b,other,s2)]:m().backward();o.step();o.zero_grad(set_to_none=True);s.step()
            self.assertEqual(other.updates,2);self.assertEqual(s2.last_epoch,2)
            for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertTrue(all(v>0 for v in other.observations[-1]['update_l2'].values()))

    def test_access_boundary_denies_gt_val_test_poolB_foreign(self):
        root=Path(self.c['dataset']['root']);records=[json.loads(l) for l in (root/'source_index.jsonl').read_text().splitlines()];train=[r for r in records if r['split']=='train']
        with tempfile.TemporaryDirectory() as tmp:
            guard=TrainEightGuard(root,tmp,train)
            with guard:
                for p in [root/train[0]['gt_volume_64_path'],Path(train[0]['source_target_path']),root/train[8]['train_projection_path'],root/next(r for r in records if r['split']=='val')['train_projection_path'],root/'geometry/ood_geometry.npz',Path('/tmp/foreign.pt')]:
                    with self.assertRaises(PermissionError):p.open('rb')
                with (root/train[0]['train_projection_path']).open('rb') as f:self.assertTrue(f.read(1))
            self.assertEqual(guard.summary()['gt_open_attempts'],2)
            self.assertEqual(guard.summary()['supervised_checkpoint_open_attempts'],1)
            self.assertEqual(guard.summary()['unique_pool_a_projection_files'],1)


if __name__=='__main__':unittest.main()
