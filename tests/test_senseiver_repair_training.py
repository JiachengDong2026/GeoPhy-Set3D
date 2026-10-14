import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from geophy_set3d.models.ray_senseiver_ss import RaySenseiverSS
from geophy_set3d.models.ray_senseiver_latent_residual import add_latent_residual
from geophy_set3d.training.senseiver_single_repair import exit_event,verify_resume
from geophy_set3d.training.senseiver_readout import ProbeGuard


class RepairTests(unittest.TestCase):
    def test_only_added_zero_residual_shared_initial_function_and_gradient(self):
        torch.set_num_threads(2);torch.manual_seed(20261011);a=RaySenseiverSS();b=add_latent_residual(a)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters()),4096)
        latent=torch.randn(2,64,64);coords=a.query_encoding[:7].unsqueeze(0).expand(2,-1,-1)
        torch.testing.assert_close(a.decoder(latent,coords),b.decoder(latent,coords),rtol=0,atol=0)
        b.decoder(latent,coords).square().mean().backward()
        self.assertGreater(float(b.decoder.latent_condition.weight.grad.norm()),0)
        with torch.no_grad():b.decoder.latent_condition.weight.fill_(.03)
        self.assertGreater(float((a.decoder(latent,coords)-b.decoder(latent,coords)).abs().max().detach()),0)

    def fixture(self,epoch=25,l1=1.,l2=1.,dependence=0):
        history=[dict(eval=dict(measurement=dict(heldout=dict(relative_l1=l1,relative_l2=l2)))) for _ in range(epoch)]
        h=dict(full_fixed_subset=dict(measurement=dict(heldout=dict(relative_l1=l1)),output=dict(mean=.001)),
            logits=dict(fraction_below_minus12=0.),diagnostic_cases=[dict(measurement_dependence=dict(relative_MA=.001 if i<dependence else 0.)) for i in range(3)])
        return history,h

    def test_exit_windows_budget80_and_dependency_prevents_false_stop(self):
        rows,h=self.fixture();s=exit_event(25,rows,h,.0015,{})
        self.assertEqual(s['failure_streak'],1);self.assertIsNone(s['reason'])
        self.assertEqual(exit_event(30,rows,h,.0015,s)['reason'],'FAILED_TO_LEARN')
        rows,h=self.fixture(40,dependence=2);s=exit_event(40,rows,h,.0015,{})
        self.assertIsNone(s['reason']);self.assertEqual(exit_event(80,rows,h,.0015,s)['reason'],'BUDGET_COMPLETE')
        rows,h=self.fixture(25);rows[-5:]=[dict(eval=dict(measurement=dict(heldout=dict(relative_l1=1.,relative_l2=.9))))]*5
        self.assertFalse(exit_event(25,rows,h,.0015,{})['failure_condition'])
        rows,h=self.fixture(20);self.assertFalse(exit_event(20,rows,h,.0015,{})['failure_condition'])
        h['logits']['fraction_below_minus12']=.96;h['full_fixed_subset']['output']['mean']=1e-8
        self.assertFalse(exit_event(5,rows,h,.0015,{})['collapse_condition'])
        s=exit_event(10,rows,h,.0015,{})
        self.assertEqual(exit_event(15,rows,h,.0015,s)['reason'],'NEAR_ZERO_COLLAPSE')

    def test_resume_own_arm_full_state_and_scientific_stop_boundary(self):
        c={'a':1};s=dict(config=c,arm='control',initial_hash='i',registration_sha256='r',source_manifest_sha256='m',relative_l1=0.,consistency_weight=0.,exit_state={},updates=160,epoch=40,scheduler=dict(last_epoch=40))
        verify_resume(s,c,'control','i','r','m')
        for key,value in [('arm','treatment'),('exit_state',{'reason':'FAILED_TO_LEARN'}),('updates',159)]:
            bad=copy.deepcopy(s);bad[key]=value
            with self.assertRaises(ValueError):verify_resume(bad,c,'control','i','r','m')

    def test_stage_guards_fit_check_and_reconstruction_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=[]
            for i in range(17):
                records.append(dict(split='train',train_projection_path=f'p{i}.npy',gt_volume_64_path=f'g{i}.npy',source_target_path=str(root/f'G{i}.npy')))
                (root/f'p{i}.npy').write_bytes(b'x')
            (root/'source_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));cp=root/'ck'/'last.pt';cp.parent.mkdir();cp.write_bytes(b'x')
            with ProbeGuard(root,cp.parent,records,16,cp) as guard:
                (root/'p15.npy').read_bytes();cp.read_bytes()
                with self.assertRaises(PermissionError):(root/'p16.npy').read_bytes()
                with self.assertRaises(PermissionError):(root/'g0.npy').read_bytes()
                with self.assertRaises(PermissionError):(cp.parent/'other.pt').read_bytes()
            with ProbeGuard(root,root/'own',records,8):
                (root/'p7.npy').read_bytes()
                with self.assertRaises(PermissionError):(root/'p8.npy').read_bytes()
                with self.assertRaises(PermissionError):cp.read_bytes()


if __name__=='__main__':unittest.main()
