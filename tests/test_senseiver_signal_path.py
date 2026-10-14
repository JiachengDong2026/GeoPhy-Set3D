import json
from pathlib import Path
import tempfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unittest
import numpy as np
import torch
import yaml
from geophy_set3d.diagnostics.senseiver_signal_path import (intervene,differentiable_input,Moments,Capture,SignalGuard,linear_contributions,model_hashes,valid_measurements)
from geophy_set3d.models.ray_senseiver_ss import RaySenseiverSS,ray_tokens,Y_SCALE


def synthetic():
    raw=torch.tensor([[[[[0.,2.,0.],[4.,0.,6.]]]]])
    valid=torch.tensor([[[[[True,True,False],[True,True,True]]]]])
    direction=torch.zeros(1,1,2,3,3);direction[...,0]=1
    return dict(case_id=['synthetic'],input_projection_raw=raw,input_projection=raw/Y_SCALE,input_valid_ray=valid,view_mask=torch.tensor([[True]]),
        input_station_indices=torch.tensor([[3]]),input_station_origin_m=torch.zeros(1,1,3),input_ray_direction=direction,
        input_t_near=torch.zeros_like(raw),input_path_length_m=torch.full_like(raw,.6))


class SignalPathTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(4)
    def test_fields_identity_wrong_alignment_and_valid_zeros(self):
        original=synthetic();source=dict(original,input_projection_raw=original['input_projection_raw']*2)
        for mode in ['normal','zero','wrong_case','gain_plus','gain_minus']:
            b=intervene(original,mode,source)
            for k in original:
                if k not in ['input_projection_raw','input_projection']:self.assertIs(b[k],original[k])
            tokens,mask,counts=ray_tokens(b);self.assertEqual(int(counts.sum()),5);self.assertEqual(tokens.shape,(1,5,203))
            torch.testing.assert_close(tokens[...,0],b['input_projection_raw'][valid_measurements(b)].reshape(1,-1)/Y_SCALE)
            if mode=='zero':self.assertEqual(int((tokens[...,0]==0).sum()),5)
        bad=dict(source,input_station_indices=torch.tensor([[4]]))
        with self.assertRaises(ValueError):intervene(original,'wrong_case',bad)
        bad=dict(source,input_valid_ray=~source['input_valid_ray'])
        with self.assertRaises(ValueError):intervene(original,'wrong_case',bad)

    def test_real_raw_field_gradient_and_no_buffers_written(self):
        original=synthetic();b,u=differentiable_input(original);tokens,_,_=ray_tokens(b)
        g=torch.autograd.grad(tokens[...,0].sum(),u)[0]
        torch.testing.assert_close(g,valid_measurements(b).float());self.assertIsNone(u.grad)
        self.assertTrue(b['input_projection_raw'].grad_fn is not None)
        # Normalized-only replacement is deliberately not what ray_tokens reads.
        unused=torch.ones_like(u,requires_grad=True);copy=dict(b,input_projection=unused)
        t,_,_=ray_tokens(copy);self.assertIsNone(torch.autograd.grad(t[...,0].sum(),unused,allow_unused=True)[0])

    def test_element_weighted_unequal_chunks_and_null_denominator(self):
        x=np.arange(15,dtype=np.float32).reshape(1,5,3);base=x*.5
        m=Moments();m.add(x[:,:4],base[:,:4]);m.add(x[:,4:],base[:,4:]);r=m.finish()
        self.assertEqual(r['effective_elements'],15);self.assertAlmostEqual(r['mean'],float(x.mean()))
        self.assertAlmostEqual(r['paired']['RMS'],float(np.sqrt(np.mean((x.astype(float)-base)**2))))
        z=Moments();z.add(np.ones((2,3)),np.zeros((2,3)));self.assertIsNone(z.finish()['paired']['relative_RMS'])
        self.assertEqual(z.finish()['paired']['relative_RMS_near_zero_denominator_count'],1)

    def test_real_shared_hook_order_linear_reconstruction_and_cleanup(self):
        model=RaySenseiverSS(query_chunk_size=3).eval();batch=synthetic();before=model_hashes(model)
        with torch.no_grad(),Capture(model,batch) as c:
            latent=model.encode(batch);model.decode_queries(latent,model.query_encoding[:7])
            self.assertEqual(c.calls,['after_cross_1','block_1','after_cross_2','block_2','after_cross_3','block_3'])
            self.assertEqual(c.offsets['decoder.logits'],7)
            self.assertEqual([s[1] for s in c.shapes['decoder.logits']],[3,3,1])
            self.assertTrue(c.decomposition['reconstruction_passed'])
            tokens,_,_=ray_tokens(batch);d=linear_contributions(model,tokens,model.encoder.preproc(tokens))
            self.assertEqual(d['groups']['fourier_geometry']['channels'],[11,203])
        self.assertEqual(before,model_hashes(model));self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules()),0)

    def test_source_measurement_selected_at_target_station_ids(self):
        from diagnose_senseiver_signal_path import aligned_source
        from geophy_set3d.training.selfsup import make_dataset
        from geophy_set3d.data.measurement_only import dual_measurement_collate
        c=yaml.safe_load(Path('configs/ray_senseiver_consistency_probe_v1.yaml').read_text())
        ds=make_dataset(c)
        checkpoints=['/data/djc/GeoPhy-Set3D/outputs/checkpoints/ray_senseiver_consistency_probe_v1/treatment/'+n for n in ['epoch0.pt','last.pt']]
        with tempfile.TemporaryDirectory() as out,SignalGuard(c['dataset']['root'],checkpoints,ds.records,out):
            sample=ds[0];original=dual_measurement_collate([sample])['A']
            aligned=aligned_source(ds,sample,1);_,source=ds._projections(1)
            ids=sample['A']['input_station_indices'].numpy()
            torch.testing.assert_close(aligned['input_projection_raw'][0,:,0],torch.from_numpy(np.array(source[ids],copy=True)),atol=0,rtol=0)
            changed=intervene(original,'wrong_case',aligned)
            self.assertIs(changed['input_valid_ray'],original['input_valid_ray'])
            self.assertTrue(torch.equal(aligned['input_station_indices'],original['input_station_indices']))

    def test_exact_checkpoint_and_four_case_access_boundary(self):
        root=Path(yaml.safe_load(Path('configs/ray_senseiver_consistency_probe_v1.yaml').read_text())['dataset']['root'])
        records=[json.loads(s) for s in (root/'source_index.jsonl').read_text().splitlines() if json.loads(s)['split']=='train']
        checkpoints=['/data/djc/GeoPhy-Set3D/outputs/checkpoints/ray_senseiver_consistency_probe_v1/treatment/'+n for n in ['epoch0.pt','last.pt']]
        with tempfile.TemporaryDirectory() as out,SignalGuard(root,checkpoints,records,out) as guard:
            for p in [root/records[0]['gt_volume_64_path'],root/records[4]['train_projection_path'],Path(checkpoints[0]).parent/'unregistered.pt',Path(checkpoints[0]).parent.parent/'control/last.pt',root/'geometry/ood_geometry.npz']:
                with self.assertRaises(PermissionError):p.open('rb')
            with (root/records[0]['train_projection_path']).open('rb') as f:self.assertTrue(f.read(1))
            self.assertEqual(len(guard.allowed),4)

if __name__=='__main__':unittest.main()
