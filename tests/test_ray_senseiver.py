import copy
import unittest
from pathlib import Path
import torch
from geophy_set3d.models.ray_senseiver_ss import RaySenseiverSS, ray_tokens, fourier
from geophy_set3d.training.ray_senseiver import load_comparison_config
from geophy_set3d.training.selfsup import validate_selfsup_config, make_dataset
from geophy_set3d.training.deep_comparison import validate_comparison_config
from geophy_set3d.data.collate import projection_pool_collate


class RaySenseiverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(123)
        self.c=load_comparison_config('configs/train_ray_senseiver_selfsup_v1.yaml')

    def test_full_chunk_prediction_and_gradient(self):
        a=RaySenseiverSS();b=copy.deepcopy(a)
        latent=torch.randn(1,64,64,requires_grad=True);latent2=latent.detach().clone().requires_grad_()
        q=fourier(torch.rand(259,3)*2-1).requires_grad_();q2=q.detach().clone().requires_grad_()
        x=a.decode_queries(latent,q,chunk_size=259,checkpoint_chunks=False)
        y=b.decode_queries(latent2,q2,chunk_size=37,checkpoint_chunks=True)
        torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        weights=torch.linspace(.1,1,259).reshape(1,-1,1)
        (x*weights).sum().backward();(y*weights).sum().backward()
        torch.testing.assert_close(latent.grad,latent2.grad,atol=2e-5,rtol=2e-4)
        torch.testing.assert_close(q.grad,q2.grad,atol=2e-6,rtol=2e-4)
        self.assertTrue((q2.grad.abs().sum(1)>0).all())
        for (n,p),(n2,p2) in zip(a.decoder.named_parameters(),b.decoder.named_parameters()):
            self.assertEqual(n,n2);torch.testing.assert_close(p.grad,p2.grad,atol=3e-5,rtol=2e-4)

    def test_official_core_and_sharing(self):
        # Execute pinned source with dependency-only substitutions; original attention path.
        s=Path('third_party/senseiver/model.py').read_text().replace('from fairscale.nn import checkpoint_wrapper','')
        ns={};exec(compile(s,'official/model.py','exec'),ns)
        official=ns['Encoder'](203,64,64,64,num_layers=3,num_cross_attention_heads=4,num_self_attention_heads=4,num_self_attention_layers_per_block=3)
        m=RaySenseiverSS();m.encoder.load_state_dict(official.state_dict())
        x=torch.randn(2,17,203);mask=torch.zeros(2,17,dtype=torch.bool);mask[1,10:]=True
        torch.testing.assert_close(m.encoder(x,mask),official(x,mask),atol=3e-6,rtol=3e-5)
        officialdec=ns['Decoder'](96,64,64,1,1,num_cross_attention_heads=4)
        m.decoder.load_state_dict(officialdec.state_dict());latent=torch.randn(2,64,64);coords=torch.randn(2,23,96)
        torch.testing.assert_close(m.decoder(latent,coords),officialdec(latent,coords),atol=2e-6,rtol=2e-5)
        calls=[];h=m.encoder.layer_n.register_forward_hook(lambda *args:calls.append(1));m.encoder(x,mask);h.remove();self.assertEqual(len(calls),2)

    def test_real_geometry_zero_and_ray_permutation(self):
        ds=make_dataset(self.c);batch=projection_pool_collate([ds[0]['A']])
        valid=batch['input_valid_ray'][:,:,0] & batch['view_mask'][:,:,None,None]
        batch['input_projection_raw'].zero_();tokens,mask,counts=ray_tokens(batch)
        self.assertEqual(int(counts.sum()),int(valid.sum()));self.assertEqual(tokens.shape[1],int(valid.sum()))
        self.assertEqual(float(tokens[:,:,0].abs().max()),0.)
        self.assertTrue((tokens[:,:,1:7].abs()<1.0001).all())
        self.assertTrue(torch.allclose(tokens[:,:,7:10].norm(dim=-1),torch.ones_like(tokens[:,:,0]),atol=2e-6))
        model=RaySenseiverSS().eval();x=model.encoder(tokens,mask)
        perm=torch.randperm(tokens.shape[1]);y=model.encoder(tokens[:,perm],mask[:,perm]);torch.testing.assert_close(x,y,atol=4e-6,rtol=3e-5)
        grid=model.query_coordinates_m.reshape(64,64,64,3)
        torch.testing.assert_close(grid[0,32,0],torch.tensor([0.,32*.6/63,0.]),atol=1e-7,rtol=1e-6)
        batch['input_valid_ray'].zero_();tokens,mask,counts=ray_tokens(batch);self.assertEqual(int(counts.sum()),0);self.assertEqual(tokens.shape,(1,1,203));self.assertFalse(mask.any())

    def test_strict_independent_validation(self):
        for old in [validate_selfsup_config,validate_comparison_config]:
            with self.assertRaises(ValueError):old(self.c)
        from geophy_set3d.training.ray_senseiver import validate_comparison_config as validate
        for key,value in [('num_latents',32),('space_bands',8),('num_layers',4)]:
            c=copy.deepcopy(self.c);c['model'][key]=value
            with self.assertRaises(ValueError):validate(c)

if __name__=='__main__':unittest.main()
