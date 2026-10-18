"""CPU-only independent OCMSF invariants; no dataset or GT access."""
import copy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import mrgr_ocmsf_v1 as candidate


class OCMSFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_descriptor_order_and_validation(self):
        for b in (1,2):
            r=3
            raw=torch.zeros(b*r**3,5)
            raw[:,0]=torch.arange(len(raw))/len(raw)
            volume=candidate.descriptor_volume(raw,b,r)
            for bi,x,y,z in ((0,1,0,2),(b-1,2,1,0)):
                expected=raw[bi*r**3+x*r*r+y*r+z,0]
                self.assertEqual(volume[bi,0,x,y,z],expected)
            self.assertEqual(volume.shape,(b,5,r,r,r))
        for bad in (torch.zeros(27,4),torch.zeros(27,5,dtype=torch.int64),
                    torch.full((27,5),float('nan')),torch.full((27,5),1.5)):
            with self.assertRaises(ValueError):candidate.descriptor_volume(bad,1,3)

    def test_masks_shapes_finite_and_learning(self):
        torch.manual_seed(41)
        for b in (1,2):
            scales=[torch.randn(b,12,32,32,32) for _ in range(3)]
            for conditioned in (False,True):
                m=candidate.ScaleFusion(conditioned)
                for mode in ('full','partial','empty'):
                    descriptor=torch.zeros(b,5,32,32,32)
                    descriptor[:,:3]=1/3;descriptor[:,3:]=1
                    if mode=='partial':descriptor[:,:,:,12:]=0
                    elif mode=='empty':descriptor[:]=0
                    m.record_diagnostics=True
                    output=m(scales,descriptor)
                    self.assertEqual(output.shape,(b,32,32,32,32))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(int(torch.count_nonzero(output)),0)
                    visible=descriptor[:,4:5]>0
                    self.assertTrue(torch.equal(output.masked_select(~visible),torch.zeros_like(output.masked_select(~visible))))
                    self.assertTrue(torch.allclose(m.last_diagnostics['alpha'].sum(1),torch.ones_like(descriptor[:,0])))
                    if mode=='empty':
                        self.assertTrue(torch.allclose(m.last_diagnostics['alpha'],torch.full_like(m.last_diagnostics['alpha'],1/3)))
                # Both arms have exact same parameter count, all parameters used.
                self.assertEqual(sum(p.numel() for p in m.parameters()),739)
                descriptor=torch.rand(b,5,32,32,32);descriptor[:,4]=1
                optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0)
                target=torch.randn(b,32,32,32,32)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss=(m(scales,descriptor)-target).square().mean();loss.backward()
                    for name,p in m.named_parameters():
                        self.assertIsNotNone(p.grad,name)
                        self.assertTrue(torch.isfinite(p.grad).all(),name)
                        if step==0 and not name.startswith('output.'):
                            self.assertEqual(int(torch.count_nonzero(p.grad)),0,name)
                        else:self.assertGreater(float(p.grad.abs().sum()),0,name)
                    optimizer.step()
                output=m(scales,descriptor)
                self.assertGreater(float(output.abs().sum()),0)
                alpha=m.last_diagnostics['alpha']
                self.assertGreater(float((alpha-1/3).abs().mean()),0)
                if conditioned:
                    old=output.detach().clone()
                    changed=descriptor.clone();changed[:,:4]=0
                    self.assertGreater(float((m(scales,changed)-old).abs().sum()),0)
                with self.assertRaises(ValueError):m(scales,descriptor.double())
                broken=descriptor.clone();broken[0,0,0,0,0]=float('nan')
                with self.assertRaises(ValueError):m(scales,broken)

    def test_decoder_preserves_existing_nonzero_context(self):
        torch.manual_seed(42)
        decoder=candidate.base.ContextDecoder(candidate.base.build_model(candidate.base.load_config(ROOT/'configs/train_selfsup_oari_mrgr_v1.yaml')).decoder)
        with torch.no_grad():
            decoder.context.output.weight.normal_(0,.01)
            decoder.context.output.bias.normal_(0,.01)
        x=torch.randn(1,32,32,32,32)
        decoder.eval()
        with torch.no_grad():expected=decoder(x)
        for conditioned in (False,True):
            extended=candidate.OCMSFDecoder(copy.deepcopy(decoder),conditioned)
            raw=torch.full((32**3,5),.5)
            extended.capture(raw)
            with torch.no_grad():actual=extended(x)
            self.assertTrue(torch.equal(expected,actual))
            self.assertEqual(len([m for m in extended.modules() if isinstance(m,candidate.ScaleFusion)]),1)


if __name__=='__main__':unittest.main()
