"""CPU invariants for global-only, mask-safe scale fusion."""
import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mrgr_scale_fusion_v2 as candidate
class Tests(unittest.TestCase):
    def test_global_mask_gradient_and_no_O_values(self):
        torch.set_num_threads(2);torch.manual_seed(8)
        m=candidate.GlobalScaleFusion();m.record_diagnostics=True
        self.assertEqual(sum(p.numel() for p in m.parameters()),419)
        x=[torch.randn(2,12,32,32,32) for _ in range(3)]
        d=torch.rand(2,5,32,32,32);d[:,4]=1;d[:,:,:,16:]=0
        y=m(x,d);self.assertEqual(float(y.abs().sum()),0.)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0)
        target=torch.randn_like(y)
        for step in range(2):
            opt.zero_grad();loss=(m(x,d)-target).square().mean();loss.backward()
            for n,p in m.named_parameters():
                self.assertTrue(torch.isfinite(p.grad).all())
                if step or n!='logits':self.assertGreater(float(p.grad.abs().sum()),0.)
            opt.step()
        y=m(x,d);changed=d.clone();changed[:,:4]=0
        self.assertTrue(torch.equal(y,m(x,changed)))
        self.assertTrue(torch.equal(y.masked_select(~(d[:,4:5]>0)),torch.zeros_like(y.masked_select(~(d[:,4:5]>0)))))
        self.assertTrue(torch.equal(m(x,torch.zeros_like(d)),torch.zeros_like(y)))
        self.assertGreater(float(m.logits.abs().sum()),0.)
if __name__=='__main__':unittest.main()
