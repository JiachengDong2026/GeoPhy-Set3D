"""CPU algebra tests; actual physics, masks and timing tested by GPU smoke."""
from pathlib import Path
import sys,unittest,copy
from unittest.mock import patch
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import mrgr_two_step_v2 as m

class Lift(nn.Module):
    def forward(self,x,b):
        return x.mean((-1,-2))[...,None,None,None].expand(-1,-1,-1,2,2,2),b['view_mask'][:,:,None,None,None].expand(-1,-1,2,2,2)
class Fusion(nn.Module):
    def forward(self,x,g,visibility,mask):return x.mean(1),None
class FakeCore(nn.Module):
    def __init__(self):
        super().__init__();self.residual_encoder=nn.Conv2d(3,16,1);self.lifting=Lift();self.fusion=Fusion()
        self.residual_adapter=nn.Sequential(nn.Conv3d(48,32,1),nn.SiLU(),nn.Conv3d(32,32,1))
        nn.init.zeros_(self.residual_adapter[-1].weight);nn.init.zeros_(self.residual_adapter[-1].bias)
        self.decoder=nn.Sequential(nn.Conv3d(32,1,1),nn.Upsample(size=(64,64,64),mode='trilinear',align_corners=True),nn.Sigmoid())
    def geometry_descriptor(self,b):return None
    def forward(self,b,return_stages=False,residual_mode='normal'):
        x=b['input_projection_raw'].mean((-1,-2,-3))[:,:,None,None,None,None].expand(-1,-1,32,2,2,2)/m.Y_SCALE
        visibility=b['view_mask'][:,:,None,None,None].expand(-1,-1,2,2,2)
        x=torch.where(visibility[:,:,None],x,0)
        y=self.decoder(x.mean(1))
        return dict(C0=y,C1=y,context1=x,visibility_base=visibility) if return_stages else y
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def batch(self):
        return {'input_projection_raw':torch.rand(2,2,1,64,64)*100,'input_valid_ray':torch.ones(2,2,1,64,64,dtype=torch.bool),
            'view_mask':torch.ones(2,2,dtype=torch.bool),'input_path_length_m':torch.ones(2,2,1,64,64)}
    def test_factory_shared_state_and_count(self):
        config=m.base.load_config(ROOT/'configs/train_selfsup_oari_mrgr_v1.yaml')
        a=m.model_for(config,'B_one','cpu');b=m.model_for(config,'T_two','cpu')
        self.assertEqual(sum(p.numel() for p in a.parameters()),344678)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertIs(a.decoder,a.core.decoder)
    def test_initial_equivalence_and_intermediate_gradient(self):
        torch.manual_seed(7);core=FakeCore();single=m.TwoStepMRGR(copy.deepcopy(core),1);double=m.TwoStepMRGR(copy.deepcopy(core),2);batch=self.batch()
        projector=lambda f,b:f.mean((2,3,4))[:,None,:,None,None].expand_as(b['input_projection_raw'])*m.Y_SCALE
        with patch.object(m,'project_input',projector):
            self.assertTrue(torch.equal(single(batch),double(batch)))
            # Once the existing shared adapter moves from zero, C1->project->C2
            # has a real, finite gradient. No extra trainable state is added.
            with torch.no_grad():double.core.residual_adapter[-1].weight.normal_(0,.01)
            stages=double(batch,return_stages=True);stages['C_feedback1'].retain_grad()
            stages['C1'].mean().backward()
            self.assertGreater(float(stages['C_feedback1'].grad.abs().sum()),0)
            self.assertTrue(torch.isfinite(stages['C_feedback1'].grad).all())
            for mode in ('partial','empty','no_rays'):
                b=copy.deepcopy(batch)
                if mode=='partial':b['view_mask'][:,1]=False;b['input_valid_ray'][:,1]=False
                elif mode=='empty':b['view_mask'][:]=False;b['input_valid_ray'][:]=False
                else:b['input_valid_ray'][:]=False
                s=double(b,return_stages=True)
                self.assertTrue(torch.isfinite(s['C1']).all())
                if mode=='empty':self.assertEqual(int(s['context2'].count_nonzero()),0)
if __name__=='__main__':unittest.main()
