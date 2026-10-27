"""Measurement-only amplitude normalization around an unchanged classical network."""
from feasible_context import *


def amplitude(batch):
    valid=batch['input_valid_ray'].bool()&batch['view_mask'][:,:,None,None,None]
    y=torch.where(valid,batch['input_projection_raw'],0).double()
    ell=torch.where(valid,batch['input_path_length_m'],0).double()
    chi=(ell*y).flatten(1).sum(1)/(1e6*ell.square().flatten(1).sum(1).clamp_min(1e-30))
    return chi.clamp(0,1-1e-8).float()


def canonical(batch,reference_chi):
    chi=amplitude(batch);factor=torch.where(chi>0,reference_chi/chi.clamp_min(1e-30),torch.zeros_like(chi))
    result=dict(batch)
    for key in ['input_projection_raw','input_projection']:result[key]=batch[key]*factor[:,None,None,None,None]
    mass=chi*16.043/(chi*16.043+(1-chi)*28.965)
    ref=reference_chi*16.043/(reference_chi*16.043+(1-reference_chi)*28.965)
    ratio=(mass/(1-mass).clamp_min(1e-8))/(ref/(1-ref))
    return result,ratio


class AmplitudeModel(torch.nn.Module):
    def __init__(self,base,reference_chi):super().__init__();self.base=base;self.reference_chi=float(reference_chi)
    def forward(self,batch):
        b,ratio=canonical(batch,self.reference_chi)
        p=self.base(b);scale=ratio[:,None,None,None,None]
        # Algebraically sigmoid(logit(p)+log(scale)), with exact zero when input amplitude is zero.
        return (p*scale)/(1-p+p*scale).clamp_min(1e-30)
