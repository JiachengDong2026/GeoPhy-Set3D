"""Li et al. (ICLR 2021) standard 3D FNO with fixed ray-adjoint input."""
from collections import OrderedDict
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from geophy_set3d.data.measurement_only import assert_no_gt
from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator

Y_SCALE=32284.894580078173

class FixedRayBackprojection:
    """CPU-deterministic T^T y and T^T1, reusing the audited K96 stencil.

    T is normalized *linear mole-fraction* integration. The nonlinear mass
    projector has no global adjoint; J(0)^T / chi'(0) equals T^T exactly.
    CPU scatter accumulation fixes ordering. The bounded cache is keyed by
    complete measurements/geometry/masks, never case labels or GT.
    """
    def __init__(self,cache_entries=512):
        self.operator=ExplicitRayOperator(cache_budget_bytes=128*1024**2,device_cache_budget_bytes=0)
        self.cache=OrderedDict();self.cache_entries=cache_entries
    @torch.no_grad()
    def __call__(self,batch):
        assert_no_gt(batch)
        keys=('input_projection_raw','input_valid_ray','view_mask','input_station_origin_m',
              'input_ray_direction','input_t_near','input_path_length_m')
        outputs=[]
        for i in range(len(batch['view_mask'])):
            b={k:batch[k][i:i+1].detach().cpu() for k in keys}
            valid=b['input_valid_ray'].bool()&b['view_mask'][:,:,None,None,None]
            # Ignore every invalid/padded value, including NaNs, before hashing.
            for k in ['input_projection_raw','input_t_near','input_path_length_m']:
                b[k]=torch.where(valid,b[k],0)
            b['input_ray_direction']=torch.where(valid[:,:,0,:,:,None],b['input_ray_direction'],0)
            b['input_station_origin_m']=torch.where(b['view_mask'][:,:,None],b['input_station_origin_m'],0)
            digest=hashlib.sha256()
            for k in keys:digest.update(b[k].contiguous().numpy().tobytes())
            key=digest.digest()
            if key in self.cache:
                value=self.cache.pop(key);self.cache[key]=value;outputs.append(value);continue
            zero=torch.zeros(1,1,64,64,64,dtype=torch.float32)
            if valid.any():
                stencil=self.operator.prepare(b,64)
                derivative=self.operator._chi_prime(torch.tensor(0.,dtype=torch.float32))
                numerator=self.operator.vjp(zero,b['input_projection_raw']/Y_SCALE,stencil)/derivative
                support=self.operator.vjp(zero,valid.float(),stencil)/derivative
                back=torch.where(support>0,numerator/support.clamp_min(torch.finfo(torch.float32).tiny),0)
                value=torch.cat((back,torch.log1p(support),(support>0).float()),1)
            else:value=torch.zeros(1,3,64,64,64)
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite fixed backprojection')
            if self.cache_entries:
                if len(self.cache)>=self.cache_entries:self.cache.popitem(last=False)
                self.cache[key]=value
            outputs.append(value)
        return torch.cat(outputs).to(batch['input_projection_raw'].device)

class SpectralConv3d(nn.Module):
    def __init__(self,width,modes):
        super().__init__();self.modes=modes
        # Real storage preserves ordinary optimizer/RNG/checkpoint semantics.
        self.weights=nn.Parameter(torch.randn(4,width,width,modes,modes,modes,2)/(width*width))
    def forward(self,x):
        n=x.shape[-3:];m=self.modes
        if any(s<2*m for s in n):raise ValueError('Fourier quadrants overlap')
        transformed=torch.fft.rfftn(x,dim=(-3,-2,-1))
        out=transformed.new_zeros(x.shape[0],x.shape[1],n[0],n[1],n[2]//2+1)
        for i,(sx,sy) in enumerate(((slice(0,m),slice(0,m)),(slice(-m,None),slice(0,m)),
                                     (slice(0,m),slice(-m,None)),(slice(-m,None),slice(-m,None)))):
            out[:,:,sx,sy,:m]=torch.einsum('bixyz,ioxyz->boxyz',transformed[:,:,sx,sy,:m],torch.view_as_complex(self.weights[i].contiguous()))
        return torch.fft.irfftn(out,s=n,dim=(-3,-2,-1))

class FourierBlock(nn.Module):
    def __init__(self,width,modes):
        super().__init__();self.spectral=SpectralConv3d(width,modes);self.pointwise=nn.Conv3d(width,width,1)
    def forward(self,x):return F.gelu(self.spectral(x)+self.pointwise(x))

class FNOV1(nn.Module):
    def __init__(self,width=16,modes=8,layers=4,padding=8,activation_checkpoint=True):
        super().__init__()
        if min(width,modes,layers)<1 or padding<0:raise ValueError('Invalid FNO settings')
        self.adapter=FixedRayBackprojection();self.padding=padding;self.activation_checkpoint=activation_checkpoint
        self.lift=nn.Conv3d(6,width,1)
        self.blocks=nn.ModuleList([FourierBlock(width,modes) for _ in range(layers)])
        self.readout=nn.Sequential(nn.Conv3d(width,64,1),nn.GELU(),nn.Conv3d(64,1,1))
        axis=torch.linspace(-1,1,64)
        self.register_buffer('coordinates',torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),0)[None],persistent=False)
    def forward(self,batch,return_logits=False):
        inputs=self.adapter(batch)
        x=self.lift(torch.cat((inputs,self.coordinates.expand(len(inputs),-1,-1,-1,-1)),1))
        p=self.padding
        if p:x=F.pad(x,(0,p,0,p,0,p))
        for block in self.blocks:
            x=checkpoint(block,x,use_reentrant=False) if self.training and self.activation_checkpoint and torch.is_grad_enabled() else block(x)
        if p:x=x[:,:,:-p,:-p,:-p]
        logits=self.readout(x)
        return (logits.sigmoid(),logits) if return_logits else logits.sigmoid()
