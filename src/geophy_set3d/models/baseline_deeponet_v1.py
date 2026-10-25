"""Lu et al. (2021) DeepONet: CNN branch, XYZ MLP trunk, inner product.

Adaptation: complete per-station ray maps in ascending station-ID order;
up to five slots, with explicit ray/view masks. No GeoLift or attention.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from geophy_set3d.data.measurement_only import assert_no_gt

Y_SCALE=32284.894580078173

def ray_maps(batch):
    assert_no_gt(batch)
    valid=batch['input_valid_ray'].bool() & batch['view_mask'][:,:,None,None,None]
    raw=torch.where(valid,batch['input_projection_raw'],0)/Y_SCALE
    length=torch.where(valid,batch['input_path_length_m'],0)
    near=torch.where(valid,batch['input_t_near'],0)
    d=torch.where(valid[:, :, 0, :, :, None],batch['input_ray_direction'],0)
    origin=torch.where(batch['view_mask'][:,:,None],batch['input_station_origin_m'],0)[:,:,None,None,:]
    entry=origin+d*near[:,:,0,:,:,None]
    exit=entry+d*length[:,:,0,:,:,None]
    geometry=torch.cat((2*entry/.6-1,2*exit/.6-1,d),-1).permute(0,1,4,2,3)
    geometry=torch.where(valid,geometry,0)
    result=torch.cat((raw,valid.to(raw.dtype),length/(math.sqrt(3)*.6),geometry),2)
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite valid input')
    return result

class DeepONetV1(nn.Module):
    def __init__(self,features=64,width=128,max_views=5,query_chunk=8192):
        super().__init__()
        if max_views!=5 or features<1 or width<1 or query_chunk<1:raise ValueError('Invalid DeepONet settings')
        self.max_views=max_views;self.query_chunk=query_chunk
        self.station_encoder=nn.Sequential(nn.Conv2d(12,16,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(32,32,3,stride=2,padding=1),nn.GELU(),
            nn.Flatten(),nn.Linear(32*8*8,64),nn.GELU())
        self.branch=nn.Sequential(nn.Linear(5*65,width),nn.GELU(),nn.Linear(width,features))
        self.trunk=nn.Sequential(nn.Linear(3,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,features))
        self.bias=nn.Parameter(torch.zeros(()))
        axis=torch.linspace(-1,1,64)
        self.register_buffer('coordinates',torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3),persistent=False)
    def encode_branch(self,batch):
        maps=ray_maps(batch);bsz=maps.shape[0]
        slots=maps.new_zeros(bsz,self.max_views,65)
        for b in range(bsz):
            indices=batch['view_mask'][b].nonzero().flatten()
            if len(indices)>self.max_views:raise ValueError('More than five real input views')
            ids=batch['input_station_indices'][b,indices]
            if len(ids.unique())!=len(ids):raise ValueError('Duplicate station IDs')
            indices=indices[ids.argsort()]
            if len(indices):
                features=self.station_encoder(maps[b,indices])
                slots[b,:len(indices)]=torch.cat((features,features.new_ones(len(indices),1)),-1)
        return self.branch(slots.flatten(1))
    def decode(self,branch,coordinates):
        values=[]
        for xyz in coordinates.split(self.query_chunk):
            if self.training and torch.is_grad_enabled():
                trunk=checkpoint(self.trunk,xyz,use_reentrant=False)
            else:trunk=self.trunk(xyz)
            values.append(branch@trunk.T+self.bias)
        return torch.cat(values,1)
    def forward(self,batch,return_logits=False):
        logits=self.decode(self.encode_branch(batch),self.coordinates).reshape(-1,1,64,64,64)
        return (logits.sigmoid(),logits) if return_logits else logits.sigmoid()
