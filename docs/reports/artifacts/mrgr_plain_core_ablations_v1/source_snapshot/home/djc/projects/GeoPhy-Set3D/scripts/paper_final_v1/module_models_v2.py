"""Additional single-factor module ablations, leaving registered v1 sources intact."""
import torch
from torch import nn
from models import model_for as original_model
from geophy_set3d.models.geolift_oari_mrgr_zero_residual import GeoLiftOARIMRGRZeroResidualControl

ARMS=('no_ray_relations','no_observability','uniform_fusion','zero_residual')


class MaskedPairBias(nn.Sequential):
    def forward(self,x):
        # Scalar fields, not attention visibility masks, are ablated.
        if self.ablation=='no_ray_relations':x=torch.cat((torch.zeros_like(x[...,:3]),x[...,3:]),-1)
        elif self.ablation=='no_observability':x=torch.cat((x[...,:3],torch.zeros_like(x[...,3:])), -1)
        else:raise ValueError(self.ablation)
        return super().forward(x)


class ZeroDescriptorEmbedding(nn.Sequential):
    def forward(self,x):return super().forward(torch.zeros_like(x))


class UniformFusion(nn.Module):
    def forward(self,features,geometry,visibility,view_mask):
        valid=visibility & view_mask[:,:,None,None,None]
        count=valid.sum(1,keepdim=True).clamp_min(1)
        weights=valid.to(features.dtype)/count
        fused=(torch.where(valid[:,:,None],features,0)*weights[:,:,None]).sum(1)
        return fused,weights


def model_for(config,arm,device):
    if arm not in ARMS:return original_model(config,arm,device)
    model=original_model(config,'multiscale','cpu')
    if arm in ('no_ray_relations','no_observability'):
        model.interaction.pairwise_bias.__class__=MaskedPairBias
        model.interaction.pairwise_bias.ablation=arm
        if arm=='no_observability':model.interaction.observability_embedding.__class__=ZeroDescriptorEmbedding
    elif arm=='uniform_fusion':model.fusion=UniformFusion()
    else:model.__class__=GeoLiftOARIMRGRZeroResidualControl
    return model.to(device)
