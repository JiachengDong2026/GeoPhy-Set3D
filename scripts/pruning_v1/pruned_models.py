"""Structural descriptor pruning; exact inherited MRGR feedback implementation."""
import math
import torch
from torch import nn
from runtime import ARMS
from module_models_v2 import model_for as historical_model
from geophy_set3d.models.geolift_oari_res3d import GeoLiftOARIRes3D
from geophy_set3d.models.geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D


class UniformFusion(nn.Module):
    def forward(self,features,geometry,visibility,view_mask):
        valid=visibility & view_mask[:,:,None,None,None]
        weights=valid.to(features.dtype)/valid.sum(1,keepdim=True).clamp_min(1)
        return (torch.where(valid[:,:,None],features,0)*weights[:,:,None]).sum(1),weights


class PrunedInteraction(nn.Module):
    def forward(self,features,geometry,visibility,view_mask,relations,return_attention=False):
        valid=visibility & view_mask[:,:,None,None,None]
        b,v,c,x,y,z=features.shape
        features=torch.where(valid[:,:,None],features,0);geometry=torch.where(valid[:,:,None],geometry,0)
        embedded=self.geometry_embedding(geometry.flatten(0,1)).reshape(b,v,16,x,y,z)
        tokens=torch.cat((features,embedded),2).permute(0,3,4,5,1,2).reshape(-1,v,48)
        mask=valid.permute(0,2,3,4,1).reshape(-1,v);token_mask=mask[:,:,None];pair_mask=mask[:,:,None]&mask[:,None,:]
        tokens=torch.where(token_mask,self.token_projection(tokens),0)
        qkv=self.qkv(self.norm1(tokens)).reshape(-1,v,3,self.num_heads,c//self.num_heads)
        q,k,values=qkv.permute(2,0,3,1,4).unbind(0)
        values=torch.where(mask[:,None,:,None],values,0)
        relations=torch.where(pair_mask[...,None],relations,0)
        bias=self.pairwise_bias(relations).permute(0,3,1,2);bias=torch.where(pair_mask[:,None],bias,0)
        logits=(q@k.transpose(-1,-2))/math.sqrt(c//self.num_heads)+bias
        logits=logits.masked_fill(~mask[:,None,None,:],-torch.inf)
        logits=torch.where(mask.any(1)[:,None,None,None],logits,0)
        weights=torch.softmax(logits,-1).masked_fill(~pair_mask[:,None],0)
        attended=(weights@values).transpose(1,2).reshape(-1,v,c)
        tokens=torch.where(token_mask,tokens+self.attention_output(attended),0)
        tokens=torch.where(token_mask,tokens+self.ffn(self.norm2(tokens)),0)
        context=tokens.reshape(b,x,y,z,v,c).permute(0,4,5,1,2,3).contiguous()
        if return_attention:
            return context,weights.reshape(b,x,y,z,self.num_heads,v,v).permute(0,4,5,6,1,2,3).contiguous(),bias
        return context


class PrunedStage0(GeoLiftOARIRes3D):
    def forward(self,batch,return_attention=False,relation_permutation=None):
        if relation_permutation is not None and self.training:raise ValueError('Evaluation-only relation probe')
        y=batch['input_projection']
        inputs=torch.cat((y,batch['input_valid_ray'].to(y.dtype),batch['input_path_length_m']/(math.sqrt(3)*.6)),2)
        inputs=torch.where(batch['view_mask'][:,:,None,None,None],inputs,0);b,v=inputs.shape[:2]
        features=self.encoder(inputs.flatten(0,1)).reshape(b,v,32,64,64)
        lifted,visibility=self.lifting(features,batch);geometry=self.geometry_descriptor(batch)
        valid=visibility & batch['view_mask'][:,:,None,None,None]
        directions,relations=self.ray_geometry(batch['input_station_origin_m'],self.lifting.voxel_xyz,batch['input_station_azimuth_deg'],valid)
        # Directions are necessary to form ray relations, but no descriptor,
        # 3x3 directional tensor, eigensolver or view-count fraction is computed.
        if relation_permutation is not None:relations=relations[:,relation_permutation][:,:,relation_permutation]
        context=self.interaction(lifted,geometry,visibility,batch['view_mask'],relations,return_attention)
        if return_attention:context,matrices,bias=context
        fused,weights=self.fusion(context,geometry,visibility,batch['view_mask']);prediction=self.decoder(fused)
        if return_attention:
            return prediction,dict(attention_weights=weights,interaction_attention=matrices,pairwise_bias=bias,
                pairwise_geometry=relations,visibility=visibility,context_features=context)
        return prediction


class PrunedMRGR(GeoLiftOARIMRGRRes3D,PrunedStage0):
    # C3 MRO makes super() inside the unchanged MRGR.forward call PrunedStage0.
    # All residual construction, internal P(C0), and C1 execution stay inherited.
    pass


def sliced_linear(original,n):
    before=torch.get_rng_state();layer=nn.Linear(n,original.out_features,bias=original.bias is not None)
    torch.set_rng_state(before)
    with torch.no_grad():
        layer.weight.copy_(original.weight[:,:n])
        if original.bias is not None:layer.bias.copy_(original.bias)
    return layer


def model_for(config,arm,device):
    if arm not in ARMS:
        m=historical_model(config,arm,'cpu')
        if arm=='uniform_fusion':m.fusion=UniformFusion()
        return m.to(device)
    m=historical_model(config,'multiscale','cpu')
    del m.observability;del m.interaction.observability_embedding
    m.interaction.token_projection=sliced_linear(m.interaction.token_projection,48)
    m.interaction.pairwise_bias[0]=sliced_linear(m.interaction.pairwise_bias[0],3)
    m.interaction.__class__=PrunedInteraction;m.__class__=PrunedMRGR
    if arm=='uniform_pruned_observability':m.fusion=UniformFusion()
    return m.to(device)
