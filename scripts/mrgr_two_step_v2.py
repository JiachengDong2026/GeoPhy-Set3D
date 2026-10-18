"""Parameter-shared second signed measurement feedback step; frozen core unchanged."""
import math
import torch
from torch import nn
import probe_mrgr_multiscale_v1 as base
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input,Y_SCALE
ARMS=('B_one','T_two')

class TwoStepMRGR(nn.Module):
    def __init__(self,core,steps):
        super().__init__()
        if steps not in (1,2):raise ValueError('Only one or two feedback steps supported')
        self.core=core;self.steps=steps
    @property
    def decoder(self):return self.core.decoder
    def forward(self,batch,return_stages=False,residual_mode='normal'):
        if self.steps==1:
            return self.core(batch,return_stages=return_stages,residual_mode=residual_mode)
        batch={k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        stages=self.core(batch,return_stages=True,residual_mode=residual_mode)
        c1=stages['C1'];context1=stages['context1'];visibility=stages['visibility_base']
        # Crucially, do not detach c1 or y1: the second feedback stays end-to-end.
        y1=project_input(c1,batch)
        valid=batch['input_valid_ray'].bool()&batch['view_mask'][:,:,None,None,None]
        residual=torch.where(valid,batch['input_projection_raw']-y1,0)
        scaled=residual/Y_SCALE
        if residual_mode=='zero':scaled=torch.zeros_like(scaled)
        elif residual_mode=='signflip':scaled=-scaled
        elif residual_mode=='wrong_pairing':
            scaled=scaled.clone()
            for i in range(len(c1)):
                ids=batch['view_mask'][i].nonzero().flatten()
                scaled[i,ids]=scaled[i,ids.roll(1)]
        scaled=torch.where(valid,scaled,0)
        path=torch.where(valid,batch['input_path_length_m']/(math.sqrt(3)*.6),0)
        inputs=torch.cat((scaled,valid.to(c1.dtype),path),2)
        b,v=inputs.shape[:2]
        features=self.core.residual_encoder(inputs.flatten(0,1)).reshape(b,v,16,64,64)
        lifted,residual_visibility=self.core.lifting(features,batch)
        delta=self.core.residual_adapter(torch.cat((context1,lifted),2).flatten(0,1)).reshape_as(context1)
        visible=visibility&batch['view_mask'][:,:,None,None,None]
        delta=torch.where(visible[:,:,None],delta,0)
        context2=torch.where(visible[:,:,None],context1+delta,0)
        fused,_=self.core.fusion(context2,self.core.geometry_descriptor(batch),visibility,batch['view_mask'])
        c2=self.core.decoder(fused)
        if return_stages:
            return dict(stages,C_feedback1=c1,C1=c2,C2=c2,context2=context2,
                measurement_prediction_Y1=y1,measurement_residual_step2=residual,
                residual_context_step2=delta,residual_features_step2=features,
                residual_lifted_step2=lifted,visibility_residual_step2=residual_visibility,
                residual_input_step2=inputs)
        return c2

def model_for(config,arm,device):
    if arm not in ARMS:raise ValueError(arm)
    core=base.model_for(config,'multiscale','cpu')
    return TwoStepMRGR(core,1 if arm=='B_one' else 2).to(device)
