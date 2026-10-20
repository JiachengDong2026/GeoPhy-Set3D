"""OARI over residual-adapter features; original first stage and physics retained."""
import copy
import math
import torch
from runtime import ARMS
from models import model_for as original_model
from geophy_set3d.models.geolift_oari_res3d import GeoLiftOARIRes3D
from geophy_set3d.models.geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D,project_input,Y_SCALE

class FeedbackOARI(GeoLiftOARIMRGRRes3D):
    def forward(self,batch,return_stages=False,residual_mode='normal'):
        if residual_mode!='normal' and self.training:raise ValueError('Evaluation-only residual probe')
        if residual_mode not in ('normal','zero','wrong_pairing','signflip'):raise ValueError(residual_mode)
        batch={k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        c0,base=GeoLiftOARIRes3D.forward(self,batch,return_attention=True)
        context0,visibility=base['context_features'],base['visibility']
        relations,observability=base['pairwise_geometry'],base['observability_descriptor'];del base
        y0=project_input(c0,batch)
        valid=batch['input_valid_ray'].bool() & batch['view_mask'][:,:,None,None,None]
        residual=torch.where(valid,batch['input_projection_raw']-y0,0);scaled=residual/Y_SCALE
        if self.zero_feedback_residual or residual_mode=='zero':scaled=torch.zeros_like(scaled)
        elif residual_mode=='signflip':scaled=-scaled
        elif residual_mode=='wrong_pairing':
            scaled=scaled.clone()
            for i in range(len(c0)):
                ids=batch['view_mask'][i].nonzero().flatten();scaled[i,ids]=scaled[i,ids.roll(1)]
        scaled=torch.where(valid,scaled,0)
        path=torch.where(valid,batch['input_path_length_m']/(math.sqrt(3)*.6),0)
        inputs=torch.cat((scaled,valid.to(c0.dtype),path),2);b,v=inputs.shape[:2]
        features=self.residual_encoder(inputs.flatten(0,1)).reshape(b,v,16,64,64)
        lifted,residual_visibility=self.lifting(features,batch)
        combined=torch.cat((context0,lifted),2).flatten(0,1)
        adapted=self.residual_adapter[1](self.residual_adapter[0](combined)).reshape_as(context0)
        geometry=self.geometry_descriptor(batch)
        # The bypass exists solely for equivalence checks; never used by training.
        interacted=adapted if self.feedback_bypass else self.feedback_interaction(adapted,geometry,visibility,batch['view_mask'],relations,observability)
        delta=self.residual_adapter[2](interacted.flatten(0,1)).reshape_as(context0)
        visible=visibility & batch['view_mask'][:,:,None,None,None]
        delta=torch.where(visible[:,:,None],delta,0);context1=torch.where(visible[:,:,None],context0+delta,0)
        fused,_=self.fusion(context1,geometry,visibility,batch['view_mask']);c1=self.decoder(fused)
        if return_stages:
            return dict(C0=c0,C1=c1,measurement_prediction_Y0=y0,measurement_residual=residual,residual_scaled=residual/Y_SCALE,
                residual_input=inputs,residual_context=delta,residual_features=features,residual_lifted=lifted,
                visibility_base=visibility,visibility_residual=residual_visibility,context1=context1)
        return c1

def model_for(config,arm,device):
    if arm not in ARMS:raise ValueError(arm)
    model=original_model(config,'multiscale','cpu')
    if arm!='baseline':
        model.feedback_interaction=copy.deepcopy(model.interaction)
        model.__class__=FeedbackOARI
        model.zero_feedback_residual=arm=='feedback_oari_zero_residual'
        model.feedback_bypass=False
    return model.to(device)
