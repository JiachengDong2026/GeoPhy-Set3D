"""Global-scale fusion candidate and independent content replication (no O values)."""
import torch
from torch import nn
import mrgr_ocmsf_v1 as previous

base=previous.base
OCMSFDecoder=previous.OCMSFDecoder
ARMS=('B_multiscale','C_content','E_global')

class GlobalScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=nn.Parameter(torch.zeros(3))
        self.output=nn.Conv3d(12,32,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.record_diagnostics=False;self.last_diagnostics=None
    def forward(self,scales,descriptor):
        x=scales[0]
        if len(scales)!=3 or any(s.shape!=x.shape for s in scales):raise ValueError('Three aligned scales required')
        if x.ndim!=5 or x.shape[1:]!=(12,32,32,32):raise ValueError('Bad feature shape')
        if descriptor.shape!=(x.shape[0],5,32,32,32):raise ValueError('Bad mask descriptor')
        if descriptor.dtype!=x.dtype or descriptor.device!=x.device:raise ValueError('dtype/device mismatch')
        if any(not bool(torch.isfinite(s).all()) for s in (*scales,descriptor)):raise ValueError('Nonfinite input')
        visible=descriptor[:,4:5]>0
        clean=[torch.where(visible,s,0) for s in scales]
        alpha=self.logits.softmax(0).reshape(1,3,1,1,1).expand(x.shape[0],3,32,32,32)
        alpha=torch.where(visible,alpha,torch.full_like(alpha,1/3))
        fused=sum(alpha[:,i:i+1]*s for i,s in enumerate(clean))
        delta=torch.where(visible,self.output(fused),0)
        if not bool(torch.isfinite(delta).all()):raise FloatingPointError('Nonfinite residual')
        if self.record_diagnostics:self.last_diagnostics=dict(alpha=alpha.detach(),delta=delta.detach(),visible=visible.detach())
        return delta

def model_for(config,arm,device):
    if arm not in ARMS:raise ValueError(arm)
    model=base.model_for(config,'multiscale','cpu')
    if arm!='B_multiscale':
        before=torch.get_rng_state()
        torch.manual_seed(config['training']['seed']+2)
        previous.attach(model,False)
        if arm=='E_global':
            # Rewind before global initialization, independent of unused content gate draws.
            torch.manual_seed(config['training']['seed']+2)
            model.decoder.scale_fusion=GlobalScaleFusion()
        torch.set_rng_state(before)
    return model.to(device)
