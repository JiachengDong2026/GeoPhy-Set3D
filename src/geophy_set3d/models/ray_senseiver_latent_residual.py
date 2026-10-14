"""Single registered adaptation: global latent residual after query cross layer."""
import copy
import torch
from torch import nn


class LatentResidualDecoder(nn.Module):
    """Preserves all original modules/keys; adds one zero-initialized 64x64 map."""
    def __init__(self, original):
        super().__init__()
        self.postproc=original.postproc
        self.preproc=original.preproc
        self.cross_attention=original.cross_attention
        self.output=original.output
        self.latent_condition=nn.Linear(64,64,bias=False)
        nn.init.zeros_(self.latent_condition.weight)

    def forward(self,x,coords):
        b=len(x)
        query=self.output.unsqueeze(0).expand(b,-1,-1).repeat_interleave(coords.shape[1],dim=1)
        query=self.preproc(torch.cat([coords,query],dim=-1))
        features=self.cross_attention(query,x)
        features=features+self.latent_condition(x.mean(dim=1)).unsqueeze(1)
        return self.postproc(features)


def add_latent_residual(control):
    treatment=copy.deepcopy(control)
    # Keep device and all inherited tensors exactly equal to the fresh Control.
    treatment.decoder=LatentResidualDecoder(treatment.decoder).to(next(control.parameters()).device)
    return treatment
