"""Independent, single-factor MRGR ablations; no frozen source modifications."""
from common import *
import torch
from torch import nn
from torch.nn import functional as F
import probe_mrgr_multiscale_v1 as original
from geophy_set3d.models.geolift_oari_res3d import GeoLiftOARIRes3D
from geophy_set3d.models.geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D

ARMS=('multiscale','single_scale','no_feedback','no_consistency')

class SingleScaleContext(original.MultiscaleContext):
    """Same 12,608 parameters and layers; every context convolution stays at 32³."""
    def forward(self,x):
        y=self.local16(self.reduce(x));z=self.context8(y)
        return self.output(self.fuse16(y+z))

class SingleScaleDecoder(original.ContextDecoder):
    def __init__(self,decoder):
        nn.Sequential.__init__(self,*list(decoder.children()))
        self.context=SingleScaleContext()

class NoFeedback(GeoLiftOARIMRGRRes3D):
    def forward(self,batch,return_stages=False):
        # Actual removal: residual encoder/adapter do not exist, P(C0) is not called.
        c=GeoLiftOARIRes3D.forward(self,batch)
        return dict(C0=c,C1=c) if return_stages else c


def model_for(config,arm,device):
    if arm not in ARMS:raise ValueError(arm)
    if arm in ('multiscale','no_consistency'):return original.model_for(config,'multiscale',device)
    model=original.model_for(config,'reference','cpu');before=torch.get_rng_state()
    torch.manual_seed(config['training']['seed']+1)
    if arm=='single_scale':model.decoder=SingleScaleDecoder(model.decoder)
    else:
        model.decoder=original.ContextDecoder(model.decoder)
        del model.residual_encoder;del model.residual_adapter;model.__class__=NoFeedback
    torch.set_rng_state(before)
    return model.to(device)


def install_loss(config):
    from geophy_set3d.training import selfsup
    # Used only within this worker process; original FROZEN_LOSS remains canonical.
    canonical=dict(hold=1.,cross=.5,observed=.1,consistency=.05,beta=.01,relative_l1=.1,relative_epsilon=1e-6,measurement_scale=32284.894580078173)
    assert all(config['loss'][k]==v for k,v in canonical.items() if k!='consistency')
    selfsup.FROZEN_LOSS=dict(canonical,consistency=config['loss']['consistency'])
