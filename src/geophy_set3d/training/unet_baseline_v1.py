"""Independent U-Net registry: frozen original training configuration."""
import copy,math
from pathlib import Path
import torch,yaml
from geophy_set3d.models.baseline_unet3d_v1 import UNet3DV1
from .classical_operators import load_config as historical_config
PROJECT=Path(__file__).resolve().parents[3]
SETTINGS=dict(channels=[16,32,64,128,256],activation_checkpoint=True)
REGISTRY={'unet':UNet3DV1}

def load_config(arm='unet'):
 if arm!='unet':raise ValueError('Independent U-Net registry only')
 c=yaml.safe_load((PROJECT/'configs/unet_baseline_v1/unet.yaml').read_text())
 expected=copy.deepcopy(historical_config('fno'))
 expected['model']=dict(name='unet3d_v1',settings=SETTINGS)
 expected['experiment']=dict(name='unet3d_selfsup_v1',output_root='/scratch/djc/GeoPhy-Set3D/unet_baseline_v1')
 expected['comparison']['output']='/scratch/djc/GeoPhy-Set3D/unet_baseline_v1/unet'
 assert c==expected,'Frozen training/data/physics/loss/loader configuration must not change'
 return c

@torch.no_grad()
def initialize_scale(model,c0,alpha):
 if not 0<c0<1 or not 0<alpha<=1:raise ValueError('Invalid registered scale')
 z0=math.log(c0/(1-c0))
 if abs(z0)+.1>10:raise ValueError('Scale conflicts with nonsaturation condition')
 model.readout.weight.mul_(alpha);model.readout.bias.fill_(z0)
 return model

def build_model(c,c0,alpha,device='cpu'):
 assert c==load_config()
 return initialize_scale(UNet3DV1(**c['model']['settings']),c0,alpha).to(device)
