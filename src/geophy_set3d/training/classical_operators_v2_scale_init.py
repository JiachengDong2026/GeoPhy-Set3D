"""Independent registry: identical v1 graphs, existing-parameter initialization only."""
import copy,math
from pathlib import Path
import torch
import yaml
from . import classical_operators as v1

REGISTRY=dict(v1.REGISTRY)
PROJECT=Path(__file__).resolve().parents[3]

def load_config(arm):
 if arm not in REGISTRY:raise ValueError(arm)
 c=yaml.safe_load((PROJECT/f'configs/classical_operators_v2_scale_init/{arm}.yaml').read_text())
 expected=copy.deepcopy(v1.load_config(arm))
 expected['model']['name']=f'{arm}_v2_scale_init'
 expected['experiment']=dict(name=f'{arm}_selfsup_v2_scale_init',output_root='/scratch/djc/GeoPhy-Set3D/classical_operators_v2_scale_init')
 expected['comparison']['output']=f'/scratch/djc/GeoPhy-Set3D/classical_operators_v2_scale_init/{arm}'
 assert c==expected,'Only experiment identity/output paths can change'
 return c

@torch.no_grad()
def initialize_scale(model,arm,c0,alpha):
 if not 0<c0<1 or not 0<alpha<=1:raise ValueError('Invalid registered scale')
 z0=math.log(c0/(1-c0))
 if abs(z0)+.1>10:raise ValueError('Physical center incompatible with registered nonsaturation bound')
 if arm=='deeponet':
  model.branch[-1].weight.mul_(alpha);model.branch[-1].bias.mul_(alpha);model.bias.fill_(z0)
 elif arm=='fno':
  model.readout[-1].weight.mul_(alpha);model.readout[-1].bias.fill_(z0)
 else:raise ValueError(arm)
 return model

def build_model(c,c0,alpha,device='cpu'):
 arm=c['model']['name'].removesuffix('_v2_scale_init');assert c==load_config(arm)
 return initialize_scale(REGISTRY[arm](**c['model']['settings']),arm,c0,alpha).to(device)
