"""Independent model registry/policy; historical builders and losses untouched."""
import copy
from pathlib import Path
import yaml
from geophy_set3d.models.baseline_deeponet_v1 import DeepONetV1
from geophy_set3d.models.baseline_fno_v1 import FNOV1

REGISTRY={'deeponet':DeepONetV1,'fno':FNOV1}
SETTINGS={'deeponet':dict(features=64,width=128,max_views=5,query_chunk=8192),
          'fno':dict(width=16,modes=8,layers=4,padding=8,activation_checkpoint=True)}
PROJECT=Path(__file__).resolve().parents[3]

def load_config(arm):
    if arm not in REGISTRY:raise ValueError('Unknown classical operator')
    c=yaml.safe_load((PROJECT/f'configs/classical_operators_v1/{arm}.yaml').read_text())
    return validate_config(c,arm)

def validate_config(c,arm):
    old=yaml.safe_load((PROJECT/'configs/train_deepsets_selfsup_v1.yaml').read_text())
    actual=copy.deepcopy(c)
    assert actual.pop('model')==dict(name=f'{arm}_v1',settings=SETTINGS[arm])
    old.pop('model');actual.pop('experiment');old.pop('experiment')
    comp=actual.pop('comparison');prior=old.pop('comparison')
    assert comp['output']==f'/scratch/djc/GeoPhy-Set3D/classical_operators_v1/{arm}'
    comp.pop('output');prior.pop('output');assert comp==prior
    assert actual==old,'Frozen training/data/loss/loader configuration changed'
    assert c['experiment']==dict(name=f'{arm}_selfsup_v1',output_root='/scratch/djc/GeoPhy-Set3D/classical_operators_v1')
    return c

def build_model(c,device=None):
    arm=c['model']['name'].removesuffix('_v1');validate_config(c,arm)
    return REGISTRY[arm](**c['model']['settings']).to(device or c['training']['device'])
