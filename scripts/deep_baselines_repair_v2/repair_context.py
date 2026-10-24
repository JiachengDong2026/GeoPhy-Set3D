"""New repair-stage identity; original baselines, losses and main remain immutable."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import torch
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'scripts/deep_baselines_v1')]
import baseline_context as b
c=b.c
sys.path.insert(0,str(HERE))
from geophy_set3d.training import selfsup as ss
from geophy_set3d.training.utils import seed_everything, rng_state, restore_rng
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.physics.projectors import mass_to_mole
from torch.utils.data import Subset
ROOT=Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_repair_v2')
SEED=20261011
ARMS=('deepsets','gnot')
UUID='GPU-37424437-e208-62d1-e617-53c421c94a49'

def freeze(name,value):c.freeze(ROOT/name,value)
def read(name):return c.verified(ROOT/name)
def config(arm,target):
    cfg=copy.deepcopy(c.verified(b.ROOT/'main_anchor.json')['model']['config'])
    cfg['training']['seed']=SEED;cfg['training']['device']='cuda:0'
    cfg['experiment']=dict(name=target.name,output_root=str(target))
    cfg['model']=dict(name='ray_'+arm+'_repair_v2')
    cfg['loader']=dict(num_workers=0,pin_memory=False,persistent_workers=False)
    return cfg

def old_checkpoint(arm):return b.ROOT/'training_registered_v2'/f'{arm}_seed{SEED}'/'checkpoints/best.pt'

def guard(checkpoints=()):return c.AccessGuard(splits=['train','val'],checkpoints=checkpoints,output=ROOT)

def sources():
    paths=list((PROJECT/'src').rglob('*.py'))+list(HERE.glob('*.py'))
    return {str(p):c.sha(p) for p in sorted(paths)}

def fixed_batch(cfg,validation=False,indices=(0,1),device='cuda:0'):
    ds=ss.make_dataset(cfg,validation=validation)
    loader=ss.make_loader(Subset(ds,list(indices)),cfg)
    return ss.move_dual_batch(next(iter(loader)),device)

def tensor_stats(x):
    x=x.detach().double()
    return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),std=float(x.std(unbiased=False)))

def health(model,batch):
    mode=model.training;model.eval()
    with torch.no_grad():
        normal=model(batch['A'])
        zero=dict(batch['A'],input_projection_raw=batch['A']['input_projection_raw']*0)
        half=dict(batch['A'],input_projection_raw=batch['A']['input_projection_raw']*.5)
        z,h=model(zero),model(half)
        value=tensor_stats(normal)
        value.update(relative_measurement_zero_change=float((normal-z).abs().mean()/(normal.abs().mean()+1e-15)),
                     relative_measurement_half_change=float((normal-h).abs().mean()/(normal.abs().mean()+1e-15)),
                     case_relative_difference=float((normal[0]-normal[1]).abs().mean()/(normal.abs().mean()+1e-15)),
                     fraction_below_1e_10=float((normal<1e-10).float().mean()),
                     fraction_above_0999=float((normal>.999).float().mean()),
                     sigmoid_derivative_mean=float((normal*(1-normal)).mean()))
    model.train(mode)
    return value
