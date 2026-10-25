"""Read-only neural diagnostics. All new arrays live in an isolated scratch root."""
import os, sys, json, hashlib, csv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
PROJECT=Path(__file__).resolve().parents[2]
OLD=Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_identifiability_v2')
ROOT=Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_latent_readout_v1')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'scripts/deep_baselines_identifiability_v2'))
import ctx_v2 as old
EXPECTED={'gnot':'a69e1b2cbeda198be13d34feb90a8fd41a56940a1e10d8c03ca5f99c4419c6e2','deepsets':'ca2abaab4bf06a0d5d39d81187b8f08eba36aabd7c5714d2f97683cf04486607'}
sha,read,write,append,freeze=old.sha,old.read,old.write,old.append,old.freeze
now=old.now

def checkpoint(arm):return OLD/f'runs/{arm}/checkpoint_200.pt'
def csvwrite(p,rows):
    p=Path(p);assert not p.exists(),p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def verify_seal():
    reg=read(ROOT/'registration.json')
    for group in ('inputs','checkpoints','sources','history'):
        for p,h in reg[group].items():assert sha(p)==h,p
    return reg

def samples():
    return torch.load(OLD/'inputs/train_AB.pt',weights_only=False)+torch.load(OLD/'inputs/unseen_AB.pt',weights_only=False)
def netload(arm,device):
    assert sha(checkpoint(arm))==EXPECTED[arm]
    net=old.model(arm,device);ck=torch.load(checkpoint(arm),map_location='cpu',weights_only=False)
    assert ck['updates']==200;net.load_state_dict(ck['model'],strict=True);net.eval().requires_grad_(False)
    return net

def zero(inp):
    out=dict(inp)
    for k in ('input_projection','input_projection_raw'):out[k]=torch.zeros_like(inp[k])
    return out

def swap(inp,donor):
    out=dict(inp)
    for k in ('input_projection','input_projection_raw'):out[k]=donor[k].clone()
    assert old.value_hash(old.nonmeasurement(out))==old.value_hash(old.nonmeasurement(inp))
    assert old.value_hash(out)==old.value_hash(donor)
    return out

class Access:
    """Auditable file capabilities, including stage-specific unseen-label denial."""
    def __init__(self,phase):self.phase=phase;self.active=False;self.reads={};self.denied=[]
    def __enter__(self):sys.addaudithook(self.audit);self.active=True;return self
    def __exit__(self,*args):self.active=False;freeze(ROOT/f'access_{self.phase}.json',dict(phase=self.phase,reads=self.reads,denied=self.denied))
    def audit(self,event,args):
        if not self.active or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        writing=any(c in mode for c in 'wax+') if isinstance(mode,str) else bool(flags&(os.O_WRONLY|os.O_RDWR))
        reason=None;array=p.suffix in ('.pt','.pth','.ckpt','.npy','.npz','.h5','.hdf5')
        if writing:
            if not p.is_relative_to(ROOT):reason='writes outside isolated scratch forbidden'
        elif p.is_relative_to(Path('/data')):reason='real data forbidden'
        elif array and not p.is_relative_to(ROOT):
            allowed=p in [checkpoint(a) for a in EXPECTED] or p in (OLD/'inputs/train_AB.pt',OLD/'inputs/unseen_AB.pt',OLD/'inputs/geometry.pt')
            if p.parent==OLD/'inputs' and p.name.startswith('truth_'):
                i=int(p.stem.split('_')[1]);allowed=self.phase.split('_')[0] in ('evaluate','review') or (self.phase.split('_')[0]=='fit' and i<6)
            if p.parent in [OLD/f'runs/{a}/raw' for a in EXPECTED] and p.name.startswith('update200_'):allowed=self.phase.split('_')[0]=='extract'
            if not allowed:reason='array outside stage capability'
        if reason:self.denied.append(dict(path=str(p),reason=reason));raise PermissionError(reason+': '+str(p))
        if not writing and (array or p.is_relative_to(OLD)):self.reads[str(p)]=self.reads.get(str(p),0)+1

def feature_path(arm,b,kind,i,stage='prehead'):
    return ROOT/f'features/{arm}/{b}/{kind}/{i if kind=="normal" else 0}/{stage}.npy'
