"""Isolated fixed-geometry diagnostics; no historical dataset getters."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_identifiability_v2')
DATA = Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
GEOMETRY = DATA / 'geometry/train_geometry.npz'
ART = PROJECT / 'docs/reports/artifacts/deep_baselines_identifiability_v2'
CONFIG = PROJECT / 'configs/baselines/identifiability_v2/design.json'
UUID = 'GPU-37424437-e208-62d1-e617-53c421c94a49'
sys.path.insert(0, str(PROJECT / 'src'))
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch, seed_everything
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.models.baseline_measurement_init_v2 import RayDeepSetsMeasurementInitV2, RayGNOTMeasurementInitV2
from geophy_set3d.models.ray_senseiver_ss import Y_SCALE

def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def value_hash(d):
    return hashlib.sha256(json.dumps({k:tensor_hash(v) if torch.is_tensor(v) else v for k,v in sorted(d.items())},sort_keys=True).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def freeze(p,v):
    p=Path(p);assert not p.exists(),p
    write(p,v);p.with_suffix(p.suffix+'.sha256').write_text(sha(p)+'\n')
def append(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush()
def verified(p):
    p=Path(p);assert sha(p)==p.with_suffix(p.suffix+'.sha256').read_text().strip(),p
    return read(p)
def verify(mapping):
    for p,h in mapping.items():assert sha(p)==h,p
def design():return read(CONFIG)
def backend():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
def model(arm,device='cpu'):
    seed_everything(design()['seed'])
    config=read(PROJECT/f'configs/baselines/{arm}/model_repair_v2.json')
    cls=RayGNOTMeasurementInitV2 if arm=='gnot' else RayDeepSetsMeasurementInitV2
    return cls(**config['kwargs']).to(device)
def batch(samples,branch,device='cpu'):
    return move_batch(projection_pool_collate([s[branch] for s in samples]),device)
def save_pt(p,value):
    p=Path(p);assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(p)

class Boundary:
    """CPython open audit allowlist, installed for each entire worker lifetime.

    Training cannot read H, truth, unseen AB, earlier arrays or checkpoints.
    This is an auditable file boundary, not an OS sandbox against malicious code.
    """
    def __init__(self,phase):
        self.phase=phase;self.active=False;self.opens={};self.denied=[]
    def __enter__(self):
        sys.addaudithook(self.audit);self.active=True;return self
    def __exit__(self,*args):self.active=False
    def audit(self,event,args):
        if not self.active or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        writing=any(k in mode for k in 'wax+') if isinstance(mode,str) else bool(flags&(os.O_WRONLY|os.O_RDWR))
        reason=None;label=None
        if p.is_relative_to(DATA):
            if writing or p!=GEOMETRY.resolve() or self.phase not in ('generate','test'):reason='real data outside geometry-only capability'
            else:label='train_geometry_only'
        elif p.is_relative_to(Path('/data')) and not writing:reason='all other real data forbidden'
        elif p.is_relative_to(ROOT):
            if not writing and p.suffix in ('.pt','.npy','.npz'):
                relative=p.relative_to(ROOT)
                if self.phase=='train' and str(relative)!='inputs/train_AB.pt':reason='training cannot read H/truth/unseen/checkpoints'
                elif self.phase=='smoke' and str(relative)!='inputs/train_AB.pt':reason='smoke AB-only'
                else:label='own_'+self.phase+'_array'
        elif not writing and p.suffix in ('.pt','.pth','.ckpt','.npz','.npy','.h5','.hdf5'):
            reason='external array/checkpoint forbidden'
        if reason:
            self.denied.append(dict(path=str(p),reason=reason));raise PermissionError(reason+': '+str(p))
        if label:self.opens[str(p)]=self.opens.get(str(p),0)+1
    def summary(self):return dict(phase=self.phase,reads=self.opens,denied=self.denied)

def sources():
    paths=list(HERE.glob('*.py'))+[CONFIG,PROJECT/'docs/deep_baselines_identifiability_v2_protocol_zh.md',PROJECT/'tests/test_baseline_identifiability_v2.py']
    return {str(p):sha(p) for p in paths}

def geometry_only():
    """Direct geometry schema conversion; never instantiate a real dataset."""
    with np.load(GEOMETRY,allow_pickle=False) as z:g={k:z[k].copy() for k in z.files}
    d=design();out={}
    for branch,indices in d['stations'].items():
        indices=np.asarray(indices,dtype=np.int64)
        assert len(indices)==4 and indices.min()>=0 and indices.max()<len(g['station_origin_m'])
        sample=dict(case_id='fixed_geometry',sample_id='fixed_geometry',split='synthetic',geometry_pool='train',
                    projection_scale_ppm_m=torch.tensor(Y_SCALE,dtype=torch.float64))
        for prefix,ids in [('input',indices),('hold',np.empty(0,dtype=np.int64))]:
            sample[prefix+'_station_indices']=torch.from_numpy(ids.copy())
            for k,arr in g.items():
                if k in ('horizontal_angle_deg','vertical_angle_deg'):continue
                value=torch.from_numpy(arr[ids].copy())
                value=value.bool() if k=='valid_ray' else value.float()
                if k in ('valid_ray','path_length_m','t_near','t_far'):value=value[:,None]
                sample[prefix+'_'+k]=value
            zero=torch.zeros_like(sample[prefix+'_path_length_m'])
            sample[prefix+'_projection_raw']=zero
            sample[prefix+'_projection']=zero.clone()
        for k in ('horizontal_angle_deg','vertical_angle_deg'):sample[k]=torch.from_numpy(g[k].copy()).float()
        out[branch]=sample
    assert all(not set(d['stations'][a])&set(d['stations'][b]) for a,b in [('A','B'),('A','H'),('B','H')])
    return out

def nonmeasurement(inp):
    return {k:v for k,v in inp.items() if k not in ('input_projection','input_projection_raw')}
