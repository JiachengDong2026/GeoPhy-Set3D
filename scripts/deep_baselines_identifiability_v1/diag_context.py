"""Isolated, train-only diagnostic capabilities; historical code is read-only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT/'scripts/deep_baselines_v1')]
import baseline_context as legacy
c = legacy.c
sys.path.insert(0, str(HERE))
from geophy_set3d.data.measurement_only import DualMeasurementDataset, DisjointABHSampler
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models.ray_senseiver_ss import Y_SCALE
from geophy_set3d.training.utils import move_batch, seed_everything
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.models.baseline_measurement_init_v2 import RayDeepSetsMeasurementInitV2, RayGNOTMeasurementInitV2

ROOT = Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_identifiability_v1')
ART = PROJECT/'docs/reports/artifacts/deep_baselines_identifiability_v1'
DESIGN = PROJECT/'configs/baselines/identifiability_v1/design.json'
PROTOCOL = PROJECT/'docs/deep_baselines_identifiability_v1_protocol_zh.md'
UUID = 'GPU-37424437-e208-62d1-e617-53c421c94a49'
ARMS = ('deepsets', 'gnot')

def design(): return c.read(DESIGN)
def freeze(name, value): c.freeze(ROOT/name, value)
def verified(name): return c.verified(ROOT/name)
def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)

def model(arm, device='cpu', original=False):
    seed_everything(design()['seed'])
    if original: return legacy.build_model(arm,device,design()['seed'])
    cfg=c.read(PROJECT/f'configs/baselines/{arm}/model_repair_v2.json')
    cls=RayDeepSetsMeasurementInitV2 if arm=='deepsets' else RayGNOTMeasurementInitV2
    return cls(**cfg['kwargs']).to(device)

def checkpoint_states(arm):
    out=[dict(name='fresh_v1',path=None,original=True),dict(name='fresh_v2',path=None,original=False)]
    old=legacy.ROOT/f'training_registered_v2/{arm}_seed20261011/checkpoints/best.pt'
    if old.exists():out.append(dict(name='original_failed_best',path=str(old),original=True))
    for version in ('v2','v3'):
        for kind in ('best','last'):
            p=ROOT.parent/f'deep_baselines_repair_v2/pilots/{arm}_{version}_seed20261011/{kind}.pt'
            if p.exists():out.append(dict(name=version+'_'+kind,path=str(p),original=False))
    return out

class TrainGuard(c.AccessGuard):
    def __init__(self, checkpoints=(), synthetic_truth=False):
        super().__init__(splits=['train'],checkpoints=checkpoints,output=ROOT)
        ids=set(design()['train_case_indices'])
        train=[r for r in self.records if r['split']=='train']
        allowed={str((c.DATA/train[i]['train_projection_path']).resolve()) for i in ids}
        self.allowed={k:v for k,v in self.allowed.items() if k in allowed}
        self.synthetic_truth=synthetic_truth
    def audit(self,event,args):
        if self.active and event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            writing=(any(x in args[1] for x in 'wa+') if isinstance(args[1],str) else bool(args[2]&(os.O_WRONLY|os.O_RDWR)))
            forbidden=(not writing and (p==c.DATA/'train_statistics.json' or
                ('synthetic_truth' in p.name and not self.synthetic_truth)))
            if forbidden:
                self.denied.append(dict(path=str(p),reason='diagnostic phase boundary'))
                raise PermissionError('diagnostic phase boundary: '+str(p))
        super().audit(event,args)

def source_dataset():
    """Reuse frozen measurement getters, without reading historical GT statistics."""
    ds=DualMeasurementDataset.__new__(DualMeasurementDataset)
    ds.root=c.DATA;ds.split='train';ds.geometry_pool='train';ds.load_gt128=False
    ds.records=[r for r in c.rows(c.DATA/'source_index.jsonl') if r['split']=='train']
    ds.y_ref=Y_SCALE;ds.normalization='linear_p99_5';ds.measurement_only=True
    with np.load(c.DATA/'geometry/train_geometry.npz',allow_pickle=False) as z:
        ds.geometry={k:z[k] for k in z.files}
    ds.station_count=len(ds.geometry['station_origin_m']);ds.projection_shape=ds.geometry['valid_ray'].shape
    ds._epoch=torch.tensor(0,dtype=torch.int64)
    ds.sampler=DisjointABHSampler((4,4),(4,4),4,design()['seed'])
    return ds

def selected_sample(ds, index, branches=('A','B')):
    row,p=ds._projections(index)
    ids=ds.sampler.sample(ds.station_count,epoch=0,sample_index=index,case_id=row['case_id'])
    groups=dict(zip(('A','B','H'),ids));empty=np.empty(0,dtype=np.int64)
    assert all(not set(groups[a])&set(groups[b]) for a,b in [('A','B'),('A','H'),('B','H')])
    return {k:ds._measurement_sample(row,p,groups[k],empty) for k in branches}

def batch(samples, branch, device='cuda:0'):
    return move_batch(projection_pool_collate([s[branch] for s in samples]),device)

def h_sample(index, synthetic=False):
    if synthetic:
        # Only evaluation has this path capability.
        return torch.load(ROOT/f'inputs/synthetic_H_{index}.pt',map_location='cpu',weights_only=False)
    ds=source_dataset()
    return selected_sample(ds,index,('H',))['H']

def sources():
    paths=list((PROJECT/'src').rglob('*.py'))+list(HERE.glob('*.py'))
    paths+=list((PROJECT/'configs/baselines/identifiability_v1').glob('*.json'))
    paths += [PROTOCOL,PROJECT/'tests/test_baseline_identifiability_v1.py']
    return {str(p):c.sha(p) for p in sorted(paths)}

def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def input_hash(value):
    h=hashlib.sha256()
    for key in sorted(value):
        h.update(key.encode())
        item=value[key]
        h.update((tensor_hash(item) if torch.is_tensor(item) else json.dumps(item,sort_keys=True)).encode())
    return h.hexdigest()

def main_train_anchor(name):
    anchor=c.verified(legacy.ROOT/'main_anchor.json')
    c.verify_hashes(anchor['training_sources']);c.verify_hashes(anchor['protocol_hashes'])
    for role in ('model','reference'):
        assert c.sha(anchor[role]['checkpoint'])==anchor[role]['checkpoint_sha256']
    m=anchor['model'];check=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
    net=c.model_for(m['config'],c.PLAIN,'cpu').eval().requires_grad_(False)
    net.load_state_dict(check['model'],strict=True)
    with TrainGuard(checkpoints=[m['checkpoint']]) as access,torch.no_grad():
        sample=selected_sample(source_dataset(),0,('A',))
        inp=batch([sample],'A','cpu');out=net(inp,return_stages=True)
        result=dict(at=c.now(),input_hash=input_hash(inp),outputs={k:tensor_hash(out[k]) for k in ('C0','C1')},
                    checkpoint_sha256=m['checkpoint_sha256'],reference_sha256=anchor['reference']['checkpoint_sha256'],access=access.summary())
    if name!='main_before.json':
        before=verified('main_before.json')
        assert result['input_hash']==before['input_hash'] and result['outputs']==before['outputs']
    result['passed']=True;freeze(name,result)
    return result
