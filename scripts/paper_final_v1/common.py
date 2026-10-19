"""Isolated paper experiment utilities; historical numerical sources stay intact."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
DATA = Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
OUT = ROOT/'outputs/research/mrgr_paper_final_v1'
ORIGINAL = ROOT/'outputs/research/mrgr_multiscale_formal_v1_seed20261011'
LR = ROOT/'outputs/research/mrgr_optimization_v2_lr4e4_seed20261011'
OLD_EVAL = ORIGINAL/'evaluation/frozen_model_test_v1'


def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def jsonl(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in v));t.replace(p)
def append(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush()
def freeze(p,v):
    p=Path(p)
    if p.exists(): raise FileExistsError(p)
    write(p,v);p.with_suffix(p.suffix+'.sha256').write_text(sha(p)+'\n')
def verified(p):
    p=Path(p);assert sha(p)==p.with_suffix(p.suffix+'.sha256').read_text().strip(),p
    return read(p)
def sources():
    paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'src').rglob('*.cpp'))+list((ROOT/'scripts/paper_final_v1').glob('*.py'))
    paths += [ROOT/'scripts/probe_mrgr_multiscale_v1.py',ROOT/'scripts/train_mrgr_multiscale_formal_v1.py',ROOT/'scripts/report_historical_reconstruction_v1.py']
    return {str(p):sha(p) for p in sorted(paths)}
def verify_hashes(mapping):
    for p,h in mapping.items():assert sha(p)==h,p

def backend():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False

def resources():
    def run(args):return subprocess.check_output(args,text=True).strip()
    return dict(at=now(),gpus=run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory','--format=csv,noheader,nounits']),
        gpu_processes=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']),
        processes=run(['ps','-eo','user,pid,etime,pcpu,pmem,args']),memory=run(['free','-b']),disk=run(['df','-B1',str(OUT),'/data','/scratch']),load=os.getloadavg())

class AccessGuard:
    """Allowlist read/write audit. Test GT requires a verified global prediction seal."""
    def __init__(self, *, splits=(), gt_splits=(), checkpoints=(), output=OUT):
        self.active=False;self.output=Path(output).resolve();self.allowed={};self.counts={};self.denied=[]
        self.records=rows(DATA/'source_index.jsonl')
        self.gt={str((DATA/r['gt_volume_64_path']).resolve()):r['split'] for r in self.records}
        self.raw={str(Path(r['source_target_path']).resolve()) for r in self.records}
        self.ckpts={str(Path(p).resolve()) for p in checkpoints}
        for r in self.records:
            if r['split'] in splits:self.allowed[str((DATA/r['train_projection_path']).resolve())]='projection_'+r['split']
            if r['split'] in gt_splits:self.allowed[str((DATA/r['gt_volume_64_path']).resolve())]='GT64_'+r['split']
        self.touched=set()
    def audit(self,event,args):
        if not self.active or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p);mode,flags=args[1:3]
        writing=any(x in mode for x in 'wa+') if isinstance(mode,str) else bool(flags&(os.O_WRONLY|os.O_RDWR))
        reason=None;label=None
        if writing:
            if p.is_relative_to(DATA) or s in self.raw:reason='dataset is immutable'
        elif s in self.raw:reason='GT128 forbidden'
        elif s in self.allowed:label=self.allowed[s]
        elif s in self.gt:reason='GT64 outside registered phase'
        elif p.suffix in ('.pt','.pth','.ckpt'):
            if s in self.ckpts or p.is_relative_to(self.output):label='checkpoint'
            else:reason='foreign checkpoint'
        elif p.suffix in ('.npy','.npz'):
            if p==DATA/'geometry/train_geometry.npz':label='Pool_A_geometry'
            elif p.is_relative_to(self.output):label='own_array'
            else:reason='array outside phase allowlist'
        if reason:
            self.denied.append(dict(path=s,reason=reason));raise PermissionError(reason+': '+s)
        if label:self.counts[label]=self.counts.get(label,0)+1;self.touched.add(s)
    def __enter__(self):sys.addaudithook(self.audit);self.active=True;return self
    def __exit__(self,*args):self.active=False
    def summary(self):return dict(counts=self.counts,denied=self.denied,touched=sorted(self.touched))

class Replay:
    def __init__(self,s):self.s=s
    def sample(self,station_count,*,epoch,sample_index,case_id):
        import numpy as np
        assert station_count==64
        return np.array(self.s['input_ids'][case_id]),np.array(self.s['heldout_ids'].get(case_id, []),dtype=np.int64)

def dataset(split,s):
    # Explicit independent adapter: the historical selfsup dataset deliberately forbids test.
    # Reuse its measurement-only getter, initialized by the metadata-only base class.
    from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
    from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
    ds=MeasurementOnlyPoolDataset.__new__(MeasurementOnlyPoolDataset)
    ProjectionPoolDataset.__init__(ds,DATA,split=split,geometry_pool='train',min_input_views=s['views'],max_input_views=s['views'],hold_views=0,
                                  base_seed=20261011,normalization='linear_p99_5',statistics_file='train_statistics.json')
    ds.measurement_only=True;ds.sampler=Replay(s)
    return ds
