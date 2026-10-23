"""Independent full train/Val GT64 access, original sampling, and resumable metrics."""
import json,os,sys
from collections import Counter
from pathlib import Path
import torch
from geophy_set3d.training.engine import make_dataset
from geophy_set3d.training.senseiver_supervised_probe import separate,volume_metrics,metrics_mean
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch


class FormalGT64Guard:
    def __init__(self,root,own_root,phase='train'):
        self.root=Path(root).resolve();self.own=Path(own_root).resolve();self.phase=phase
        rows=[json.loads(x) for x in (self.root/'source_index.jsonl').read_text().splitlines() if x.strip()]
        self.allowed={str((self.root/r[k]).resolve()):(r['split'],k) for r in rows if r['split'] in ['train','val'] for k in ['train_projection_path','gt_volume_64_path']}
        self.reads=Counter();self.denied=[];self.active=False
    def _audit(self,event,args):
        if not self.active or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        reading=('r' in mode or '+' in mode) if isinstance(mode,str) else not bool(flags&os.O_WRONLY)
        writing=any(x in mode for x in ['w','a','+']) if isinstance(mode,str) else bool(flags&(os.O_WRONLY|os.O_RDWR))
        if p.is_relative_to(self.root) and writing:self.denied.append(str(p));raise PermissionError('Frozen data read-only')
        if not reading:return
        if p.suffix=='.npy':
            info=self.allowed.get(str(p))
            if info is None and not p.is_relative_to(self.own):self.denied.append(str(p));raise PermissionError('Only official train/Val Pool A and GT64')
            if info:self.reads[(str(p),*info)]+=1
        if p.suffix=='.npz' and p!=self.root/'geometry/train_geometry.npz' and not p.is_relative_to(self.own):self.denied.append(str(p));raise PermissionError('Pool B/foreign array forbidden')
        if p.suffix in ['.pt','.pth','.ckpt'] and not p.is_relative_to(self.own):self.denied.append(str(p));raise PermissionError('Only this fresh experiment checkpoints')
    def __enter__(self):self.active=True;sys.addaudithook(self._audit);return self
    def __exit__(self,*args):self.active=False
    def summary(self):return dict(policy='Official train/Val GT64 and Pool A only; no GT128/test/Pool B/foreign weights; single-process auditable loader',denied_attempts=self.denied,reads=[dict(path=p,split=s,kind=k,count=n) for (p,s,k),n in sorted(self.reads.items())])


def datasets(c):
    bridge=dict(dataset=dict(root=c['dataset_root']),train=c['train'],validation=c['validation'],normalization=c['normalization'],training=dict(seed=c['seed']))
    return make_dataset(bridge),make_dataset(bridge,validation=True)


def batch(ds,indices,device):
    inputs=[];targets=[]
    for i in indices:
        x,g=separate(ds[i]);inputs.append(x);targets.append(g)
    return move_batch(projection_pool_collate(inputs),device),torch.stack(targets).to(device)


def metric_rows(pred,gt,ids,indices):
    return [dict(case_id=ids[j],index=indices[j],**volume_metrics(pred[j],gt[j])) for j in range(len(indices))]


def summarize(rows):
    return dict(mean=metrics_mean(rows),cases=rows,mean_peak=sum(r['peak'] for r in rows)/len(rows),
        mean_spatial_std=sum(r['spatial_std'] for r in rows)/len(rows),mean_total_ratio=sum(r['total_ratio'] for r in rows)/len(rows),
        mean_plume_relative_l1=sum(r['plume']['relative_l1'] for r in rows)/len(rows))


def validate_last(saved,c,registration):
    if saved['config']!=c or saved['registration_sha256']!=registration:raise ValueError('Different formal registration/config')
    if saved['exit_state'].get('reason'):raise ValueError('Scientifically stopped run cannot resume')
    for key in ['optimizer','scheduler','rng','epoch','position','order','train_rows','history','updates','elapsed_seconds','best','best_epoch','exit_state']:
        if key not in saved:raise ValueError('Incomplete formal resume: '+key)
    if saved['position']>83 or saved['scheduler']['last_epoch']!=saved['epoch']:raise ValueError('Epoch/cursor/scheduler mismatch')
