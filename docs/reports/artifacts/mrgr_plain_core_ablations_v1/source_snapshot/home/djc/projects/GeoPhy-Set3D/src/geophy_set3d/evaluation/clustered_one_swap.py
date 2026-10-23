"""Geometry-only single replacements, case-level pairing and independent read audit."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def directions(origins, center=(.3,.3,.3)):
    origins=np.asarray(origins,dtype=np.float64);center=np.asarray(center,dtype=np.float64)
    if origins.ndim!=2 or origins.shape[1]!=3 or not np.isfinite(origins).all():raise ValueError('Finite 3D station origins required')
    d=center-origins;n=np.linalg.norm(d,axis=1)
    if (n==0).any():raise ValueError('Station at domain center has no direction')
    return d/n[:,None]


def geometry_description(ids,origins,center=(.3,.3,.3)):
    if len(ids)!=4 or len(set(ids))!=4:raise ValueError('Exactly four distinct station IDs')
    d=directions(origins,center)[sorted(ids)]
    dot=np.clip(np.abs(d@d.T),0,1)[np.triu_indices(4,1)]
    angles=np.degrees(np.arccos(dot));q=d.T@d/4
    return dict(score=float(np.mean(1-dot)),acute_crossing_angles_deg=angles.tolist(),
        crossing_angle_min_deg=float(angles.min()),crossing_angle_mean_deg=float(angles.mean()),
        crossing_angle_max_deg=float(angles.max()),direction_tensor_eigenvalues=np.linalg.eigvalsh(q).tolist())


def targeted_replacement(original,origins,center=(.3,.3,.3)):
    """No measurements, GT, model or errors are accepted by this interface."""
    original=list(map(int,original));n=len(origins)
    if len(original)!=4 or len(set(original))!=4 or not set(original)<=set(range(n)):raise ValueError('Invalid Original V4')
    candidates=sorted(set(range(n))-set(original))
    if not candidates:raise ValueError('No replacement candidate')
    best=None
    for deleted in sorted(original):
        for added in candidates:
            ids=[added if i==deleted else i for i in original]
            score=geometry_description(ids,origins,center)['score']
            key=(-score,deleted,added)
            if best is None or key<best[0]:best=(key,ids,deleted,added,score)
    return dict(station_ids=best[1],deleted=best[2],added=best[3],score=best[4])


def keyed_rng(seed,draw,case_id,purpose):
    identity=hashlib.blake2b((purpose+'\0'+case_id).encode(),digest_size=16).digest()
    words=np.frombuffer(identity,dtype='<u4').tolist()
    return np.random.default_rng(np.random.SeedSequence([seed,draw,*words]))


def paired_station_sets(original,origins,case_id,draw,random_seed,heldout_seed,center=(.3,.3,.3)):
    t=targeted_replacement(original,origins,center)
    outside=sorted(set(range(len(origins)))-set(original))
    added=int(keyed_rng(random_seed,draw,case_id,'random-replacement').choice(outside))
    random=[added if i==t['deleted'] else int(i) for i in original]
    available=sorted(set(range(len(origins)))-set(original)-set(t['station_ids'])-set(random))
    if len(available)<4:raise ValueError('Insufficient disjoint common H4')
    heldout=sorted(keyed_rng(heldout_seed,draw,case_id,'common-H4').choice(available,4,replace=False).tolist())
    return dict(Original=list(map(int,original)),Targeted=t['station_ids'],Random=random,H=heldout,
        deleted=t['deleted'],targeted_added=t['added'],random_added=added,random_equals_targeted=added==t['added'])


def case_aggregate(rows,metric,condition,model):
    selected=[r for r in rows if r['condition']==condition and r['model']==model]
    cases={}
    for r in selected:cases.setdefault(r['case_id'],[]).append(r)
    result={}
    for case,items in cases.items():
        if len(items)!=5 or sorted(r['draw'] for r in items)!=list(range(5)):raise ValueError('Exactly five unique paired draws per case')
        result[case]=float(np.mean([r[metric] for r in items]))
    return result


def paired_statistics(rows,metric,model,left,right,samples=10000,seed=2026101205):
    a=case_aggregate(rows,metric,left,model);b=case_aggregate(rows,metric,right,model)
    if set(a)!=set(b) or not a:raise ValueError('Paired case membership mismatch')
    cases=sorted(a);av=np.array([a[k] for k in cases]);bv=np.array([b[k] for k in cases]);delta=av-bv
    rng=np.random.default_rng(seed)
    ci=np.quantile(delta[rng.integers(0,len(delta),(samples,len(delta)))].mean(1),[.025,.975])
    decision='improvement' if delta.mean()<0 and ci[1]<0 else 'worsening' if delta.mean()>0 and ci[0]>0 else 'inconclusive'
    return dict(model=model,metric=metric,comparison=left+'-'+right,case_count=len(cases),draws_per_case=5,
        left_mean=float(av.mean()),right_mean=float(bv.mean()),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),
        relative_improvement=-float(delta.mean())/float(bv.mean()) if bv.mean()!=0 else None,
        wins=int((delta<0).sum()),losses=int((delta>0).sum()),ties=int((delta==0).sum()),
        case_paired_95ci=ci.tolist(),decision=decision,bootstrap_samples=samples,bootstrap_seed=seed,
        per_case=[dict(case_id=k,left_mean=a[k],right_mean=b[k],delta=a[k]-b[k]) for k in cases])


class ClusteredEvaluationReadGuard:
    """Independent GT64 evaluation policy; original training guards remain untouched."""
    def __init__(self,dataset_root,checkpoints,output,geometry_only=False):
        self.root=Path(dataset_root).resolve();self.output=Path(output).resolve();self.geometry_only=geometry_only
        records=[json.loads(l) for l in (self.root/'source_index.jsonl').read_text().splitlines() if l.strip()]
        self.projections={(self.root/r['train_projection_path']).resolve() for r in records if r['split']=='val'}
        self.gt64={(self.root/r['gt_volume_64_path']).resolve() for r in records if r['split']=='val'}
        self.gt128={Path(r['source_target_path']).resolve() for r in records}
        self.checkpoints={Path(p).resolve() for p in checkpoints}
        self.counts=dict(val_pool_a_projection_opens=0,val_gt64_opens=0,checkpoint_opens=0,geometry_opens=0,
            forbidden_attempts=0,gt128_attempts=0,foreign_checkpoint_attempts=0,dataset_write_attempts=0,
            pool_b_attempts=0,nonval_or_unknown_npy_attempts=0,checkpoint_write_attempts=0)
        self.active=False;self.touched_gt=set();self.touched_projection=set()

    def deny(self,key,message):
        self.counts[key]+=1;self.counts['forbidden_attempts']+=1;raise PermissionError(message)

    def _audit(self,event,args):
        if not self.active or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        writing=any(c in mode for c in 'wa+') if isinstance(mode,str) else bool(flags&(os.O_WRONLY|os.O_RDWR))
        if writing:
            if p.is_relative_to(self.root) or p in self.gt128:self.deny('dataset_write_attempts','Frozen data writes forbidden')
            if p.suffix in ('.pt','.pth','.ckpt'):self.deny('checkpoint_write_attempts','Frozen checkpoints cannot be written')
            return
        if p in self.gt128:self.deny('gt128_attempts','GT128 is outside this diagnostic')
        if p.suffix in ('.pt','.pth','.ckpt'):
            if p not in self.checkpoints:self.deny('foreign_checkpoint_attempts','Only the two registered frozen checkpoints')
            self.counts['checkpoint_opens']+=1
        elif p.suffix=='.npy':
            if self.geometry_only:self.deny('nonval_or_unknown_npy_attempts','Registration reads geometry only, never measurements/GT')
            if p in self.projections:self.counts['val_pool_a_projection_opens']+=1;self.touched_projection.add(str(p))
            elif p in self.gt64:self.counts['val_gt64_opens']+=1;self.touched_gt.add(str(p))
            else:
                key='pool_b_attempts' if 'ood' in p.name else 'nonval_or_unknown_npy_attempts'
                self.deny(key,'Only Val Pool A measurements and Val GT64 are permitted')
        elif p.suffix=='.npz':
            if p==self.root/'geometry/train_geometry.npz':self.counts['geometry_opens']+=1
            elif not self.geometry_only and p.is_relative_to(self.output/'case_images'):pass
            else:self.deny('pool_b_attempts' if p.name=='ood_geometry.npz' else 'nonval_or_unknown_npy_attempts','Unregistered geometry/cache forbidden')

    def __enter__(self):self.active=True;sys.addaudithook(self._audit);return self
    def __exit__(self,*args):
        self.active=False
        with (self.output/'access_sessions.jsonl').open('a') as f:
            f.write(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),failed=args[0] is not None,**self.summary()))+'\n')
    def summary(self):
        return dict(**self.counts,geometry_only=self.geometry_only,unique_val_gt64_files=len(self.touched_gt),
            unique_val_pool_a_projections=len(self.touched_projection),optimizer_created=False,optimizer_steps=0,
            policy='Registration: metadata/geometry/two frozen checkpoints only. Evaluation: add only Val GT64 and Pool A measurements; no GT128, Pool B or test.')
