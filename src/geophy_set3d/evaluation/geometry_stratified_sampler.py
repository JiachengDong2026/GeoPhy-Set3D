"""Matched geometry-only A/B/H schedules; objective and original sampler untouched."""
import hashlib
from pathlib import Path
import json
import numpy as np
from geophy_set3d.data.measurement_only import DisjointABHSampler
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.evaluation.geometry_strata import pair_scores,membership


class StratifiedABHSampler:
    def __init__(self,origins,boundaries,seed=20261011,max_attempts=10000):
        self.pairs=pair_scores(origins);self.boundaries=boundaries;self.seed=seed;self.max_attempts=max_attempts
        self.original=DisjointABHSampler(base_seed=seed)
        self.triu={v:np.triu_indices(v,1) for v in (3,4,5)}
    def score(self,ids):
        a,b=self.triu[len(ids)];return float(self.pairs[ids[a],ids[b]].mean())
    def sample(self,station_count,*,epoch=0,sample_index=0,case_id=''):
        ca,cb,_=self.original.sample(station_count,epoch=epoch,sample_index=sample_index,case_id=case_id)
        words=np.frombuffer(hashlib.blake2b(('stratified-ab-v1\0'+case_id).encode(),digest_size=16).digest(),dtype='<u4')
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,epoch,sample_index,*words]))
        q=('low','middle','high')[int(rng.integers(3))]
        selected=[];available=np.arange(station_count)
        for n in (len(ca),len(cb)):
            for _ in range(self.max_attempts):
                ids=rng.choice(available,n,replace=False)
                if membership(self.score(ids),self.boundaries[str(n)])==q:break
            else:raise RuntimeError('Registered rejection cap reached; no fallback/reweighting')
            selected.append(ids);available=np.asarray(sorted(set(available)-set(ids)),dtype=np.int64)
        h=rng.choice(available,2,replace=False)
        return *selected,h,q


class PlannedABHSampler:
    def __init__(self,epochs):self.epochs=epochs
    def sample(self,station_count,*,epoch=0,sample_index=0,case_id=''):
        r=self.epochs[epoch][sample_index];assert r['case_id']==case_id
        return tuple(np.asarray(r[k],dtype=np.int64) for k in ('A','B','H'))


class SamplerTrainingReadGuard(GTReadGuard):
    """Extend the original guard prospectively: count Pool B/test and block data writes."""
    def __init__(self,dataset_root,checkpoint_root,output):
        super().__init__(dataset_root,checkpoint_root);self.output=Path(output).resolve()
        self.pool_b_attempts=0;self.test_attempts=0;self.data_write_attempts=0
    def _audit(self,event,args):
        if self.active and event=='open' and isinstance(args[0],(str,bytes,Path)):
            path=Path(args[0]).resolve();mode,flags=args[1:3]
            writing=any(c in mode for c in 'wa+') if isinstance(mode,str) else bool(flags&3)
            if writing and (path.is_relative_to(self.root) or str(path) in self.gt64 or str(path) in self.gt128):
                self.data_write_attempts+=1;raise PermissionError('Frozen data write denied')
            if not writing:
                if path.name=='ood_geometry.npz' or ('projections' in path.parts and 'ood' in path.parts):
                    self.pool_b_attempts+=1;raise PermissionError('Pool B denied')
                if path.suffix=='.npy' and str(path) not in self.allowed and str(path) not in self.gt64 and str(path) not in self.gt128:
                    self.test_attempts+=1
                if path.suffix=='.npz' and path!=self.root/'geometry/train_geometry.npz' and not path.is_relative_to(self.output):
                    raise PermissionError('Only frozen Pool A geometry and this experiment caches')
        super()._audit(event,args)
    def summary(self):
        return dict(**super().summary(),pool_b_open_attempts=self.pool_b_attempts,test_or_unknown_npy_attempts=self.test_attempts,data_write_attempts=self.data_write_attempts)
