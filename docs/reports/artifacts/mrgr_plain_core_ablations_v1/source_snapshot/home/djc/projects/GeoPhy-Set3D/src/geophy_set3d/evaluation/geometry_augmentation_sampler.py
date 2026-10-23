"""Explicit natural/moderate/clustered augmentation, independent of GT/errors.

Compatible with DualMeasurementDataset.sampler. Original sampler is unchanged.
"""
import hashlib
import itertools
import numpy as np
from geophy_set3d.data.measurement_only import DisjointABHSampler
from geophy_set3d.evaluation.geometry_strata import pair_scores
from geophy_set3d.evaluation.geometry_sensitivity import circular_span

MODES=('natural','moderate','hard')


def hard_subsets(azimuth, views, span_max_deg):
    azimuth=np.asarray(azimuth,float)%360
    subsets=set()
    for angle in azimuth:
        window=np.flatnonzero((azimuth-angle)%360<=span_max_deg+1e-12)
        for ids in itertools.combinations(window,views):subsets.add(tuple(sorted(map(int,ids))))
    return np.asarray(sorted(subsets),dtype=np.int64).reshape(-1,views)


class GeometryAugmentationABHSampler:
    def __init__(self,origins,azimuth,boundaries,seed=20261011,probabilities=(.5,.25,.25),hard_span_deg=10.,max_attempts=10000):
        self.pairs=pair_scores(origins);self.azimuth=np.asarray(azimuth);self.boundaries=boundaries
        self.seed=seed;self.probabilities=np.asarray(probabilities,float);self.cap=max_attempts
        if len(self.probabilities)!=3 or (self.probabilities<0).any() or not np.isclose(self.probabilities.sum(),1):
            raise ValueError('Three nonnegative mode probabilities summing to one required')
        self.original=DisjointABHSampler(base_seed=seed)
        self.hard={v:hard_subsets(self.azimuth,v,hard_span_deg) for v in (3,4,5)}
        if any(len(h)==0 for h in self.hard.values()):raise ValueError('No hard subset at a requested V')
        # Both independently hard inputs must have at least one disjoint pair.
        for va,vb in itertools.product((3,4,5),repeat=2):
            feasible=any(any(not set(a)&set(b) for b in self.hard[vb]) for a in self.hard[va])
            if not feasible:raise ValueError(f'No feasible disjoint hard V{va}/V{vb}; cannot silently change categories')

    def score(self,ids):
        a,b=np.triu_indices(len(ids),1)
        return float(self.pairs[ids[a],ids[b]].mean())

    def _subset(self,rng,n,mode):
        if mode=='hard':return rng.permutation(self.hard[n][int(rng.integers(len(self.hard[n])))])
        for _ in range(self.cap):
            ids=rng.choice(len(self.azimuth),n,replace=False)
            if mode=='natural':return ids
            low,high=self.boundaries[str(n)]
            if low<=self.score(ids)<high:return ids
        raise RuntimeError('Moderate rejection cap exhausted; no hidden fallback')

    def sample_with_metadata(self,station_count,*,epoch=0,sample_index=0,case_id=''):
        if station_count!=len(self.azimuth):raise ValueError('Pool identity mismatch')
        ca,cb,_=self.original.sample(station_count,epoch=epoch,sample_index=sample_index,case_id=case_id)
        words=np.frombuffer(hashlib.blake2b(('geometry-augmentation-v1\0'+case_id).encode(),digest_size=16).digest(),dtype='<u4')
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,epoch,sample_index,*words]))
        # Targets chosen once. Retry only joint station proposals, never modes or V.
        modes=tuple(rng.choice(MODES,2,p=self.probabilities).tolist())
        for attempt in range(self.cap):
            a=self._subset(rng,len(ca),modes[0]);b=self._subset(rng,len(cb),modes[1])
            if not set(a)&set(b):break
        else:raise RuntimeError('Disjoint pair cap exhausted; no mode/geometry fallback')
        available=sorted(set(range(station_count))-set(a)-set(b));h=rng.choice(available,2,replace=False)
        return (a,b,h),dict(A_mode=modes[0],B_mode=modes[1],joint_attempts=attempt+1,
            A_score=self.score(a),B_score=self.score(b),A_span_deg=float(circular_span(self.azimuth[a])),B_span_deg=float(circular_span(self.azimuth[b])))

    def sample(self,station_count,**identity):return self.sample_with_metadata(station_count,**identity)[0]
