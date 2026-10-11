"""Strict measurement-only access and deterministic disjoint A/B/H sampling."""
import copy
import hashlib
from dataclasses import dataclass
import numpy as np
import torch
from .projection_pool_dataset import ProjectionPoolDataset, artifact_path
from .collate import projection_pool_collate


def assert_no_gt(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.startswith(('gt_', 'source_target', 'fluent_volume')):
                raise ValueError(f'Volumetric ground truth is forbidden: {key}')
            assert_no_gt(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_gt(item)


@dataclass(frozen=True)
class DisjointABHSampler:
    a_range: tuple = (3, 5)
    b_range: tuple = (3, 5)
    hold_views: int = 2
    base_seed: int = 20261011

    def __post_init__(self):
        if any(not 1 <= r[0] <= r[1] for r in (self.a_range, self.b_range)) or self.hold_views < 1:
            raise ValueError('Invalid A/B/H view counts')

    def sample(self, station_count, *, epoch=0, sample_index=0, case_id=''):
        if station_count < self.a_range[1]+self.b_range[1]+self.hold_views:
            raise ValueError('Insufficient stations for pairwise-disjoint A/B/H')
        if min(epoch, sample_index, self.base_seed) < 0:
            raise ValueError('Seeds/epoch/index must be nonnegative')
        identity=hashlib.blake2b(('selfsup-mrgr-v1\0'+str(case_id)).encode(),digest_size=16).digest()
        words=np.frombuffer(identity,dtype='<u4').tolist()
        rng=np.random.default_rng(np.random.SeedSequence([self.base_seed,epoch,sample_index,*words]))
        na=int(rng.integers(self.a_range[0],self.a_range[1]+1))
        nb=int(rng.integers(self.b_range[0],self.b_range[1]+1))
        selected=rng.choice(station_count,na+nb+self.hold_views,replace=False).astype(np.int64)
        return selected[:na],selected[na:na+nb],selected[na+nb:]


class MeasurementOnlyPoolDataset(ProjectionPoolDataset):
    """The inherited constructor reads only metadata/statistics/geometry.

    This independent __getitem__ never opens any GT path, even when GT files
    are absent. Frozen ProjectionPoolDataset and its supervised users are intact.
    """
    def __init__(self, dataset_root, *, load_gt=False, load_gt128=False,
                 evaluation_only=False, **settings):
        if load_gt or load_gt128:
            raise ValueError('Measurement-only access forbids GT64 and GT128')
        if settings.get('split','train') == 'test':
            raise ValueError('The test split is outside selfsup v1')
        if settings.get('geometry_pool','train') != 'train' and not evaluation_only:
            raise ValueError('Pool B is forbidden before post-training evaluation')
        super().__init__(dataset_root,load_gt128=False,**settings)
        self.measurement_only=True

    def _projections(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        row=self.records[index]
        p=np.load(artifact_path(self.root,row[f'{self.geometry_pool}_projection_path']),mmap_mode='r',allow_pickle=False)
        if p.shape != self.projection_shape or p.dtype != np.float32:
            raise ValueError('Invalid frozen projection schema')
        return row,p

    def _measurement_sample(self, row, projections, inputs, holds):
        result=dict(case_id=row['case_id'],sample_id=row['sample_id'],split=self.split,
                    geometry_pool=self.geometry_pool,condition_meta=copy.deepcopy(row['condition_meta']),
                    projection_scale_ppm_m=torch.tensor(self.y_ref,dtype=torch.float64))
        result.update(self._views(projections,inputs,'input'))
        result.update(self._views(projections,holds,'hold'))
        for key in ('horizontal_angle_deg','vertical_angle_deg'):
            result[key]=torch.from_numpy(self.geometry[key].copy()).float()
        assert_no_gt(result)
        return result

    def __getitem__(self, index):
        row,p=self._projections(index)
        inputs,holds=self.sampler.sample(self.station_count,epoch=int(self._epoch.item()),sample_index=index,case_id=row['case_id'])
        return self._measurement_sample(row,p,inputs,holds)


class DualMeasurementDataset(MeasurementOnlyPoolDataset):
    def __init__(self, dataset_root, *, a_range=(3,5),b_range=(3,5),hold_views=2,**settings):
        super().__init__(dataset_root,min_input_views=a_range[0],max_input_views=a_range[1],hold_views=0,**settings)
        self.sampler=DisjointABHSampler(tuple(a_range),tuple(b_range),hold_views,settings.get('base_seed',20261011))
        if self.station_count < a_range[1]+b_range[1]+hold_views:
            raise ValueError('Insufficient disjoint candidate stations')

    def __getitem__(self,index):
        row,p=self._projections(index)
        a,b,h=self.sampler.sample(self.station_count,epoch=int(self._epoch.item()),sample_index=index,case_id=row['case_id'])
        empty=np.empty(0,dtype=np.int64)
        return {k:self._measurement_sample(row,p,ids,empty) for k,ids in (('A',a),('B',b),('H',h))}


def dual_measurement_collate(samples):
    assert_no_gt(samples)
    return {k:projection_pool_collate([s[k] for s in samples]) for k in ('A','B','H')}
