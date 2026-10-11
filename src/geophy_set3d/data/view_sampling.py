"""Stateless, reproducible station sampling without global RNG state."""
from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class ViewSampler:
    min_input_views: int = 3
    max_input_views: int = 6
    hold_views: int = 1
    base_seed: int = 20261011

    def __post_init__(self):
        if not 1 <= self.min_input_views <= self.max_input_views:
            raise ValueError('Require 1 <= min_input_views <= max_input_views')
        if self.hold_views < 0 or self.base_seed < 0:
            raise ValueError('hold_views and base_seed must be nonnegative')

    def sample(self, station_count, *, epoch=0, sample_index=0, case_id=''):
        if station_count < self.max_input_views + self.hold_views:
            raise ValueError('Insufficient candidate stations for disjoint input/hold views')
        if min(epoch, sample_index) < 0:
            raise ValueError('epoch and sample_index must be nonnegative')
        # Python hash() is process-dependent; this seed is stable across workers/runs.
        identity = hashlib.blake2b(str(case_id).encode('utf-8'), digest_size=16).digest()
        words = np.frombuffer(identity, dtype='<u4').astype(np.uint32).tolist()
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.base_seed, epoch, sample_index, *words]))
        count = int(rng.integers(self.min_input_views, self.max_input_views + 1))
        selected = rng.choice(station_count, count + self.hold_views, replace=False)
        inputs = rng.permutation(selected[:count]).astype(np.int64)
        holds = selected[count:].astype(np.int64)
        return inputs, holds
