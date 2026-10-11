"""B3-only held-out sampling that preserves every legacy input station choice."""
import hashlib
import numpy as np
from geophy_set3d.training.engine import make_dataset


class IndependentHoldSampler:
    def __init__(self, input_sampler, base_seed=20261011):
        self.input_sampler, self.base_seed = input_sampler, base_seed

    def sample(self, station_count, *, epoch=0, sample_index=0, case_id=''):
        inputs, _ = self.input_sampler.sample(station_count, epoch=epoch,
                                              sample_index=sample_index, case_id=case_id)
        remaining = np.setdiff1d(np.arange(station_count), inputs)
        if not len(remaining):
            raise ValueError('No disjoint held-out station available')
        identity = hashlib.blake2b(('b3-hold\0'+str(case_id)).encode(), digest_size=16).digest()
        words = np.frombuffer(identity, dtype='<u4').tolist()
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.base_seed, epoch, sample_index, *words]))
        return inputs, rng.choice(remaining, 1, replace=False).astype(np.int64)


def attach_hold(dataset, seed=20261011):
    dataset.sampler = IndependentHoldSampler(dataset.sampler, seed)
    return dataset


def measurement_dataset(config, validation=False):
    settings = config['validation'] if validation else config['train']
    if settings['geometry_pool'] != 'train' or settings['split'] != ('val' if validation else 'train'):
        raise ValueError('B3 training/selection is restricted to Pool A train/val')
    # Construct the original hold=0 sampler; adding a hold must not change its RNG consumption.
    import copy
    legacy = copy.deepcopy(config)
    legacy['train']['hold_views'] = legacy['validation']['hold_views'] = 0
    return attach_hold(make_dataset(legacy, validation), config['training']['seed'])
