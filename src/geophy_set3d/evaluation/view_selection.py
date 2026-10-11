"""Evaluation-only station selection; the training ViewSampler remains unchanged."""
import hashlib
import numpy as np


def coverage_descriptor(azimuths):
    a = np.sort(np.asarray(azimuths, dtype=np.float64))
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('At least two finite azimuths required')
    pairwise = np.abs(a[:, None]-a[None, :])[np.triu_indices(len(a), 1)]
    return dict(azimuth_min_deg=float(a[0]), azimuth_max_deg=float(a[-1]),
                azimuth_span_deg=float(a[-1]-a[0]), largest_azimuth_gap_deg=float(np.diff(a).max()),
                mean_pairwise_azimuth_distance_deg=float(pairwise.mean()))


class EvaluationViewSampler:
    def __init__(self, azimuth_deg, views, draw_id, pool, base_seed=20261011, protocol='random'):
        self.azimuths = np.asarray(azimuth_deg, dtype=np.float64).copy()
        if pool not in ('train', 'ood') or protocol not in ('random', 'spread', 'clustered'):
            raise ValueError('Unknown pool or selection protocol')
        if not 2 <= views <= len(self.azimuths) or draw_id < 0:
            raise ValueError('Invalid view count/draw')
        self.views, self.draw_id, self.pool = views, draw_id, pool
        self.base_seed, self.protocol = base_seed, protocol

    def select(self, case_id):
        identity = f'{self.pool}\0{case_id}'.encode()
        words = np.frombuffer(hashlib.blake2b(identity, digest_size=16).digest(), dtype='<u4')
        rng = np.random.default_rng(np.random.SeedSequence([self.base_seed, self.views, self.draw_id, *words]))
        if self.protocol == 'random':
            return rng.choice(len(self.azimuths), self.views, replace=False).astype(np.int64)
        order = np.argsort(self.azimuths, kind='stable')
        if self.protocol == 'spread':
            # Endpoints maximize span; remaining stations fill gaps via deterministic farthest-point selection.
            chosen = [int(order[0]), int(order[-1])]
            while len(chosen) < self.views:
                distance = np.abs(self.azimuths[:, None]-self.azimuths[chosen][None]).min(1)
                distance[chosen] = -1
                chosen.append(int(np.argmax(distance)))
            return np.asarray(sorted(chosen, key=lambda i: (self.azimuths[i], i)), dtype=np.int64)
        windows = [order[i:i+self.views] for i in range(len(order)-self.views+1)]
        windows.sort(key=lambda w: (self.azimuths[w[-1]]-self.azimuths[w[0]], tuple(w)))
        if self.draw_id >= len(windows):
            raise ValueError('Clustered draw exceeds number of available windows')
        return windows[self.draw_id].astype(np.int64)

    def sample(self, station_count, *, epoch, sample_index, case_id):
        if station_count != len(self.azimuths):
            raise ValueError('Geometry pool station count mismatch')
        return self.select(case_id), np.empty(0, dtype=np.int64)
