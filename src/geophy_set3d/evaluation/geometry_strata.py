"""Geometry-only populations and immutable, V-specific evaluation strata.

Boundary functions accept station origins only. No measurement, model or GT API.
"""
import itertools
import math
import numpy as np

STRATA = ('low', 'middle', 'high')
SECTIONS = ('controlled', 'natural', 'clustered')


def pair_scores(origins):
    d = .3 - np.asarray(origins, dtype=np.float64)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return 1 - np.abs(d @ d.T)


def subset_scores(subsets, pairs):
    ids = np.asarray(subsets, dtype=np.int64)
    a, b = np.triu_indices(ids.shape[1], 1)
    return pairs[ids[:, a], ids[:, b]].mean(1)


def population(origins, views, seed, mc_count=100000):
    n = len(origins)
    total = math.comb(n, views)
    if views in (3, 4) or total <= mc_count:
        ids = np.asarray(list(itertools.combinations(range(n), views)), dtype=np.int16)
        mode = 'exhaustive'
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed, views]))
        chosen = set()
        while len(chosen) < mc_count:
            chosen.add(tuple(sorted(rng.choice(n, views, replace=False).tolist())))
        ids = np.asarray(sorted(chosen), dtype=np.int16)
        mode = 'unique_deterministic_monte_carlo'
    scores = subset_scores(ids, pair_scores(origins))
    quantiles = np.quantile(scores, [.01, .05, .25, 1/3, .5, 2/3, .75, .95, .99])
    stats = dict(V=views, population_size=len(ids), possible_combinations=total,
                 generation_mode=mode, seed=seed, min=float(scores.min()), max=float(scores.max()),
                 mean=float(scores.mean()), std=float(scores.std()),
                 **{k:float(v) for k,v in zip(('q01','q05','q25','q33','q50','q67','q75','q95','q99'), quantiles)},
                 boundaries=[float(quantiles[3]), float(quantiles[5])])
    return ids, scores, stats


def membership(score, boundaries):
    return STRATA[int(np.searchsorted(boundaries, score, side='right'))]


def frozen_subsets(ids, scores, boundaries, seed, views, draws=5):
    rng = np.random.default_rng(np.random.SeedSequence([seed, views]))
    bins = np.searchsorted(boundaries, scores, side='right')
    result = []
    for index, stratum in enumerate(STRATA):
        eligible = np.flatnonzero(bins == index)
        for draw, j in enumerate(rng.choice(eligible, draws, replace=False)):
            # Random order is part of the frozen acquisition input identity.
            ordered = rng.permutation(ids[j]).astype(int).tolist()
            result.append(dict(geometry_id=f'V{views}-{stratum}-d{draw}', V=views,
                               stratum=stratum, draw=draw, station_ids=ordered,
                               center_direction_score=float(scores[j])))
    return result


def main_strata(rows):
    return [r for r in rows if r['section'] == 'controlled']


def validate_shared_inputs(rows):
    by_key = {}
    for r in rows:
        key = (r['section'], r['V'], r['stratum'], r['case_id'], r['draw'])
        value = (r['input_station_ids'], r['H_station_ids'])
        if key in by_key and by_key[key] != value:
            raise ValueError('Paired models have different station IDs/order/H')
        by_key[key] = value
