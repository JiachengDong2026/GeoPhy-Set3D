"""Independent geometry diagnostics; never consumes measurements, GT or a model."""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull, QhullError
from scipy.linalg import eigvalsh
from geophy_set3d.reconstruction.classical import RayMatrix


def circular_span(azimuth):
    a = np.sort(np.asarray(azimuth) % 360, axis=-1)
    gaps = np.diff(np.concatenate([a, a[..., :1] + 360], axis=-1), axis=-1)
    return 360 - gaps.max(axis=-1)


def group_masks(scores, spans, boundaries, max_span):
    q05, q40, q60, q80 = boundaries
    return dict(G1=scores >= q80, G2=(scores >= q60) & (scores < q80),
                G3=(scores >= q40) & (scores < q60),
                G4=(scores >= q05) & (scores < q40),
                G5=(scores < q05) & (spans <= max_span))


def spatial_descriptors(origins, azimuth):
    o = np.asarray(origins, dtype=np.float64)
    d = np.linalg.norm(o[:, None] - o[None], axis=-1)[np.triu_indices(len(o), 1)]
    center = .3 - o
    center /= np.linalg.norm(center, axis=1, keepdims=True)
    cosine = np.clip(np.abs(center @ center.T)[np.triu_indices(len(o), 1)], 0, 1)
    angles = np.degrees(np.arccos(cosine))
    def hull(points):
        try:
            h = ConvexHull(points)
            return float(h.volume), float(h.area)
        except QhullError:
            return 0., 0.
    area, perimeter = hull(o[:, [0, 2]])
    volume, surface = hull(o)
    return dict(pairwise_station_distance_m=d.tolist(), min_pairwise_distance_m=float(d.min()),
                mean_pairwise_distance_m=float(d.mean()), max_pairwise_distance_m=float(d.max()),
                horizontal_hull_area_m2=area, horizontal_hull_perimeter_m=perimeter,
                station_hull_volume_m3=volume, station_hull_surface_area_m2=surface,
                center_acute_angles_deg=angles.tolist(), min_center_acute_angle_deg=float(angles.min()),
                mean_center_acute_angle_deg=float(angles.mean()),
                center_abs_cosine_redundancy=float(cosine.mean()),
                station_azimuth_span_deg=float(circular_span(azimuth)))


def spectrum_from_gram(gram, relative_tolerance=1e-6):
    """Full coarse nodal spectrum, float64 Gram; never a full-resolution condition."""
    lam = eigvalsh(np.asarray(gram, dtype=np.float64))
    maximum = max(float(lam[-1]), 0.)
    if lam[0] < -max(maximum * 1e-12, 1e-12):
        raise ValueError('Non-PSD Gram beyond rounding tolerance')
    sigma = np.sqrt(np.maximum(lam, 0))[::-1]
    if sigma[0] == 0:
        raise ValueError('Empty operator')
    positive = sigma > relative_tolerance * sigma[0]
    rank = int(positive.sum())
    prob = sigma / sigma.sum()
    prob = prob[prob > 0]
    return dict(coarse_sigma_min_raw_ppm_m=float(sigma[-1]),
                coarse_sigma_max_ppm_m=float(sigma[0]),
                coarse_sigma_min_retained_ppm_m=float(sigma[positive][-1]),
                coarse_sigma_min_over_max=float(sigma[-1] / sigma[0]),
                coarse_condition_number=float(sigma[0] / sigma[-1]) if rank == len(sigma) else None,
                coarse_retained_condition_number=float(sigma[0] / sigma[positive][-1]),
                coarse_numeric_rank=rank, coarse_nullity=len(sigma)-rank,
                coarse_effective_rank=float(np.exp(-(prob*np.log(prob)).sum())),
                coarse_stable_rank=float((sigma**2).sum()/sigma[0]**2),
                singular_values_ppm_m=sigma.tolist(), relative_sigma_tolerance=relative_tolerance,
                raw_smallest_is_roundoff_sensitive=bool(sigma[-1] <= 1e-6*sigma[0]))


def station_gram(geometry, station, n, samples):
    import torch
    keys = ('station_origin_m', 'ray_direction', 't_near', 'path_length_m', 'valid_ray')
    g = {k:geometry[k][station:station+1] for k in keys}
    a = RayMatrix(g, n=n, samples=samples, dtype=torch.float64)
    sparse = sp.csr_matrix((a.a.values().numpy(), a.a.col_indices().numpy(), a.a.crow_indices().numpy()), shape=a.a.shape)
    rng = np.random.default_rng(712 + station)
    x = torch.from_numpy(rng.normal(size=n**3))
    y = torch.from_numpy(rng.normal(size=a.a.shape[0]))
    left = float(torch.dot(a.forward(x), y)); right = float(torch.dot(x, a.adjoint(y)))
    error = abs(left-right)/max(abs(left), abs(right), 1.)
    if error > 1e-11:
        raise ValueError('Coarse adjoint check failed')
    # A stays sparse; only 512 x 512 geometry-only Gram is dense.
    return (sparse.T @ sparse).toarray(), dict(station_id=int(station), rows=a.a.shape[0],
            columns=n**3, nnz=a.nnz, build_seconds=a.build_seconds, adjoint_relative_error=error,
            sparse_operator_bytes=a.bytes)


def correlation_rows(x, y, rank=False):
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if rank:
        x = rankdata(x); y = rankdata(y, axis=-1)
    x = x-x.mean(); y = y-y.mean(axis=-1, keepdims=True)
    denom = np.sqrt((x*x).sum()*(y*y).sum(axis=-1))
    return np.divide((x*y).sum(axis=-1), denom, out=np.full(denom.shape, np.nan), where=denom>0)
