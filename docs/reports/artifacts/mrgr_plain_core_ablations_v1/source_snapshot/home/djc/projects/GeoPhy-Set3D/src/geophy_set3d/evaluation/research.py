"""Measurement-only inference adapter and independent, frozen-model metrics.

No volume reader is imported here. Loading evaluation labels belongs to the
separate CLI, after predictions have been written and hashed.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.measurement_only import assert_no_gt
from geophy_set3d.data.projection_pool_dataset import artifact_path


class EvaluationWithHoldout:
    """Keep selected input layouts unchanged and choose four disjoint H stations."""
    def __init__(self, input_sampler, heldout_views=4):
        self.input_sampler = input_sampler
        self.heldout_views = heldout_views

    def sample(self, station_count, *, epoch, sample_index, case_id):
        selected, _ = self.input_sampler.sample(station_count, epoch=epoch,
                                              sample_index=sample_index, case_id=case_id)
        available = np.setdiff1d(np.arange(station_count), selected)
        if len(available) < self.heldout_views:
            raise ValueError('Not enough stations for disjoint evaluation holdout')
        identity = f'plan_a_H_v1:{case_id}:{self.input_sampler.pool}:{self.input_sampler.protocol}:{self.input_sampler.views}:{self.input_sampler.draw_id}'
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little')
        heldout = np.random.default_rng(seed).choice(available, self.heldout_views, replace=False)
        return selected, heldout.astype(np.int64)


def synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class ResearchReconstructor:
    """Wrap A without changing historical model forward contracts.

    Times include the full initialization and all unrolled iterations. They are
    individual synchronized observations, not a steady-state benchmark.
    """
    def __init__(self, model, *, checkpoint_hash=None, config_hash=None):
        self.model = model
        self.provenance = dict(method=type(model).__name__,
                               checkpoint_hash=checkpoint_hash,
                               config_hash=config_hash)

    @torch.no_grad()
    def reconstruct(self, input_batch, output_grid=64, options=None):
        assert_no_gt(input_batch)
        if output_grid != 64:
            raise ValueError('Scheme A uses the fixed 64 cubed XYZ endpoint grid')
        options = options or {}
        if set(options) - {'diagnostics'}:
            raise ValueError('Unknown reconstruction options')
        self.model.eval()
        device = input_batch['input_projection'].device
        synchronize(device)
        start = time.perf_counter()
        result = self.model(input_batch, return_diagnostics=options.get('diagnostics', False))
        field = result['volume'] if isinstance(result, dict) else result
        synchronize(device)
        reconstructed = time.perf_counter()
        if field.shape[1:] != (1, 64, 64, 64) or not torch.isfinite(field).all():
            raise ValueError('Expected a finite [B,1,64,64,64] field')
        reprojection = self.model.operator(field, input_batch)
        synchronize(device)
        projected = time.perf_counter()
        diagnostics = {k: v for k, v in result.items() if k not in ('volume', 'stencil')} if isinstance(result, dict) else {}
        return dict(field_pred=field, field_variable='methane_mass_fraction',
                    axes='XYZ', domain_m=[[0., 0., 0.], [.6, .6, .6]],
                    input_reprojection_ppm_m=reprojection,
                    diagnostics=diagnostics,
                    timing=dict(reconstruct_s=reconstructed-start,
                                project_s=projected-reconstructed),
                    provenance=self.provenance)


def frozen_record(run_dir):
    """Fail closed before any evaluation-label access, including on a gate run."""
    root = Path(run_dir).resolve()
    record = json.loads((root/'frozen_checkpoint.json').read_text())
    if (record.get('stage') != 'formal' or record.get('completed_epochs') != 300
            or record.get('volumetric_gt_used', True)
            or record.get('selection_metric') != 'mean held-out measurement relative L1'):
        raise ValueError('Independent GT evaluation requires a completed, GT-free formal run')
    checkpoint = Path(record['checkpoint']).resolve()
    if not checkpoint.is_relative_to(root) or digest(checkpoint) != record['sha256']:
        raise ValueError('Frozen checkpoint path or hash mismatch')
    summary = json.loads((root/'training_summary.json').read_text())
    access = summary.get('gt_access', {})
    if (not summary.get('completed') or summary.get('completed_epochs') != 300
            or any(access.get(k, 1) != 0 for k in ('gt_open_attempts',
                'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts'))):
        raise ValueError('Missing or failed measurement-only training access audit')
    if digest(root/'provenance.json') != record.get('provenance_sha256'):
        raise ValueError('Frozen provenance changed')
    provenance = json.loads((root/'provenance.json').read_text())
    repo = Path(__file__).resolve().parents[3]
    for name, expected in provenance['code'].items():
        if digest(artifact_path(repo, name)) != expected:
            raise ValueError(f'Evaluation implementation differs from training: {name}')
    import yaml
    config = yaml.safe_load((root/'config_resolved.yaml').read_text())
    for name, expected in provenance['data']['files'].items():
        if digest(artifact_path(config['dataset']['root'], name)) != expected:
            raise ValueError(f'Frozen dataset changed: {name}')
    return record


def volume_metrics(prediction, reference, thresholds=(1e-4, 1e-3)):
    """One case, XYZ endpoint grid; physical integrals use trapezoidal weights."""
    p = np.asarray(prediction, dtype=np.float64).squeeze()
    g = np.asarray(reference, dtype=np.float64).squeeze()
    if p.shape != g.shape or p.ndim != 3 or len(set(p.shape)) != 1 or p.shape[0] < 2:
        raise ValueError('Expected matching cubic XYZ endpoint grids')
    if (not np.isfinite(p).all() or not np.isfinite(g).all()
            or np.min(p) < 0 or np.min(g) < 0 or np.max(p) > 1 or np.max(g) > 1):
        raise ValueError('Concentrations must be finite mass fractions in [0,1]')
    n = p.shape[0]
    axis = np.linspace(0, .6, n)
    w = np.ones(n); w[[0, -1]] = .5
    weights = w[:, None, None]*w[None, :, None]*w[None, None, :]*(.6/(n-1))**3
    error = p-g
    absolute_sum = float(np.abs(error).sum())
    squared_sum = float(np.square(error).sum())
    reference_sum = float(np.abs(g).sum())
    reference_square = float(np.square(g).sum())
    p_integral, g_integral = float((p*weights).sum()), float((g*weights).sum())
    peak_p, peak_g = np.unravel_index(p.argmax(), p.shape), np.unravel_index(g.argmax(), g.shape)
    def centroid(c, integral):
        if integral <= 1e-15:
            return None
        weighted = c*weights
        return np.array([(weighted*axis.reshape(tuple(n if j == i else 1 for j in range(3)))).sum()/integral
                         for i in range(3)])
    cp, cg = centroid(p, p_integral), centroid(g, g_integral)
    result = dict(relative_l1=absolute_sum/max(reference_sum, 1e-6),
                  relative_l2=float(np.sqrt(squared_sum/max(reference_square, 1e-12))),
                  nmse=squared_sum/max(reference_square, 1e-12),
                  mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                  peak_error=float(p.max()-g.max()), peak_absolute_error=float(abs(p.max()-g.max())),
                  peak_location_error_mm=float(np.linalg.norm(np.array(peak_p)-peak_g)*.6/(n-1)*1000)
                  if p.max() > 0 and g.max() > 0 else None,
                  peak_tie_rule='first lexicographic XYZ node',
                  centroid_error_mm=float(np.linalg.norm(cp-cg)*1000) if cp is not None and cg is not None else None,
                  predicted_concentration_integral_m3=p_integral,
                  reference_concentration_integral_m3=g_integral,
                  concentration_integral_relative_error=abs(p_integral-g_integral)/max(g_integral, 1e-15))
    for threshold in thresholds:
        pm, gm = p >= threshold, g >= threshold
        union, intersection = int((pm | gm).sum()), int((pm & gm).sum())
        count = int(pm.sum()+gm.sum())
        result[f'dice_{threshold:g}'] = 2*intersection/count if count else 1.
        result[f'iou_{threshold:g}'] = intersection/union if union else 1.
        background = ~gm
        result[f'background_integral_m3_{threshold:g}'] = float((p*weights*background).sum())
    return result


def paired_case_bootstrap(rows_a, rows_b, *, metric='relative_l1', seed=20261015, draws=10000):
    """Average layouts within case before bootstrapping paired case differences."""
    if not isinstance(draws, int) or draws < 1:
        raise ValueError('bootstrap draws must be a positive integer')
    def keyed(rows):
        result = {}
        for row in rows:
            key = (row['case_id'], row['geometry_id'], row.get('noise_id', 'clean'))
            if key in result:
                raise ValueError('Duplicate case/geometry/noise key; compare each seed separately')
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError('Bootstrap metric must be finite')
            result[key] = value
        return result
    a, b = keyed(rows_a), keyed(rows_b)
    if not a or a.keys() != b.keys():
        raise ValueError('Comparison requires the same nonempty case/geometry/noise keys')
    groups = {}
    for key in a:
        groups.setdefault(key[0], []).append(a[key]-b[key])
    differences = np.array([np.mean(groups[k]) for k in sorted(groups)])
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(draws)])
    return dict(metric=metric, direction='A minus B', cases=len(differences),
                mean_difference=float(differences.mean()),
                confidence_interval_95=np.quantile(samples, [.025, .975]).tolist(),
                bootstrap_draws=draws, seed=seed,
                scope='case uncertainty for these fixed layouts and this one training seed')
