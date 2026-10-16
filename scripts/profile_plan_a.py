#!/usr/bin/env python3
"""Measure one real-data Plan A A/B/H forward/backward without optimizer updates.

K=4, Q=96, lift32, C64 and width32 are checked, never silently reduced. Only
measurement-only train/Pool A data are read. An OOM is saved as a failed JSON
result and returns exit status 2. This standalone diagnostic script is outside
training's source-provenance file set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.geometry.tdlas import ray_box_intersection
from geophy_set3d.models.set_unrolled3d import build_set_unrolled_model
from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.research_selfsup import (
    _sha, load_research_config, make_research_dataset, research_objective,
)
from geophy_set3d.training.selfsup import move_dual_batch
from geophy_set3d.training.utils import seed_everything


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def cuda_snapshot(device):
    if not str(device).startswith('cuda'):
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                device_free_bytes=free, device_total_bytes=total)


def json_finite(value):
    """Keep a usable failure report even when diagnostic gradients are NaN/Inf."""
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def geometry_axis_audit(geometry):
    """Validate the frozen scanner's XYZ/y-up and beta-row/alpha-column metadata."""
    origin, direction = geometry['station_origin_m'], geometry['ray_direction']
    f, r, u = (geometry[key] for key in ('center_direction', 'right_direction', 'up_direction'))
    expected_f = np.asarray([.3, .3, .3])-origin
    expected_f /= np.linalg.norm(expected_f, axis=-1, keepdims=True)
    expected_r = np.cross(expected_f, np.asarray([0., 1., 0.]))
    expected_r /= np.linalg.norm(expected_r, axis=-1, keepdims=True)
    expected_u = np.cross(expected_r, expected_f)
    alpha, beta = np.meshgrid(np.deg2rad(geometry['horizontal_angle_deg']),
                             np.deg2rad(geometry['vertical_angle_deg']), indexing='xy')
    expected_direction = (np.cos(beta)[None, ..., None]*np.cos(alpha)[None, ..., None]*f[:, None, None]
        + np.cos(beta)[None, ..., None]*np.sin(alpha)[None, ..., None]*r[:, None, None]
        + np.sin(beta)[None, ..., None]*u[:, None, None])
    near, far, length, valid = ray_box_intersection(origin[:, None, None], direction)
    maximum = lambda x: float(np.max(np.abs(x)))
    errors = dict(ray_unit_norm=maximum(np.linalg.norm(direction, axis=-1)-1),
                  forward_points_to_center=maximum(f-expected_f),
                  right_y_up_convention=maximum(r-expected_r),
                  up_y_up_convention=maximum(u-expected_u),
                  forward_right_orthogonality=maximum((f*r).sum(-1)),
                  forward_up_orthogonality=maximum((f*u).sum(-1)),
                  right_up_orthogonality=maximum((r*u).sum(-1)),
                  row_beta_column_alpha_direction=maximum(direction-expected_direction),
                  near_matches_slab_m=maximum(geometry['t_near']-near),
                  far_matches_slab_m=maximum(geometry['t_far']-far),
                  path_length_matches_slab_m=maximum(geometry['path_length_m']-length))
    checks = {key: dict(error=value, tolerance=1e-5,
                       passed=bool(np.isfinite(value) and value <= 1e-5))
              for key, value in errors.items()}
    checks['valid_matches_slab'] = dict(passed=bool(np.array_equal(geometry['valid_ray'], valid)))
    checks['angle_axes_increasing'] = dict(passed=bool(
        np.all(np.diff(geometry['horizontal_angle_deg']) > 0)
        and np.all(np.diff(geometry['vertical_angle_deg']) > 0)))
    return dict(checks=checks, passed=all(row['passed'] for row in checks.values()),
                volume_axes='XYZ', vertical_axis='y', image_rows='beta', image_columns='alpha')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/research/scheme_a_smoke.yaml')
    parser.add_argument('--dataset', type=Path, help='Override the real projection-pool root')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--case-index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=0, help='Deterministic training A/B/H sampling epoch')
    parser.add_argument('--output', required=True, type=Path, help='A fresh JSON path; existing results are protected')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; select a fresh JSON path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(protocol='plan_a_real_single_case_profile_v1', status='started',
                  optimizer_updates=0, gt_used=False, case_index=args.case_index,
                  sampling_epoch=args.epoch, config_path=str(args.config.resolve()),
                  device=args.device, phases={},
                  limitations='One batch of one case; no training trajectory or reconstruction accuracy claims.')
    guard = None
    phase = 'setup'
    began = time.perf_counter()
    code = 1
    try:
        config = load_research_config(args.config)
        if args.dataset:
            config['dataset']['root'] = str(args.dataset.resolve())
        if config['dataset']['kind'] != 'projection_pool':
            raise ValueError('This profile requires the real measurement-only projection pool')
        settings = config['model']
        required = dict(variant='A2', steps=4, samples=96, prior_width=32,
                        lift_resolution=32, output_resolution=64)
        if any(settings.get(key) != val for key, val in required.items()):
            raise ValueError(f'Full-size profile requires {required}; parameters are never reduced')
        if args.case_index < 0 or args.epoch < 0:
            raise ValueError('case-index and epoch must be nonnegative')
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA device is unavailable')
        torch.set_num_threads(config['training'].get('cpu_threads', 2))
        seed_everything(config['training']['seed'])
        report['environment'] = dict(python=platform.python_version(), torch=str(torch.__version__),
            cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(args.device)
            if args.device.startswith('cuda') else None,
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        source_paths = sorted((ROOT/'src').rglob('*.py')) + [ROOT/'scripts/train_research_selfsup.py']
        report['training_source_hashes'] = {str(path.relative_to(ROOT)): _sha(path) for path in source_paths}
        report['training_source_sha256'] = hashlib.sha256(json.dumps(
            report['training_source_hashes'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        report['resolved_model'] = settings
        report['dataset_root'] = str(Path(config['dataset']['root']).resolve())
        report['train_sampling'] = config['train']
        report['training_seed'] = config['training']['seed']
        report['precision'] = 'fp32'
        report['diagnostics'] = True
        report['effective_batch_note'] = 'One case without optimizer step; formal effective batch of two is not simulated.'
        guard = GTReadGuard(config['dataset']['root'], args.output.parent/'unused_checkpoints')
        with guard:
            phase = 'load_measurements'
            started = time.perf_counter()
            dataset = make_research_dataset(config)
            report['geometry_axis_audit'] = geometry_axis_audit(dataset.geometry)
            if not report['geometry_axis_audit']['passed']:
                raise ValueError('Frozen geometry failed the XYZ/y-up and ray-box consistency audit')
            dataset.set_epoch(args.epoch)
            if args.case_index >= len(dataset):
                raise IndexError('case-index is outside the train manifest')
            sample = dataset[args.case_index]
            batch = move_dual_batch(dual_measurement_collate([sample]), args.device)
            report['case_id'] = batch['A']['case_id'][0]
            report['branches'] = {key: dict(
                station_indices=batch[key]['input_station_indices'][0].tolist(),
                views=int(batch[key]['view_mask'][0].sum()),
                valid_rays=int((batch[key]['input_valid_ray'] & batch[key]['view_mask'][:, :, None, None, None]).sum()),
                projection_shape=list(batch[key]['input_projection_raw'].shape)) for key in ('A', 'B', 'H')}
            data_root = Path(config['dataset']['root']).resolve()
            filenames = ['source_index.jsonl', 'train_statistics.json', 'geometry/train_geometry.npz',
                         dataset.records[args.case_index]['train_projection_path']]
            report['data_file_sha256'] = {name: _sha(data_root/name) for name in filenames}
            report['phases'][phase] = dict(seconds=time.perf_counter()-started)
            phase = 'construct_model'
            started = time.perf_counter()
            model = build_set_unrolled_model(config).to(args.device).train()
            operator = ExplicitRayOperator(samples_per_ray=96, ray_chunk=settings.get('ray_chunk', 256))
            sync(args.device)
            report['parameter_count'] = sum(p.numel() for p in model.parameters())
            report['phases'][phase] = dict(seconds=time.perf_counter()-started,
                                          cuda=cuda_snapshot(args.device))
            phase = 'forward_objective'
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            loss, parts, volumes, projections, outputs = research_objective(model, batch, operator, diagnostics=True)
            sync(args.device)
            report['phases'][phase] = dict(seconds=time.perf_counter()-started,
                loss=float(loss.detach()), parts={key: float(val.detach()) for key, val in parts.items()},
                cuda=cuda_snapshot(args.device))
            phase = 'backward'
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            loss.backward()
            sync(args.device)
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            finite = bool(all(torch.isfinite(g).all() for g in grads))
            norm = float(torch.stack([g.norm() for g in grads]).norm()) if grads else 0.
            report['phases'][phase] = dict(seconds=time.perf_counter()-started,
                gradient_parameter_count=len(grads), gradients_finite=finite, gradient_norm=norm,
                cuda=cuda_snapshot(args.device))
            report['geometry_cpu_cache_bytes'] = dict(model=model.operator.cache_bytes,
                                                     outer_projection=operator.cache_bytes)
            if not finite or not grads or not math.isfinite(norm) or norm <= 0:
                raise FloatingPointError('Backward must produce finite, nonzero parameter gradients')
            report['status'] = 'passed'
            code = 0
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status='oom', failed_phase=phase, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        try:
            report['cuda_at_failure'] = cuda_snapshot(args.device)
        except Exception as memory_error:
            report['cuda_at_failure_error'] = str(memory_error)
        code = 2
    except Exception as exc:
        report.update(status='failed', failed_phase=phase, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
    finally:
        report['elapsed_seconds'] = time.perf_counter()-began
        report['gt_read_audit'] = guard.summary() if guard is not None else dict(guard_not_started=True)
        # Exclusive creation protects an earlier measurement even if two profiles race.
        with args.output.open('x') as stream:
            json.dump(json_finite(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(status=report['status'], output=str(args.output),
                              failed_phase=report.get('failed_phase'),
                              elapsed_seconds=report['elapsed_seconds'])))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
