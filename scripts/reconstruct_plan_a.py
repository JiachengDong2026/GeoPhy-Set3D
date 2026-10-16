#!/usr/bin/env python3
"""Reconstruct an external measurement-only NPZ using a frozen Scheme A run.

NPZ arrays follow the canonical input_* batch contract in the development guide.
Floating arrays are read as FP32 before any processing; raw PIC remains ppm m.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch

from geophy_set3d.data.build_utils import canonical_hash, digest, save_npy, write_json
from geophy_set3d.data.measurement_only import assert_no_gt
from geophy_set3d.evaluation.research import ResearchReconstructor, frozen_record
from geophy_set3d.models.set_unrolled3d import build_set_unrolled_model

REQUIRED = ('input_projection_raw', 'input_valid_ray', 'input_path_length_m',
    'input_t_near', 'input_ray_direction', 'input_station_origin_m',
    'input_center_direction', 'input_right_direction', 'input_up_direction',
    'horizontal_angle_deg', 'vertical_angle_deg', 'view_mask')
OPTIONAL = ('input_projection', 'input_t_far', 'input_station_indices',
            'input_station_azimuth_deg', 'projection_scale_ppm_m')


def load_measurements(path, device, scale):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    assert_no_gt(arrays)
    missing = set(REQUIRED)-arrays.keys()
    if missing or set(arrays)-set(REQUIRED)-set(OPTIONAL):
        raise ValueError(f'NPZ must follow canonical measurement fields; missing={sorted(missing)}')
    batch = {}
    for key, value in arrays.items():
        dtype = torch.bool if key in ('input_valid_ray', 'view_mask') else (
            torch.int64 if key == 'input_station_indices' else torch.float32)
        batch[key] = torch.as_tensor(value, dtype=dtype, device=device)
    raw = batch['input_projection_raw']
    if raw.ndim != 5 or raw.shape[2:] != (1, 64, 64):
        raise ValueError('input_projection_raw must be [B,V,1,64,64]')
    if batch['input_valid_ray'].shape != raw.shape or batch['view_mask'].shape != raw.shape[:2]:
        raise ValueError('Validity and view masks must match measurement dimensions')
    bsz, views = raw.shape[:2]
    valid = batch['input_valid_ray'] & batch['view_mask'][:, :, None, None, None]
    if not valid.flatten(1).any(1).all() or not torch.isfinite(raw[valid]).all():
        raise ValueError('Every case needs finite valid measurements')
    # Invalid raw contents never enter division or the encoder.
    raw = torch.where(valid, raw, 0)
    if 'projection_scale_ppm_m' in batch and not torch.allclose(
            batch['projection_scale_ppm_m'], torch.full_like(batch['projection_scale_ppm_m'], scale)):
        raise ValueError('Input scale differs from the checkpoint training scale')
    if 'input_projection' in batch and not torch.allclose(
            torch.where(valid, batch['input_projection'], 0), raw/scale, rtol=1e-5, atol=1e-7):
        raise ValueError('Normalized measurements must equal raw / fixed training scale')
    batch.update(input_projection_raw=raw, input_projection=raw/scale,
                 input_valid_ray=valid, case_id=[f'external_{i}' for i in range(bsz)])
    batch.setdefault('input_station_indices', torch.arange(views, device=device).expand(bsz, -1))
    batch.setdefault('input_t_far', batch['input_t_near']+batch['input_path_length_m'])
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--input', required=True, help='Canonical measurement-only NPZ')
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    freeze = json.loads((run/'frozen_checkpoint.json').read_text())
    if freeze['stage'] == 'formal':
        freeze = frozen_record(run)
    checkpoint = Path(freeze['checkpoint']).resolve()
    if not freeze.get('completed') or not checkpoint.is_relative_to(run) or digest(checkpoint) != freeze['sha256']:
        raise ValueError('Checkpoint is incomplete or its frozen hash changed')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = payload['config']
    model = build_set_unrolled_model(config).to(args.device)
    model.load_state_dict(payload['model'])
    batch = load_measurements(args.input, args.device, model.y_scale)
    loaded = time.perf_counter()
    adapter = ResearchReconstructor(model, checkpoint_hash=freeze['sha256'], config_hash=canonical_hash(config))
    result = adapter.reconstruct(batch, options=dict(diagnostics=True))
    output.mkdir(parents=True)
    save_npy(output/'field_mass_fraction_xyz.npy', result['field_pred'].cpu().numpy())
    save_npy(output/'input_reprojection_ppm_m.npy', result['input_reprojection_ppm_m'].cpu().numpy())
    diagnostics = result['diagnostics']
    if 'coverage' in diagnostics:
        save_npy(output/'coverage.npy', diagnostics['coverage'].cpu().numpy())
    steps = [{key: value.detach().cpu().tolist() for key, value in step.items()
              if isinstance(value, torch.Tensor) and value.numel() <= len(result['field_pred'])}
             for step in diagnostics.get('steps', [])]
    write_json(output/'reconstruction.json', dict(
        field_variable=result['field_variable'], axes=result['axes'], domain_m=result['domain_m'],
        provenance=result['provenance'], input_sha256=digest(args.input),
        stage=freeze['stage'], engineering_checkpoint=freeze['stage'] != 'formal',
        timing=dict(load_s=loaded-started, **result['timing']),
        diagnostics=dict(steps=steps, iterations=model.steps, status='ok'),
        gt_read=False))
    print(output/'reconstruction.json')


if __name__ == '__main__':
    main()
