#!/usr/bin/env python3
"""Measurement-only checks of a frozen B field, without changing its context.

Q96/192/384 evaluate the same field, including its fixed Q96 coverage features.
The exported C64 is projected separately as a representation diagnostic.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import numpy as np
import torch

from geophy_set3d.data.build_utils import digest, write_json
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.research_selfsup import make_research_dataset
from geophy_set3d.training.selfsup import move_dual_batch
from research.plan_b.physics import ContinuousRayIntegrator
from research.plan_b.training import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    p.add_argument('--export-input', help='Optional external-inference NPZ fixture; no labels')
    args = p.parse_args()
    run = Path(args.run_dir).resolve()
    freeze = json.loads((run/'frozen_checkpoint.json').read_text())
    checkpoint = Path(freeze['checkpoint']).resolve()
    if not freeze.get('completed') or not checkpoint.is_relative_to(run) or digest(checkpoint) != freeze['sha256']:
        raise ValueError('Requires a complete, frozen checkpoint')
    provenance = json.loads((run/'provenance.json').read_text())
    if provenance['protocol'] not in ('scheme_b_continuous_measurement_only_v1',
                                     'scheme_b_continuous_measurement_only_perf_v2'):
        raise ValueError('Scheme B provenance required')
    for name, expected in provenance['code'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Source changed: {name}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = payload['config']
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_model(config).to(args.device).eval()
    model.load_state_dict(payload['model'])
    records, predictions = [], {}
    guard = GTReadGuard(config['dataset']['root'], run/'checkpoints')
    with guard, torch.no_grad():
        dataset = make_research_dataset(config, validation=True)
        dataset.set_epoch(0)
        batch = move_dual_batch(dual_measurement_collate([dataset[0]]), args.device)
        # Encode once with the training Q96 geometry; do not change the field later.
        context = model.encode(batch['A'])
        original_physics = model.physics
        for q in (96, 192, 384):
            # The integral callback refers to model.physics.samples_per_ray.
            model.physics = ContinuousRayIntegrator(samples_per_ray=q, ray_chunk=256)
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            prediction = model.integrate(context, batch['H'])
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize(args.device)
            predictions[q] = prediction
            records.append(dict(samples=q, seconds=time.perf_counter()-started,
                finite=bool(torch.isfinite(prediction).all()), mean_ppm_m=float(prediction.mean())))
        model.physics = original_physics
        volume = model.sample_grid(context, 64)
        voxel = ExplicitRayOperator(samples_per_ray=384, ray_chunk=256)(volume, batch['H'])
        valid = batch['H']['input_valid_ray'] & batch['H']['view_mask'][:, :, None, None, None]
        ref = predictions[384][valid]

        def compare(value):
            delta = (value[valid]-ref).abs()
            return dict(relative_l1=float(delta.sum()/ref.abs().sum().clamp_min(1e-12)),
                        maximum_absolute_ppm_m=float(delta.max()))

        results = dict(protocol=config['protocol'], checkpoint_sha256=freeze['sha256'],
            stage=freeze['stage'], fixed_context=True, case_id=batch['A']['case_id'],
            observed_station_ids=batch['A']['input_station_indices'].cpu().tolist(),
            heldout_station_ids=batch['H']['input_station_indices'].cpu().tolist(),
            q_results=records, q96_vs_q384=compare(predictions[96]), q192_vs_q384=compare(predictions[192]),
            sampled_c64_q384_vs_continuous_q384=compare(voxel),
            note='Numerical diagnostics on one engineering field, not formal accuracy or universal Q96 qualification')
        if args.export_input:
            from reconstruct_plan_a import REQUIRED, OPTIONAL
            arrays = {key: batch['A'][key].cpu().numpy() for key in (*REQUIRED, *OPTIONAL) if key in batch['A']}
            np.savez(args.export_input, **arrays)
    results['gt_access'] = guard.summary()
    write_json(args.output, results)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
