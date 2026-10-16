#!/usr/bin/env python3
"""Export a B checkpoint on external measurement-only canonical NPZ inputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]

import torch

from geophy_set3d.data.build_utils import canonical_hash, digest, save_npy, write_json
from research.plan_b.inference import ContinuousReconstructor
from research.plan_b.model import SetConditionalField
from reconstruct_plan_a import load_measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resolution', type=int, choices=(64, 128), default=64)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    record = json.loads((run/'frozen_checkpoint.json').read_text())
    protocols = ('scheme_b_continuous_measurement_only_v1',
                 'scheme_b_continuous_measurement_only_perf_v2')
    if record.get('protocol') not in (None, *protocols):
        raise ValueError('This is not a scheme B checkpoint')
    checkpoint = Path(record['checkpoint']).resolve()
    if (not record.get('completed') or not checkpoint.is_relative_to(run)
            or digest(checkpoint) != record['sha256']):
        raise ValueError('Checkpoint is incomplete or its frozen hash changed')
    provenance = json.loads((run/'provenance.json').read_text())
    if provenance['protocol'] not in protocols:
        raise ValueError('Scheme B provenance is required')
    if digest(run/'provenance.json') != record['provenance_sha256']:
        raise ValueError('Frozen provenance changed')
    for name, expected in provenance['code'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Implementation differs from frozen checkpoint: {name}')
    if record['stage'] == 'formal':
        from geophy_set3d.evaluation.research import frozen_record
        frozen_record(run)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = payload['config']
    if config['protocol'] != provenance['protocol']:
        raise ValueError('Checkpoint protocol mismatch')
    settings = config['model']
    model = SetConditionalField(**{k: settings[k] for k in
        ('query_chunk', 'ray_chunk', 'samples', 'checkpoint_query', 'lift_resolution')}).to(args.device)
    model.load_state_dict(payload['model'], strict=True)
    batch = load_measurements(args.input, args.device, config['loss']['measurement_scale'])
    adapter = ContinuousReconstructor(model, checkpoint_hash=record['sha256'], config_hash=canonical_hash(config))
    result = adapter.reconstruct(batch, args.resolution)
    result['provenance']['protocol'] = config['protocol']
    output.mkdir(parents=True)
    save_npy(output/'field_mass_fraction_xyz.npy', result.pop('field_pred').cpu().numpy())
    save_npy(output/'input_reprojection_ppm_m.npy', result.pop('input_reprojection_ppm_m').cpu().numpy())
    save_npy(output/'coverage_feature_grid.npy', result.pop('coverage').cpu().numpy())
    write_json(output/'reconstruction.json', dict(**result, input_sha256=digest(args.input),
        resolution=args.resolution, stage=record['stage'], engineering_checkpoint=record['stage']!='formal', gt_read=False))
    print(output/'reconstruction.json')


if __name__ == '__main__':
    main()
