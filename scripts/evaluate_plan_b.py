#!/usr/bin/env python3
"""Frozen B validation predictions, then a separate GT64 scoring command.

Formal 300-epoch completion is required. Final test access remains locked.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import torch

from geophy_set3d.data.build_utils import digest, canonical_hash, save_npy, write_json, write_jsonl, timestamp
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.evaluation.research import EvaluationWithHoldout, frozen_record
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler
from geophy_set3d.training.utils import move_batch
from research.plan_b.training import PROTOCOL, build_model
from evaluate_research_selfsup import score_saved_predictions


@torch.no_grad()
def generate_predictions(run_dir, device, *, views=4, layout='random', draw=0, pool='train'):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run = Path(run_dir).resolve()
    frozen = frozen_record(run)
    checkpoint = torch.load(frozen['checkpoint'], map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if config['protocol'] not in (PROTOCOL, 'scheme_b_continuous_measurement_only_perf_v2'):
        raise ValueError('Scheme B protocol required')
    model = build_model(config).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    dataset = MeasurementOnlyPoolDataset(config['dataset']['root'], split='val',
        geometry_pool=pool, evaluation_only=True, min_input_views=views, max_input_views=views,
        hold_views=0, base_seed=20261011, normalization=config['normalization']['method'],
        statistics_file=config['normalization']['statistics_file'])
    dataset.sampler = EvaluationWithHoldout(EvaluationViewSampler(
        dataset.geometry['station_azimuth_deg'], views, draw, pool, 20261011, layout))
    directory = run/'evaluation'/f'val_{pool}_{layout}_v{views}_draw{draw}'
    if directory.exists():
        raise FileExistsError('Refusing to overwrite an evaluation')
    directory.mkdir(parents=True)

    def sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)

    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = move_batch(projection_pool_collate([sample]), device)
        inputs = {k: v for k, v in batch.items() if not k.startswith('hold_')}
        heldout = {k: v for k, v in batch.items() if not k.startswith(('input_', 'hold_')) and k != 'view_mask'}
        heldout.update({'input_'+k[5:]: v for k,v in batch.items() if k.startswith('hold_') and k != 'hold_view_mask'})
        heldout['view_mask'] = batch['hold_view_mask']
        sync()
        started = time.perf_counter()
        context = model.encode(inputs)
        field = model.sample_grid(context, 64)
        # Both measurement metrics evaluate the same encoded continuous function.
        observed_prediction = model.integrate(context, inputs)
        heldout_prediction = model.integrate(context, heldout)
        sync()
        if not all(bool(torch.isfinite(x).all()) for x in (field, observed_prediction, heldout_prediction)):
            raise FloatingPointError('Nonfinite continuous reconstruction')
        elapsed = time.perf_counter()-started
        prediction = directory/'predictions'/f'{index:03d}.npy'
        save_npy(prediction, field[0,0].cpu().numpy())

        def metrics(pred, target):
            valid = target['input_valid_ray'] & target['view_mask'][:, :, None, None, None]
            return relative_projection_metrics(pred[0], target['input_projection_raw'][0], valid[0])

        rows.append(dict(case_id=sample['case_id'], prediction=str(prediction.relative_to(directory)),
            prediction_sha256=digest(prediction), geometry_id=f'{pool}_{layout}_v{views}_draw{draw}',
            noise_id='clean', split='val', views=views, method='B_continuous',
            seed=config['training']['seed'], status='ok', checkpoint_hash=frozen['sha256'],
            selected_station_ids=sample['input_station_indices'].tolist(),
            heldout_station_ids=sample['hold_station_indices'].tolist(),
            timing=dict(encode_sample_and_continuous_projects_s=elapsed),
            observed=metrics(observed_prediction, inputs), heldout=metrics(heldout_prediction, heldout)))
    write_jsonl(directory/'predictions_manifest.jsonl', rows)
    write_json(directory/'prediction_boundary.json', dict(all_predictions_saved_before_gt=True,
        generated_at=timestamp(), prediction_manifest_sha256=digest(directory/'predictions_manifest.jsonl'),
        checkpoint_hash=frozen['sha256'], run_dir=str(run), dataset_root=str(dataset.root),
        case_count=len(rows), membership_sha256=digest(dataset.root/'source_index.jsonl'),
        config_hash=canonical_hash(config), gt_opened=False,
        timing_note='synchronized per-case continuous inference; not a steady-state benchmark'))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    predict = commands.add_parser('predict')
    predict.add_argument('--run-dir', required=True)
    predict.add_argument('--device', default='cpu')
    predict.add_argument('--views', type=int, default=4)
    predict.add_argument('--layout', choices=('random', 'spread', 'clustered'), default='random')
    predict.add_argument('--draw', type=int, default=0)
    predict.add_argument('--pool', choices=('train', 'ood'), default='train')
    score = commands.add_parser('score')
    score.add_argument('--predictions-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == 'predict':
        print(generate_predictions(args.run_dir, args.device, views=args.views, layout=args.layout, draw=args.draw, pool=args.pool))
    else:
        print(json.dumps(dict(cases=len(score_saved_predictions(args.predictions_dir)))))


if __name__ == '__main__':
    main()
