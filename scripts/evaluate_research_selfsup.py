#!/usr/bin/env python3
"""Save frozen Scheme A predictions before a separate, offline GT64 pass.

Only validation development evaluation is exposed here. Final test access needs
its own frozen confirmatory protocol; the original dataset protection is intact.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch

from geophy_set3d.data.build_utils import digest, canonical_hash, save_npy, write_json, write_jsonl, timestamp
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.data.projection_pool_dataset import artifact_path
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.evaluation.research import ResearchReconstructor, EvaluationWithHoldout, frozen_record, volume_metrics
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler
from geophy_set3d.models.set_unrolled3d import build_set_unrolled_model
from geophy_set3d.training.utils import move_batch


def generate_predictions(run_dir, device, *, views=4, layout='random', draw=0, pool='train'):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_dir = Path(run_dir).resolve()
    frozen = frozen_record(run_dir)
    # trusted local checkpoint, frozen hash verified above
    checkpoint = torch.load(frozen['checkpoint'], map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = build_set_unrolled_model(config).to(device)
    model.load_state_dict(checkpoint['model']); model.eval()
    adapter = ResearchReconstructor(model, checkpoint_hash=frozen['sha256'],
                                   config_hash=canonical_hash(config))
    dataset = MeasurementOnlyPoolDataset(config['dataset']['root'], split='val',
        geometry_pool=pool, evaluation_only=True, min_input_views=views,
        max_input_views=views, hold_views=0, base_seed=20261011,
        normalization=config['normalization']['method'],
        statistics_file=config['normalization']['statistics_file'])
    dataset.sampler = EvaluationWithHoldout(EvaluationViewSampler(
        dataset.geometry['station_azimuth_deg'], views, draw, pool, 20261011, layout))
    directory = run_dir/'evaluation'/f'val_{pool}_{layout}_v{views}_draw{draw}'
    if directory.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation: {directory}')
    directory.mkdir(parents=True)
    rows = []
    for index in range(len(dataset)):
        started = time.perf_counter()
        sample = dataset[index]
        batch = move_batch(projection_pool_collate([sample]), device)
        inputs = {k: v for k, v in batch.items() if not k.startswith('hold_')}
        heldout = {k: v for k, v in batch.items() if not k.startswith(('input_', 'hold_')) and k != 'view_mask'}
        heldout.update({'input_'+k[5:]: v for k, v in batch.items() if k.startswith('hold_') and k != 'hold_view_mask'})
        heldout['view_mask'] = batch['hold_view_mask']
        loaded = time.perf_counter()
        result = adapter.reconstruct(inputs)
        prediction = directory/'predictions'/f'{index:03d}.npy'
        save_npy(prediction, result['field_pred'][0, 0].cpu().numpy())
        valid = batch['input_valid_ray'].bool() & batch['view_mask'][:, :, None, None, None]
        metrics = relative_projection_metrics(result['input_reprojection_ppm_m'][0],
                                             batch['input_projection_raw'][0], valid[0])
        with torch.no_grad():
            hp = model.operator(result['field_pred'], heldout)
        hv = heldout['input_valid_ray'].bool() & heldout['view_mask'][:, :, None, None, None]
        heldout_metrics = relative_projection_metrics(hp[0], heldout['input_projection_raw'][0], hv[0])
        rows.append(dict(case_id=sample['case_id'], prediction=str(prediction.relative_to(directory)),
            prediction_sha256=digest(prediction), geometry_id=f'{pool}_{layout}_v{views}_draw{draw}',
            noise_id='clean', split='val', views=views,
            selected_station_ids=sample['input_station_indices'].tolist(),
            heldout_station_ids=sample['hold_station_indices'].tolist(),
            checkpoint_hash=frozen['sha256'], seed=config['training']['seed'],
            method=config['model'].get('variant', 'A2'), status='ok',
            timing=dict(load_s=loaded-started, **result['timing']), observed=metrics,
            heldout=heldout_metrics))
    write_jsonl(directory/'predictions_manifest.jsonl', rows)
    write_json(directory/'prediction_boundary.json', dict(
        all_predictions_saved_before_gt=True, generated_at=timestamp(),
        prediction_manifest_sha256=digest(directory/'predictions_manifest.jsonl'),
        checkpoint_hash=frozen['sha256'], run_dir=str(run_dir),
        dataset_root=str(dataset.root), case_count=len(rows),
        membership_sha256=digest(dataset.root/'source_index.jsonl'),
        config_hash=canonical_hash(config), gt_opened=False,
        timing_note='individual synchronized forwards; not a steady-state benchmark'))
    return directory


def score_saved_predictions(directory):
    """The only GT open in this workflow occurs after checking every prediction."""
    directory = Path(directory).resolve()
    boundary = json.loads((directory/'prediction_boundary.json').read_text())
    frozen = frozen_record(boundary['run_dir'])
    if (not boundary.get('all_predictions_saved_before_gt')
            or boundary['checkpoint_hash'] != frozen['sha256']
            or digest(directory/'predictions_manifest.jsonl') != boundary['prediction_manifest_sha256']):
        raise ValueError('Prediction boundary changed')
    root = Path(boundary['dataset_root'])
    if digest(root/'source_index.jsonl') != boundary['membership_sha256']:
        raise ValueError('Dataset membership changed')
    source = {r['case_id']: r for r in
              map(json.loads, (root/'source_index.jsonl').read_text().splitlines())}
    rows = [json.loads(line) for line in (directory/'predictions_manifest.jsonl').read_text().splitlines()]
    if len(rows) != boundary['case_count'] or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate prediction cases')
    # Check all hashes and split membership BEFORE opening the first label.
    for row in rows:
        if row['split'] != 'val' or source[row['case_id']]['split'] != 'val':
            raise ValueError('This tool permits validation development evaluation only')
        if digest(artifact_path(directory, row['prediction'])) != row['prediction_sha256']:
            raise ValueError('Saved prediction changed')
    if (directory/'per_case_metrics.jsonl').exists():
        raise FileExistsError('Refusing to overwrite scored evaluation')
    scored = []
    for row in rows:
        prediction = np.load(artifact_path(directory, row['prediction']), allow_pickle=False)
        # Deliberate boundary: first and only evaluation-label loading location.
        reference = np.load(artifact_path(root, source[row['case_id']]['gt_volume_64_path']), allow_pickle=False)
        scored.append(dict(row, **volume_metrics(prediction, reference)))
    write_jsonl(directory/'per_case_metrics.jsonl', scored)
    names = ('relative_l1', 'relative_l2', 'nmse', 'mae', 'rmse', 'peak_absolute_error',
             'concentration_integral_relative_error', 'dice_0.0001', 'dice_0.001')
    write_json(directory/'summary.json', dict(
        cases=len(scored), checkpoint_hash=frozen['sha256'],
        mean_per_case={name: float(np.mean([r[name] for r in scored])) for name in names},
        measurement={domain: {name: float(np.mean([r[domain][name] for r in scored]))
                             for name in ('relative_l1', 'relative_l2')}
                     for domain in ('observed', 'heldout')},
        gt_for_training=False, gt_for_selection=False, development_gt_evaluation=True,
        final_test_evaluated=False, prediction_boundary=boundary))
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    predict = sub.add_parser('predict')
    predict.add_argument('--run-dir', required=True)
    predict.add_argument('--device', default='cpu')
    predict.add_argument('--views', type=int, default=4)
    predict.add_argument('--layout', choices=('random', 'spread', 'clustered'), default='random')
    predict.add_argument('--draw', type=int, default=0)
    predict.add_argument('--pool', choices=('train', 'ood'), default='train')
    score = sub.add_parser('score')
    score.add_argument('--predictions-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == 'predict':
        path = generate_predictions(args.run_dir, args.device, views=args.views,
                                    layout=args.layout, draw=args.draw, pool=args.pool)
        print(path)
    else:
        rows = score_saved_predictions(args.predictions_dir)
        print(json.dumps(dict(cases=len(rows), relative_l1=np.mean([r['relative_l1'] for r in rows]))))


if __name__ == '__main__':
    main()
