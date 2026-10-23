"""Reference hashes, protected paths and a unified read-only evaluation pipeline."""
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from geophy_set3d.data.build_utils import digest, source_snapshot
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models import build_model
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.losses import supervised_loss
from geophy_set3d.training.utils import load_config, move_batch, write_json, permute_views
from .view_selection import EvaluationViewSampler, coverage_descriptor

REPO = Path(__file__).resolve().parents[3]
SOURCE = Path('/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1')


def load_protocol(path='configs/evaluate_b0_generalization_v1.yaml'):
    p = yaml.safe_load(Path(path).read_text())
    if p['test_evaluation'] or p['pool_b_training'] or p['reference_split'] != 'val':
        raise ValueError('This phase permits val evaluation and Pool A-only training')
    root = Path(p['output_root']).resolve()
    cfg = load_config(REPO/p['reference_config'])
    frozen = Path(cfg['dataset']['root']).resolve()
    if root.is_relative_to(frozen) or root.is_relative_to(SOURCE) or root.name == 'supervised_b0':
        raise ValueError('Outputs must be new and outside frozen datasets/reference experiment')
    root.mkdir(parents=True, exist_ok=True)
    for name in ('plume_generalization', 'geometry_generalization', 'view_count', 'angular_coverage',
                 'condition_stratification', 'multiseed', 'horizon_probe', 'figures'):
        (root/name).mkdir(exist_ok=True)
    return p, cfg, root


def evaluation_dataset(config, *, pool='train', views=4, selection='reference', draw=0, evaluation_seed=20261011):
    dataset = ProjectionPoolDataset(config['dataset']['root'], split='val', geometry_pool=pool,
                                    min_input_views=views, max_input_views=views, hold_views=0,
                                    base_seed=evaluation_seed, normalization=config['normalization']['method'],
                                    statistics_file=config['normalization']['statistics_file'])
    if selection != 'reference':
        dataset.sampler = EvaluationViewSampler(dataset.geometry['station_azimuth_deg'], views, draw,
                                                pool, evaluation_seed, selection)
    dataset.set_epoch(0)
    return dataset


def load_model(checkpoint_path, device):
    # Match the original FP32 evaluation exactly; cuDNN otherwise defaults to TF32.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model']); model.eval()
    return model, checkpoint


def evaluate(model, dataset, config, output_dir=None, *, condition, draw_id=0, shortcut='normal'):
    model.eval(); metrics = VolumeMetrics(); rows = []; loss_sum = 0.
    device = config['training']['device']
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            batch = move_batch(projection_pool_collate([sample]), device)
            if shortcut == 'zero_measurement':
                batch['input_projection'] = torch.zeros_like(batch['input_projection'])
                batch['input_projection_raw'] = torch.zeros_like(batch['input_projection_raw'])
            elif shortcut == 'wrong_geometry':
                batch = permute_views(batch, torch.arange(batch['view_mask'].shape[1], device=device).roll(1), geometry_only=True)
            elif shortcut != 'normal':
                raise ValueError('Unknown shortcut')
            prediction = model(batch)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite evaluation output')
            metrics.update(prediction, batch['gt_volume_64'], [sample['case_id']])
            loss, _ = supervised_loss(prediction, batch['gt_volume_64'], **config['loss'])
            loss_sum += float(loss)
            ids = sample['input_station_indices'].tolist()
            azimuths = dataset.geometry['station_azimuth_deg'][ids]
            row = dict(metrics.rows[-1], condition=condition, draw_id=draw_id, pool=dataset.geometry_pool,
                       views=len(ids), selected_station_ids=ids, station_azimuth_deg=azimuths.tolist(),
                       condition_meta=sample['condition_meta'], **coverage_descriptor(azimuths))
            rows.append(row)
    summary = dict(**metrics.compute(), supervised_loss=loss_sum/len(dataset), condition=condition,
                   split='val', pool=dataset.geometry_pool, views=rows[0]['views'], draw_id=draw_id,
                   shortcut=shortcut, selection=type(dataset.sampler).__name__, normalization_scale_ppm_m=dataset.y_ref)
    if output_dir:
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        write_json(output/'metrics.json', summary)
        (output/'per_case_metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    return summary, rows


def freeze_reference(protocol, config, root):
    checkpoint_path = Path(protocol['reference_checkpoint'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    effective = yaml.safe_load((Path(config['experiment']['output_root'])/'logs/supervised_b0/effective_config.yaml').read_text())
    if checkpoint['config'] != config or effective != config:
        raise ValueError('Reference config/checkpoint/effective config mismatch')
    if checkpoint['git_commit'] != 'e30bbcd5f2962a509bffe7a24ebdd1db709fc231':
        raise ValueError('Unexpected B0 training source commit')
    dataset = Path(config['dataset']['root'])
    hashes = {str(p.relative_to(dataset)): digest(p) for p in
              [dataset/'dataset_build_manifest.json', dataset/'train_statistics.json', dataset/'source_index.jsonl',
               *sorted((dataset/'splits').glob('*')), *sorted((dataset/'geometry').glob('*.npz'))] if p.is_file()}
    models = {str(p.relative_to(REPO)): digest(p) for p in sorted((REPO/'src/geophy_set3d/models').glob('*.py'))}
    reference = dict(checkpoint_path=str(checkpoint_path), checkpoint_sha256=digest(checkpoint_path),
                     last_checkpoint_sha256=digest(checkpoint_path.parent/'last.pt'),
                     training_source_git_commit=checkpoint['git_commit'], report_sha256=digest(REPO/'docs/reports/supervised_b0_v1.md'),
                     effective_config=config, best_epoch_zero_based=checkpoint['epoch'], dataset_hashes=hashes,
                     model_source_hashes=models, reference_metrics=json.loads((Path(config['experiment']['output_root'])/'logs/supervised_b0/evaluation.json').read_text())['results']['normal'],
                     test_evaluation=False, pool_b_training=False)
    target = root/'reference.json'
    if target.exists() and json.loads(target.read_text()) != reference:
        raise ValueError('Frozen reference changed since evaluation began')
    if not target.exists():
        write_json(target, reference)
    protocol_path = root/'effective_protocol.yaml'
    if protocol_path.exists() and yaml.safe_load(protocol_path.read_text()) != protocol:
        raise ValueError('Frozen evaluation protocol changed')
    if not protocol_path.exists():
        protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False))
    before_path = root/'inventory_before.json'
    if not before_path.exists():
        old = Path(config['experiment']['output_root'])
        write_json(before_path, dict(dataset=source_snapshot(dataset), source=source_snapshot(SOURCE),
            old_experiment={kind: source_snapshot(old/kind/'supervised_b0') for kind in ('logs','checkpoints','figures')}))
    return reference


def reproduce_reference(protocol, config, root):
    reference = freeze_reference(protocol, config, root)
    model, _ = load_model(protocol['reference_checkpoint'], config['training']['device'])
    dataset = evaluation_dataset(config, evaluation_seed=protocol['reference_seed'])
    result, _ = evaluate(model, dataset, config, root/'plume_generalization/Val-A-V4', condition='Val-A-V4')
    differences = {}
    for aggregation in ('mean_per_case', 'pooled'):
        for metric in ('mae','rmse','relative_l1','relative_l2'):
            expected = reference['reference_metrics'][aggregation][metric]
            actual = result[aggregation][metric]
            tolerance = protocol['reference_tolerance']['absolute'] + protocol['reference_tolerance']['relative']*abs(expected)
            differences[f'{aggregation}.{metric}'] = dict(expected=expected, actual=actual, absolute_difference=abs(actual-expected), tolerance=tolerance)
    passed = all(r['absolute_difference'] <= r['tolerance'] for r in differences.values())
    write_json(root/'reference_reproduction.json', dict(passed=passed, comparisons=differences))
    if not passed:
        raise AssertionError('Reference mismatch: stop all subsequent evaluations')
    return model, result


def require_reference(root):
    path = Path(root)/'reference_reproduction.json'
    if not path.exists() or not json.loads(path.read_text())['passed']:
        raise RuntimeError('Reference reproduction must pass first')
