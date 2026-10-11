#!/usr/bin/env python3
"""Validate real frozen membership, physical fields, padding and worker sampling."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from geophy_set3d.data.build_utils import write_json, timestamp
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset, load_source_index


def dataset_options(config):
    return dict(dataset_root=config['dataset']['root'], **config['train'],
                normalization=config['normalization']['method'],
                statistics_file=config['normalization']['statistics_file'],
                load_gt128=config['gt']['load_gt128'], base_seed=config['random']['seed'])


def signatures(loader):
    result = []
    for batch in loader:
        for i, case_id in enumerate(batch['case_id']):
            result.append((case_id, batch['input_station_indices'][i][batch['view_mask'][i]].tolist(),
                           batch['hold_station_indices'][i][batch['hold_view_mask'][i]].tolist()))
    return result


def validate(config, output):
    options = dataset_options(config)
    if options['split'] != 'train' or options['geometry_pool'] != 'train' or config['gt']['resolution'] != 64:
        raise ValueError('Validation requires production train / Pool A / GT64 configuration')
    root, output = Path(options['dataset_root']).resolve(), Path(output).resolve()
    build = json.loads((root / 'dataset_build_manifest.json').read_text())
    source = Path(build['complete_effective_config']['dataset']['source_root']).resolve()
    if output.is_relative_to(root) or output.is_relative_to(source):
        raise ValueError('Validation output must be outside frozen and source datasets')
    rows = load_source_index(root)
    counts = {split: sum(row['split'] == split for row in rows) for split in ('train', 'val', 'test')}
    assert counts == build['split_counts']
    identities = {split: {r['case_id'] for r in rows if r['split'] == split} for split in counts}
    assert not (identities['train'] & identities['val'] or identities['train'] & identities['test']
                or identities['val'] & identities['test'])
    train = ProjectionPoolDataset(**options)
    observed_counts, unsorted_order = set(), False
    for index in range(len(train)):
        sample, row = train[index], train.records[index]
        raw = np.load(root / row['train_projection_path'], mmap_mode='r', allow_pickle=False)
        assert sample['case_id'] == row['case_id'] and sample['split'] == 'train'
        assert sample['gt_volume_64'].shape == (1, 64, 64, 64)
        ids = sample['input_station_indices'].tolist() + sample['hold_station_indices'].tolist()
        assert len(ids) == len(set(ids))
        observed_counts.add(len(sample['input_station_indices']))
        unsorted_order |= sample['input_station_indices'].tolist() != sorted(sample['input_station_indices'].tolist())
        for prefix in ('input', 'hold'):
            stations = sample[f'{prefix}_station_indices'].numpy()
            np.testing.assert_array_equal(sample[f'{prefix}_projection_raw'][:, 0].numpy(), raw[stations])
            np.testing.assert_array_equal(sample[f'{prefix}_valid_ray'][:, 0].numpy(), train.geometry['valid_ray'][stations])
            for key in ('path_length_m', 'ray_direction', 'station_origin_m'):
                value = sample[f'{prefix}_{key}']
                if key == 'path_length_m': value = value[:, 0]
                np.testing.assert_allclose(value.numpy(), train.geometry[key][stations], rtol=1e-6, atol=1e-7)
            expected = sample[f'{prefix}_projection_raw'] / train.y_ref
            if train.normalization == 'log1p_p99_5': expected = torch.log1p(expected)
            torch.testing.assert_close(sample[f'{prefix}_projection'], expected)
    assert observed_counts.issubset(set(range(options['min_input_views'], options['max_input_views']+1)))
    assert unsorted_order
    # Pool B is used here only for interface smoke checks, never audit thresholds or training.
    eval_membership = {}
    for split in ('val', 'test'):
        for pool in ('train', 'ood'):
            evaluation = ProjectionPoolDataset(**{**options, 'split': split, 'geometry_pool': pool})
            assert {r['case_id'] for r in evaluation.records} == identities[split]
            assert evaluation[0]['geometry_pool'] == pool
            eval_membership[f'{split}/{pool}'] = len(evaluation)
    try:
        ProjectionPoolDataset(**{**options, 'geometry_pool': 'ood'})
    except ValueError:
        pass
    else:
        raise AssertionError('Train/Pool B must be rejected')
    small = ProjectionPoolDataset(**{**options, 'min_input_views': 3, 'max_input_views': 3})[0]
    large = ProjectionPoolDataset(**{**options, 'min_input_views': 6, 'max_input_views': 6})[1]
    batch = projection_pool_collate([small, large])
    assert batch['input_projection'].shape == (2, 6, 1, 64, 64)
    assert batch['gt_volume_64'].shape == (2, 1, 64, 64, 64)
    assert batch['view_mask'][0].tolist() == [True]*3 + [False]*3
    for key in small:
        if key.startswith('input_'):
            expected = -1 if key.endswith('station_indices') else 0
            assert torch.all(batch[key][0, 3:] == expected)
    # Independent loaders must replay with the SAME worker count and assignment.
    worker_count = max(1, config['loader']['num_workers'])
    workers = [ProjectionPoolDataset(**options), ProjectionPoolDataset(**options)]
    loaders = [DataLoader(Subset(ds, list(range(min(8, len(ds))))),
                          batch_size=config['loader']['batch_size'], num_workers=worker_count,
                          collate_fn=projection_pool_collate, persistent_workers=True,
                          pin_memory=False) for ds in workers]
    try:
        epoch0 = signatures(loaders[0])
        assert epoch0 == signatures(loaders[1])
        for dataset in workers: dataset.set_epoch(1)
        epoch1 = signatures(loaders[0])
        assert epoch1 == signatures(loaders[1]) and epoch0 != epoch1
        workers[0].set_epoch(0)
        assert epoch0 == signatures(loaders[0])
    finally:
        for loader in loaders:
            if loader._iterator is not None: loader._iterator._shutdown_workers()
    # Smoke-test the configured normal loader as well as persistent-worker validation above.
    configured = DataLoader(train, collate_fn=projection_pool_collate,
                            generator=torch.Generator().manual_seed(config['random']['seed']),
                            **config['loader'])
    configured_batch = next(iter(configured))
    assert configured_batch['gt_volume_64'].shape[1:] == (1, 64, 64, 64)
    shapes = {}
    for key, value in sorted(batch.items()):
        description = list(value.shape) if isinstance(value, torch.Tensor) else f'list[{len(value)}]'
        shapes[key] = description
        print(f'{key}: {description}')
    report = dict(passed=True, completed_at=timestamp(), split_counts=counts,
                  train_geometry_pool='A only', train_ood_rejected=True,
                  evaluation_membership=eval_membership, train_cases_checked=len(train),
                  observed_view_counts=sorted(observed_counts), random_view_order=True,
                  projection_raw_exact_disk_match=True, valid_ray_exact_geometry_match=True,
                  y_ref_ppm_m=train.y_ref, statistics_file=str(train.statistics_path),
                  normalization=train.normalization, clips_normalized_projection=False,
                  multiworker_reproducibility=True, persistent_workers_observe_set_epoch=True,
                  worker_count=worker_count, configured_loader_smoke_passed=True,
                  explicit_variable_view_batch_shapes=shapes)
    write_json(output / 'validation_summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'explicit_variable_view_batch_shapes'}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training_data_v1.yaml')
    parser.add_argument('--output-root', default='/data/djc/GeoPhy-Set3D/outputs/audits/training_interface_v1')
    args = parser.parse_args()
    torch.set_num_threads(2)
    validate(yaml.safe_load(Path(args.config).read_text()), args.output_root)


if __name__ == '__main__':
    main()
