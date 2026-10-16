#!/usr/bin/env python3
"""Read-only audit and optional measurement replay of the historical baseline."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import torch
import yaml

from geophy_set3d.data.build_utils import digest, write_json, timestamp
from geophy_set3d.models import build_model
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.selfsup import make_dataset, make_loader, run_epoch, selection_metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-baseline', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='docs/reports/artifacts/plan_a_v1/assets_audit.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    repo = Path(__file__).resolve().parents[1]
    frozen = json.loads((repo/'protocols/plan_a_v1/assets_before.json').read_text())
    root = Path(frozen['dataset_root'])
    metadata = {name: digest(root/name) == expected for name, expected in frozen['dataset_metadata_hashes'].items()}
    historical_source = {name: digest(repo/name) == expected for name, expected in frozen['tracked_source_hashes'].items()}
    baseline = frozen['historical_baseline_freeze']
    checkpoint = Path(baseline['checkpoint'])
    checkpoint_ok = digest(checkpoint) == baseline['sha256']
    result = dict(created_at=timestamp(), split_counts=frozen['split_counts'],
                  dataset_metadata_unchanged=all(metadata.values()), metadata_checks=metadata,
                  historical_source_unchanged=all(historical_source.values()),
                  changed_historical_source=[k for k, v in historical_source.items() if not v],
                  baseline_checkpoint_unchanged=checkpoint_ok,
                  gt_body_read=False, baseline_measurement_replayed=False)
    if not all((all(metadata.values()), all(historical_source.values()), checkpoint_ok)):
        write_json(repo/args.output, result)
        raise RuntimeError('Frozen assets changed')
    if args.replay_baseline:
        config = yaml.safe_load((repo/'configs/train_selfsup_oari_mrgr_v1.yaml').read_text())
        settings = copy.deepcopy(config)
        settings['training'].update(device=args.device, batch_size=1, gradient_accumulation=2)
        settings['loader'].update(num_workers=0, persistent_workers=False, pin_memory=False)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        with GTReadGuard(root, checkpoint.parent) as guard:
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if payload['config'] != config:
                raise ValueError('Historical baseline config mismatch')
            model = build_model(config).to(args.device)
            model.load_state_dict(payload['model'])
            actual = run_epoch(model, make_loader(make_dataset(settings, validation=True), settings), settings)
            result['baseline_measurement_replayed'] = True
            result['baseline_measurements'] = actual
            result['gt_access'] = guard.summary()
        log = Path(config['experiment']['output_root'])/'logs'/config['experiment']['name']/'training_summary.json'
        expected = json.loads(log.read_text())['best_val_heldout_relative_l1']
        actual_metric = selection_metric(actual)
        tolerance = 1e-8+abs(expected)*1e-5
        result['baseline_reproduction'] = dict(expected=expected, actual=actual_metric,
            absolute_difference=abs(actual_metric-expected), tolerance=tolerance,
            passed=abs(actual_metric-expected) <= tolerance)
    result['passed'] = all((result['dataset_metadata_unchanged'], result['historical_source_unchanged'],
                            result['baseline_checkpoint_unchanged'],
                            result.get('baseline_reproduction', {}).get('passed', True)))
    write_json(repo/args.output, result)
    print(json.dumps({k: result[k] for k in ('passed', 'split_counts', 'baseline_measurement_replayed')}))
    if not result['passed']:
        raise RuntimeError('Historical baseline reproduction failed')


if __name__ == '__main__':
    main()
