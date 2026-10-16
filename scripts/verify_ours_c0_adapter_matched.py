#!/usr/bin/env python3
"""Verify existing experiment artifacts without model forwards or data/GT reads."""
import json
from pathlib import Path

import pandas as pd
import torch

import run_ours_c0_adapter_matched as experiment


def main():
    c, base, out = experiment.configuration()
    protocol = experiment.verify(c, out)
    analysis = json.loads((out / 'analysis_source_manifest.json').read_text())
    for name, expected in analysis['source_sha256'].items():
        assert experiment.digest(experiment.REPO / name) == expected, name
    frozen = json.loads((out / 'frozen_best.json').read_text())
    sealed = json.loads((out / 'predictions_complete.json').read_text())
    gt = json.loads((out / 'gt_complete.json').read_text())
    assert experiment.digest(out / 'metrics_1800.csv') == gt['metrics_sha256']
    old = torch.load(c['source_checkpoint'], map_location='cpu', weights_only=False)
    plans = json.loads((out / 'station_plan.json').read_text())
    orders = json.loads((out / 'case_orders.json').read_text())
    assert len(plans) == len(orders) == 20
    for epoch, order in enumerate(orders):
        assert sorted(order) == list(range(166)) and len(plans[epoch]) == 166
        for row in plans[epoch]:
            a, b, h = [set(row[k]) for k in ('A', 'B', 'H')]
            assert 3 <= len(a) <= 5 and 3 <= len(b) <= 5 and len(h) == 2
            assert not (a & b or a & h or b & h)
    summary = {}
    training_rows = {}
    for arm in experiment.ARMS:
        complete = frozen['completed'][arm]
        assert complete['epochs'] == 20 and complete['optimizer_steps'] == 1660
        root = out / arm
        ckpath = Path(complete['checkpoint'])
        assert experiment.digest(ckpath) == complete['best_sha256']
        assert experiment.digest(ckpath.parent / 'last.pt') == complete['last_sha256']
        best = torch.load(ckpath, map_location='cpu', weights_only=False)
        last = torch.load(ckpath.parent / 'last.pt', map_location='cpu', weights_only=False)
        assert not best['volumetric_gt_used'] and not last['volumetric_gt_used']
        assert last['completed_epochs'] == 20 and last['optimizer_steps'] == 1660
        steps = [int(state['step']) for state in last['optimizer']['state'].values()]
        assert steps and min(steps) == max(steps) == 1660
        assert best['best_epoch'] == complete['best_epoch'] == best['completed_epochs']
        if best['completed_epochs'] == 0:
            for key, value in old['model'].items():
                assert torch.equal(value, best['model'][key]), key
        training_rows[arm] = [json.loads(x) for x in (root / 'metrics.jsonl').read_text().splitlines()]
        rows = training_rows[arm]
        assert len(rows) == 21 and [x['epoch'] for x in rows] == list(range(21))
        assert min(x['val']['measurement']['heldout']['relative_l1'] for x in rows) == complete['best_val_heldout_relative_l1']
        for key in ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts',
                    'pool_b_open_attempts', 'test_or_unknown_npy_attempts', 'data_write_attempts'):
            assert complete['access'][key] == 0
        infer = json.loads((root / 'inference_access.json').read_text())
        assert infer['val_gt64_opens'] == infer['forbidden_attempts'] == 0
        assert json.loads((root / 'selector_replay.json').read_text())['passed']
        for token, expected in sealed['row_sha256'][arm].items():
            path = root / 'prediction_rows' / (token + '.json')
            assert experiment.digest(path) == expected
            row = json.loads(path.read_text())
            assert row['checkpoint_sha256'] == complete['best_sha256']
            assert experiment.digest(row['prediction_file']) == row['prediction_sha256']
        experiment.seed_everything(c['seed'])
        initial_model = experiment.build_arm(arm, base, old['model'], 'cpu')
        assert experiment.state_hash(initial_model.state_dict()) == protocol['initial_state_sha256'][arm]
        parameters = sum(p.numel() for p in initial_model.parameters())
        assert parameters == protocol['parameter_count'][arm]
        del initial_model
        summary[arm] = dict(epochs=20, optimizer_steps=1660, optimizer_parameter_steps_all_1660=True,
                            parameters=parameters,
                            best_epoch=best['completed_epochs'], best_sha256=complete['best_sha256'],
                            training_access=complete['access'], inference_access=infer,
                            epoch_seconds_sum=sum(x.get('seconds', 0) for x in rows),
                            training_peak_allocated_bytes=max(x.get('peak_gpu_allocated_bytes', 0) for x in rows))
        shared_delta = sum(float((last['model'][k].double() - v.double()).square().sum())
                           for k, v in old['model'].items()) ** .5
        summary[arm]['last_shared_parameter_buffer_change_l2'] = shared_delta
        assert shared_delta > 0
        if arm == 'Treatment':
            summary[arm]['adapter_last_final_weight_l2'] = float(last['model']['c0_adapter.body.2.weight'].double().norm())
            summary[arm]['adapter_best_final_weight_l2'] = float(best['model']['c0_adapter.body.2.weight'].double().norm())
            assert summary[arm]['adapter_last_final_weight_l2'] > 0
    matches = []
    for epoch in range(1, 21):
        a, b = [training_rows[arm][epoch] for arm in experiment.ARMS]
        for split in ('train', 'val'):
            assert a[split]['case_station_digest'] == b[split]['case_station_digest']
        matches.append(dict(epoch=epoch, optimizer_steps=a['optimizer_steps'],
                            train_case_station_digest=a['train']['case_station_digest'],
                            val_case_station_digest=a['val']['case_station_digest'], paired=True))
    pd.DataFrame(matches).to_csv(out / 'epoch_matching.csv', index=False)
    assert sealed['rows_per_arm'] == 900 and all(len(v) == 900 for v in sealed['row_sha256'].values())
    assert sealed['paired_complete_input_hashes_match']
    gt_access = json.loads((out / 'gt_access.json').read_text())
    assert gt_access['forbidden_attempts'] == gt_access['checkpoint_opens'] == gt_access['val_pool_a_projection_opens'] == 0
    assert gt_access['unique_val_gt64_files'] == 36
    artifact = dict(at=experiment.now(), source_and_input_hashes_verified=True, scientific_source_unchanged=True,
                    frozen_original_checkpoint_unchanged=True, no_forward_or_projector_calls=True,
                    paired_inputs=True, predictions_per_arm=900, all_saved_prediction_hashes_verified=True,
                    all_row_hashes_verified=True, GT_phase_access=gt_access, arms=summary,
                    analysis_source_manifest_sha256=experiment.digest(out / 'analysis_source_manifest.json'),
                    verifier_source_sha256=experiment.digest(__file__), registered_parameter_count=protocol['parameter_count'])
    experiment.write_json(out / 'final_integrity_audit.json', artifact)
    experiment.archive(out, ['final_integrity_audit.json', 'epoch_matching.csv'])
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
