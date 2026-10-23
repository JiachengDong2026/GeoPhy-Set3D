"""Independent saved-JSON audit; no imports from the evaluated implementation.

Run with the project's gas-torch Python. Reads only a fixed set of JSON files
under evaluation_mini_v2; never loads checkpoint, measurement, prediction or GT
arrays. Recomputes case bootstrap via sample-count matrix multiplication.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import numpy.random  # Resolve lazy runtime dependencies before the JSON-only guard.
import numpy.ma
np.quantile(np.array([0., 1.]), [.025, .975])

ROOT = Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_v1')
EVALUATION = ROOT / 'evaluation_mini_v2'
OUTPUT = ROOT / 'independent_review'
ARMS = ('plain', 'deepsets', 'senseiver', 'gnot')
GROUPS = ('G1', 'G2', 'G3', 'G4', 'G5')
METRICS = ('relative_l1', 'relative_l2', 'mae', 'rmse',
           'integrated_concentration_relative_error', 'peak_value_absolute_error',
           'observed_relative_l1', 'observed_relative_l2',
           'heldout_relative_l1', 'heldout_relative_l2')
JSON_NAMES = ('protocol.json', 'predictions_sealed.json', 'first_GT_boundary.json',
              'GT_complete.json', 'GT_access.json', 'scored_rows.json', 'summary.json')
allowed = {(EVALUATION / arm / name).resolve() for arm in ARMS for name in JSON_NAMES}
allowed |= {(EVALUATION / arm / 'paired_to_plain.json').resolve() for arm in ARMS[1:]}
inputs = {}
opened = []
active = True


def audit_open(event, args):
    if not active or event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = any(c in mode for c in 'wxa+') if isinstance(mode, str) else bool(flags & (os.O_WRONLY | os.O_RDWR))
    if writing:
        if path.parent != OUTPUT or path.suffix != '.json':
            raise PermissionError(f'Audit output outside allowed JSON: {path}')
    elif path not in allowed:
        raise PermissionError(f'Audit reads fixed saved JSON only: {path}')
    else:
        opened.append(str(path))


sys.addaudithook(audit_open)


def read(path):
    path = path.resolve()
    raw = path.read_bytes()
    inputs[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return json.loads(raw)


field_count = 0
max_difference = 0.
max_difference_field = None


def compare(actual, expected, field):
    global field_count, max_difference, max_difference_field
    field_count += 1
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise AssertionError(f'Null mismatch: {field}')
        return
    if not np.isfinite(actual) or not np.isfinite(expected):
        raise AssertionError(f'Nonfinite statistic: {field}')
    difference = abs(float(actual) - float(expected))
    if difference > max_difference:
        max_difference, max_difference_field = difference, field
    if difference > 1.e-12 * max(1., abs(float(expected))):
        raise AssertionError(f'Statistical discrepancy: {field}: {actual} != {expected}')


rng = np.random.default_rng(2026101801)
resample_counts = np.stack([np.bincount(sample, minlength=2)
                           for sample in rng.integers(0, 2, (10000, 2))])


def values_for(rows, cases, group, metric):
    # Independent grouping, with draw averaging before case resampling.
    return np.array([np.mean([r[metric] for r in rows
                              if r['record']['case_id'] == case and r['record']['group'] == group])
                     for case in cases], dtype=np.float64)


def interval(values):
    return np.quantile(resample_counts @ values / 2., [.025, .975])


reference = None
arm_checks = {}
for arm in ARMS:
    folder = EVALUATION / arm
    objects = {name: read(folder / name) for name in JSON_NAMES}
    protocol, summary, rows = [objects[name] for name in ('protocol.json', 'summary.json', 'scored_rows.json')]
    seal, access = objects['predictions_sealed.json'], objects['GT_access.json']
    assert protocol['mini'] is True and protocol['split'] == 'val'
    assert protocol['test_allowed'] is False and protocol['Pool_B_allowed'] is False
    assert protocol['GT128_allowed'] is False
    assert summary['formal'] is False and summary['scope'] == 'NONFORMAL_MINI_INTEGRATION'
    assert summary['cases'] == 2 and summary['arm'] == arm
    assert summary['bootstrap']['seed'] == 2026101801 and summary['bootstrap']['samples'] == 10000
    assert len(protocol['records']) == len(rows) == seal['rows'] == 10
    assert sorted({r['record']['case_index'] for r in rows}) == [0, 1]
    assert all(r['record']['draw'] == 0 for r in rows)
    assert all(len(r['record']['station_ids']) == len(r['record']['H_station_ids']) == 4 for r in rows)
    assert all(not (set(r['record']['station_ids']) & set(r['record']['H_station_ids'])) for r in rows)
    cases = sorted({r['record']['case_id'] for r in rows})
    assert len(cases) == 2
    expected_keys = {(case, group, 0) for case in cases for group in GROUPS}
    assert {(r['record']['case_id'], r['record']['group'], r['record']['draw']) for r in rows} == expected_keys
    assert access['forbidden_attempts'] == access['checkpoint_opens'] == access['val_pool_a_projection_opens'] == 0
    assert access['unique_val_gt64_files'] == 2
    assert seal['all_predictions_saved_before_GT'] is True
    assert seal['protocol_sha256'] == inputs[str((folder / 'protocol.json').resolve())]['sha256']
    assert objects['GT_complete.json']['scored_rows_sha256'] == inputs[str((folder / 'scored_rows.json').resolve())]['sha256']
    assert objects['first_GT_boundary.json']['saved_outputs_only'] is True
    assert objects['first_GT_boundary.json']['checkpoint_reads_during_GT'] is False
    assert objects['first_GT_boundary.json']['seal_sha256'] == inputs[str((folder / 'predictions_sealed.json').resolve())]['sha256']
    assert set(summary_row['group'] for summary_row in summary['summary']) == set(GROUPS)
    assert len(summary['summary']) == 50 and len(summary['contrasts']) == 10
    for result in summary['summary']:
        vals = values_for(rows, cases, result['group'], result['metric'])
        low, high = interval(vals)
        expected = dict(mean_per_case=vals.mean(), median=np.median(vals),
                        p90=np.quantile(vals, .9), p95=np.quantile(vals, .95),
                        maximum=vals.max(), ci_low=low, ci_high=high)
        for field, value in expected.items():
            compare(result[field], value, f'{arm}/summary/{result["group"]}/{result["metric"]}/{field}')
        assert result['worst_case_id'] == cases[int(vals.argmax())]
    for result in summary['contrasts']:
        first = values_for(rows, cases, 'G1', result['metric'])
        last = values_for(rows, cases, 'G5', result['metric'])
        low, high = interval(last - first)
        denominator = resample_counts @ first / 2.
        ratio_valid = first.mean() != 0 and bool(np.all(denominator != 0))
        ratio = last.mean() / first.mean() if ratio_valid else None
        ratio_low, ratio_high = np.quantile((resample_counts @ last / 2.) / denominator, [.025, .975]) if ratio_valid else (None, None)
        expected = dict(G5_minus_G1=(last-first).mean(), difference_ci_low=low, difference_ci_high=high,
                        G5_div_G1_grand_mean_ratio=ratio, ratio_ci_low=ratio_low, ratio_ci_high=ratio_high)
        for field, value in expected.items():
            compare(result[field], value, f'{arm}/contrasts/{result["metric"]}/{field}')
    if arm == 'plain':
        reference = rows
    else:
        paired = read(folder / 'paired_to_plain.json')
        assert paired['formal'] is False and paired['same_training_seed'] is True
        assert paired['plain_seed'] == paired['baseline_seed'] == protocol['seed'] == 20261011
        assert len(paired['rows']) == 50
        identity = lambda r: (r['record']['case_id'], r['record']['group'], r['record']['draw'])
        plain_map, baseline_map = {identity(r): r for r in reference}, {identity(r): r for r in rows}
        assert set(plain_map) == set(baseline_map)
        for key, full in plain_map.items():
            baseline = baseline_map[key]
            assert full['record'] == baseline['record']
            assert all(full[field] == baseline[field] for field in ('complete_input_sha256', 'complete_H_sha256'))
        for result in paired['rows']:
            full = values_for(reference, cases, result['group'], result['metric'])
            base = values_for(rows, cases, result['group'], result['metric'])
            delta = base - full
            low, high = interval(delta)
            expected = dict(plain_mean=full.mean(), baseline_mean=base.mean(), mean_delta=delta.mean(),
                            median_delta=np.median(delta), ci_low=low, ci_high=high,
                            baseline_worse=int((delta > 1.e-8).sum()), baseline_better=int((delta < -1.e-8).sum()),
                            ties=int((np.abs(delta) <= 1.e-8).sum()))
            for field, value in expected.items():
                compare(result[field], value, f'{arm}/paired/{result["group"]}/{result["metric"]}/{field}')
    arm_checks[arm] = dict(rows=10, cases=2, groups=5, draws=1, seed=protocol['seed'],
                          checkpoint_sha256=protocol['checkpoint_sha256'],
                          formal=False, scope=summary['scope'],
                          gt_stage_unique_val_gt64_files_reported=access['unique_val_gt64_files'],
                          gt_stage_forbidden_attempts_reported=access['forbidden_attempts'])

result = dict(status='PASS', at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              evaluation_root=str(EVALUATION), scope='NONFORMAL_MINI_INTEGRATION_ONLY',
              independent_implementation='standalone grouping plus multinomial-count bootstrap matrix multiplication; no evaluated-module import',
              statistical_numeric_fields_recomputed=field_count, max_absolute_difference=max_difference,
              max_difference_field=max_difference_field, bootstrap_seed=2026101801, bootstrap_samples=10000,
              resampling_unit='case after draw mean', ratio_definition='ratio of group grand means',
              delta_definition='baseline minus plain', seeds_never_pooled=True,
              input_json_hashes=inputs, input_json_count=len(inputs), audit_read_events=len(opened),
              audit_read_boundary='fixed saved JSON only, enforced by Python audit hook during recomputation',
              prediction_arrays_read=0, measurement_arrays_read=0, gt_arrays_read=0, checkpoint_arrays_read=0,
              original_gt_access_evidence='GT_access.json contents audited as records; this review does not open or hash GT arrays',
              prediction_byte_hashes_rechecked=False,
              scientific_limit='2 validation cases x 5 groups x 1 draw cannot support formal accuracy or independent-test claims',
              arms=arm_checks)
with (OUTPUT / 'summary.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({k:result[k] for k in ('status', 'scope', 'statistical_numeric_fields_recomputed', 'max_absolute_difference', 'input_json_count', 'gt_arrays_read', 'prediction_arrays_read', 'checkpoint_arrays_read')}, indent=2))
