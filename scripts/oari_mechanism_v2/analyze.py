"""Offline case-level aggregation, without GT or model execution."""
from runtime import *


def case_rows(records):
    result = []
    for r in records:
        b = [x for x in r['bias_rows'] if x['valid_views'] == 'all']
        a = [x for x in r['attention_rows'] if x['valid_views'] == 'all' and x['query_scope'] == 'all']
        c = next(x for x in r['context_rows'] if x['valid_views'] == 'all')
        row = {k: r[k] for k in ('plan_id', 'case_id', 'condition', 'group')}
        for name in ('js_mean', 'mean_abs_mean', 'max_abs_max', 'entropy_delta_mean'):
            row['attention_' + name] = float(np.mean([x[name] for x in a]))
        for name in ('bias_rms', 'centered_bias_rms', 'content_rms', 'bias_content_rms_ratio', 'bias_delta_rms', 'content_delta_rms'):
            row[name] = float(np.mean([x[name] for x in b]))
        row.update(context_relative_l1=c['relative_l1'], context_relative_l2=c['relative_l2'],
                   context_near_zero_fraction=c['near_zero_baseline_fraction'],
                   context_changed_voxel_fraction=c['fraction_relative_above_threshold'],
                   relation_changed_pair_fraction=r['intervention'].get('changed_valid_pair_fraction', 0),
                   source_contamination_fraction=r['intervention']['source_contamination_fraction'])
        for stage in ('C0', 'C1'):
            out = next(x for x in r['output_rows'] if x['stage'] == stage)
            for name in ('relative_l1', 'relative_l2', 'mean_abs', 'max_abs', 'peak_amplitude_delta', 'peak_distance_mm', 'changed_voxel_fraction'):
                row[stage + '_' + name] = out[name]
        result.append(row)
    return result


def collapsed_cases(records):
    groups = {}
    for row in records:
        group = 'A3_P1star_mean8' if row['group'].startswith('A3_') else row['group']
        groups.setdefault((row['case_id'], row['condition'], group), []).append(row)
    out = []
    for (case, condition, group), rows_ in groups.items():
        assert len(rows_) == (8 if group.startswith('A3_') else 1)
        row = dict(case_id=case, condition=condition, group=group)
        row.update({k: float(np.mean([r[k] for r in rows_])) for k in rows_[0] if k not in ('plan_id', 'case_id', 'condition', 'group')})
        out.append(row)
    return out


def summarize(records):
    output = []
    cases = verified(BASE / 'cases.json')
    for condition in (*CONDITIONS, 'across5_within_case_mean'):
        for group in dict.fromkeys(r['group'] for r in records):
            selected = [r for r in records if r['group'] == group and (condition == 'across5_within_case_mean' or r['condition'] == condition)]
            values = {k: [] for k in selected[0] if k not in ('case_id', 'condition', 'group')}
            for case in cases:
                rows_ = [r for r in selected if r['case_id'] == case]
                assert len(rows_) == (5 if condition == 'across5_within_case_mean' else 1)
                for k in values:
                    values[k].append(float(np.mean([r[k] for r in rows_])))
            for metric, x in values.items():
                arr = np.asarray(x)
                output.append(dict(condition=condition, group=group, metric=metric, n_cases=35,
                                   mean=float(arr.mean()), median=float(np.median(arr)), p90=float(np.quantile(arr, .9)),
                                   maximum=float(arr.max()), minimum=float(arr.min())))
            # Affected fraction is based on the case's mean response after repetitions/layouts.
            for stage in ('C0', 'C1'):
                output.append(dict(condition=condition, group=group, metric=stage + '_affected_case_fraction', n_cases=35,
                                   mean=float((np.asarray(values[stage + '_max_abs']) > CHANGE).mean())))
    return output


def run():
    validate()
    assert verified(BASE / 'inference_complete.json')['completed']
    records = []
    with Guard(projections=False) as guard:
        for plan in verified(BASE / 'plans.json'):
            unit = BASE / 'units' / f'p{plan["plan_id"]:03d}'
            verify_hashes(verified(unit / 'complete.json')['files'])
            for group in GROUPS:
                r = verified(unit / f'{group}.json')
                verify_hashes(r['files'])
                assert r['checkpoint_sha256'] == HIST.EXPECTED and r['eval'] and r['no_grad']
                records.append(r)
        assert len(records) == 3150 and sum(not r['prediction_reused'] for r in records) == 2625
        execution = rows(BASE / 'forward_log.jsonl')
        kinds = {kind: sum(r['kind'] == kind for r in execution) for kind in sorted({r['kind'] for r in execution})}
        uuids = sorted({r['gpu_uuid'] for r in execution})
        assert len(uuids) == 1
        if not (BASE / 'forward_execution_audit.json').exists():
            freeze(BASE / 'forward_execution_audit.json', dict(at=now(), total_logged_calls=len(execution),
                calls_by_kind=kinds, GPU_UUIDs=uuids, new_prediction_count=2625, reused_prediction_count=525,
                engineering_complete=verified(BASE / 'engineering_checks.json'),
                forward_log_sha256=sha(BASE / 'forward_log.jsonl'), sources_unchanged=True,
                all_group_input_model_identities_verified=True, all_outputs_finite_in_range=True))
        # Seal all identities before any downstream interpretation (GT remains forbidden).
        seal = BASE / 'prediction_manifest.json'
        if not seal.exists():
            freeze(seal, [dict(**{k: r[k] for k in ('plan_id', 'case_id', 'condition', 'group', 'input_sha256',
                    'checkpoint_sha256', 'prediction', 'prediction_sha256', 'prediction_reused', 'probe_path')},
                    probe_sha256=r['files'][r['probe_path']]) for r in records])
        tables = {'bias_statistics': 'bias_rows', 'attention_change_statistics': 'attention_rows',
                  'context_change_statistics': 'context_rows', 'output_change_statistics': 'output_rows'}
        for name, key in tables.items():
            path = BASE / (name + '.csv')
            if not path.exists():
                csv_write(path, [row for r in records for row in r[key]])
        per_case = case_rows(records)
        collapsed = collapsed_cases(per_case)
        for name, table in [('per_case_results', per_case), ('case_results_collapsed', collapsed)]:
            path = BASE / (name + '.csv')
            if not path.exists():
                csv_write(path, table)
        summary = summarize(collapsed)
        if not (BASE / 'paired_statistics.csv').exists():
            csv_write(BASE / 'paired_statistics.csv', summary)
        if not (BASE / 'summary.json').exists():
            freeze(BASE / 'summary.json', dict(at=now(), n_cases=35, inputs=175, records=3150,
                summaries=summary, all_GT_forbidden=True, threshold=CHANGE,
                interpretation='frozen sensitivity on previously used test cases; no accuracy/performance contribution identified',
                A3='8 fields averaged within case/layout;5 layouts averaged within case for across-layout analysis',
                nonlinear_interaction='joint response minus sum of single response is descriptive nonadditivity, not independent contribution',
                guard=guard.summary()))
        assert not guard.denied
    print('OFFLINE AGGREGATION COMPLETE', flush=True)


if __name__ == '__main__':
    run()
