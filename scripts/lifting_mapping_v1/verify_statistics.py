"""Independent CSV regrouping and weighted-bootstrap verification; no GT access."""
from runtime import *
import csv


def table(name):
    with (BASE / name).open() as f:
        return list(csv.DictReader(f))


def main():
    verified(BASE / 'all_predictions_sealed.json')
    stats = verified(BASE / 'statistics.json')
    metric = table('per_case_metrics.csv'); changes = table('prediction_changes.csv')
    cases = verified(BASE / 'cases.json')
    raw = {(r['case_id'], r['condition'], r['intervention'], r['stage']): r for r in metric}
    assert len(raw) == len(metric) == 1400
    sensitivity = {(r['case_id'], r['condition'], r['intervention'], r['stage']): r for r in changes}
    assert len(sensitivity) == 1050
    # Separate grouping implementation: construct a dense case x layout x mode
    # tensor from raw absolute metrics, subtract the baseline only afterward.
    indices = np.random.default_rng(20261022).choice(35, size=(20000, 35), replace=True)
    weights = np.stack([(indices == i).sum(axis=1) / 35 for i in range(35)], axis=1)
    max_error = 0.
    for panel in stats['panels']:
        layouts = CONDITIONS if panel['condition'] == 'ALL_5_LAYOUTS' else (panel['condition'],)
        modes = MODES if panel['intervention'] == 'mean_3_interventions' else (panel['intervention'],)
        key = panel['metric']; stage = panel['stage']; values = []; baseline = []; changed = []
        missing = False
        for case in cases:
            bases = []; outputs = []
            for layout in layouts:
                b = raw[case, layout, 'baseline', stage][key]
                rr = [raw[case, layout, mode, stage][key] for mode in modes]
                if b == '' or any(v == '' for v in rr):
                    missing = True; break
                bases.append(float(b)); outputs.append(sum(float(v) for v in rr)/len(modes))
            if missing:
                break
            baseline.append(sum(bases)/len(layouts)); changed.append(sum(outputs)/len(layouts))
            values.append(changed[-1]-baseline[-1])
        if missing:
            assert panel['ci95'] is None
            continue
        a = np.array(values)
        np.testing.assert_allclose(a, panel['case_deltas'], atol=1e-12, rtol=1e-10)
        ci = np.percentile(weights @ a, [2.5, 97.5])
        np.testing.assert_allclose(ci, panel['ci95'], atol=1e-12, rtol=1e-10)
        for actual, expected in [(a.mean(), panel['mean_delta']), (np.median(a), panel['median_delta']),
                                 (np.mean(baseline), panel['baseline_mean']), (np.mean(changed), panel['intervention_mean'])]:
            assert abs(actual-expected) < 1e-12
            max_error = max(max_error, abs(actual-expected))
    for panel in stats['prediction_changes']:
        layouts = CONDITIONS if panel['condition'] == 'ALL_5_LAYOUTS' else (panel['condition'],)
        modes = MODES if panel['intervention'] == 'mean_3_interventions' else (panel['intervention'],)
        a = np.array([sum(float(sensitivity[case, layout, mode, panel['stage']][panel['metric']])
                          for layout in layouts for mode in modes) / (len(layouts)*len(modes)) for case in cases])
        np.testing.assert_allclose(a, panel['case_values'], atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(np.percentile(weights @ a, [2.5, 97.5]), panel['ci95'], atol=1e-12, rtol=1e-10)
    primary = stats['primary']; assert primary['n_cases'] == 35
    assert primary['condition'] == 'ALL_5_LAYOUTS' and primary['intervention'] == 'mean_3_interventions'
    freeze(BASE / 'independent_statistics_verification.json', dict(at=now(), passed=True,
        panels_checked=len(stats['panels']), sensitivity_panels_checked=len(stats['prediction_changes']),
        n_cases=35, resamples=20000, seed=20261022, max_scalar_disagreement=max_error,
        method='independent dense grouping from absolute metric CSV, then subtraction; multinomial-weight bootstrap vs indexed paired bootstrap',
        GT_read=False, files={str(BASE / n): sha(BASE / n) for n in ('per_case_metrics.csv', 'prediction_changes.csv', 'statistics.json')}))
    print('INDEPENDENT STATISTICS VERIFIED', len(stats['panels']), len(stats['prediction_changes']), flush=True)


if __name__ == '__main__':
    main()
