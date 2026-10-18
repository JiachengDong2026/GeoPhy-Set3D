#!/usr/bin/env python3
"""Independent scalar-only verification of the registered four-arm OCMSF screen."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/mrgr_ocmsf_screen_v1'
ARMS = ('A_original', 'B_multiscale', 'C_content', 'D_ocmsf')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert np.isclose(a, b, atol=1e-14, rtol=1e-12), (a, b)


def check_audit(audit):
    assert audit['gt_file_opened'] is False
    assert all(audit[k] == 0 for k in ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts'))


def main():
    p = read(OUT / 'protocol.json')
    assert digest(OUT / 'protocol.json') == (OUT / 'protocol.sha256').read_text().strip()
    for path, expected in p['sources'].items():
        assert digest(ROOT / path) == expected, path
    assert p['no_gt'] and p['no_final_test'] and p['no_automatic_formal300']
    assert p['epochs'] == 20 and p['updates_per_arm'] == 1660
    assert p['parameters'] == dict(A_original=332070, B_multiscale=344678, C_content=345417, D_ocmsf=345417)
    assert p['initial_state_sha256']['C_content'] == p['initial_state_sha256']['D_ocmsf']
    assert digest(Path(p['panel_source'])) == p['panel_source_sha256']
    check_audit(p['registration_audit'])
    smoke = read(OUT / 'smoke.json')
    assert smoke['passed'] and smoke['initial_B_C_D_bitwise_equal'] and smoke['all_smoke_models_discarded']
    check_audit(smoke['GT_access'])
    result = read(OUT / 'comparison.json')
    assert result['protocol_sha256'] == digest(OUT / 'protocol.json')
    assert result['gt_used'] is False and result['final_test_used'] is False and result['formal300_started'] is False
    logs, ordinary, hard, times = {}, {}, {}, {}
    ids = p['validation_case_ids']
    assert len(ids) == len(set(ids)) == 36 and len(p['train_case_ids']) == 166
    for arm in ARMS:
        folder = OUT / arm
        assert digest(folder / 'effective_config.json') == p['effective_config_sha256'][arm]
        cfg = read(folder / 'effective_config.json')
        assert cfg['training']['device'] == p['arm_device'][arm]
        logs[arm] = rows(folder / 'metrics.jsonl')
        assert [r['epoch'] for r in logs[arm]] == list(range(1, 21))
        for record in logs[arm]:
            for split, count in (('train', 166), ('val', 36)):
                assert record[split]['sample_count'] == count and record[split]['gt_used'] is False
        summary = read(folder / 'summary.json')
        assert summary['epochs'] == 20 and summary['updates'] == 1660 and summary['output_sanity_passed']
        assert digest(OUT / 'checkpoints' / arm / 'last.pt') == summary['checkpoint_sha256']
        check_audit(summary['gt_audit'])
        check_audit(read(folder / 'gt_access_audit.json'))
        raw = rows(folder / 'epoch20_per_case.jsonl')
        assert [r['case_id'] for r in raw] == ids
        for r in raw:
            close(r['heldout_relative_l1'], .5 * (r['A']['relative_l1'] + r['B']['relative_l1']))
            assert r['stations'] == p['station_map'][r['case_id']]
            for s in r['output_sanity'].values():
                assert 0 <= s['min'] < s['max'] <= 1 and s['std'] > 1e-8
            if arm in ('C_content', 'D_ocmsf'):
                for g in r['gate'].values():
                    assert np.isclose(sum(g['alpha_mean']), 1., atol=2e-6, rtol=0)
                    assert g['delta_norm'] > 0 and g['visible_voxels'] > 0
        ordinary[arm] = np.array([r['heldout_relative_l1'] for r in raw])
        close(ordinary[arm].mean(), summary['final_H'])
        close(ordinary[arm].mean(), logs[arm][-1]['val']['measurement']['heldout']['relative_l1'])
        cluster = rows(folder / 'cluster5_per_case.jsonl')
        assert len(cluster) == len({(r['case_id'], r['draw']) for r in cluster}) == 180
        settings = {s['draw']: s for s in p['cluster_settings']}
        for r in cluster:
            assert r['input_ids'] == settings[r['draw']]['input_ids'][r['case_id']]
            assert r['heldout_ids'] == settings[r['draw']]['heldout_ids'][r['case_id']]
        hard[arm] = np.array([np.mean([r['relative_l1'] for r in cluster if r['case_id'] == case]) for case in ids])
        cluster_summary = read(folder / 'cluster5_summary.json')
        close(hard[arm].mean(), cluster_summary['mean'])
        check_audit(cluster_summary['gt_audit'])
        times[arm] = float(np.mean([r['seconds'] for r in logs[arm][2:]]))
        close(times[arm], summary['warm_epoch_seconds'])
        for panel, values in (('ordinary', ordinary[arm]), ('cluster5', hard[arm])):
            for key, value in dict(mean=np.mean(values), p90=np.quantile(values, .9), median=np.median(values), maximum=np.max(values)).items():
                close(value, result[panel][arm][key])
    for arm in ARMS:
        for a, b in zip(logs['B_multiscale'], logs[arm]):
            assert a['learning_rate'] == b['learning_rate'] and a['loader_rng_sha256'] == b['loader_rng_sha256']
            for split in ('train', 'val'):
                assert a[split]['case_station_digest'] == b[split]['case_station_digest']
    for control, treatment in itertools.combinations(ARMS, 2):
        contrast = result['contrasts'][f'{treatment}_vs_{control}']
        for name, arrays in (('ordinary', ordinary), ('cluster5', hard)):
            a, b = arrays[control], arrays[treatment]
            close(b.mean() / a.mean(), contrast[name]['mean_ratio'])
            close(np.quantile(b, .9) / np.quantile(a, .9), contrast[name]['p90_ratio'])
            delta = b - a
            rng = np.random.default_rng(20261017)
            ci = np.quantile(delta[rng.integers(0, 36, (20000, 36))].mean(1), [.025, .975])
            for x, y in zip(ci, contrast[name]['paired_ci95']):
                close(x, y)
            assert contrast[name]['wins'] == int((delta < -1e-8).sum())
            assert contrast[name]['losses'] == int((delta > 1e-8).sum())
    gates = {}
    t = p['promotion_gate']
    for control in ('B_multiscale', 'C_content'):
        a, b = ordinary[control], ordinary['D_ocmsf']
        checks = dict(ordinary_mean=b.mean() / a.mean() <= t['ordinary_mean_ratio_max'],
                      ordinary_p90=np.quantile(b, .9) / np.quantile(a, .9) <= t['ordinary_p90_ratio_max'],
                      cluster5_mean=hard['D_ocmsf'].mean() / hard[control].mean() <= t['cluster5_mean_ratio_max'],
                      runtime=times['D_ocmsf'] / times[control] <= t['runtime_ratio_max'],
                      five_hour=times['D_ocmsf'] * 300 / 3600 <= t['projected300_hours_max'],
                      correctness=smoke['passed'], output_sanity=True)
        assert checks == result['gates'][control]['checks']
        gates[control] = bool(all(checks.values()))
        assert gates[control] == result['gates'][control]['gate_passed']
    assert bool(all(gates.values())) == result['gate_passed']
    verified = dict(independent_verification_passed=True, protocol_sha256=digest(OUT / 'protocol.json'),
                    verifier_sha256=digest(Path(__file__)), gate_passed=result['gate_passed'], gates=gates,
                    cases_per_arm=36, cluster_rows_per_arm=180, total_epochs=80, total_updates=6640,
                    gt_arrays_opened=False, final_test_used=False,
                    note='Scalar/checksum audit; no independent new volume prediction or3D GT scoring.')
    (OUT / 'independent_verification.json').write_text(json.dumps(verified, indent=2) + '\n')
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
