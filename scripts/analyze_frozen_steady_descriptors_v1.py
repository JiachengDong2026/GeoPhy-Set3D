"""Matched, CPU-only spatial descriptors of sealed validation reconstructions.

This computes concentration descriptors, never temporal or transport residuals.
All predictions and ordered input stations are verified before opening GT64.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
OUT = ROOT / 'outputs/research/frozen_steady_descriptors_v1'
HISTORY = ROOT / 'outputs/research/historical_reconstruction_morphology_v1'
MS = ROOT / 'outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1'
RUNS = {'A': ROOT / 'outputs/research/scheme_a_budget_v4_wide_seed20261011',
        'B': ROOT / 'outputs/research/scheme_b_perf_v2_formal_seed20261011'}
MODELS = ('History', 'Multiscale', 'A', 'B')
CONDITIONS = ('Val-A-V4', 'clustered-V4')
Y = np.linspace(0, .6, 64)
W = np.full(64, .6 / 63)
W[[0, -1]] *= .5
AREA = W[:, None] * W[None, :]


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def section(c):
    return np.einsum('ijk,i,k->j', c, W, W, optimize=True)


def integrate(c): return float(section(c) @ W)


def faces(c):
    return dict(x0=c[0, :, :], x1=c[-1, :, :], y0=c[:, 0, :],
                y1=c[:, -1, :], z0=c[:, :, 0], z1=c[:, :, -1])


def metrics(p, g):
    p, g = np.asarray(p, dtype=np.float64), np.asarray(g, dtype=np.float64)
    assert p.shape == g.shape == (64, 64, 64)
    assert np.isfinite(p).all() and np.isfinite(g).all()
    assert p.min() >= 0 and p.max() <= 1 and g.min() >= 0 and g.max() <= 1
    ip, ig = section(p), section(g)
    qp, qg = float(ip @ W), float(ig @ W)
    assert qp > 0 and qg > 0
    pf, gf = faces(p), faces(g)
    diff = np.abs(p - g)
    # Source-band endpoint is exactly grid plane y=5*0.6/63, not rounded to .05 m.
    near_w = W[:6].copy()
    near_w[-1] *= .5
    near_error = float(section(diff)[:6] @ near_w)
    near_gt = float(ig[:6] @ near_w)
    assert near_gt > 0
    peak_index = np.unravel_index(g.argmax(), g.shape)
    result = dict(
        weighted_relative_l1=integrate(diff) / qg,
        integral_absolute_relative_error=abs(qp / qg - 1),
        integral_ratio=qp / qg,
        height_marginal_relative_l1=float(np.abs(ip - ig) @ W / qg),
        height_distribution_TV=float(.5 * (np.abs(ip / qp - ig / qg) @ W)),
        centroid_y_error_mm=float(abs((ip * Y) @ W / qp - (ig * Y) @ W / qg) * 1000),
        centroid_y_signed_error_mm=float(((ip * Y) @ W / qp - (ig * Y) @ W / qg) * 1000),
        source_band_relative_l1=near_error / near_gt,
        global_peak_ratio=float(p.max() / g.max()),
        gt_peak_location_ratio=float(p[peak_index] / g.max()),
        six_face_relative_l1=float(sum((np.abs(pf[f]-gf[f]) * AREA).sum() for f in pf)
                                  / sum((gf[f] * AREA).sum() for f in gf)),
    )
    for f in pf:
        result[f + '_error_over_mean_section'] = float((np.abs(pf[f]-gf[f]) * AREA).sum() / (qg / .6))
        result[f + '_integral_signed_error_m2'] = float(((pf[f]-gf[f]) * AREA).sum())
    # The height marginal cannot have more L1 error than the entire volume.
    assert result['height_marginal_relative_l1'] <= result['weighted_relative_l1'] + 1e-12
    assert 0 <= result['height_distribution_TV'] <= 1 + 1e-12
    return result, ip / (qg / .6), ig / (qg / .6)


def verify_predictions(cases):
    found, evidence = {}, []
    for model, root in [('History', HISTORY), ('Multiscale', MS)]:
        manifest = root / 'prediction_manifest.jsonl'
        complete = read(root / 'predictions_complete.json')
        assert sha(manifest) == complete['manifest_sha256']
        assert complete['all_predictions_saved_before_gt']
        assert complete['audit']['gt_open_attempts'] == 0
        for r in rows(manifest):
            if r['condition'] not in CONDITIONS or r.get('draw', 0) != 0: continue
            key = model, r['condition'], r['case_id']
            assert key not in found
            path = root / r['prediction']
            assert sha(path) == r['sha256']
            found[key] = dict(path=path, ids=r.get('station_ids', r.get('input_ids')), sha256=r['sha256'])
        evidence.append(dict(model=model, manifest=str(manifest), sha256=sha(manifest)))
    for model, run in RUNS.items():
        for cond in CONDITIONS:
            directory = Path(read(run / f'evaluation/frozen_model_test/{cond}_prediction.json')['directory'])
            manifest = directory / 'predictions_manifest.jsonl'
            boundary = read(directory / 'prediction_boundary.json')
            assert sha(manifest) == boundary['prediction_manifest_sha256']
            assert boundary['all_predictions_saved_before_gt']
            for r in rows(manifest):
                path = directory / r['prediction']
                assert sha(path) == r['prediction_sha256']
                found[model, cond, r['case_id']] = dict(path=path, ids=r['selected_station_ids'], sha256=r['prediction_sha256'])
            evidence.append(dict(model=model, condition=cond, manifest=str(manifest), sha256=sha(manifest)))
    assert set(found) == {(m, c, case) for m in MODELS for c in CONDITIONS for case in cases}
    for cond in CONDITIONS:
        for case in cases:
            assert all(found[m, cond, case]['ids'] == found['History', cond, case]['ids'] for m in MODELS)
    return found, evidence


def main():
    index = [r for r in rows(DATA / 'source_index.jsonl') if r['split'] == 'val']
    cases = {r['case_id'] for r in index}
    assert len(index) == len(cases) == 36
    assert cases == set((DATA / 'splits/val_cases.txt').read_text().split())
    predictions, evidence = verify_predictions(cases)
    records, profiles, gt_reads = [], {}, []
    for r in index:
        case = r['case_id']
        gt_path = DATA / r['gt_volume_64_path']
        g = np.load(gt_path, allow_pickle=False).squeeze()
        gt_reads.append(dict(case_id=case, path=str(gt_path), sha256=sha(gt_path)))
        for cond in CONDITIONS:
            for model in MODELS:
                item = predictions[model, cond, case]
                if item['path'].suffix == '.npz':
                    with np.load(item['path'], allow_pickle=False) as a: p = a['C1'].copy()
                else:
                    p = np.load(item['path'], allow_pickle=False).squeeze()
                values, ip, ig = metrics(p, g)
                records.append(dict(model=model, condition=cond, case_id=case,
                                    wind_speed_m_s=r['condition_meta']['wind_speed_m_s'], **values))
                profiles.setdefault((model, cond), []).append(ip)
                if model == 'History': profiles.setdefault(('GT', cond), []).append(ig)
    keys = [k for k, v in records[0].items() if isinstance(v, float) and k != 'wind_speed_m_s']
    summary = []
    for cond in CONDITIONS:
        for model in MODELS:
            selected = [r for r in records if r['condition'] == cond and r['model'] == model]
            summary.append(dict(model=model, condition=cond, n=len(selected),
                mean={k: float(np.mean([r[k] for r in selected])) for k in keys},
                median={k: float(np.median([r[k] for r in selected])) for k in keys}))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'per_case_metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    write(OUT / 'summary.json', dict(completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        scope='36 validation cases; two exact matched draw0 layouts; four previously frozen models; exploratory development',
        physical_quadrature='endpoint trapezoidal; C integral, not methane mass',
        source_band_y_max_m=float(Y[5]), summary=summary, prediction_manifests=evidence,
        gt_reads=gt_reads, final_test_opened=False, checkpoints_opened=False,
        temporal_or_PDE_residual_computed=False, source_sha256=sha(__file__)))
    arrays = {'y_m': Y, 'case_ids': np.array([r['case_id'] for r in index])}
    for (model, cond), p in profiles.items(): arrays[model + '_' + cond] = np.array(p)
    np.savez_compressed(OUT / 'height_profiles.npz', **arrays)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, cond in enumerate(CONDITIONS):
        for model in ('GT',) + MODELS:
            axes[row, 0].plot(Y, np.mean(profiles[model, cond], axis=0), label=model)
        axes[row, 0].set(title=cond + ': mean height sections', xlabel='Height y (m)', ylabel='I(y) / GT mean-height I')
        axes[row, 0].legend(fontsize=8)
        for col, key, title in [(1, 'source_band_relative_l1', 'Source-band relative L1 (%)'),
                                (2, 'six_face_relative_l1', 'Six-face concentration relative L1 (%)')]:
            groups = [[r[key]*100 for r in records if r['condition'] == cond and r['model'] == m] for m in MODELS]
            axes[row, col].boxplot(groups, tick_labels=MODELS, showfliers=False)
            for i, values in enumerate(groups):
                axes[row, col].scatter(np.full(len(values), i+1), values, s=8, alpha=.35)
            axes[row, col].set(title=title, ylabel='Per-case error (%)')
    fig.suptitle('Frozen validation concentration descriptors: no temporal / flux residual test')
    fig.savefig(OUT / 'comparison.png', dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
