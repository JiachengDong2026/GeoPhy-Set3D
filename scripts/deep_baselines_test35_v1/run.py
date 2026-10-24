"""Frozen two-model exploratory test: registration -> inference/seal -> GT64 -> case statistics.

Never launches/resumes training. Historical Plain predictions are hash-checked and reused.
Each phase runs in a separate process. No GT128 or Pool B is authorized.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT / 'src'), str(PROJECT / 'scripts/deep_baselines_v1')]
import baseline_context as b
c = b.c
import numpy as np
import torch

ROOT = Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_test35_v1')
TRAIN = b.ROOT / 'training_registered_v2'
ARMS = ('deepsets', 'gnot')
SEED = 20261011
UUID = 'GPU-37424437-e208-62d1-e617-53c421c94a49'
METRICS = ('volume.relative_l1', 'volume.relative_l2', 'volume.mae', 'volume.rmse',
           'morphology.absolute_peak_ratio_error', 'volume.peak_absolute_error',
           'volume.peak_location_error_mm', 'volume.centroid_error_mm',
           'morphology.centroid_distance_mm', 'morphology.dice_0.5',
           'measurement.observed.relative_l1', 'measurement.heldout.relative_l1')


def method(arm):
    return f'ray_{arm}_v1_seed{SEED}'


def frozen(name, value):
    c.freeze(ROOT / name, value)


def load(name):
    return c.verified(ROOT / name)


def key(row):
    return row['method'], row['condition'], row['draw'], row['case_id']


def verify():
    p = load('protocol.json')
    c.verify_hashes(p['sources'])
    c.verify_hashes(p['evidence'])
    c.verify_hashes(p['data_identity'])
    for m in p['models']:
        assert c.sha(m['checkpoint']) == m['checkpoint_sha256']
    return p


def register():
    assert not ROOT.exists(), 'Use a new audited directory; never overwrite a historical evaluation'
    ROOT.mkdir()
    models = []
    evidence = {}
    for arm in ARMS:
        run = TRAIN / f'{arm}_seed{SEED}'
        s = c.read(run / 'summary.json')
        p = c.verified(run / 'protocol.json')
        c.verify_hashes(p['sources'])
        logs = c.rows(run / 'metrics.jsonl')
        assert s['completed'] and s['epochs'] == 300 and s['updates'] == 24900
        assert len(logs) == 300 and [r['epoch'] for r in logs] == list(range(1, 301))
        selected = min(logs, key=lambda r: r['val']['measurement']['heldout']['relative_l1'])
        assert selected['epoch'] == s['best_epoch']
        assert selected['val']['measurement']['heldout']['relative_l1'] == s['best_val_H']
        checkpoint = run / 'checkpoints/best.pt'
        assert c.sha(checkpoint) == s['best_checkpoint_sha256']
        models.append(dict(method=method(arm), arm=arm, seed=SEED, run=str(run),
                           checkpoint=str(checkpoint), checkpoint_sha256=c.sha(checkpoint),
                           parameters=s['parameters'], best_epoch=s['best_epoch'],
                           best_val_H=s['best_val_H'], train_epochs=300, train_updates=24900,
                           train_elapsed_seconds=s['elapsed_seconds'], config=p['config'],
                           training_protocol_sha256=c.sha(run / 'protocol.json')))
        for path in (run / 'protocol.json', run / 'summary.json', run / 'metrics.jsonl'):
            evidence[str(path)] = c.sha(path)
    for name in ('cases.json', 'layouts.json', 'input_manifest.json', 'prediction_manifest.json',
                 'metrics.jsonl', 'GT64_access.json', 'model_manifest.json', 'protocol.json',
                 'all_predictions_sealed.json'):
        path = c.BASE / name
        evidence[str(path)] = c.sha(path)
    cases, layouts = c.verified(c.BASE / 'cases.json'), c.verified(c.BASE / 'layouts.json')
    inputs = c.verified(c.BASE / 'input_manifest.json')
    assert len(cases) == len(set(cases)) == 35 and len(layouts) == 17 and len(inputs) == 595
    main = c.verified(c.BASE / 'model_manifest.json')['models'][0]
    assert main['key'] == c.MAIN and c.sha(main['checkpoint']) == main['checkpoint_sha256']
    reused = [dict(r, reused=True) for r in c.verified(c.BASE / 'prediction_manifest.json') if r['method'] == c.MAIN]
    assert len(reused) == 595 and len({key(r) for r in reused}) == 595
    for r in reused:
        assert r['checkpoint_sha256'] == main['checkpoint_sha256']
        assert c.sha(r['prediction']) == r['prediction_sha256']
    main_record = dict(method=c.MAIN, arm='plain', seed=SEED, checkpoint=main['checkpoint'],
                       checkpoint_sha256=main['checkpoint_sha256'], parameters=342209,
                       best_epoch=298, train_epochs=300, train_updates=24900, reused=True)
    sources = b.source_identity()
    sources.update(c.sources())
    sources[str(Path(__file__).resolve())] = c.sha(__file__)
    identities = {str(c.DATA / name): c.sha(c.DATA / name) for name in
                  ('source_index.jsonl', 'train_statistics.json', 'geometry/train_geometry.npz', 'splits/test_cases.txt')}
    frozen('protocol.json', dict(at=c.now(), scope='EXPLORATORY_HISTORICAL_TEST35',
        authorization='User 2026-10-24: pause training, test the two completed seed20261011 models, report and sync repository; supersedes waiting for all three algorithms',
        independent_confirmation=False, no_architecture_or_checkpoint_selection_from_test=True,
        cases=cases, layouts=layouts, input_manifest=inputs, reused_predictions=reused,
        models=models + [main_record], sources=sources, evidence=evidence, data_identity=identities,
        output='single field from each baseline scored as final C1 comparison; no artificial C0/PRF stage',
        statistics=dict(seed=20261019, resamples=20000, unit='case after within-layout draw mean',
                        tie=1e-8, primary='Val-A-V4 relative_l1', multiplicity='unadjusted exploratory CIs'),
        GT64_only_after_global_prediction_seal=True, GT128_allowed=False, Pool_B_allowed=False,
        training_resume_allowed=False, checkpoint_selection='strict minimum validation H4 over all300epochs'))
    print('Registered two preselected completed models and595 reused Plain predictions', flush=True)


class PhaseGuard(c.AccessGuard):
    def __init__(self, phase, p):
        super().__init__(splits=['test'] if phase == 'inference' else [],
                         gt_splits=['test'] if phase == 'score' else [],
                         checkpoints=[m['checkpoint'] for m in p['models'][:2]] if phase == 'inference' else [],
                         output=ROOT)
        self.phase = phase
        for r in p['reused_predictions']:
            self.allowed[str(Path(r['prediction']).resolve())] = 'audited_prediction'
    def audit(self, event, args):
        if self.active and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if self.phase == 'score' and (path.suffix in ('.pt', '.pth', '.ckpt') or
                                        path == c.DATA / 'geometry/train_geometry.npz'):
                self.denied.append(dict(path=str(path), reason='checkpoint/geometry denied in saved-output GT phase'))
                raise PermissionError(self.denied[-1]['reason'])
        return super().audit(event, args)


def check_array(x):
    assert x.shape == (64, 64, 64) and x.dtype == np.float32
    assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1


def construct(m, device):
    model = b.build_model(m['arm'], device, SEED)
    ck = torch.load(m['checkpoint'], map_location='cpu', weights_only=False)
    assert ck['volumetric_gt_used'] is False and ck['config'] == m['config']
    assert ck['epoch'] == m['best_epoch'] and ck['arm'] == m['arm']
    assert ck['protocol_sha256'] == m['training_protocol_sha256']
    model.load_state_dict(ck['model'], strict=True)
    return model.eval().requires_grad_(False)


def cpu():
    p = verify()
    torch.set_num_threads(2)
    result = b.regression()
    checks = []
    with PhaseGuard('inference', p) as guard, torch.inference_mode():
        ds = c.dataset('test', p['layouts'][0])
        batch = c.batch_for(ds, 0)
        identity = p['input_manifest'][0]
        assert c.input_hash(batch) == identity['input_sha256']
        for m in p['models'][:2]:
            model = construct(m, 'cpu')
            x = model(batch)
            assert tuple(x.shape) == (1, 1, 64, 64, 64)
            check_array(x[0, 0].numpy())
            assert sum(v.numel() for v in model.parameters()) == m['parameters']
            checks.append(dict(method=m['method'], parameters=m['parameters'], output_min=x.min().item(),
                               output_max=x.max().item(), output_mean=x.mean().item()))
    # Analytical perfect/zero fields exercise the official scorer without reading GT.
    g = np.full((64, 64, 64), .001)
    assert c.official.volume_metrics(g, g)['relative_l1'] == 0
    assert c.official.volume_metrics(g * 0, g)['relative_l1'] == 1
    assert c.official.volume_metrics(g * 0, g)['centroid_error_mm'] is None
    assert not guard.denied
    frozen('cpu_checks.json', dict(at=c.now(), passed=True, model_checks=checks,
                                   frozen_main_bitwise_regression=result, audit=guard.summary()))
    print('CPU checks and frozen main bitwise regression passed', flush=True)


def diagnostics(field, geometry, projection, setting, case):
    result = {}
    for label, ids in [('observed', setting['input_ids'][case]), ('heldout', setting['heldout_ids'][case])]:
        g = c.geometry_tensor({k: v[ids] for k, v in geometry.items()
                               if k not in ('horizontal_angle_deg', 'vertical_angle_deg')}, 'cuda:0')
        y = torch.as_tensor(np.array(projection[ids]), device='cuda:0')
        pred = c.CachedProjector(g)(torch.as_tensor(field, device='cuda:0'))
        result[label] = c.relative_projection_metrics(pred, y, g['valid_ray'])
    return result


def infer():
    p = verify()
    assert load('cpu_checks.json')['passed']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == UUID
    c.backend()
    torch.set_num_threads(2)
    out = ROOT / 'predictions'
    out.mkdir()  # No silent partial replay; failure requires inspected separate recovery.
    records = []
    inp = {(r['condition'], r['draw'], r['case_id']): r for r in p['input_manifest']}
    source = {r['case_id']: r for r in c.rows(c.DATA / 'source_index.jsonl')}
    with PhaseGuard('inference', p) as guard, torch.inference_mode():
        with np.load(c.DATA / 'geometry/train_geometry.npz', allow_pickle=False) as z:
            geometry = {k: z[k] for k in z.files}
        for m in p['models'][:2]:
            model = construct(m, 'cuda:0')
            state = c.canonical_worker.probe.state_hash(model.state_dict())
            folder = out / m['arm']
            folder.mkdir()
            # One excluded warmup. Every saved row is a single synchronized forward.
            warm = c.batch_for(c.dataset('test', p['layouts'][0]), 0, 'cuda:0')
            check_array(model(warm)[0, 0].cpu().numpy())
            del warm
            for setting in p['layouts']:
                ds = c.dataset('test', setting)
                for i, case in enumerate(p['cases']):
                    identity = inp[setting['condition'], setting['draw'], case]
                    batch = c.batch_for(ds, i, 'cuda:0')
                    assert ds.records[i]['case_id'] == case
                    assert c.input_hash(batch) == identity['input_sha256']
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    began = time.perf_counter()
                    x = model(batch)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - began
                    peak = torch.cuda.max_memory_allocated()
                    assert tuple(x.shape) == (1, 1, 64, 64, 64)
                    field = x[0, 0].cpu().numpy().copy()
                    check_array(field)
                    prediction = folder / f"{setting['condition']}_d{setting['draw']}_{i:03d}.npz"
                    with prediction.open('xb') as stream:
                        np.savez_compressed(stream, C1=field)
                    projection = np.load(c.DATA / source[case]['train_projection_path'], allow_pickle=False)
                    measurement = diagnostics(field, geometry, projection, setting, case)
                    record = dict(identity, method=m['method'], checkpoint_sha256=m['checkpoint_sha256'],
                        prediction=str(prediction), prediction_sha256=c.sha(prediction),
                        seconds=seconds, peak_allocated_bytes=peak, gpu_uuid=UUID, reused=False,
                        forward_calls=1, measurement=measurement, output_min=float(field.min()),
                        output_max=float(field.max()), output_mean=float(field.mean()), output_std=float(field.std()))
                    records.append(record)
                    c.append(ROOT / 'inference_manifest.jsonl', record)
                    del x, batch, projection, field
                print(m['arm'], setting['condition'], setting['draw'], len(records), '/1190', flush=True)
            assert state == c.canonical_worker.probe.state_hash(model.state_dict())
            del model
            torch.cuda.empty_cache()
        assert len(records) == 1190 and not guard.denied
    allrows = records + p['reused_predictions']
    assert len({key(r) for r in allrows}) == 1785
    for r in allrows:
        assert c.sha(r['prediction']) == r['prediction_sha256']
    frozen('prediction_manifest.json', allrows)
    frozen('all_predictions_sealed.json', dict(at=c.now(), count=1785, new_predictions=1190,
        reused_plain_predictions=595, all_predictions_saved_before_GT64=True, model_state_unchanged=True,
        protocol_sha256=c.sha(ROOT / 'protocol.json'), prediction_manifest_sha256=c.sha(ROOT / 'prediction_manifest.json'),
        GT64_read=False, GT128_read=False, Pool_B_read=False, audit=guard.summary()))
    print('ALL1785FIELDS SEALED before GT64', flush=True)


def score():
    p = verify()
    seal = load('all_predictions_sealed.json')
    assert seal['all_predictions_saved_before_GT64']
    assert c.sha(ROOT / 'protocol.json') == seal['protocol_sha256']
    assert c.sha(ROOT / 'prediction_manifest.json') == seal['prediction_manifest_sha256']
    manifest = load('prediction_manifest.json')
    old = {(r['condition'], r['draw'], r['case_id']): r for r in c.rows(c.BASE / 'metrics.jsonl')
           if r['method'] == c.MAIN and r['stage'] == 'C1'}
    known = c.verified(c.BASE / 'GT64_access.json')['GT64_sha256']
    source = {r['case_id']: r for r in c.rows(c.DATA / 'source_index.jsonl')}
    records, hashes, maximum, compared, first = [], {}, 0., 0, None
    with PhaseGuard('score', p) as guard:
        for r in manifest:
            assert c.sha(r['prediction']) == r['prediction_sha256']
        for case in p['cases']:
            path = c.DATA / source[case]['gt_volume_64_path']
            assert source[case]['split'] == 'test'
            if first is None:
                first = c.now()
            hashes[str(path)] = c.sha(path)
            assert hashes[str(path)] == known[str(path)]
            gt = np.load(path, allow_pickle=False).astype(np.float64).squeeze()
            for r in [r for r in manifest if r['case_id'] == case]:
                with np.load(r['prediction'], allow_pickle=False) as z:
                    prediction = z['C1'].astype(np.float64)
                volume = c.official.volume_metrics(prediction, gt)
                morphology = c.official.safe_shape(prediction, gt)
                morphology['absolute_peak_ratio_error'] = abs(morphology['peak_ratio'] - 1)
                prior = old[r['condition'], r['draw'], case]
                measurement = prior['measurement'] if r['method'] == c.MAIN else r['measurement']
                values = c.flatten(volume, morphology, measurement)
                if r['method'] == c.MAIN:
                    expected = c.flatten(prior['volume'], prior['morphology'], measurement)
                    for k, value in expected.items():
                        if value is None:
                            assert values[k] is None
                        else:
                            diff = abs(value - values[k])
                            maximum = max(maximum, diff)
                            assert diff < 1e-11, (k, diff)
                    compared += 1
                records.append(dict(method=r['method'], stage='C1', condition=r['condition'], draw=r['draw'],
                    case_id=case, **values, seconds=r['seconds'], peak_allocated_bytes=r['peak_allocated_bytes'],
                    prediction=r['prediction'], prediction_sha256=r['prediction_sha256'], reused=r['reused']))
            print('scored', case, len(records), '/1785', flush=True)
        assert len(records) == 1785 and compared == 595 and not guard.denied
    c.jsonl(ROOT / 'metrics.jsonl', records)
    c.csv_write(ROOT / 'per_case_metrics.csv', records)
    frozen('GT64_access.json', dict(at=c.now(), first_GT64_at=first, seal_at=seal['at'], unique_GT64_files=len(hashes),
        GT64_sha256=hashes, fields=len(records), reused_plain_fields_recomputed=compared,
        maximum_old_metric_difference=maximum, metrics_sha256=c.sha(ROOT / 'metrics.jsonl'),
        GT128_read=False, Pool_B_read=False, audit=guard.summary()))


def summarize(values, indices):
    a = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    valid = np.isfinite(a)
    finite = a[valid]
    # A fixed35-case CI is not identifiable when any cases are undefined.
    bs = a[indices].mean(1) if valid.all() else None
    return dict(valid_cases=int(valid.sum()), missing_cases=int((~valid).sum()),
                mean=float(finite.mean()) if len(finite) else None,
                median=float(np.median(finite)) if len(finite) else None,
                p90=float(np.quantile(finite, .9)) if len(finite) else None,
                p95=float(np.quantile(finite, .95)) if len(finite) else None,
                maximum=float(finite.max()) if len(finite) else None,
                ci95=np.quantile(bs, [.025, .975]).tolist() if bs is not None else None)


def analyze():
    p = verify()
    gt = load('GT64_access.json')
    assert gt['metrics_sha256'] == c.sha(ROOT / 'metrics.jsonl')
    raw = c.rows(ROOT / 'metrics.jsonl')
    lookup = {key(r): r for r in raw}
    assert len(lookup) == 1785
    cases = p['cases']
    idx = np.random.default_rng(20261019).integers(0, 35, (20000, 35))
    case_rows, summary, paired, vectors = [], [], [], {}
    for condition in c.CONDITIONS:
        draws = [s['draw'] for s in p['layouts'] if s['condition'] == condition]
        for m in p['models']:
            name = m['method']
            for case in cases:
                rows = [lookup[name, condition, d, case] for d in draws]
                vals = {k: None if any(r[k] is None for r in rows) else float(np.mean([r[k] for r in rows])) for k in METRICS}
                case_rows.append(dict(method=name, condition=condition, case_id=case, draws=len(draws), **vals))
            for metric in METRICS:
                vals = [r[metric] for r in case_rows if r['method'] == name and r['condition'] == condition]
                vectors[name, condition, metric] = vals
                stats = summarize(vals, idx)
                worst = None
                if stats['valid_cases']:
                    valid = [(case, v) for case, v in zip(cases, vals) if v is not None]
                    worst = (min if metric == 'morphology.dice_0.5' else max)(valid, key=lambda x: x[1])[0]
                summary.append(dict(method=name, condition=condition, metric=metric, worst_case_id=worst, **stats))
        for arm in ARMS:
            for metric in METRICS:
                a, ref = vectors[method(arm), condition, metric], vectors[c.MAIN, condition, metric]
                delta = [None if v is None or w is None else v - w for v, w in zip(a, ref)]
                stats = summarize(delta, idx)
                paired.append(dict(method=method(arm), reference=c.MAIN, seed=SEED, condition=condition, metric=metric,
                    delta_definition='baseline minus frozen Plain; positive worse for errors, negative worse for Dice',
                    baseline_mean=summarize(a, idx)['mean'], plain_mean=summarize(ref, idx)['mean'],
                    increased=sum(v is not None and v > 1e-8 for v in delta),
                    decreased=sum(v is not None and v < -1e-8 for v in delta),
                    unchanged=sum(v is not None and abs(v) <= 1e-8 for v in delta), **stats))
    c.csv_write(ROOT / 'case_aggregated_metrics.csv', case_rows)
    c.csv_write(ROOT / 'summary.csv', [{**r, 'ci95': json.dumps(r['ci95'])} for r in summary])
    c.csv_write(ROOT / 'paired_statistics.csv', [{**r, 'ci95': json.dumps(r['ci95'])} for r in paired])
    frozen('statistics.json', dict(at=c.now(), exploratory=True, seed=SEED, summary=summary, paired=paired,
        case_count=35, draw_aggregation='arithmetic mean within case and condition; no draw-as-independent samples',
        bootstrap=p['statistics'], missing_rule='available-case point estimates; CI suppressed unless all35 cases defined'))
    # Independent algorithm: raw Python draw differences, count-weight resamples,
    # and manually sorted/interpolated percentiles; no reuse of summary vectors.
    weights = np.array([np.bincount(i, minlength=35) for i in idx], dtype=float)
    def quantile(x, q):
        x = sorted(x)
        pos = (len(x) - 1) * q
        lo = int(pos)
        return x[lo] + (x[min(lo + 1, len(x) - 1)] - x[lo]) * (pos - lo)
    maximum, checked = 0., 0
    for row in paired:
        if row['valid_cases'] != 35:
            continue
        draws = [s['draw'] for s in p['layouts'] if s['condition'] == row['condition']]
        v = []
        for case in cases:
            differences = [lookup[row['method'], row['condition'], d, case][row['metric']] -
                           lookup[c.MAIN, row['condition'], d, case][row['metric']] for d in draws]
            v.append(sum(differences) / len(differences))
        bs = weights @ np.array(v) / 35
        comparisons = [(sum(v) / 35, row['mean']), (quantile(v, .5), row['median']),
                       (quantile(bs, .025), row['ci95'][0]), (quantile(bs, .975), row['ci95'][1])]
        for actual, expected in comparisons:
            maximum = max(maximum, abs(actual - expected))
            assert abs(actual - expected) < 1e-10
        assert sum(x > 1e-8 for x in v) == row['increased']
        assert sum(x < -1e-8 for x in v) == row['decreased']
        checked += 1
    frozen('independent_statistics_verification.json', dict(at=c.now(), passed=True,
        paired_vectors_checked=checked, maximum_difference=maximum,
        algorithm='Python raw draw differences + multinomial count weights + independently sorted interpolated quantiles'))
    cost = []
    predictions = load('prediction_manifest.json')
    for m in p['models']:
        rr = [r for r in predictions if r['method'] == m['method']]
        seconds = [r['seconds'] for r in rr]
        cost.append(dict(method=m['method'], parameters=m['parameters'],
            delta_parameters=m['parameters'] - 342209, checkpoint_bytes=Path(m['checkpoint']).stat().st_size,
            train_epochs=300, committed_updates=24900, train_elapsed_seconds=m.get('train_elapsed_seconds'),
            mean_forward_ms=float(np.mean(seconds) * 1000), median_forward_ms=float(np.median(seconds) * 1000),
            p95_forward_ms=float(np.quantile(seconds, .95) * 1000),
            peak_allocated_MiB=max(r['peak_allocated_bytes'] for r in rr) / 1024**2,
            samples=len(rr), timing_scope='single synchronized forward per layout/case; Plain historical GPU timing reused; not controlled benchmark'))
    c.csv_write(ROOT / 'cost.csv', cost)
    verify()
    pause = c.read(b.ROOT / 'queue_status.json')
    assert pause['state'] == 'paused_by_user'
    frozen('acceptance.json', dict(at=c.now(), completed=True, training_remains_paused=True,
        evaluated_models=list(map(method, ARMS)), models_completed300epochs24900updates=True,
        predictions=1785, newly_generated=1190, historical_plain_reused=595,
        input_case_layout_draw_alignment=True, shape_finite_range_checks_passed=True,
        checkpoint_hashes_unchanged=True, source_hashes_unchanged=True,
        frozen_main_CPU_bitwise_regression=True, GT64_files=35, GT128_read=False, Pool_B_read=False,
        independent_statistics_passed=True, exploratory=True, independent_confirmation=False,
        scientific_outcome='Both baselines collapsed to nearly zero outputs under frozen training protocol; no test-driven repair/search',
        statistics_sha256=c.sha(ROOT / 'statistics.json'), pause=pause))
    print('Statistics independently verified, acceptance complete', flush=True)


def launch():
    # Reuse established global UUID locks,15-second double snapshots and conflict
    # handling, while all task records live in this new evaluation directory.
    import controller
    original = c.scheduler
    def restricted():
        s = original()
        devices = s.devices
        s.devices = lambda snapshot: [d for d in devices(snapshot) if d['uuid'] == UUID]
        return s
    c.scheduler = restricted
    controller.ROOT = ROOT
    controller.run('seed11_test35_inference', [sys.executable, str(Path(__file__).resolve()), 'infer'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register', 'cpu', 'launch', 'infer', 'score', 'analyze'])
    args = parser.parse_args()
    globals()[args.phase]()
