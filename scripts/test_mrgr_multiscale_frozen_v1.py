#!/usr/bin/env python3
"""Register matched layouts, seal predictions, then independently score validation GT."""
import argparse
import json
from pathlib import Path
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import train_mrgr_multiscale_formal_v1 as formal
import probe_mrgr_multiscale_v1 as probe
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.utils import move_batch
from geophy_set3d.training.projection_losses import project_measurements
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.evaluation.research import volume_metrics
from report_historical_reconstruction_v1 import metrics as morphology_metrics

RUN = formal.DEFAULT_OUT
OUT = RUN / 'evaluation/frozen_model_test_v1'
DATA = Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
HISTORY = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v1/selfsup_oari_mrgr_v1')
AB_RUNS = {'A': ROOT / 'outputs/research/scheme_a_budget_v4_wide_seed20261011',
           'B': ROOT / 'outputs/research/scheme_b_perf_v2_formal_seed20261011'}
AB_EXTRA = ROOT / 'outputs/research/ab_remaining_draws_v1'


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def write(path, value): formal.write(path, value)
def sha(path): return probe.digest(Path(path))
def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def jsonl(path, values): Path(path).write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in values))


def verify_registration():
    r = read(OUT / 'registration.json')
    assert sha(OUT / 'registration.json') == (OUT / 'registration.sha256').read_text().strip()
    assert sha(Path(__file__)) == r['evaluation_source_sha256']
    formal.verify(RUN)
    assert sha(RUN / 'checkpoints/best.pt') == r['checkpoint_sha256']
    for path, digest in r['evidence_sha256'].items():
        assert sha(path) == digest, path
    return r


def register():
    p = formal.verify(RUN)
    summary = read(RUN / 'summary.json')
    assert summary['completed'] and summary['epochs'] == 300 and summary['updates'] == 24900
    assert summary['sources_preserved'] and not summary['final_test_evaluated']
    assert all(summary['gt_audit'][k] == 0 for k in ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts'))
    checkpoint = RUN / 'checkpoints/best.pt'
    assert sha(checkpoint) == summary['best_checkpoint_sha256']
    source = [x for x in rows(DATA / 'source_index.jsonl') if x['split'] == 'val']
    cases = [x['case_id'] for x in source]
    assert len(cases) == len(set(cases)) == 36
    assert set(cases) == set(p['validation_case_ids'])
    extra = read(AB_EXTRA / 'registration.json')
    extra_metrics = rows(AB_EXTRA / 'per_case_metrics.jsonl')
    settings = []
    evidence = {str(path): sha(path) for path in [RUN / 'protocol.json', RUN / 'summary.json',
                DATA / 'source_index.jsonl', DATA / 'train_statistics.json',
                DATA / 'geometry/train_geometry.npz', AB_EXTRA / 'registration.json',
                AB_EXTRA / 'per_case_metrics.jsonl']}
    for condition, views in [('Val-A-V4', 4), ('spread-V4', 4), ('V3', 3), ('V6', 6), ('clustered-V4', 4)]:
        draws = [0] if condition in ('Val-A-V4', 'spread-V4') else range(5)
        for draw in draws:
            ref = HISTORY / condition
            if condition not in ('Val-A-V4', 'spread-V4'): ref = ref / f'draw{draw}'
            hpath = ref / 'C1/per_case_metrics.jsonl'
            historical = rows(hpath)
            assert len(historical) == 36 and {x['case_id'] for x in historical} == set(cases)
            ids = {x['case_id']: x['selected_station_ids'] for x in historical}
            assert all(len(v) == len(set(v)) == views and 0 <= min(v) <= max(v) < 64 for v in ids.values())
            refs = {'History': str(hpath)}
            c0ref = ref / 'C0/per_case_metrics.jsonl'
            if c0ref.exists(): refs['History-C0'] = str(c0ref)
            if draw == 0:
                ab_rows = {}
                for model, run in AB_RUNS.items():
                    pointer = run / f'evaluation/frozen_model_test/{condition}_prediction.json'
                    directory = Path(read(pointer)['directory'])
                    refs[model] = str(directory / 'per_case_metrics.jsonl')
                    ab_rows[model] = rows(refs[model])
                    evidence[str(pointer)] = sha(pointer)
                holds = {x['case_id']: x['heldout_station_ids'] for x in ab_rows['A']}
                for model in ab_rows:
                    assert len(ab_rows[model]) == 36
                    for x in ab_rows[model]:
                        assert x['selected_station_ids'] == ids[x['case_id']]
                        assert x['heldout_station_ids'] == holds[x['case_id']]
            else:
                s = next(s for s in extra['settings'] if s['condition'] == condition and s['draw'] == draw)
                assert s['input_ids'] == ids
                holds = s['heldout_ids']
                for model in AB_RUNS:
                    selected = [x for x in extra_metrics if x['condition'] == condition and x['draw'] == draw and x['model'] == model]
                    assert len(selected) == 36 and {x['case_id'] for x in selected} == set(cases)
                    for x in selected:
                        assert x['input_ids'] == ids[x['case_id']] and x['heldout_ids'] == holds[x['case_id']]
            for case in cases:
                assert len(holds[case]) == len(set(holds[case])) == 4
                assert not set(holds[case]) & set(ids[case])
            evidence.update({path: sha(path) for path in refs.values()})
            settings.append(dict(condition=condition, draw=draw, views=views,
                                 input_ids=ids, heldout_ids=holds, references=refs))
    assert len(settings) == 17
    evidence.update({str(DATA / x['train_projection_path']): sha(DATA / x['train_projection_path']) for x in source})
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'predictions').mkdir()
    registration = dict(registered_at=now(), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        best_epoch=summary['best_epoch'], completed_epochs=300, parameters=344678,
        evaluation_source_sha256=sha(Path(__file__)), evidence_sha256=evidence,
        settings=settings, cases=cases, case_count=36, layouts=17, reconstructions=612, saved_fields=1224,
        split='val', geometry_pool='train', precision='FP32, TF32 off, cudnn benchmark off',
        primary='ordinary fixed V4 C1 3D relative L1, paired vs historical C1',
        secondary='V3/V6/clustered each 5 draws averaged within case; spread V4; C0 and morphology exploratory',
        statistics='36 paired cases; 10000 percentile case bootstrap, seed20261016; no seed uncertainty or multiplicity correction',
        gt_for_inference=False, gt_for_selection=False, final_test_evaluated=False,
        selection='unchanged lowest formal validation H; no GT-based checkpoint selection',
        morphology='same equal-weight endpoint moments and GT-relative thresholds as prior image report; physical volume metrics use trapezoidal weights',
        config=p['config'])
    write(OUT / 'registration.json', registration)
    (OUT / 'registration.sha256').write_text(sha(OUT / 'registration.json') + '\n')
    print('Registered 17 layouts x 36 validation cases; C0/C1 saved before independent scoring.', flush=True)


class Replay:
    def __init__(self, setting): self.setting = setting
    def sample(self, station_count, *, epoch, sample_index, case_id):
        return np.array(self.setting['input_ids'][case_id], dtype=np.int64), np.array(self.setting['heldout_ids'][case_id], dtype=np.int64)


def predict(device):
    r = verify_registration()
    assert not (OUT / 'prediction_manifest.jsonl').exists(), 'Refuse to replace prediction manifest'
    assert not list((OUT / 'predictions').iterdir()), 'Refuse to overwrite partial predictions'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    records = []
    with GTReadGuard(DATA, RUN / 'checkpoints') as guard:
        ckpt = torch.load(r['checkpoint'], map_location='cpu', weights_only=False)
        assert ckpt['epoch'] == r['best_epoch'] and not ckpt['volumetric_gt_used']
        assert ckpt['protocol_sha256'] == sha(RUN / 'protocol.json')
        model = probe.model_for(ckpt['config'], 'multiscale', device)
        model.load_state_dict(ckpt['model'], strict=True)
        model.eval()
        assert sum(p.numel() for p in model.parameters()) == r['parameters']
        for setting in r['settings']:
            ds = MeasurementOnlyPoolDataset(DATA, split='val', geometry_pool='train', evaluation_only=True,
                min_input_views=setting['views'], max_input_views=setting['views'], hold_views=0,
                base_seed=20261011, normalization=r['config']['normalization']['method'],
                statistics_file=r['config']['normalization']['statistics_file'])
            ds.sampler = Replay(setting)
            assert [x['case_id'] for x in ds.records] == r['cases']
            with torch.no_grad():
                for i in range(len(ds)):
                    sample = ds[i]
                    assert sample['input_station_indices'].tolist() == setting['input_ids'][sample['case_id']]
                    assert sample['hold_station_indices'].tolist() == setting['heldout_ids'][sample['case_id']]
                    batch = move_batch(projection_pool_collate([sample]), device)
                    inputs = {k: v for k, v in batch.items() if not k.startswith('hold_')}
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    stages = model(inputs, return_stages=True)
                    torch.cuda.synchronize(device)
                    reconstruct_s = time.perf_counter() - t0
                    measurement = {}
                    for prefix, label in [('input', 'observed'), ('hold', 'heldout')]:
                        prediction, reference, valid = project_measurements(stages['C1'], batch, prefix)[0]
                        measurement[label] = relative_projection_metrics(prediction, reference, valid)
                    torch.cuda.synchronize(device)
                    project_s = time.perf_counter() - t0 - reconstruct_s
                    fields = {k: stages[k][0, 0].cpu().numpy() for k in ('C0', 'C1')}
                    assert all(v.shape == (64, 64, 64) and np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1 for v in fields.values())
                    path = OUT / 'predictions' / f"{setting['condition']}_draw{setting['draw']}_{i:03d}.npz"
                    np.savez_compressed(path, **fields)
                    records.append(dict(condition=setting['condition'], draw=setting['draw'], case_id=sample['case_id'],
                        input_ids=sample['input_station_indices'].tolist(), heldout_ids=sample['hold_station_indices'].tolist(),
                        prediction=str(path.relative_to(OUT)), sha256=sha(path), measurement=measurement,
                        reconstruct_s=reconstruct_s, external_project_s=project_s))
                    del stages, fields, batch, inputs
            print(f"{setting['condition']} draw{setting['draw']}: 36/36; {len(records)}/612; elapsed={time.perf_counter()-started:.1f}s", flush=True)
        audit = guard.summary()
        assert all(audit[k] == 0 for k in ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts'))
    assert len(records) == 612
    jsonl(OUT / 'prediction_manifest.jsonl', records)
    for x in records: assert sha(OUT / x['prediction']) == x['sha256']
    write(OUT / 'predictions_complete.json', dict(completed_at=now(), reconstructions=612, fields=1224,
        manifest_sha256=sha(OUT / 'prediction_manifest.jsonl'), registration_sha256=sha(OUT / 'registration.json'),
        checkpoint_sha256=r['checkpoint_sha256'], all_predictions_saved_before_gt=True,
        elapsed_seconds=time.perf_counter()-started, audit=audit, final_test_evaluated=False,
        device=device, cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
    print('All predictions sealed; separate score process may now read validation GT.', flush=True)


def paired(current, reference):
    a, b = np.asarray(current), np.asarray(reference)
    d = a-b
    rng = np.random.default_rng(20261016)
    ci = np.quantile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), [.025, .975])
    return dict(mean_current=float(a.mean()), mean_reference=float(b.mean()), mean_delta=float(d.mean()),
        relative_error_reduction=float(1-a.mean()/b.mean()), ci95=ci.tolist(),
        wins=int((d < -1e-8).sum()), ties=int((abs(d) <= 1e-8).sum()), losses=int((d > 1e-8).sum()))


def score():
    r = verify_registration()
    boundary = read(OUT / 'predictions_complete.json')
    assert boundary['all_predictions_saved_before_gt'] and boundary['reconstructions'] == 612
    assert boundary['registration_sha256'] == sha(OUT / 'registration.json')
    assert boundary['manifest_sha256'] == sha(OUT / 'prediction_manifest.jsonl')
    manifest = rows(OUT / 'prediction_manifest.jsonl')
    assert len(manifest) == 612
    settings = {(s['condition'], s['draw']): s for s in r['settings']}
    for x in manifest:
        assert sha(OUT / x['prediction']) == x['sha256']
        s = settings[x['condition'], x['draw']]
        assert x['case_id'] in r['cases'] and x['input_ids'] == s['input_ids'][x['case_id']]
    assert len({(x['condition'], x['draw'], x['case_id']) for x in manifest}) == 612
    # Full checkpoint/reference/prediction boundary verified before the first label read.
    source = {x['case_id']: x for x in rows(DATA / 'source_index.jsonl') if x['split'] == 'val'}
    gt = {}
    scored = []
    gt_paths = []
    for x in manifest:
        case = x['case_id']
        if case not in gt:
            path = DATA / source[case]['gt_volume_64_path']
            gt[case] = np.load(path, allow_pickle=False).squeeze().astype(np.float64)
            gt_paths.append(str(path))
        with np.load(OUT / x['prediction'], allow_pickle=False) as fields:
            for stage in ('C0', 'C1'):
                p = fields[stage].astype(np.float64)
                scored.append(dict(case_id=case, condition=x['condition'], draw=x['draw'], stage=stage,
                    volume=volume_metrics(p, gt[case]), morphology=morphology_metrics(p, gt[case])))
        if len(scored) % 144 == 0: print(f'Scored {len(scored)}/1224 fields', flush=True)
    jsonl(OUT / 'per_case_metrics.jsonl', scored)
    current = {(x['condition'], x['draw'], x['case_id'], x['stage']): x for x in scored}
    reference = {}
    for s in r['settings']:
        for model, path in s['references'].items():
            for x in rows(path):
                assert x['selected_station_ids'] == s['input_ids'][x['case_id']]
                reference[model, s['condition'], s['draw'], x['case_id']] = x['relative_l1']
    for x in rows(AB_EXTRA / 'per_case_metrics.jsonl'):
        reference[x['model'], x['condition'], x['draw'], x['case_id']] = x['volume']['relative_l1']
    panels = []
    for condition in ['Val-A-V4', 'V3', 'V6', 'spread-V4', 'clustered-V4']:
        selections = [[0]] if condition in ('Val-A-V4', 'spread-V4') else [[0], list(range(5))]
        for draws in selections:
            values = [np.mean([current[condition, d, c, 'C1']['volume']['relative_l1'] for d in draws]) for c in r['cases']]
            row = dict(condition=condition, draws=draws, cases=36, mean=float(np.mean(values)),
                       median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                       max=float(np.max(values)), comparisons={})
            for model in ('History', 'A', 'B'):
                old = [np.mean([reference[model, condition, d, c] for d in draws]) for c in r['cases']]
                row['comparisons'][model] = paired(values, old)
            panels.append(row)
    shape_summary = []
    for condition in ('Val-A-V4', 'clustered-V4'):
        for stage in ('C0', 'C1'):
            rr = [x for x in scored if x['condition'] == condition and x['draw'] == 0 and x['stage'] == stage]
            keys = rr[0]['morphology'].keys()
            means = {k: float(np.mean([x['morphology'][k] for x in rr])) for k in keys
                     if np.isscalar(rr[0]['morphology'][k])}
            means['absolute_peak_ratio_error'] = float(np.mean([abs(x['morphology']['peak_ratio']-1) for x in rr]))
            shape_summary.append(dict(condition=condition, stage=stage, cases=36, mean=means))
    measurement = []
    for s in r['settings']:
        rr = [x for x in manifest if (x['condition'], x['draw']) == (s['condition'], s['draw'])]
        measurement.append(dict(condition=s['condition'], draw=s['draw'],
            **{label: float(np.mean([x['measurement'][label]['relative_l1'] for x in rr])) for label in ('observed', 'heldout')}))
    timing = [x['reconstruct_s'] for x in manifest][1:]
    summary = dict(completed_at=now(), checkpoint_sha256=r['checkpoint_sha256'], best_epoch=r['best_epoch'],
        panels=panels, morphology=shape_summary, measurement=measurement,
        timing=dict(reconstruct_seconds_median_excluding_first=float(np.median(timing)),
                    reconstruct_seconds_p90_excluding_first=float(np.quantile(timing, .9)),
                    scope='single-case synchronized GPU3 observations, C0 and C1; excludes external projection and I/O; not controlled cross-model benchmark'),
        sources_preserved=True, development_validation_only=True, final_test_evaluated=False,
        training_seeds=1, prediction_boundary=boundary,
        gt_access=dict(first_label_phase='separate score process after complete prediction seal',
                       unique_gt64_files=36, gt64_paths=gt_paths, gt128_loaded=False, test_loaded=False))
    write(OUT / 'summary.json', summary)
    for row in panels: print(json.dumps(row), flush=True)
    print('Scoring complete; no final test accessed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'predict', 'score'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == 'register': register()
    elif args.action == 'predict': predict(args.device)
    else: score()


if __name__ == '__main__': main()
