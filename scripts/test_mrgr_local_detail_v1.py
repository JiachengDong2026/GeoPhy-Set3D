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
import test_mrgr_multiscale_frozen_v1 as base
import mrgr_local_detail_v1 as local_model
import probe_mrgr_multiscale_v1 as probe
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.utils import move_batch
from geophy_set3d.training.projection_losses import project_measurements
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.evaluation.research import volume_metrics
from report_historical_reconstruction_v1 import metrics as morphology_metrics

RUN = ROOT / 'outputs/research/mrgr_local_detail_formal_v1_seed20261011'
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


def verify_run():
    p = read(RUN / 'protocol.json')
    assert sha(RUN / 'protocol.json') == (RUN / 'protocol.sha256').read_text().strip()
    for name, digest in p['sources'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        assert sha(path) == digest, f'Registered source changed: {path}'
    if 'data_files' in p:
        data_root = Path(p['config']['dataset']['root'])
        for name, digest in p['data_files'].items():
            assert sha(data_root / name) == digest, f'Registered data changed: {name}'
    return p


def verify_registration():
    r = read(OUT / 'registration.json')
    assert sha(OUT / 'registration.json') == (OUT / 'registration.sha256').read_text().strip()
    assert sha(Path(__file__)) == r['evaluation_source_sha256']
    assert sha(ROOT / 'scripts/mrgr_local_detail_v1.py') == r['model_source_sha256']
    verify_run()
    assert sha(RUN / 'checkpoints/best.pt') == r['checkpoint_sha256']
    for path, digest in r['evidence_sha256'].items():
        assert sha(path) == digest, path
    return r


def register():
    p = verify_run()
    summary = read(RUN / 'summary.json')
    assert summary['completed'] and summary['epochs'] == 300 and summary['updates'] == 24900
    assert summary['sources_preserved'] and not summary['final_test_evaluated']
    assert all(summary['gt_audit'][k] == 0 for k in ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts'))
    checkpoint = RUN / 'checkpoints/best.pt'
    assert sha(checkpoint) == summary['best_checkpoint_sha256']
    # Reuse the already frozen 17-layout protocol, never derive layouts from candidate predictions or GT.
    old = base.verify_registration()
    source = [x for x in rows(DATA / 'source_index.jsonl') if x['split'] == 'val']
    cases = [x['case_id'] for x in source]
    assert len(cases) == len(set(cases)) == 36
    assert cases == old['cases'] and set(cases) == set(p['validation_case_ids'])
    settings = old['settings']
    assert len(settings) == 17
    evidence = dict(old['evidence_sha256'])
    for path in [RUN / 'protocol.json', RUN / 'summary.json',
                 base.OUT / 'registration.json', base.OUT / 'registration.sha256',
                 base.OUT / 'per_case_metrics.jsonl', base.OUT / 'summary.json',
                 base.OUT / 'predictions_complete.json']:
        evidence[str(path)] = sha(path)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'predictions').mkdir()
    registration = dict(registered_at=now(), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        best_epoch=summary['best_epoch'], completed_epochs=300, parameters=p['parameters'],
        evaluation_source_sha256=sha(Path(__file__)), model_source_sha256=sha(ROOT / 'scripts/mrgr_local_detail_v1.py'),
        evidence_sha256=evidence, baseline_multiscale_metrics=str(base.OUT / 'per_case_metrics.jsonl'),
        settings=settings, cases=cases, case_count=36, layouts=17, reconstructions=612, saved_fields=1224,
        split='val', geometry_pool='train', precision='FP32, TF32 off, cudnn benchmark off',
        primary='ordinary fixed V4 C1 3D relative L1, paired vs frozen original multiscale C1',
        secondary='relative L2 and V3/V6/clustered each 5 draws averaged within case; spread V4; C0 and morphology exploratory',
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
    assert not (OUT / 'predictions_complete.json').exists(), 'Refuse to replace prediction boundary'
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
        model = local_model.model_for(ckpt['config'], arm='local', device=device)
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
    assert not (OUT / 'summary.json').exists() and not (OUT / 'per_case_metrics.jsonl').exists(), 'Refuse to overwrite scoring results'
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
        assert x['heldout_ids'] == s['heldout_ids'][x['case_id']]
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
                for metric in ('relative_l1', 'relative_l2'):
                    reference[model, s['condition'], s['draw'], x['case_id'], metric] = x[metric]
    for x in rows(AB_EXTRA / 'per_case_metrics.jsonl'):
        for metric in ('relative_l1', 'relative_l2'):
            reference[x['model'], x['condition'], x['draw'], x['case_id'], metric] = x['volume'][metric]
    multiscale = rows(r['baseline_multiscale_metrics'])
    assert len(multiscale) == 1224
    for x in multiscale:
        if x['stage'] == 'C1':
            for metric in ('relative_l1', 'relative_l2'):
                reference['Multiscale', x['condition'], x['draw'], x['case_id'], metric] = x['volume'][metric]
    panels = []
    for condition in ['Val-A-V4', 'V3', 'V6', 'spread-V4', 'clustered-V4']:
        selections = [[0]] if condition in ('Val-A-V4', 'spread-V4') else [[0], list(range(5))]
        for draws in selections:
            for metric in ('relative_l1', 'relative_l2'):
                values = [np.mean([current[condition, d, c, 'C1']['volume'][metric] for d in draws]) for c in r['cases']]
                row = dict(condition=condition, draws=draws, metric=metric, cases=36, mean=float(np.mean(values)),
                           median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                           max=float(np.max(values)), comparisons={})
                for model in ('History', 'Multiscale', 'A', 'B'):
                    old = [np.mean([reference[model, condition, d, c, metric] for d in draws]) for c in r['cases']]
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
                    scope='single-case synchronized device observations, C0 and C1; excludes external projection and I/O; not controlled cross-model benchmark'),
        sources_preserved=True, development_validation_only=True, final_test_evaluated=False,
        training_seeds=1, prediction_boundary=boundary,
        gt_access=dict(first_label_phase='separate score process after complete prediction seal',
                       unique_gt64_files=36, gt64_paths=gt_paths, gt128_loaded=False, test_loaded=False))
    write(OUT / 'summary.json', summary)
    for row in panels: print(json.dumps(row), flush=True)
    print('Scoring complete; no final test accessed.', flush=True)


def main():
    global RUN, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'predict', 'score'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    OUT = RUN / 'evaluation/frozen_model_test_v1'
    torch.set_num_threads(2)
    if args.action == 'register': register()
    elif args.action == 'predict': predict(args.device)
    else: score()


if __name__ == '__main__': main()
