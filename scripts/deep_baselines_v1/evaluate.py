"""Frozen Val G1--G5 comparison: register -> infer/seal -> saved-output GT -> stats.

This entry point does not schedule GPUs or launch training. The caller must hold
the repository resource lock before invoking inference on a GPU. ``--mini`` is
an explicitly nonformal 2-case, 1-draw integration check, never a 900-row result.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / 'src'), str(PROJECT / 'scripts')]
GROUPS = ('G1', 'G2', 'G3', 'G4', 'G5')
VOLUME_METRICS = ('relative_l1', 'relative_l2', 'mae', 'rmse',
                  'integrated_concentration_relative_error', 'peak_value_absolute_error')
METRICS = VOLUME_METRICS + ('observed_relative_l1', 'observed_relative_l2',
                            'heldout_relative_l1', 'heldout_relative_l2')
BOOTSTRAP_SEED = 2026101801
BOOTSTRAP_SAMPLES = 10000
RECORDS = PROJECT / 'docs/reports/artifacts/selfsup_v1_c0c1_geometry/station_subsets.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def csv_write(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def token(record):
    return f"case{record['case_index']:02d}_{record['geometry_id']}"


def validate_records(records, mini=False):
    expected_cases = 2 if mini else 36
    expected_draws = [0] if mini else list(range(5))
    keys = [(r['case_id'], r['group'], r['draw']) for r in records]
    if len(set(keys)) != len(records) or len(records) != expected_cases * 5 * len(expected_draws):
        raise ValueError('Incomplete or duplicate frozen Val case/group/draw membership')
    cases = sorted({r['case_id'] for r in records})
    if len(cases) != expected_cases:
        raise ValueError('Unexpected number of Val cases')
    for case in cases:
        for group in GROUPS:
            part = [r for r in records if r['case_id'] == case and r['group'] == group]
            if sorted(r['draw'] for r in part) != expected_draws:
                raise ValueError('Missing draw/group')
            for row in part:
                if len(row['station_ids']) != 4 or len(row['H_station_ids']) != 4:
                    raise ValueError('Exactly four observed and four H stations required')
                if set(row['station_ids']) & set(row['H_station_ids']):
                    raise ValueError('Observed/H station overlap')
    if mini and sorted({r['case_index'] for r in records}) != [0, 1]:
        raise ValueError('Mini must use fixed case indices 0,1')


def aggregate_cases(rows, mini=False):
    """Draw means first. No voxel pooling and no draw-as-independent bootstrap."""
    records = [row['record'] for row in rows]
    validate_records(records, mini)
    output = []
    for case in sorted({r['case_id'] for r in records}):
        for group in GROUPS:
            part = [r for r in rows if r['record']['case_id'] == case and r['record']['group'] == group]
            result = dict(case_id=case, case_index=part[0]['record']['case_index'], group=group, draws=len(part))
            for metric in METRICS:
                values = np.asarray([r[metric] for r in part], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite metric; retain and report failures before statistics')
                result[metric] = float(values.mean())
            output.append(result)
    return output


def summarize(case_rows, mini=False):
    cases = sorted({r['case_id'] for r in case_rows})
    if len(cases) != (2 if mini else 36):
        raise ValueError('Wrong case-level sample count')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(cases), (BOOTSTRAP_SAMPLES, len(cases)))
    summary, contrasts = [], []
    for metric in METRICS:
        arrays = {}
        for group in GROUPS:
            part = sorted([r for r in case_rows if r['group'] == group], key=lambda r: r['case_id'])
            if [r['case_id'] for r in part] != cases:
                raise ValueError('Case alignment mismatch')
            values = np.asarray([r[metric] for r in part], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Finite case-level metrics required')
            arrays[group] = values
            low, high = np.quantile(values[indices].mean(1), [.025, .975])
            summary.append(dict(group=group, metric=metric, cases=len(cases), mean_per_case=float(values.mean()),
                                median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                                p95=float(np.quantile(values, .95)), maximum=float(values.max()),
                                worst_case_id=cases[int(values.argmax())], ci_low=float(low), ci_high=float(high)))
        difference = arrays['G5'] - arrays['G1']
        low, high = np.quantile(difference[indices].mean(1), [.025, .975])
        denominator = arrays['G1'].mean()
        denominators = arrays['G1'][indices].mean(1)
        valid_ratio = denominator != 0 and bool(np.all(denominators != 0))
        ratio = arrays['G5'].mean() / denominator if valid_ratio else None
        ratio_ci = np.quantile(arrays['G5'][indices].mean(1) / denominators, [.025, .975]).tolist() if valid_ratio else [None, None]
        contrasts.append(dict(metric=metric, G5_minus_G1=float(difference.mean()), difference_ci_low=float(low),
                              difference_ci_high=float(high), G5_div_G1_grand_mean_ratio=None if ratio is None else float(ratio),
                              ratio_ci_low=ratio_ci[0], ratio_ci_high=ratio_ci[1],
                              ratio_definition='ratio of group grand means; recomputed per case bootstrap; zero denominators yield null'))
    return dict(formal=not mini, scope='NONFORMAL_MINI_INTEGRATION' if mini else 'VAL_G1_G5_900',
                cases=len(cases), bootstrap=dict(seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES, unit='case after draw mean'),
                aggregation='unweighted mean per case; seeds kept separate; no pooled voxels', summary=summary, contrasts=contrasts)


def context():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import baseline_context
    return baseline_context


def sources():
    paths = list((PROJECT / 'src').rglob('*.py')) + list(Path(__file__).resolve().parent.glob('*.py'))
    paths += list((PROJECT / 'configs/baselines').rglob('*.json'))
    return {str(p): sha(p) for p in sorted(paths)}


def verify_run(run, checkpoint=False):
    run = Path(run)
    protocol = read(run / 'protocol.json')
    if sha(run / 'protocol.json') != (run / 'protocol.sha256').read_text().strip():
        raise ValueError('Evaluation protocol changed')
    for path, expected in protocol['sources'].items():
        if sha(path) != expected:
            raise ValueError(f'Numerical source changed: {path}')
    for path, expected in protocol['data_identity'].items():
        if sha(path) != expected:
            raise ValueError(f'Frozen Val data identity changed: {path}')
    if sha(RECORDS) != protocol['records_source_sha256']:
        raise ValueError('Original 900 station protocol changed')
    if sha(protocol['selection_record']) != protocol['selection_record_sha256']:
        raise ValueError('Selected checkpoint record changed')
    if checkpoint and sha(protocol['checkpoint']) != protocol['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    validate_records(protocol['records'], protocol['mini'])
    return protocol


def register(args):
    b = context()
    path = Path(args.checkpoint).resolve()
    if sha(path) != args.checkpoint_sha256:
        raise ValueError('Explicit selected checkpoint hash mismatch')
    selection = Path(args.selection_record).resolve()
    if not selection.is_file():
        raise ValueError('A pre-existing checkpoint selection record is required')
    selected = read(selection)
    selected_hash = selected.get('checkpoint_sha256', selected.get('sha256'))
    if args.arm == 'plain':
        selected_hash = selected.get('model', {}).get('checkpoint_sha256', selected_hash)
    if selected_hash != args.checkpoint_sha256:
        raise ValueError('Selection record does not identify this checkpoint')
    if 'seed' in selected and int(selected['seed']) != args.seed:
        raise ValueError('Selection record training seed differs')
    if selected.get('formal') is False and not args.mini:
        raise ValueError('Engineering pilot checkpoint is limited to --mini')
    records = read(RECORDS)
    validate_records(records)
    if args.mini:
        records = [r for r in records if r['case_index'] in (0, 1) and r['draw'] == 0]
    validate_records(records, args.mini)
    run = Path(args.run).resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError('Evaluation directory must be new; old results are immutable')
    run.mkdir(parents=True, exist_ok=True)
    config = b.model_config(args.arm) if args.arm != 'plain' else read(b.ROOT / 'main_anchor.json')['model']['config']
    data_root = Path(b.c.DATA)
    data_identity = {str(path): sha(path) for path in [data_root / 'source_index.jsonl',
                     data_root / 'train_statistics.json', data_root / 'geometry/train_geometry.npz',
                     data_root / 'splits/val_cases.txt']}
    value = dict(registered_at=now(), arm=args.arm, seed=args.seed, mini=args.mini, records=records,
                 split='val', geometry_pool='train', GT128_allowed=False, test_allowed=False, Pool_B_allowed=False,
                 checkpoint=str(path), checkpoint_sha256=args.checkpoint_sha256,
                 selection_record=str(selection), selection_record_sha256=sha(selection), config=config,
                 records_source=str(RECORDS), records_source_sha256=sha(RECORDS), sources=sources(),
                 data_identity=data_identity,
                 output='C1 final field for frozen plain; single field for baselines',
                 checkpoint_use='preselected only; no selection or architecture changes from these results',
                 bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, unit='case after five-draw mean'))
    write(run / 'protocol.json', value)
    (run / 'protocol.sha256').write_text(sha(run / 'protocol.json') + '\n')
    print(json.dumps(dict(run=str(run), rows=len(records), formal=not args.mini)))


def saved_rows(run, protocol):
    rows = []
    for record in protocol['records']:
        path = Path(run) / 'rows' / (token(record) + '.json')
        if path.exists():
            row = read(path)
            if row['record'] != record or row['checkpoint_sha256'] != protocol['checkpoint_sha256']:
                raise ValueError('Saved row identity mismatch')
            prediction = Path(run) / row['prediction']
            if sha(prediction) != row['prediction_sha256']:
                raise ValueError('Saved prediction hash mismatch')
            rows.append(row)
    return rows


def seal(run, protocol):
    run = Path(run)
    rows = saved_rows(run, protocol)
    if len(rows) != len(protocol['records']):
        raise ValueError('Cannot seal incomplete predictions')
    value = dict(rows=len(rows), protocol_sha256=sha(run / 'protocol.json'), checkpoint_sha256=protocol['checkpoint_sha256'],
                 all_predictions_saved_before_GT=True,
                 row_sha256={token(r): sha(run / 'rows' / (token(r) + '.json')) for r in protocol['records']})
    if (run / 'predictions_sealed.json').exists():
        if read(run / 'predictions_sealed.json') != value:
            raise ValueError('Prediction seal changed')
    else:
        write(run / 'predictions_sealed.json', value)
    return rows


def infer(args):
    import torch
    from geophy_set3d.data.collate import projection_pool_collate
    from geophy_set3d.training.utils import move_batch, seed_everything
    from geophy_set3d.training.deep_comparison import tensor_identity
    from geophy_set3d.training.rvtc_matched import state_hash
    from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
    from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
    from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import DiagnosticReadGuard
    from diagnose_clustered_one_swap import dataset
    b = context()
    protocol = verify_run(args.run, checkpoint=True)
    run = Path(args.run)
    torch.set_num_threads(2)
    seed_everything(protocol['seed'])
    b.c.backend()
    data_root = str(b.c.DATA)
    existing = {token(r['record']) for r in saved_rows(run, protocol)}
    for folder in ('case_images', 'rows'):
        (run / folder).mkdir(exist_ok=True)
    with DiagnosticReadGuard(data_root, [protocol['checkpoint']], run, 'inference') as guard:
        ds = dataset(dict(dataset=data_root, original_seed=20261011))
        if len(ds) != 36:
            raise ValueError('Expected 36 Val cases')
        if protocol['arm'] == 'plain':
            anchor = read(b.ROOT / 'main_anchor.json')
            if protocol['checkpoint_sha256'] != anchor['model']['checkpoint_sha256']:
                raise ValueError('Only the anchored final plain checkpoint is allowed')
            model = b.c.model_for(protocol['config'], b.c.PLAIN, args.device)
        else:
            model = b.build_model(protocol['arm'], args.device, protocol['seed'])
        checkpoint = torch.load(protocol['checkpoint'], map_location='cpu', weights_only=False)
        if checkpoint.get('volumetric_gt_used') is not False:
            raise ValueError('Explicit GT-free checkpoint provenance required')
        trained_config = checkpoint['config']
        if trained_config['training']['seed'] != protocol['seed']:
            raise ValueError('Loaded checkpoint training seed differs')
        if protocol['arm'] == 'plain':
            if trained_config != protocol['config']:
                raise ValueError('Anchored plain training configuration differs')
        elif trained_config['model']['baseline'] != protocol['config']:
            raise ValueError('Baseline architecture differs from registered configuration')
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        model.eval().requires_grad_(False)
        before = state_hash(model.state_dict())
        with torch.inference_mode():
            for record in protocol['records']:
                name = token(record)
                if name in existing:
                    continue
                row, projections = ds._projections(record['case_index'])
                if row['case_id'] != record['case_id']:
                    raise ValueError('Case/order mismatch')
                def batch(ids):
                    sample = ds._measurement_sample(row, projections, np.asarray(ids), np.empty(0, dtype=np.int64))
                    return move_batch(projection_pool_collate([sample]), args.device)
                observed, heldout = batch(record['station_ids']), batch(record['H_station_ids'])
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize(args.device)
                    torch.cuda.reset_peak_memory_stats(args.device)
                began = time.perf_counter()
                prediction = model(observed)
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize(args.device)
                seconds = time.perf_counter() - began
                peak = torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None
                if prediction.dtype != torch.float32:
                    raise ValueError('The common output contract requires FP32')
                array = prediction[0, 0].float().cpu().numpy()
                check_array(array)
                if tuple(prediction.shape) != (1, 1, 64, 64, 64):
                    raise ValueError('Incorrect full field shape')
                metrics = {}
                for label, measured in [('observed', observed), ('heldout', heldout)]:
                    projected = project_input(prediction, measured)
                    valid = measured['input_valid_ray'][0, :, 0].bool() & measured['view_mask'][0, :, None, None]
                    values = relative_projection_metrics(projected[0, :, 0], measured['input_projection_raw'][0, :, 0], valid)
                    metrics.update({label + '_' + key: value for key, value in values.items()})
                relative = f'case_images/{name}.npz'
                with (run / relative).open('xb') as stream:
                    np.savez(stream, prediction=array)
                identity = lambda measured: tensor_identity(*[measured[k] for k in sorted(measured) if isinstance(measured[k], torch.Tensor)])
                write(run / 'rows' / (name + '.json'), dict(record=record, prediction=relative,
                      prediction_sha256=sha(run / relative), checkpoint_sha256=protocol['checkpoint_sha256'],
                      complete_input_sha256=identity(observed), complete_H_sha256=identity(heldout),
                      forward_seconds=seconds, parameter_count=sum(p.numel() for p in model.parameters()),
                      checkpoint_bytes=Path(protocol['checkpoint']).stat().st_size,
                      peak_gpu_allocated_bytes=peak,
                      timing_scope='one synchronized forward, excludes measurement diagnostics/I/O; no warmup benchmark claim',
                      memory_scope='absolute CUDA allocated peak through model forward, before extra Pdiff diagnostics',
                      output_min=float(array.min()), output_max=float(array.max()), output_mean=float(array.mean()),
                      output_std=float(array.std()), output_near_constant=bool(array.std() <= 1e-12), **metrics))
                print(f"{protocol['arm']} {name} saved", flush=True)
        if state_hash(model.state_dict()) != before:
            raise ValueError('Inference modified model state')
        if guard.counts['val_gt64_opens'] or guard.counts['forbidden_attempts']:
            raise ValueError('Inference data boundary failed')
        audit_path = run / f'inference_access_{time.time_ns()}.json'
        write(audit_path, guard.summary())
    seal(run, protocol)


def check_array(array):
    if array.shape != (64, 64, 64) or array.dtype != np.float32 or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError('Expected finite float32 64³ mass fraction in [0,1]')


def gt(args):
    import torch
    from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import DiagnosticReadGuard
    from geophy_set3d.training.metrics import VolumeMetrics
    from diagnose_clustered_one_swap import dataset, target_gt
    b = context()
    protocol = verify_run(args.run)
    run = Path(args.run)
    if not (run / 'predictions_sealed.json').exists():
        raise ValueError('GT requires a complete inference seal from a prior stage')
    rows = seal(run, protocol)
    torch.set_num_threads(2)
    write(run / 'first_GT_boundary.json', dict(at=now(), seal_sha256=sha(run / 'predictions_sealed.json'),
          protocol_sha256=sha(run / 'protocol.json'), saved_outputs_only=True, checkpoint_reads_during_GT=False))
    scored = []
    with DiagnosticReadGuard(str(b.c.DATA), [], run, 'gt') as guard:
        ds = dataset(dict(dataset=str(b.c.DATA), original_seed=20261011))
        for index in sorted({row['record']['case_index'] for row in rows}):
            target = target_gt(ds, index)
            for row in [r for r in rows if r['record']['case_index'] == index]:
                with np.load(run / row['prediction'], allow_pickle=False) as archive:
                    array = archive['prediction'].copy()
                check_array(array)
                metric = VolumeMetrics()
                metric.update(torch.from_numpy(array)[None, None], target, [row['record']['case_id']])
                scored.append(dict(row, **{k: metric.rows[0][k] for k in VOLUME_METRICS}))
        write(run / 'GT_access.json', guard.summary())
    write(run / 'scored_rows.json', scored)
    flat = [dict(row['record'], **{k: v for k, v in row.items() if k != 'record'}) for row in scored]
    csv_write(run / 'per_draw_metrics.csv', flat)
    write(run / 'GT_complete.json', dict(rows=len(scored), scored_rows_sha256=sha(run / 'scored_rows.json'),
          predictions_seal_sha256=sha(run / 'predictions_sealed.json'), GT128_read=False, test_read=False, pool_B_read=False))


def stats(args):
    protocol = verify_run(args.run)
    run = Path(args.run)
    complete = read(run / 'GT_complete.json')
    if sha(run / 'scored_rows.json') != complete['scored_rows_sha256']:
        raise ValueError('Scored rows changed')
    cases = aggregate_cases(read(run / 'scored_rows.json'), protocol['mini'])
    value = summarize(cases, protocol['mini'])
    value.update(arm=protocol['arm'], seed=protocol['seed'], checkpoint_sha256=protocol['checkpoint_sha256'])
    csv_write(run / 'case_group_metrics.csv', cases)
    write(run / 'summary.json', value)
    csv_write(run / 'group_summary.csv', value['summary'])
    csv_write(run / 'geometry_contrasts.csv', value['contrasts'])


def paired_statistics(reference, baseline, mini=False, tie_epsilon=1e-8):
    """Positive delta means baseline error exceeds plain, one seed at a time."""
    reference_map = {token(r['record']): r for r in reference}
    baseline_map = {token(r['record']): r for r in baseline}
    if set(reference_map) != set(baseline_map):
        raise ValueError('Paired case/layout membership differs')
    for key, row in reference_map.items():
        other = baseline_map[key]
        if row['record'] != other['record']:
            raise ValueError('Paired station IDs/order differ')
        for field in ('complete_input_sha256', 'complete_H_sha256'):
            if row[field] != other[field]:
                raise ValueError('Paired input/H tensors differ')
    full = aggregate_cases(reference, mini)
    ablated = aggregate_cases(baseline, mini)
    cases = sorted({r['case_id'] for r in full})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(cases), (BOOTSTRAP_SAMPLES, len(cases)))
    output = []
    for group in GROUPS:
        a = sorted([r for r in full if r['group'] == group], key=lambda r: r['case_id'])
        b = sorted([r for r in ablated if r['group'] == group], key=lambda r: r['case_id'])
        for metric in METRICS:
            plain = np.asarray([r[metric] for r in a])
            base = np.asarray([r[metric] for r in b])
            difference = base - plain
            low, high = np.quantile(difference[indices].mean(1), [.025, .975])
            output.append(dict(group=group, metric=metric, cases=len(cases), plain_mean=float(plain.mean()),
                baseline_mean=float(base.mean()), mean_delta=float(difference.mean()),
                median_delta=float(np.median(difference)), ci_low=float(low), ci_high=float(high),
                baseline_worse=int((difference > tie_epsilon).sum()), baseline_better=int((difference < -tie_epsilon).sum()),
                ties=int((np.abs(difference) <= tie_epsilon).sum()), delta_definition='baseline minus plain',
                interval='exploratory marginal 95%; no multiplicity-adjusted confirmatory claim'))
    return output


def compare(args):
    if not args.reference_run:
        raise ValueError('compare requires --reference-run for frozen plain')
    run, reference_run = Path(args.run), Path(args.reference_run)
    protocol, reference = verify_run(run), verify_run(reference_run)
    if reference['arm'] != 'plain' or protocol['arm'] == 'plain' or protocol['mini'] != reference['mini']:
        raise ValueError('Must compare one baseline to plain with the same scope')
    # Plain seed 20261011 can be a fixed benchmark for seed19, but is not a
    # same-seed pairing. The output explicitly labels the actual training seeds.
    scored = []
    for folder in (reference_run, run):
        complete = read(folder / 'GT_complete.json')
        if sha(folder / 'scored_rows.json') != complete['scored_rows_sha256']:
            raise ValueError('Scored rows changed')
        scored.append(read(folder / 'scored_rows.json'))
    paired = paired_statistics(*scored, mini=protocol['mini'])
    write(run / 'paired_to_plain.json', dict(arm=protocol['arm'], baseline_seed=protocol['seed'],
        plain_seed=reference['seed'], same_training_seed=protocol['seed'] == reference['seed'],
        formal=not protocol['mini'], paired_unit='case after draw mean; no seed pooling', rows=paired))
    csv_write(run / 'paired_to_plain.csv', paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register', 'infer', 'gt', 'stats', 'compare'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--arm', choices=['plain', 'deepsets', 'senseiver', 'gnot'])
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--selection-record')
    parser.add_argument('--mini', action='store_true')
    parser.add_argument('--reference-run')
    args = parser.parse_args()
    if args.phase == 'register' and not all((args.arm, args.checkpoint, args.checkpoint_sha256, args.selection_record)):
        parser.error('register requires arm, checkpoint, checkpoint-sha256 and selection-record')
    globals()[args.phase](args)


if __name__ == '__main__':
    main()
