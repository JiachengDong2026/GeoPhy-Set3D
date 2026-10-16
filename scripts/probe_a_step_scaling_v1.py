#!/usr/bin/env python3
"""Frozen A late-step displacement intervention; independent development probe.

All four globally fixed interventions are saved before a separate GT process.
No model, source manifest, checkpoint selector or final test is modified.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest, timestamp, write_json
from plan_a_budget_v4 import build_budget_model
from plan_a_runtime_candidates import configure_backend

BASE = ROOT / 'outputs/research/a_step_scaling_v1'
FORMAL = ROOT / 'outputs/research/scheme_a_budget_v4_wide_seed20261011'
STAGES = ROOT / 'outputs/research/plan_a_mechanisms_v1/stages'
SCALES = (0., .25, .5, 1.)
CONDITIONS = ('Val-A-V4', 'clustered-V4')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def register():
    from geophy_set3d.evaluation.research import frozen_record
    frozen = frozen_record(FORMAL)
    prior = json.loads((FORMAL / 'evaluation/frozen_model_test/registration.json').read_text())
    settings = [s for s in prior['settings'] if s['name'] in CONDITIONS]
    assert len(settings) == 2 and all(len(s['input_ids']) == 36 for s in settings)
    boundary = json.loads((STAGES / 'boundary.json').read_text())
    assert boundary['checkpoint_sha256'] == frozen['sha256']
    assert digest(STAGES / 'manifest.jsonl') == boundary['manifest_sha256']
    BASE.mkdir(parents=True, exist_ok=False)
    write_json(BASE / 'registration.json', dict(
        registered_at=timestamp(), script_sha256=digest(Path(__file__)),
        checkpoint_sha256=frozen['sha256'], checkpoint=frozen['checkpoint'],
        source_manifest_sha256=digest(STAGES / 'manifest.jsonl'), scales=SCALES,
        settings=settings, dataset_root=prior['dataset_root'],
        intervention='Resume each archived FP32 C2; on steps 3 and 4 set z=clip(c-scale*tau*update). Keep prior_signal=tanh(update/.02) and all learned priors unchanged.',
        scale_zero='Physical displacement exactly zero on both late steps; residual/gradient signal is still supplied and both priors run.',
        selection='All scales global and fixed before prediction; no per-case oracle, no best-scale checkpoint or threshold selection.',
        budget=dict(max_prediction_seconds=1200, cases=72, final_fields=288, estimated_minutes='2-15; stop at 20 min; no additional draws'),
        correctness=dict(scale_one_max_abs_to_archived_C4=0.0015,
            scale_one_mean_abs_to_archived_C4=0.00001,
            first_case_full_forward_max_abs=0.0015),
        final_test_evaluated=False, gt_for_training=False, gt_for_selection=False,
        prediction_gt_reads_allowed=False, development_diagnostic_only=True))
    print('Registered 72 cases x 4 fixed scales; <=20 min prediction budget.', flush=True)


def setup_late(model, batch, c):
    from geophy_set3d.models.set_unrolled3d import _clean_inputs
    batch = _clean_inputs(batch)
    stencil = model.operator.prepare(batch, c.shape[-1])
    valid = stencil.valid_ray.to(c.device)
    counts = stencil.valid_counts.to(device=c.device, dtype=c.dtype)
    y = torch.where(valid, batch['input_projection_raw'] / model.y_scale, 0)
    tt1, gram = model.operator.coverage(stencil, device=c.device, dtype=c.dtype)
    d0 = gram * (28.965 / 16.043) ** 2 / counts[:, None, None, None, None]
    positive = d0 > 0
    eps = (1e-3 * d0.flatten(1).sum(1) / positive.flatten(1).sum(1)).clamp_min(1e-12)
    denominator = torch.maximum(d0, eps[:, None, None, None, None])
    log_coverage = torch.log1p(tt1)
    maximum = log_coverage.flatten(1).amax(1).reshape(-1, 1, 1, 1, 1)
    coverage = torch.where(maximum > 0, log_coverage / maximum.clamp_min(1e-30), 0)
    return stencil, valid, counts, y, positive, denominator, coverage, model._visible_fraction(batch, c)


def late_forward(model, c, scale, context):
    stencil, valid, counts, y, positive, denominator, coverage, visible = context
    physical_max = []
    tau = model.step_logits.sigmoid()
    for k in (2, 3):
        _, gradient = model.operator.residual_gradient(c, y, stencil)
        g = gradient / counts[:, None, None, None, None]
        update = torch.where(positive, g / denominator, 0)
        signal = torch.tanh(update / .02)
        z = (c - scale * tau[k] * update).clamp(0, 1)
        physical_max.append(float((z-c).abs().max()))
        embedding = model.step_embedding.weight[k].reshape(1, 4, 1, 1, 1).expand(c.shape[0], -1, *c.shape[2:])
        inputs = torch.cat((c, z, signal, coverage, visible, embedding), 1)
        correction = model.prior_amplitude * model.prior(inputs).tanh()
        c = (z + correction).clamp(0, 1)
    if scale == 0:
        assert physical_max == [0., 0.]
    return c, physical_max


def predict(device):
    from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
    from geophy_set3d.data.collate import projection_pool_collate
    from geophy_set3d.training.utils import move_batch
    from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
    from geophy_set3d.evaluation.research import EvaluationWithHoldout, frozen_record
    from geophy_set3d.evaluation.view_selection import EvaluationViewSampler
    from geophy_set3d.training.gt_read_guard import GTReadGuard
    registration = json.loads((BASE / 'registration.json').read_text())
    assert digest(Path(__file__)) == registration['script_sha256']
    assert frozen_record(FORMAL)['sha256'] == registration['checkpoint_sha256']
    assert digest(STAGES / 'manifest.jsonl') == registration['source_manifest_sha256']
    archive = {(r['condition'], r['case_id']): r for r in rows(STAGES / 'manifest.jsonl')}
    config = yaml.safe_load((FORMAL / 'config_resolved.yaml').read_text())
    configure_backend(config['execution_runtime'])
    model = build_budget_model(config).to(device)
    dataset = MeasurementOnlyPoolDataset(config['dataset']['root'], split='val', geometry_pool='train',
        evaluation_only=True, min_input_views=4, max_input_views=4, hold_views=0,
        base_seed=20261011, normalization='linear_p99_5', statistics_file='train_statistics.json')
    output = BASE / 'predictions'; output.mkdir(exist_ok=False)
    manifest, checks = [], []
    began = time.monotonic()
    with GTReadGuard(config['dataset']['root'], FORMAL / 'checkpoints') as guard:
        checkpoint = torch.load(registration['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model']); model.eval()
        for setting in registration['settings']:
            class Replay(EvaluationViewSampler):
                def sample(self, station_count, *, epoch, sample_index, case_id):
                    return np.array(setting['input_ids'][case_id], dtype=np.int64), np.empty(0, dtype=np.int64)
            dataset.sampler = EvaluationWithHoldout(Replay(dataset.geometry['station_azimuth_deg'], 4, 0, 'train', 20261011, 'random'))
            for index in range(len(dataset)):
                assert time.monotonic()-began < 1200, 'Registered 20-minute prediction budget exhausted'
                batch = move_batch(projection_pool_collate([dataset[index]]), device)
                case = batch['case_id'][0]
                source = archive[(setting['name'], case)]
                assert source['input_ids'] == setting['input_ids'][case]
                assert source['heldout_ids'] == batch['hold_station_indices'][0].tolist()
                source_path = STAGES / source['prediction']
                assert digest(source_path) == source['sha256']
                with np.load(source_path, allow_pickle=False) as saved:
                    c2 = torch.from_numpy(saved['C2'].copy())[None,None].to(device)
                    baseline = saved['C4'].copy()
                inputs = {k:v for k,v in batch.items() if not k.startswith('hold_')}
                heldout = {k:v for k,v in batch.items() if not k.startswith(('input_', 'hold_')) and k!='view_mask'}
                heldout.update({'input_'+k[5:]:v for k,v in batch.items() if k.startswith('hold_') and k!='hold_view_mask'})
                heldout['view_mask'] = batch['hold_view_mask']
                fields, measurements, timings, displacement = {}, {}, {}, {}
                with torch.no_grad():
                    context = setup_late(model, inputs, c2)
                    for scale in SCALES:
                        key = str(scale)
                        torch.cuda.synchronize(); start = time.monotonic()
                        field, phys = late_forward(model, c2, scale, context)
                        torch.cuda.synchronize(); timings[key] = time.monotonic()-start
                        fields[key] = field[0,0].cpu().numpy()
                        displacement[key] = phys
                        measurements[key] = {}
                        for domain, target in [('observed', inputs), ('heldout', heldout)]:
                            projected = model.operator(field, target)
                            valid = target['input_valid_ray'].bool() & target['view_mask'][:,:,None,None,None]
                            measurements[key][domain] = relative_projection_metrics(projected[0], target['input_projection_raw'][0], valid[0])
                    difference = np.abs(fields['1.0'] - baseline)
                    check = dict(case_id=case, condition=setting['name'], max_abs=float(difference.max()), mean_abs=float(difference.mean()))
                    assert check['max_abs'] <= .0015 and check['mean_abs'] <= .00001, check
                    if index == 0:
                        full = model(inputs)[0,0].cpu().numpy()
                        check['full_forward_max_abs'] = float(np.abs(full-fields['1.0']).max())
                        check['full_forward_mean_abs'] = float(np.abs(full-fields['1.0']).mean())
                        assert check['full_forward_max_abs'] <= .0015, check
                    checks.append(check)
                path = output / f"{setting['name']}_{index:03d}.npz"
                np.savez_compressed(path, **fields)
                manifest.append(dict(condition=setting['name'], case_id=case, prediction=str(path.relative_to(BASE)),
                    sha256=digest(path), measurement=measurements, seconds=timings,
                    late_physical_displacement_max_abs=displacement, input_ids=source['input_ids'], heldout_ids=source['heldout_ids']))
                if index % 9 == 0:
                    print(f"{setting['name']} {index+1}/36; elapsed={time.monotonic()-began:.1f}s", flush=True)
        audit = guard.summary()
    manifest_path = BASE / 'manifest.jsonl'
    manifest_path.write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    write_json(BASE / 'correctness.json', dict(checks=checks,
        scale_zero_exactly_skips_both_displacements=True, scale_zero_preserves_two_prior_calls=True))
    write_json(BASE / 'prediction_boundary.json', dict(
        completed_at=timestamp(), elapsed_seconds=time.monotonic()-began,
        cases=72, final_fields=288, checkpoint_sha256=registration['checkpoint_sha256'],
        all_predictions_saved_before_gt=True, manifest_sha256=digest(manifest_path),
        gt_read_audit=audit, final_test_evaluated=False))
    print('All 288 fields saved; independent score process may now read Val GT.', flush=True)


def paired(value, reference):
    delta = value-reference
    rng = np.random.default_rng(20261016)
    boots = delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
    return dict(mean_delta=float(delta.mean()), ci95=np.quantile(boots,[.025,.975]).tolist(),
                wins=int((delta < -1e-8).sum()), losses=int((delta > 1e-8).sum()), ties=int((abs(delta)<=1e-8).sum()),
                relative_error_reduction=float(1-value.mean()/reference.mean()))


def score():
    from geophy_set3d.evaluation.research import volume_metrics
    registration = json.loads((BASE / 'registration.json').read_text())
    boundary = json.loads((BASE / 'prediction_boundary.json').read_text())
    assert boundary['all_predictions_saved_before_gt']
    assert digest(BASE / 'manifest.jsonl') == boundary['manifest_sha256']
    manifest = rows(BASE / 'manifest.jsonl')
    assert len(manifest) == 72
    index = {r['case_id']:r for r in rows(Path(registration['dataset_root']) / 'source_index.jsonl')}
    scored = []
    for row in manifest:
        assert index[row['case_id']]['split'] == 'val'
        path = BASE / row['prediction']; assert digest(path) == row['sha256']
        target = np.load(Path(registration['dataset_root']) / index[row['case_id']]['gt_volume_64_path'], allow_pickle=False)
        with np.load(path, allow_pickle=False) as saved:
            assert sorted(saved.files) == sorted(str(s) for s in SCALES)
            scored.append(dict(row, volume={k:volume_metrics(saved[k],target) for k in saved.files}))
    (BASE / 'per_case_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    summary = []
    for setting in registration['settings']:
        group = [r for r in scored if r['condition'] == setting['name']]
        historical = {r['case_id']:r for r in rows(setting['reference'])}
        reference = np.array([historical[r['case_id']]['relative_l1'] for r in group])
        baseline = np.array([r['volume']['1.0']['relative_l1'] for r in group])
        for scale in SCALES:
            key = str(scale); values = np.array([r['volume'][key]['relative_l1'] for r in group])
            summary.append(dict(condition=setting['name'], scale=scale, cases=len(group),
                relative_l1=float(values.mean()), relative_l1_p90=float(np.quantile(values,.9)),
                relative_l1_median=float(np.median(values)),
                observed_l1=float(np.mean([r['measurement'][key]['observed']['relative_l1'] for r in group])),
                heldout_l1=float(np.mean([r['measurement'][key]['heldout']['relative_l1'] for r in group])),
                late_two_steps_seconds_median=float(np.median([r['seconds'][key] for r in group])),
                versus_scale_one=paired(values,baseline), versus_history=paired(values,reference)))
    write_json(BASE / 'summary.json', dict(results=summary, prediction_elapsed_seconds=boundary['elapsed_seconds'],
        final_test_evaluated=False, gt_for_training=False, gt_for_selection=False,
        development_diagnostic_only=True, seed=20261011,
        scope='36 validation cases, fixed legacy V4 and clustered draw0; global predetermined scales; no per-case oracle'))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('register','predict','score'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    {'register':register, 'predict':lambda:predict(args.device), 'score':score}[args.action]()
