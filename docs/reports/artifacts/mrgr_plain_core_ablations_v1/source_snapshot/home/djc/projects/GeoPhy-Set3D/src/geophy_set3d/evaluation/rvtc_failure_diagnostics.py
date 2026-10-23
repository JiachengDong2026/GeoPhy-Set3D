"""Read-only RVTC diagnostics. No optimizer, GT data or alternative training loss."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from geophy_set3d.data.measurement_only import assert_no_gt
from geophy_set3d.data.rich_measurement import rich_measurement_collate
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training.selfsup_v2 import FROZEN_LOSS, V2ReadGuard, assert_rich_union, objective
from geophy_set3d.training.utils import move_batch


class DiagnosticReadGuard(V2ReadGuard):
    """Add an exact checkpoint whitelist; retain every original GT/Pool A rule."""
    def __init__(self, dataset_root, checkpoint):
        self.checkpoint = Path(checkpoint).resolve()
        super().__init__(dataset_root, self.checkpoint.parent)
        import multiprocessing as mp
        self.checkpoint_opens = mp.Value('i', 0)

    def _audit(self, event, args):
        if self.active and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix in ('.pt', '.pth', '.ckpt'):
                mode, flags = args[1:3]
                writing = ('w' in mode or 'a' in mode or '+' in mode) if isinstance(mode, str) else bool(flags & (os.O_WRONLY | os.O_RDWR))
                if path != self.checkpoint or writing:
                    with self.supervised_checkpoint_attempts.get_lock():
                        self.supervised_checkpoint_attempts.value += 1
                    raise PermissionError('Diagnosis permits reading only the existing v2 gate checkpoint')
                with self.checkpoint_opens.get_lock():
                    self.checkpoint_opens.value += 1
        super()._audit(event, args)

    def summary(self):
        return dict(super().summary(), allowed_checkpoint=str(self.checkpoint),
                    allowed_checkpoint_open_count=self.checkpoint_opens.value,
                    checkpoint_writes_permitted=False)


def unseen_station_ids(station_count, protocol, case_id, seed, count):
    excluded = set().union(*(set(protocol[key]) for key in ('A', 'B', 'H')))
    available = np.array(sorted(set(range(station_count)) - excluded), dtype=np.int64)
    if len(available) < count or count < 1:
        raise ValueError('Insufficient new stations; do not substitute another case/protocol')
    words = np.frombuffer(hashlib.blake2b(case_id.encode(), digest_size=16).digest(), dtype='<u4').tolist()
    rng = np.random.default_rng(np.random.SeedSequence([seed, *words]))
    return sorted(rng.choice(available, count, replace=False).tolist())


def fixed_sample(dataset, index, protocol, new_ids=None):
    """Reuse the strict projection loader and field indexing, never GT __getitem__."""
    row, projections = dataset._projections(index)
    empty = np.empty(0, dtype=np.int64)
    keys = ('A', 'B', 'H', 'T')
    sample = {key: dataset._measurement_sample(row, projections, np.asarray(protocol[key], dtype=np.int64), empty)
              for key in keys}
    if new_ids is not None:
        sample['U'] = dataset._measurement_sample(row, projections, np.asarray(new_ids, dtype=np.int64), empty)
    assert_no_gt(sample)
    return sample


def collate_sample(sample, device):
    batch = {key: move_batch(value, device) for key, value in rich_measurement_collate([sample]).items()}
    if 'U' in sample:
        from geophy_set3d.data.collate import projection_pool_collate
        batch['U'] = move_batch(projection_pool_collate([sample['U']]), device)
    assert_rich_union(batch)
    return batch


def history_review(root):
    root = Path(root)
    rows = [json.loads(line) for line in (root / 'gate_metrics.jsonl').read_text().splitlines() if line.strip()]
    summary = json.loads((root / 'gate_summary.json').read_text())
    with (root / 'teacher_training_diagnostics.csv').open(newline='') as f:
        compact = list(csv.DictReader(f))
    if len(rows) != 80 or [row['epoch'] for row in rows] != list(range(80)) or len(compact) != 80:
        raise ValueError('Incomplete or duplicate historical gate records')
    flattened = []
    for row, saved in zip(rows, compact):
        tr = row['train']
        if saved['phase'] != 'gate' or saved['split'] != 'train' or int(saved['epoch']) != row['epoch'] + 1:
            raise ValueError('Historical CSV and JSON epoch identities differ')
        checks = {'sparse_heldout_l1': tr['measurement']['heldout']['relative_l1'],
                  'teacher_heldout_l1': tr['measurement']['teacher_heldout']['relative_l1'],
                  **tr['disagreement'], **{'teacher_' + key: value for key, value in tr['teacher_output'].items()}}
        if any(not math.isclose(float(saved[key]), value, rel_tol=1e-12, abs_tol=1e-14) for key, value in checks.items()):
            raise ValueError('Historical CSV and JSON values differ')
        if not math.isclose(tr['loss']['teacher'], .5 * (tr['disagreement']['D_A_T'] + tr['disagreement']['D_B_T']), rel_tol=1e-6):
            raise ValueError('Recorded teacher loss and disagreements differ')
        flattened.append(dict(epoch=row['epoch'] + 1, statistic='within-epoch changing-parameter train mean',
            teacher_discrepancy=tr['loss']['teacher'], sparse_H_l1=tr['measurement']['heldout']['relative_l1'],
            sparse_H_l2=tr['measurement']['heldout']['relative_l2'], teacher_H_l1=tr['measurement']['teacher_heldout']['relative_l1'],
            teacher_H_l2=tr['measurement']['teacher_heldout']['relative_l2'],
            student_mean=tr['output']['mean'], student_std=tr['output']['std'],
            teacher_mean=tr['teacher_output']['mean'], teacher_std=tr['teacher_output']['std'],
            teacher_max=tr['teacher_output']['max'], teacher_fraction_near_zero=tr['teacher_output']['fraction_near_zero']))
    first = float(np.mean([row['train']['loss']['teacher'] for row in rows[:10]]))
    last = float(np.mean([row['train']['loss']['teacher'] for row in rows[-10:]]))
    if not math.isclose(first, summary['student_teacher_decrease']['first10_mean'], rel_tol=1e-12) or not math.isclose(last, summary['student_teacher_decrease']['last10_mean'], rel_tol=1e-12):
        raise ValueError('Recomputed historical gate windows differ from the frozen gate')
    minimum = min(flattened, key=lambda row: row['teacher_discrepancy'])
    fixed = {phase: dict(statistic='fixed-parameter evaluation; historical record',
        teacher_discrepancy=summary[phase]['loss']['teacher'],
        sparse_H_l1=summary[phase]['measurement']['heldout']['relative_l1'],
        teacher_H_l1=summary[phase]['measurement']['teacher_heldout']['relative_l1'],
        student_output=summary[phase]['output'], teacher_output=summary[phase]['teacher_output']) for phase in ('initial', 'final')}
    return dict(first10_mean=first, last10_mean=last, original_protocol_passed=summary['passed'],
        original_failed_criteria=[key for key, value in summary['criteria'].items() if not value],
        fixed_parameter_evaluations=fixed, minimum_training_discrepancy=minimum, epoch80_training=flattened[-1],
        historical_absolute_voxel_disagreement='not recoverable: only mean ratios and output moments, no per-case ratio-denominator pairs or volumes',
        initial_and_intermediate_checkpoint_status='not saved by gate implementation; only final gate.pt exists',
        all_history_csv_values_crosschecked=True), flattened


def volume_statistics(c):
    c = c.detach()
    return dict(mean=float(c.mean()), std=float(c.std(unbiased=False)), min=float(c.min()), max=float(c.max()),
                fraction_near_zero=float((c <= 1e-5).float().mean()), fraction_near_one=float((c >= .999).float().mean()))


def measurement_rows(volumes, batch, domains, cohort):
    rows = []
    ca, cb, ct = (value.detach() for value in volumes)
    denominator = float(ct.abs().mean()) + 1e-6
    absolute_a = float((ca - ct).abs().mean()); absolute_b = float((cb - ct).abs().mean())
    for domain in domains:
        target = batch[domain]
        valid = target['input_valid_ray'].bool() & target['view_mask'][:, :, None, None, None]
        scores = {key: relative_projection_metrics(project_input(c, target)[0], target['input_projection_raw'][0], valid[0])
                  for key, c in zip(('A', 'B', 'T'), (ca, cb, ct))}
        row = dict(cohort=cohort, domain=domain, case_id=target['case_id'][0], target_station_ids=target['input_station_indices'][0].tolist(),
            **{f'{key}_{metric}': score[metric] for key, score in scores.items() for metric in ('relative_l1', 'relative_l2')},
            valid_rays=scores['T']['valid_rays'], teacher_denominator=denominator,
            absolute_voxel_disagreement_A_T=absolute_a, absolute_voxel_disagreement_B_T=absolute_b,
            D_A_T=absolute_a/denominator, D_B_T=absolute_b/denominator,
            **{f'{key}_output_{field}': value for key, c in zip(('A', 'B', 'T'), (ca, cb, ct)) for field, value in volume_statistics(c).items()})
        for metric in ('relative_l1', 'relative_l2'):
            row['student_' + metric] = .5 * (scores['A'][metric] + scores['B'][metric])
            row['teacher_minus_student_' + metric] = scores['T'][metric] - row['student_' + metric]
        rows.append(row)
    return rows


def paired_summary(rows, bootstrap_samples, seed):
    if not rows or len({row['case_id'] for row in rows}) != len(rows):
        raise ValueError('Resampling unit must be unique cases, not rays/repeated views')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(bootstrap_samples, len(rows)))
    result = dict(case_count=len(rows), bootstrap_unit='paired case', bootstrap_samples=bootstrap_samples, bootstrap_seed=seed)
    for metric in ('relative_l1', 'relative_l2'):
        delta = np.array([row['teacher_minus_student_' + metric] for row in rows])
        boot = delta[indices].mean(axis=1)
        result[metric] = dict(student_mean=float(np.mean([row['student_' + metric] for row in rows])),
            teacher_mean=float(np.mean([row['T_' + metric] for row in rows])),
            teacher_minus_student_mean=float(delta.mean()), teacher_minus_student_median=float(np.median(delta)),
            teacher_win_rate=float(np.mean(delta < 0)), teacher_worse_case_count=int(np.sum(delta > 0)),
            paired_mean_delta_percentile_95ci=np.quantile(boot, [.025, .975]).tolist(),
            worse_cases=[dict(case_id=row['case_id'], teacher_minus_student=row['teacher_minus_student_' + metric])
                for row in sorted(rows, key=lambda row: row['teacher_minus_student_' + metric], reverse=True)
                if row['teacher_minus_student_' + metric] > 0])
    return result


def weighted_losses(parts):
    return dict(student_measurement=sum(FROZEN_LOSS[key] * parts[key] for key in ('hold', 'cross', 'observed')),
                teacher_measurement=sum(FROZEN_LOSS[key] * parts[key] for key in ('hold_teacher', 'observed_teacher')),
                consistency=FROZEN_LOSS['teacher'] * parts['teacher'])


def vector_geometry(vectors):
    s, t, d = (vectors[key].double() for key in ('student_measurement', 'teacher_measurement', 'consistency'))
    def norm(v): return float(v.norm())
    def cosine(a, b):
        n = norm(a) * norm(b)
        return None if n == 0 else float(torch.dot(a, b)) / n
    measurement = s + t
    m = norm(measurement)
    return dict(student_norm=norm(s), teacher_norm=norm(t), consistency_norm=norm(d), measurement_sum_norm=m,
        cosine_student_teacher=cosine(s, t), cosine_student_consistency=cosine(s, d),
        cosine_teacher_consistency=cosine(t, d), cosine_measurement_consistency=cosine(measurement, d),
        consistency_to_student_norm=None if norm(s) == 0 else norm(d)/norm(s),
        consistency_to_teacher_norm=None if norm(t) == 0 else norm(d)/norm(t),
        consistency_to_measurement_sum_norm=None if m == 0 else norm(d)/m)


def gradient_diagnostic(model, batch):
    assert_no_gt(batch); assert_rich_union(batch)
    total, parts, volumes, _ = objective(model, batch)
    losses = weighted_losses(parts)
    if not torch.allclose(total, sum(losses.values())):
        raise ValueError('Diagnostic decomposition differs from frozen training objective')
    ca, cb, ct = volumes
    ga, gb, gt = torch.autograd.grad(losses['consistency'], (ca, cb, ct), allow_unused=True, retain_graph=True)
    gm, = torch.autograd.grad(losses['teacher_measurement'], ct, retain_graph=True)
    branch = dict(teacher_consistency_output_gradient_zero=gt is None or bool((gt == 0).all()),
        student_A_gradient_dot_error=float((ga.detach() * (ca.detach() - ct.detach())).sum()),
        student_B_gradient_dot_error=float((gb.detach() * (cb.detach() - ct.detach())).sum()),
        teacher_measurement_output_gradient_norm=float(gm.norm()))
    parameters = [p for p in model.parameters() if p.requires_grad]
    vectors = {}
    for i, (name, loss) in enumerate(losses.items()):
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=i < len(losses) - 1)
        vectors[name] = torch.cat([torch.zeros_like(p).reshape(-1).detach().cpu() if g is None else g.detach().reshape(-1).cpu()
                                   for p, g in zip(parameters, gradients)]).double()
        if not torch.isfinite(vectors[name]).all():
            raise FloatingPointError('Nonfinite unclipped shared-parameter gradient')
    if any(p.grad is not None for p in parameters):
        raise AssertionError('Diagnostic must not accumulate parameter .grad')
    return dict(**vector_geometry(vectors), **branch, **{name+'_weighted_loss': float(loss.detach()) for name, loss in losses.items()}), vectors
