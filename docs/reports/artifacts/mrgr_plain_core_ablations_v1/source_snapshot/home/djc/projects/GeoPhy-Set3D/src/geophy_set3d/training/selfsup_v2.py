"""Frozen RVTC objective on a single shared, unchanged MRGR network."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from geophy_set3d.data.measurement_only import assert_no_gt
from geophy_set3d.data.rich_measurement import RichMeasurementDataset, rich_measurement_collate
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input, Y_SCALE
from . import selfsup as v1
from .gt_read_guard import GTReadGuard
from .utils import load_config, move_batch

FROZEN_LOSS = dict(hold=1., cross=.5, observed=.1, hold_teacher=1., observed_teacher=.1,
                   teacher=.1, beta=.01, relative_l1=.1, relative_epsilon=1e-6,
                   measurement_scale=Y_SCALE)
SELECTION = 'mean sparse held-out measurement relative L1'
KEYS = ('A', 'B', 'H', 'T')


def validate_selfsup_config(config):
    legacy = copy.deepcopy(config)
    legacy['loss'] = v1.FROZEN_LOSS
    v1.validate_selfsup_config(legacy)
    frozen = load_config(Path(__file__).resolve().parents[3] / 'configs/train_selfsup_oari_mrgr_v1.yaml')
    if config['model'] != frozen['model'] or config['loss'] != FROZEN_LOSS:
        raise ValueError('v2 freezes the complete MRGR model configuration and RVTC weights')
    if config['experiment']['name'] != 'selfsup_oari_mrgr_v2_rvtc':
        raise ValueError('Use an independent v2 experiment')
    if (config['training']['batch_size'], config['training']['gradient_accumulation']) != (1, 2):
        raise ValueError('v2 uses batch1 / accumulation2')
    if any(k in config for k in ('checkpoint', 'pretrained', 'teacher_model', 'ema')):
        raise ValueError('v2 permits fresh initialization only, with no separate teacher')


initialize_fresh_model = v1.initialize_fresh_model
measurement_discrepancy = v1.measurement_discrepancy


class V2ReadGuard(GTReadGuard):
    """Keep v1 hooks/shared worker counters; additionally count Pool B attempts."""
    def __init__(self, dataset_root, checkpoint_root):
        super().__init__(dataset_root, checkpoint_root)
        import multiprocessing as mp
        self.pool_b_attempts = mp.Value('i', 0)
        rows = [json.loads(line) for line in (self.root / 'source_index.jsonl').read_text().splitlines() if line.strip()]
        self.pool_b = {str((self.root / row['ood_projection_path']).absolute()) for row in rows}

    def _audit(self, event, args):
        if self.active and event == 'open' and isinstance(args[0], (str, bytes, Path)):
            import os
            path = os.path.abspath(os.fsdecode(args[0]))
            if path in self.pool_b or path.endswith('ood_geometry.npz'):
                with self.pool_b_attempts.get_lock():
                    self.pool_b_attempts.value += 1
        super()._audit(event, args)

    def summary(self):
        return dict(super().summary(), Pool_B_access_attempts=self.pool_b_attempts.value,
                    forbidden_projection_attempts=self.other_npy_attempts.value,
                    external_checkpoint_attempts=self.supervised_checkpoint_attempts.value)


def audit_is_clean(audit):
    return all(audit[key] == 0 for key in ('gt_open_attempts', 'supervised_checkpoint_open_attempts',
                                          'forbidden_projection_attempts', 'Pool_B_access_attempts'))


def make_dataset(config, validation=False):
    settings = config['validation'] if validation else config['train']
    return RichMeasurementDataset(config['dataset']['root'], split=settings['split'],
        geometry_pool=settings['geometry_pool'], a_range=settings['a_views'], b_range=settings['b_views'],
        hold_views=settings['heldout_views'], base_seed=config['training']['seed'],
        normalization=config['normalization']['method'], statistics_file=config['normalization']['statistics_file'])


def make_loader(dataset, config, shuffle=False, generator=None):
    workers = config['loader']['num_workers']
    return DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=shuffle,
        num_workers=workers, pin_memory=config['loader']['pin_memory'],
        persistent_workers=config['loader']['persistent_workers'] and workers > 0,
        collate_fn=rich_measurement_collate, generator=generator,
        **({'multiprocessing_context': 'fork'} if workers else {}))


def move_rich_batch(batch, device):
    assert_no_gt(batch)
    return {key: move_batch(batch[key], device) for key in KEYS}


def assert_rich_union(batch):
    v1.assert_disjoint(batch)
    for i, case_id in enumerate(batch['A']['case_id']):
        if batch['T']['case_id'][i] != case_id:
            raise ValueError('Rich teacher must reconstruct the same plume')
        ids = {key: batch[key]['input_station_indices'][i, batch[key]['view_mask'][i]].tolist()
               for key in KEYS}
        if len(ids['T']) != len(ids['A']) + len(ids['B']) or set(ids['T']) != set(ids['A']) | set(ids['B']):
            raise ValueError('T must be the exact A/B union, without duplicate stations')
        if set(ids['T']) & set(ids['H']):
            raise ValueError('T cannot contain held-out H')


def sample_relative_l1(student, target):
    return ((student - target).abs().flatten(1).mean(1) /
            (target.abs().flatten(1).mean(1) + 1e-6)).mean()


def teacher_discrepancy(ca, cb, ct):
    teacher = ct.detach()
    return .5 * (sample_relative_l1(ca, teacher) + sample_relative_l1(cb, teacher))


def objective(model, batch):
    assert_no_gt(batch)
    assert_rich_union(batch)
    # Three calls to precisely the same network. C_T retains its measurement graph.
    ca, cb, ct = (model(batch[key]) for key in ('A', 'B', 'T'))
    if not all(bool(torch.isfinite(c).all()) for c in (ca, cb, ct)):
        raise FloatingPointError('Nonfinite MRGR output')
    projections = {label: {key: project_input(c, batch[key]) for key in targets}
                   for label, c, targets in (('C_A', ca, ('A', 'B', 'H')),
                                             ('C_B', cb, ('A', 'B', 'H')),
                                             ('C_T', ct, ('H', 'T')))}
    def m(label, key):
        return measurement_discrepancy(projections[label][key], batch[key])
    parts = dict(hold=.5*(m('C_A', 'H') + m('C_B', 'H')),
                 cross=.5*(m('C_A', 'B') + m('C_B', 'A')),
                 observed=.5*(m('C_A', 'A') + m('C_B', 'B')),
                 hold_teacher=m('C_T', 'H'), observed_teacher=m('C_T', 'T'),
                 teacher=teacher_discrepancy(ca, cb, ct))
    total = sum(FROZEN_LOSS[key] * value for key, value in parts.items())
    if not all(bool(torch.isfinite(value)) for value in (total, *parts.values())):
        raise FloatingPointError('Nonfinite RVTC objective')
    return total, parts, (ca, cb, ct), projections


def projection_scores(projections, batch):
    groups = dict(heldout=(('C_A', 'H'), ('C_B', 'H')), cross=(('C_A', 'B'), ('C_B', 'A')),
                  observed=(('C_A', 'A'), ('C_B', 'B')), teacher_heldout=(('C_T', 'H'),),
                  teacher_observed=(('C_T', 'T'),), A_heldout=(('C_A', 'H'),), B_heldout=(('C_B', 'H'),))
    result = {key: [] for key in groups}
    for group, pairs in groups.items():
        for label, key in pairs:
            target = batch[key]
            valid = target['input_valid_ray'].bool() & target['view_mask'][:, :, None, None, None]
            for i in range(len(target['case_id'])):
                result[group].append(relative_projection_metrics(projections[label][key][i],
                                      target['input_projection_raw'][i], valid[i]))
    return result


def selection_metric(validation):
    return validation['measurement']['heldout']['relative_l1']


def output_statistics(moments, examples):
    mean = sum(v[2] for v in moments) / len(moments)
    ordered = [examples[key] for key in sorted(examples)]
    return dict(min=min(v[0] for v in moments), max=max(v[1] for v in moments), mean=mean,
        std=math.sqrt(max(0., sum(v[3] for v in moments)/len(moments) - mean*mean)),
        fraction_near_one=sum(v[4] for v in moments)/len(moments),
        fraction_near_zero=sum(v[5] for v in moments)/len(moments),
        case_to_case_mean_absolute_difference=sum(float((a-b).abs().mean())
            for a, b in zip(ordered, ordered[1:])) / max(len(ordered)-1, 1))


def run_epoch(model, loader, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0. for key in ('total', 'hold', 'cross', 'observed', 'hold_teacher', 'observed_teacher', 'teacher')}
    scores = {}; moments = {key: [] for key in ('A', 'B', 'T')}; examples = {key: {} for key in moments}
    disagreements = {key: 0. for key in ('D_A_T', 'D_B_T', 'D_A_B')}
    count = 0; pairings = []; began = time.perf_counter(); gradient_max = 0.
    accumulation = config['training']['gradient_accumulation']
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, raw in enumerate(loader):
            batch = move_rich_batch(raw, config['training']['device'])
            loss, parts, volumes, projections = objective(model, batch)
            n = len(volumes[0]); count += n
            if training:
                start = (step // accumulation) * accumulation
                stop = min(start + accumulation, len(loader))
                group_samples = min(stop * loader.batch_size, len(loader.dataset)) - start * loader.batch_size
                (loss * n / group_samples).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'], error_if_nonfinite=True)
                    gradient_max = max(gradient_max, float(norm))
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for key, value in dict(total=loss, **parts).items():
                totals[key] += float(value.detach()) * n
            for key, rows in projection_scores(projections, batch).items():
                scores.setdefault(key, []).extend(rows)
            ca, cb, ct = (c.detach() for c in volumes)
            for key, student, target in (('D_A_T', ca, ct), ('D_B_T', cb, ct), ('D_A_B', ca, cb)):
                disagreements[key] += float(sample_relative_l1(student, target)) * n
            for key, c in zip(('A', 'B', 'T'), volumes):
                for i, pred in enumerate(c.detach()):
                    mean = float(pred.mean()); std = float(pred.std(unbiased=False))
                    moments[key].append((float(pred.min()), float(pred.max()), mean, std*std+mean*mean,
                                         float((pred >= .999).float().mean()), float((pred <= 1e-5).float().mean())))
                    case_id = batch['A']['case_id'][i]
                    if len(examples[key]) < 8 and case_id not in examples[key]:
                        examples[key][case_id] = pred.cpu()
            for i, case_id in enumerate(batch['A']['case_id']):
                pairings.append((case_id, *[batch[key]['input_station_indices'][i, batch[key]['view_mask'][i]].tolist() for key in KEYS]))
            # Release every sample's three graphs before allocating the next.
            del loss, parts, volumes, projections, ca, cb, ct, batch
    return dict(loss={key: value/count for key, value in totals.items()},
        measurement={key: {metric: sum(v[metric] for v in rows)/len(rows) for metric in ('relative_l1', 'relative_l2')}
                     for key, rows in scores.items()},
        disagreement={key: value/count for key, value in disagreements.items()},
        output=output_statistics(moments['A'] + moments['B'], examples['A']),
        teacher_output=output_statistics(moments['T'], examples['T']), gradient_norm_max=gradient_max,
        case_station_digest=hashlib.sha256(json.dumps(sorted(pairings)).encode()).hexdigest(),
        sample_count=count, gt_used=False, seconds=time.perf_counter()-began)


def zero_volume_baseline(loader, config):
    rows = []
    for raw in loader:
        assert_no_gt(raw)
        h = raw['H']; valid = h['input_valid_ray'].bool() & h['view_mask'][:, :, None, None, None]
        for i in range(len(h['case_id'])):
            rows.append(relative_projection_metrics(torch.zeros_like(h['input_projection_raw'][i]), h['input_projection_raw'][i], valid[i]))
    return {metric: sum(row[metric] for row in rows)/len(rows) for metric in ('relative_l1', 'relative_l2')}


def output_collapsed(stats):
    return (stats['max'] <= 1e-5 or stats['mean'] <= 1e-6 or stats['std'] <= 1e-6 or
            stats['fraction_near_one'] >= .01 or stats['case_to_case_mean_absolute_difference'] <= 1e-6)


def teacher_gradient_diagnostic(model, batch):
    """Audit branch gradients separately, despite shared network parameters."""
    model.eval(); model.zero_grad(set_to_none=True)
    _, parts, (ca, cb, ct), _ = objective(model, batch)
    ga, gb, gt = torch.autograd.grad(parts['teacher'], (ca, cb, ct), allow_unused=True, retain_graph=True)
    gm, = torch.autograd.grad(parts['hold_teacher'], ct)
    result = dict(student_A_gradient_abs_sum=float(ga.abs().sum()), student_B_gradient_abs_sum=float(gb.abs().sum()),
                  teacher_consistency_gradient_abs_sum=0. if gt is None else float(gt.abs().sum()),
                  teacher_measurement_gradient_abs_sum=float(gm.abs().sum()))
    result['teacher_stop_gradient_verified'] = gt is None or bool((gt == 0).all())
    result['passed'] = (result['teacher_stop_gradient_verified'] and result['student_A_gradient_abs_sum'] > 0 and
                        result['student_B_gradient_abs_sum'] > 0 and result['teacher_measurement_gradient_abs_sum'] > 0 and
                        bool(torch.isfinite(gm).all()))
    model.zero_grad(set_to_none=True)
    return result
