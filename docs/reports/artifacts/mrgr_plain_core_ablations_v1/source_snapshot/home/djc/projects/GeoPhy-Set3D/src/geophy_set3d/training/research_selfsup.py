"""Independent, measurement-only training protocol for the scheme A research models.

The frozen historical selfsup validator and its experiment files are never changed.
Synthetic runs are engineering checks, not evidence for a real-data acceptance gate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from geophy_set3d.data.measurement_only import (
    DisjointABHSampler, DualMeasurementDataset, assert_no_gt, dual_measurement_collate,
)
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.selfsup import (
    FROZEN_LOSS, assert_disjoint, consistency_discrepancy, measurement_discrepancy,
    move_dual_batch, projection_scores, selection_metric, zero_volume_baseline,
)
from geophy_set3d.training.utils import (
    atomic_checkpoint, restore_rng, rng_state, seed_everything, write_json,
)

PROTOCOL = 'scheme_a_measurement_only_v1'
TRAIN = dict(split='train', geometry_pool='train', a_views=[3, 5], b_views=[3, 5], heldout_views=2)
VALIDATION = dict(split='val', geometry_pool='train', a_views=[4, 4], b_views=[4, 4], heldout_views=4)


def validate_research_config(config):
    """Fail closed on design YAML and changes to the shared scientific protocol."""
    if config.get('status') != 'executable' or config.get('protocol') != PROTOCOL:
        raise ValueError('Research config must be executable; design_template_not_executable is forbidden')
    stage = config.get('stage')
    if stage not in ('synthetic_smoke', 'smoke', 'gate', 'formal'):
        raise ValueError('Unknown research stage')
    if config['train'] != TRAIN or config['validation'] != VALIDATION or config['loss'] != FROZEN_LOSS:
        raise ValueError('Freeze Pool A disjoint A/B/H and the historical external loss protocol')
    if config['normalization'] != dict(method='linear_p99_5', statistics_file='train_statistics.json'):
        raise ValueError('Only fixed train Pool A p99.5 linear normalization is supported')
    model, training = config['model'], config['training']
    if model.get('name') != 'set_unrolled3d' or model.get('variant') not in ('A0', 'A1', 'A2', 'A3', 'A-control'):
        raise ValueError('Unknown research model/variant')
    steps = model.get('steps')
    if isinstance(steps, bool) or not isinstance(steps, int) or (steps != 0 if model['variant'] == 'A0' else steps not in (1, 2, 4, 6)):
        raise ValueError('A0 requires K=0; A1/A2/A3 require K in 1/2/4/6')
    if stage == 'formal' and steps != (0 if model['variant'] == 'A0' else 4):
        raise ValueError('First formal protocol fixes K=4 (A0=0)')
    if training['precision'] != 'fp32' or training['gradient_clip_norm'] != 1.0 or training['early_stopping_patience'] is not None:
        raise ValueError('Research starts in FP32, clip norm 1, without early stopping')
    if (training['batch_size'], training['gradient_accumulation']) not in ((2, 1), (1, 2)):
        raise ValueError('Exactly two effective cases per update are required, except the final tail')
    if training['seed'] not in (20261011, 20261012, 20261013):
        raise ValueError('Use a preregistered research seed')
    if not isinstance(training['max_epochs'], int) or training['max_epochs'] < 1:
        raise ValueError('max_epochs must be positive')
    if config['optimizer'] != dict(name='adamw', lr=2e-4, weight_decay=1e-4):
        raise ValueError('The shared AdamW starting protocol is fixed')
    if config['scheduler'] != dict(name='cosine', T_max=training['max_epochs']):
        raise ValueError('Cosine T_max must match complete run length')
    loader = config['loader']
    if loader['num_workers'] < 0 or (loader['num_workers'] and platform.system() != 'Linux'):
        raise ValueError('Non-Linux development requires num_workers=0')
    dataset = config['dataset']
    if dataset['kind'] not in ('projection_pool', 'synthetic'):
        raise ValueError('Unknown dataset kind')
    if (stage == 'synthetic_smoke') != (dataset['kind'] == 'synthetic'):
        raise ValueError('Synthetic data must be labelled synthetic_smoke')
    if stage == 'formal':
        if training['max_epochs'] != 300 or dataset.get('train_limit') is not None or dataset.get('val_limit') is not None:
            raise ValueError('Formal runs require all cases and 300 epochs')
    if stage in ('smoke', 'synthetic_smoke') and (dataset.get('train_limit') != 2 or training['max_epochs'] != 20):
        raise ValueError('Smoke is two cases and twenty optimizer updates')
    if stage == 'gate' and dataset.get('train_limit') != 16:
        raise ValueError('The new research gate has exactly sixteen training cases')
    if stage == 'gate':
        budget = config.get('budget', {})
        if not 1 <= budget.get('max_updates', 0) <= 2000 or not 0 < budget.get('max_seconds', 0) <= 1800:
            raise ValueError('Gate requires an explicit budget <=2000 updates and <=1800 seconds')
    if model.get('preconditioner', 'geometry') not in ('geometry', 'none'):
        raise ValueError('Unknown preconditioner ablation')
    if stage != 'synthetic_smoke':
        if model.get('samples', 96) != 96 or model.get('lift_resolution', 32) != 32 or model.get('output_resolution', 64) != 64:
            raise ValueError('Real-data protocol requires Q96, lift32, C64')
    if float(model.get('y_scale', FROZEN_LOSS['measurement_scale'])) != FROZEN_LOSS['measurement_scale']:
        raise ValueError('Model and frozen measurement losses must use the same physical scale')
    return config


def load_research_config(path):
    return validate_research_config(yaml.safe_load(Path(path).read_text()))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class SyntheticMeasurementDataset(Dataset):
    """Analytic constant-field line measurements; never stores or reads a GT volume."""
    def __init__(self, config, validation=False):
        from geophy_set3d.geometry.tdlas import make_geometry
        self.split = 'val' if validation else 'train'
        self.records = [dict(case_id=f'synthetic_{self.split}_{i}', sample_id=f'{self.split}_{i}') for i in range(2)]
        self.geometry = make_geometry(dict(azimuth_deg=[0., 180.], radius_m=[6., 6.], height_m=[.3, .3],
            target_m=[.3, .3, .3], projection_hw=[8, 8], horizontal_fan_deg=10., vertical_scan_deg=8.), 16, 41)
        self.station_count = 16
        self.y_ref = FROZEN_LOSS['measurement_scale']
        settings = config['validation'] if validation else config['train']
        self.sampler = DisjointABHSampler(tuple(settings['a_views']), tuple(settings['b_views']), settings['heldout_views'], config['training']['seed'])
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        row = self.records[index]
        ids = self.sampler.sample(self.station_count, epoch=self.epoch, sample_index=index, case_id=row['case_id'])
        concentration = (.01, .03)[index]
        chi = (concentration / 16.043) / (concentration / 16.043 + (1 - concentration) / 28.965)
        result = {}
        for label, indices in zip(('A', 'B', 'H'), ids):
            branch = dict(**row, split=self.split, geometry_pool='train', projection_scale_ppm_m=torch.tensor(self.y_ref, dtype=torch.float64))
            for key, array in self.geometry.items():
                if key in ('horizontal_angle_deg', 'vertical_angle_deg'):
                    branch[key] = torch.tensor(array, dtype=torch.float32)
                    continue
                value = torch.tensor(array[indices], dtype=torch.bool if key == 'valid_ray' else torch.float32)
                if key in ('valid_ray', 'path_length_m', 't_near', 't_far'):
                    value = value[:, None]
                branch['input_' + key] = value
            raw = branch['input_path_length_m'] * (1e6 * chi)
            branch.update(input_station_indices=torch.tensor(indices), input_projection_raw=raw, input_projection=raw / self.y_ref)
            # Match the existing collate contract including the empty legacy hold branch.
            for key in list(branch):
                if key.startswith('input_'):
                    branch['hold_' + key[6:]] = branch[key][:0]
            result[label] = branch
        return result


def make_research_dataset(config, validation=False):
    if config['dataset']['kind'] == 'synthetic':
        return SyntheticMeasurementDataset(config, validation)
    settings = config['validation'] if validation else config['train']
    return DualMeasurementDataset(config['dataset']['root'], split=settings['split'], geometry_pool='train',
        a_range=settings['a_views'], b_range=settings['b_views'], hold_views=settings['heldout_views'],
        base_seed=config['training']['seed'], normalization='linear_p99_5', statistics_file='train_statistics.json')


def _limited(dataset, limit):
    if limit is None:
        return dataset
    if len(dataset) < limit:
        raise ValueError('Requested fixed case limit exceeds manifest membership')
    return Subset(dataset, range(limit))


def make_research_loader(dataset, config, *, shuffle=False, generator=None):
    settings = config['loader']
    workers = settings['num_workers']
    return DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=shuffle,
        num_workers=workers, pin_memory=settings['pin_memory'], drop_last=False,
        persistent_workers=bool(workers and settings['persistent_workers']),
        collate_fn=dual_measurement_collate, generator=generator,
        **({'multiprocessing_context': 'fork'} if workers else {}))


def research_objective(model, batch, operator, *, diagnostics=False):
    assert_no_gt(batch)
    assert_disjoint(batch)
    for branch in batch.values():
        valid = branch['input_valid_ray'] & branch['view_mask'][:, :, None, None, None]
        if not valid.flatten(1).any(1).all():
            raise ValueError('Each case/branch needs at least one valid ray')
    outputs = [model(batch[key], return_diagnostics=True) if diagnostics else model(batch[key]) for key in ('A', 'B')]
    volumes = tuple(item['volume'] if isinstance(item, dict) else item for item in outputs)
    if any(not torch.isfinite(c).all() or (c < 0).any() or (c > 1).any() for c in volumes):
        raise FloatingPointError('Model output must be finite mass fraction in [0,1]')
    # Each branch geometry is independent of the reconstructed concentration.
    # Reuse the inner stencil, or prepare it once, across both outer fields.
    stencils = {}
    for index, key in enumerate(('A', 'B', 'H')):
        existing = outputs[index].get('stencil') if index < 2 and isinstance(outputs[index], dict) else None
        stencils[key] = existing if existing is not None else operator.prepare(batch[key], volumes[0].shape[-1])
    projections = {name: {key: operator(c, stencils[key]) for key in ('A', 'B', 'H')} for name, c in zip(('C_A', 'C_B'), volumes)}
    discrepancy = lambda name, key: measurement_discrepancy(projections[name][key], batch[key])
    parts = dict(hold=.5 * (discrepancy('C_A', 'H') + discrepancy('C_B', 'H')),
        cross=.5 * (discrepancy('C_A', 'B') + discrepancy('C_B', 'A')),
        observed=.5 * (discrepancy('C_A', 'A') + discrepancy('C_B', 'B')),
        consistency=consistency_discrepancy(*volumes))
    loss = sum(FROZEN_LOSS[key] * value for key, value in parts.items())
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite research objective')
    return loss, parts, volumes, projections, outputs


def _plain(value):
    """Copy nested diagnostic tensors once per device/dtype, preserving values.

    Step diagnostics contain many tiny tensors. Transferring each separately
    synchronizes CUDA once per scalar; packing detached leaves avoids those
    repeated barriers without retaining their autograd graphs or casting them.
    """
    groups, converted = {}, {}

    def collect(item):
        if isinstance(item, torch.Tensor):
            groups.setdefault((item.device, item.dtype), []).append(item)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    def rebuild(item):
        if isinstance(item, torch.Tensor):
            return converted[id(item)]
        if isinstance(item, dict):
            return {key: rebuild(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [rebuild(child) for child in item]
        return item

    collect(value)
    for tensors in groups.values():
        packed = torch.cat([item.detach().reshape(-1) for item in tensors]).cpu()
        offset = 0
        for item in tensors:
            size = item.numel()
            converted[id(item)] = packed[offset:offset+size].reshape(item.shape).tolist()
            offset += size
    return rebuild(value)


def run_research_epoch(model, loader, config, operator, optimizer=None, *, snapshot_path=None,
                       max_updates=None, deadline=None):
    """Average losses by cases; normalize every accumulation group by actual count."""
    training = optimizer is not None
    model.train(training)
    totals = dict.fromkeys(('total', 'hold', 'cross', 'observed', 'consistency'), 0.)
    scores = {key: [] for key in ('heldout', 'cross', 'observed')}
    count = group_cases = updates = 0
    norms, moments, examples, physical = [], [], {}, []
    began = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, raw in enumerate(loader):
            batch = move_dual_batch(raw, config['training']['device'])
            loss, parts, volumes, projections, outputs = research_objective(model, batch, operator, diagnostics=True)
            if snapshot_path is not None and step == 0:
                snapshot_path = Path(snapshot_path)
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(case_id=batch['A']['case_id'][0], C0=outputs[0]['C0'][0].detach().cpu(),
                    Ck=[record['volume'][0].detach().cpu() for record in outputs[0]['steps']],
                    final=volumes[0][0].detach().cpu(),
                    heldout_prediction=projections['C_A']['H'][0].detach().cpu(),
                    heldout_reference=batch['H']['input_projection_raw'][0].detach().cpu()), snapshot_path)
            n = len(batch['A']['case_id'])
            count += n
            norm = None
            if training:
                (loss * n).backward()
                group_cases += n
                if (step + 1) % config['training']['gradient_accumulation'] == 0 or step + 1 == len(loader):
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_cases)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    group_cases = 0
            detached_projections = {name: {key: value.detach() for key, value in predictions.items()}
                                    for name, predictions in projections.items()}
            for key, values in projection_scores(detached_projections, batch).items():
                scores[key].extend(values)
            batch_moments = []
            for volume in volumes:
                for prediction in volume.detach():
                    batch_moments.append((prediction.min(), prediction.max(), prediction.mean(),
                        prediction.square().mean(), (prediction <= 1e-5).float().mean(), (prediction >= .999).float().mean()))
            for i, case_id in enumerate(batch['A']['case_id']):
                if len(examples) < 8:
                    examples[case_id] = volumes[0][i].detach().cpu()
            batch_physical = []
            for output in outputs:
                if isinstance(output, dict):
                    scalar_keys = ('E', 'update_norm', 'physical_step_norm', 'prior_norm', 'tau', 'feedback_norm', 'signal_saturation_fraction',
                        'physical_clip_fraction', 'prior_clip_fraction', 'near_zero_fraction', 'near_one_fraction')
                    batch_physical.append(dict(steps=[{key: record[key] for key in scalar_keys if key in record}
                        for record in output.get('steps', [])]))
                    if 'coverage' in output:
                        batch_physical[-1]['coverage_zero_fraction'] = (output['coverage'] == 0).float().mean()
            diagnostics = _plain(dict(loss=dict(total=loss, **parts), norm=norm,
                moments=batch_moments, physical=batch_physical))
            for key, value in diagnostics['loss'].items():
                totals[key] += value * n
            if norm is not None:
                norms.append(diagnostics['norm'])
            moments.extend(diagnostics['moments'])
            physical.extend(diagnostics['physical'])
            if training and group_cases == 0 and ((max_updates is not None and updates >= max_updates)
                                                  or (deadline is not None and time.monotonic() >= deadline)):
                break
            if not training and deadline is not None and time.monotonic() >= deadline:
                break
    if not count:
        raise ValueError('An epoch cannot be empty')
    mean = sum(v[2] for v in moments) / len(moments)
    ordered = [examples[k] for k in sorted(examples)]
    return dict(loss={k: v / count for k, v in totals.items()},
        measurement={k: {metric: sum(v[metric] for v in values) / len(values) for metric in ('relative_l1', 'relative_l2')} for k, values in scores.items()},
        output=dict(min=min(v[0] for v in moments), max=max(v[1] for v in moments), mean=mean,
            std=math.sqrt(max(0., sum(v[3] for v in moments) / len(moments) - mean**2)),
            fraction_near_zero=sum(v[4] for v in moments) / len(moments), fraction_near_one=sum(v[5] for v in moments) / len(moments),
            case_to_case_mean_absolute_difference=sum(float((a-b).abs().mean()) for a, b in zip(ordered, ordered[1:])) / max(1, len(ordered)-1)),
        gradient_norm_max=max(norms, default=0.), gradient_norms=norms, sample_count=count, optimizer_updates=updates,
        physical=physical, gt_used=False, complete_epoch=count == len(loader.dataset), seconds=time.perf_counter()-began)


def _sampling_rows(dataset, epoch, split):
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
    for index in indices:
        row = base.records[index]
        ids = base.sampler.sample(base.station_count, epoch=epoch, sample_index=index, case_id=row['case_id'])
        yield dict(epoch=epoch, split=split, case_id=row['case_id'], sample_id=row['sample_id'],
                   **{key: values.tolist() for key, values in zip(('A', 'B', 'H'), ids)})


def _provenance(config, train, val):
    root = Path(__file__).resolve().parents[3]
    source_files = sorted((root / 'src').rglob('*.py')) + [root / 'scripts/train_research_selfsup.py']
    code = {str(path.relative_to(root)): _sha(path) for path in source_files}
    data = dict(kind=config['dataset']['kind'])
    if data['kind'] == 'projection_pool':
        dataset_root = Path(config['dataset']['root']).resolve()
        files = {dataset_root / name for name in ('source_index.jsonl', 'train_statistics.json', 'geometry/train_geometry.npz')}
        for dataset in (train, val):
            base = dataset.dataset if isinstance(dataset, Subset) else dataset
            indices = dataset.indices if isinstance(dataset, Subset) else range(len(base))
            for index in indices:
                path = (dataset_root / base.records[index]['train_projection_path']).resolve()
                if not path.is_relative_to(dataset_root):
                    raise ValueError('Projection path escapes the dataset root')
                files.add(path)
        data['files'] = {str(path.relative_to(dataset_root)): _sha(path) for path in sorted(files)}
    else:
        data['generator'] = 'analytic_constant_mass_fraction_0.01_0.03_geometry_seed41_v1'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    return dict(protocol=PROTOCOL, config_sha256=_canonical(config), data_sha256=_canonical(data),
                source_sha256=_canonical(code), git_commit=commit, code=code, data=data,
                train_case_ids=[r['case_id'] for r in _sampling_rows(train, 0, 'train')],
                val_case_ids=[r['case_id'] for r in _sampling_rows(val, 0, 'val')])


def validate_resume(checkpoint, config, provenance):
    if checkpoint.get('boundary') != 'completed_epoch':
        raise ValueError('Only completed epoch boundaries can be resumed')
    if checkpoint['config'] != config:
        raise ValueError('Resume config differs; use a new run directory')
    for key in ('protocol', 'config_sha256', 'data_sha256', 'source_sha256', 'git_commit'):
        if checkpoint['provenance'][key] != provenance[key]:
            raise ValueError(f'Resume provenance changed: {key}; use a new run directory')
    expected_next = checkpoint['epoch'] + 1
    if checkpoint['sampling_epoch_next'] != expected_next or checkpoint['scheduler']['last_epoch'] != expected_next:
        raise ValueError('Resume sampling/scheduler epoch is inconsistent')
    if checkpoint['rng'].keys() != {'python', 'numpy', 'torch', 'cuda', 'loader'}:
        raise ValueError('Incomplete RNG state')


def _validate_acceptance(config, provenance):
    """Formal runs require recorded, real-data smoke and sixteen-case gate evidence."""
    if config['stage'] != 'formal':
        return
    for kind in ('smoke', 'gate'):
        directory = Path(config.get('acceptance', {}).get(kind + '_run', ''))
        if not (directory / 'acceptance_result.json').is_file():
            raise ValueError(f'Formal training requires a completed real-data {kind} run')
        result = json.loads((directory / 'acceptance_result.json').read_text())
        prior_config = yaml.safe_load((directory / 'config_resolved.yaml').read_text())
        prior_provenance = json.loads((directory / 'provenance.json').read_text())
        audit = json.loads((directory / 'gt_read_audit.json').read_text())
        if not result.get('passed') or result.get('stage') != kind or result.get('synthetic'):
            raise ValueError(f'{kind} evidence failed or is synthetic')
        if prior_config['model'] != config['model'] or prior_config['dataset']['root'] != config['dataset']['root']:
            raise ValueError(f'{kind} model/data differs from the formal run')
        if prior_provenance['source_sha256'] != provenance['source_sha256'] or prior_provenance['git_commit'] != provenance['git_commit']:
            raise ValueError(f'{kind} evidence used a different implementation')
        current_files = provenance['data'].get('files', {})
        if any(current_files.get(path) != checksum for path, checksum in prior_provenance['data'].get('files', {}).items()):
            raise ValueError(f'{kind} evidence used changed data/geometry/statistics content')
        if any(audit.get(key, 0) for key in ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts')):
            raise ValueError(f'{kind} evidence has a forbidden read attempt')


def measurement_dependence_probe(model, branch):
    """Change intensity at fixed geometry; disallow a geometry-only collapsed solution."""
    altered = dict(branch)
    altered['input_projection'] = branch['input_projection'] * 1.2
    altered['input_projection_raw'] = branch['input_projection_raw'] * 1.2
    model.eval()
    with torch.no_grad():
        original, changed = model(branch), model(altered)
    delta = float((original-changed).abs().mean())
    return dict(intervention='multiply valid measurements by 1.2; identical geometry and masks',
                mean_absolute_output_change=delta, passed=bool(math.isfinite(delta) and delta > 1e-8))


def train_research(config, *, resume=False, stop_after_epochs=None):
    """Train a new run or resume exactly at an epoch boundary.

    stop_after_epochs supports controlled interruption/replay checks. It never
    changes the resolved config or marks an unfinished run as frozen.
    """
    began_run = time.monotonic()
    from geophy_set3d.models.set_unrolled3d import build_set_unrolled_model
    from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator
    validate_research_config(config)
    config = copy.deepcopy(config)
    budget = config.get('budget', {}) if config['stage'] == 'gate' else {}
    deadline = began_run + budget['max_seconds'] if budget else None
    run = Path(config['experiment']['output_root']) / config['experiment']['name']
    checkpoints = run / 'checkpoints'
    if run.exists() and not resume:
        raise FileExistsError('Run exists; use explicit resume or a new run name')
    if resume and not (checkpoints / 'last.pt').exists():
        raise FileNotFoundError('Resume requires checkpoints/last.pt')
    checkpoints.mkdir(parents=True, exist_ok=True)
    seed_everything(config['training']['seed'])
    torch.set_num_threads(config['training'].get('cpu_threads', 2))
    device = config['training']['device']
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable')
    guard = GTReadGuard(config['dataset']['root'], checkpoints) if config['dataset']['kind'] == 'projection_pool' else nullcontext()
    try:
        with guard:
            train_base = make_research_dataset(config)
            val_base = make_research_dataset(config, True)
            if config['dataset']['kind'] == 'projection_pool':
                if len(train_base) != 166 or len(val_base) != 36:
                    raise ValueError('Expected frozen 166/36 train/val case membership')
                if train_base.y_ref != FROZEN_LOSS['measurement_scale'] or val_base.y_ref != train_base.y_ref:
                    raise ValueError('Measurement statistics scale changed')
            train = _limited(train_base, config['dataset'].get('train_limit'))
            val = _limited(val_base, config['dataset'].get('val_limit'))
            provenance = _provenance(config, train, val)
            _validate_acceptance(config, provenance)
            generator = torch.Generator().manual_seed(config['training']['seed'])
            train_loader = make_research_loader(train, config, shuffle=True, generator=generator)
            # Validation has an independent generator so it never changes training ordering.
            val_loader = make_research_loader(val, config, generator=torch.Generator().manual_seed(0))
            model = build_set_unrolled_model(config).to(device)
            # Inner updates and outer losses use identical ray geometry. Share
            # its bounded caches rather than building/transferring two copies.
            operator = getattr(model, 'operator', None)
            if operator is None:  # Lightweight trainer replay test models.
                operator = ExplicitRayOperator(samples_per_ray=config['model'].get('samples', 96), ray_chunk=config['model'].get('ray_chunk', 256))
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['scheduler']['T_max'])
            start = global_step = 0
            best = float('inf')
            if resume:
                checkpoint = torch.load(checkpoints / 'last.pt', map_location=device, weights_only=False)
                validate_resume(checkpoint, config, provenance)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                restore_rng(checkpoint['rng'], generator)
                start, global_step, best = checkpoint['epoch'] + 1, checkpoint['global_step'], checkpoint['best_val_metric']
                # Discard logs written after the last atomic checkpoint, e.g. a crash during commit.
                for name in ('metrics.jsonl', 'sampling_manifest.jsonl'):
                    path = run / name
                    if path.exists():
                        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
                        rows = [r for r in rows if r['epoch'] < start or r.get('split') == 'val']
                        path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
            else:
                (run / 'config_resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
                write_json(run / 'provenance.json', provenance)
                write_json(run / 'environment.json', dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                    numpy=np.__version__, cuda=torch.version.cuda, device=device,
                    gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
                    precision='fp32', git_commit=provenance['git_commit'],
                    git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[3], text=True).strip()),
                    parameter_count=sum(p.numel() for p in model.parameters())))
                with (run / 'sampling_manifest.jsonl').open('w') as stream:
                    for row in _sampling_rows(val, 0, 'val'):
                        stream.write(json.dumps(row) + '\n')
                val_base.set_epoch(0)
                write_json(run / 'zero_volume_baseline.json', zero_volume_baseline(val_loader, config))
                initial = run_research_epoch(model, val_loader, config, operator, deadline=deadline)
                write_json(run / 'initial_validation.json', initial)
            end = config['training']['max_epochs']
            if stop_after_epochs is not None:
                end = min(end, stop_after_epochs)
            budget_exhausted = False
            completed_epochs = start
            executed_updates = global_step
            for epoch in range(start, end):
                if budget and (global_step >= budget['max_updates'] or time.monotonic() >= deadline):
                    budget_exhausted = True
                    break
                train_base.set_epoch(epoch)
                val_base.set_epoch(0)
                with (run / 'sampling_manifest.jsonl').open('a') as stream:
                    for row in _sampling_rows(train, epoch, 'train'):
                        stream.write(json.dumps(row) + '\n')
                if str(device).startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats(device)
                lr = optimizer.param_groups[0]['lr']
                training = run_research_epoch(model, train_loader, config, operator, optimizer,
                    max_updates=budget['max_updates']-global_step if budget else None, deadline=deadline)
                executed_updates += training['optimizer_updates']
                if not training['complete_epoch']:
                    write_json(run / 'budget_partial_epoch.json', dict(epoch=epoch, result=training,
                        checkpoint_policy='Partial epoch never checkpointed; last.pt remains a completed boundary'))
                    budget_exhausted = True
                    break
                snapshot_path = run / 'diagnostics' / f'epoch_{epoch:03d}.pt' if epoch in (0, config['training']['max_epochs']-1) else None
                validation = run_research_epoch(model, val_loader, config, operator, snapshot_path=snapshot_path, deadline=deadline)
                if not validation['complete_epoch']:
                    write_json(run / 'budget_partial_epoch.json', dict(epoch=epoch, train=training, validation=validation,
                        checkpoint_policy='Partial validation never checkpointed; last.pt remains a completed boundary'))
                    budget_exhausted = True
                    break
                global_step += training['optimizer_updates']
                score = selection_metric(validation)
                improved = score < best
                best = min(best, score)
                # Save scheduler after this completed epoch: resumed next lr is exact.
                scheduler.step()
                row = dict(epoch=epoch, global_step=global_step, learning_rate=lr, next_learning_rate=scheduler.get_last_lr()[0],
                    train=training, validation=validation, selection_metric=score,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0)
                with (run / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                payload = dict(boundary='completed_epoch', epoch=epoch, sampling_epoch_next=epoch+1, global_step=global_step,
                    model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(generator),
                    best_val_metric=best, config=config, provenance=provenance)
                if improved:
                    atomic_checkpoint(checkpoints / 'best.pt', payload)
                atomic_checkpoint(checkpoints / 'last.pt', payload)
                completed_epochs = epoch + 1
                print(json.dumps(dict(run=str(run), epoch=epoch+1, global_step=global_step, loss=training['loss']['total'], heldout_relative_l1=score)), flush=True)
            completed = completed_epochs == config['training']['max_epochs']
            if budget_exhausted:
                audit = guard.summary()
                write_json(run / 'acceptance_result.json', dict(stage='gate', synthetic=False, passed=False,
                    completed=False, completed_epochs=completed_epochs, stopping_reason='budget_exhausted',
                    budget=budget, elapsed_seconds=time.monotonic()-began_run, decision='NO-GO: incomplete gate'))
                write_json(run / 'training_summary.json', dict(completed=False, completed_epochs=completed_epochs,
                    optimizer_updates=executed_updates, committed_global_step=global_step,
                    stopping_reason='budget_exhausted', gt_access=audit))
            if completed:
                best_checkpoint = torch.load(checkpoints / 'best.pt', map_location='cpu', weights_only=False)
                audit = guard.summary() if isinstance(guard, GTReadGuard) else dict(gt_open_attempts=0, gt_file_opened=False, synthetic=True)
                rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line]
                baseline = json.loads((run / 'zero_volume_baseline.json').read_text())
                initial = json.loads((run / 'initial_validation.json').read_text())
                final = rows[-1]['validation']
                probe_batch = move_dual_batch(next(iter(val_loader)), device)
                probe = measurement_dependence_probe(model, probe_batch['A'])
                write_json(run / 'measurement_dependence.json', probe)
                criteria = dict(finite=True,
                    heldout_better_than_zero=best < baseline['relative_l1'],
                    heldout_improves_initial=best < selection_metric(initial),
                    nonzero_gradient=any(row['train']['gradient_norm_max'] > 0 for row in rows),
                    output_changes_across_cases=final['output']['case_to_case_mean_absolute_difference'] > 1e-7,
                    measurement_dependence=probe['passed'],
                    no_gt_reads=not audit['gt_open_attempts'],
                    no_forbidden_reads=not any(audit.get(key, 0) for key in
                        ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts')),
                    expected_updates=(global_step == 20) if config['stage'] in ('smoke', 'synthetic_smoke') else True)
                passed = all(criteria.values())
                write_json(run / 'acceptance_result.json', dict(stage=config['stage'], synthetic=config['stage']=='synthetic_smoke',
                    passed=passed, criteria=criteria, best_heldout_relative_l1=best,
                    zero_heldout_relative_l1=baseline['relative_l1'], initial_heldout_relative_l1=selection_metric(initial)))
                frozen = dict(completed=True, completed_epochs=end, stage=config['stage'], protocol=PROTOCOL,
                    synthetic=config['dataset']['kind'] == 'synthetic', formal_gate_claim=False,
                    selection='heldout measurement relative_l1 only', best_epoch=best_checkpoint['epoch']+1,
                    best_val_metric=best_checkpoint['best_val_metric'], global_step=global_step,
                    checkpoint=str((checkpoints / 'best.pt').resolve()), sha256=_sha(checkpoints / 'best.pt'),
                    selection_metric='mean held-out measurement relative L1', volumetric_gt_used=False,
                    formal_evaluation_allowed=config['stage']=='formal' and end==300 and criteria['no_forbidden_reads'],
                    best_sha256=_sha(checkpoints / 'best.pt'), last_sha256=_sha(checkpoints / 'last.pt'),
                    provenance_sha256=_sha(run / 'provenance.json'), gt_used_for_training_or_selection=False)
                write_json(run / 'freeze.json', frozen)
                write_json(run / 'frozen_checkpoint.json', frozen)
                write_json(run / 'training_summary.json', dict(completed=True, completed_epochs=end,
                    optimizer_updates=global_step, best_val_metric=best, gt_access=audit,
                    scientific_gate_passed=passed if config['stage'] != 'synthetic_smoke' else False))
            return run
    except BaseException as error:
        # Preserve a concrete failure conclusion even for OOM/interrupt; never fabricate a completed boundary.
        audit = guard.summary() if isinstance(guard, GTReadGuard) else dict(synthetic=True, gt_file_opened=False, gt_open_attempts=0)
        write_json(run / 'failure.json', dict(completed=False, stage=config['stage'],
            error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic()-began_run))
        write_json(run / 'training_summary.json', dict(completed=False, stage=config['stage'],
            completed_epochs=locals().get('completed_epochs', 0), stopping_reason=type(error).__name__, gt_access=audit))
        write_json(run / 'acceptance_result.json', dict(stage=config['stage'], synthetic=config['stage']=='synthetic_smoke',
            completed=False, passed=False, stopping_reason=type(error).__name__))
        raise
    finally:
        if isinstance(guard, GTReadGuard):
            write_json(run / 'gt_read_audit.json', guard.summary())
        else:
            write_json(run / 'gt_read_audit.json', dict(policy='Synthetic analytic measurements; no external data files or volumetric labels', gt_file_opened=False, gt_open_attempts=0, synthetic=True))
