"""Isolated scheme B measurement-only protocol; never modifies frozen scheme A.

Training projections query continuous ray midpoints. The 64^3 endpoint field is
used only for the common consistency loss and diagnostics. Formal runs are gated
on real-data smoke/gate evidence from the same source and data identities.
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
from geophy_set3d.data.measurement_only import assert_no_gt
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.research_selfsup import (
    TRAIN, VALIDATION, _canonical, _limited, _sampling_rows, _sha,
    make_research_dataset, make_research_loader, validate_resume,
)
from geophy_set3d.training.selfsup import (
    FROZEN_LOSS, assert_disjoint, consistency_discrepancy,
    measurement_discrepancy, move_dual_batch, projection_scores,
    selection_metric, zero_volume_baseline,
)
from geophy_set3d.training.utils import (
    atomic_checkpoint, restore_rng, rng_state, seed_everything, write_json,
)

PROTOCOL = 'scheme_b_continuous_measurement_only_v1'
ROOT = Path(__file__).resolve().parents[2]
MODEL_SPEC = dict(name='set_conditional_field', feature_channels=32, residual_blocks=3,
                  lift_resolution=32, output_resolution=64, samples=96, fourier_bands=4,
                  query_hidden=64, query_hidden_layers=4, query_chunk=32768,
                  ray_chunk=256, checkpoint_query=True, initialization='train_full_pool_a_pooled_chi_bias_v1',
                  y_scale=FROZEN_LOSS['measurement_scale'], cache_budget_mib=512)


def validate_config(config):
    """Reject design templates and scientific-protocol drift before opening data."""
    if config.get('status') != 'executable' or config.get('protocol') != PROTOCOL:
        raise ValueError('Scheme B needs status=executable and its independent protocol; design templates cannot run')
    stage = config.get('stage')
    if stage not in ('synthetic_smoke', 'smoke', 'gate', 'formal'):
        raise ValueError('Unknown scheme B stage')
    if config['train'] != TRAIN or config['validation'] != VALIDATION or config['loss'] != FROZEN_LOSS:
        raise ValueError('Freeze disjoint Pool A A/B/H subsets and historical per-case loss')
    if config['normalization'] != dict(method='linear_p99_5', statistics_file='train_statistics.json'):
        raise ValueError('Use frozen train Pool A p99.5 statistics')
    if config['model'] != MODEL_SPEC:
        raise ValueError('First B protocol freezes feature32, three blocks, four bands, MLP61/64x4/1, Q96, chunks and checkpointing')
    tr = config['training']
    if tr['precision'] != 'fp32' or tr['gradient_clip_norm'] != 1. or tr['early_stopping_patience'] is not None:
        raise ValueError('Initial B is FP32, clip norm1, without early stopping')
    if (tr['batch_size'], tr['gradient_accumulation']) not in ((1, 2), (2, 1)):
        raise ValueError('Exactly two effective cases per update, except actual final tail')
    if tr['seed'] not in (20261011, 20261012, 20261013):
        raise ValueError('Use preregistered seed')
    if isinstance(tr['max_epochs'], bool) or not isinstance(tr['max_epochs'], int) or tr['max_epochs'] < 1:
        raise ValueError('max_epochs must be positive integer')
    if config['optimizer'] != dict(name='adamw', lr=2e-4, weight_decay=1e-4):
        raise ValueError('Freeze AdamW lr2e-4 weight_decay1e-4')
    if config['scheduler'] != dict(name='cosine', T_max=tr['max_epochs']):
        raise ValueError('Cosine T_max must equal complete run epochs')
    loader = config['loader']
    if loader['num_workers'] != 0 or loader['persistent_workers']:
        raise ValueError('First B implementation fixes workers=0 until separately tested')
    ds = config['dataset']
    if ds['kind'] not in ('projection_pool', 'synthetic') or ((stage == 'synthetic_smoke') != (ds['kind'] == 'synthetic')):
        raise ValueError('Analytic synthetic data must be explicitly synthetic_smoke')
    if stage in ('smoke', 'synthetic_smoke') and (ds.get('train_limit') != 2 or tr['max_epochs'] != 20):
        raise ValueError('Smoke requires two cases and twenty updates')
    if stage == 'gate' and ds.get('train_limit') != 16:
        raise ValueError('Real-data gate requires sixteen cases')
    if stage in ('smoke', 'gate', 'synthetic_smoke'):
        budget = config.get('budget', {})
        if not 1 <= budget.get('max_updates', 0) <= 2000 or not 0 < budget.get('max_seconds', 0) <= 1800:
            raise ValueError('Bounded acceptance requires <=2000 updates and <=1800 seconds')
    if stage == 'formal' and (tr['max_epochs'] != 300 or ds.get('train_limit') is not None or ds.get('val_limit') is not None):
        raise ValueError('Formal B requires all166/36 cases and 300epochs')
    return config


def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text()))


def build_model(config):
    from research.plan_b.model import SetConditionalField
    m = config['model']
    return SetConditionalField(query_chunk=m['query_chunk'], ray_chunk=m['ray_chunk'],
        samples=m['samples'], checkpoint_query=m['checkpoint_query'],
        lift_resolution=m['lift_resolution'], y_scale=m['y_scale'],
        cache_budget_bytes=m['cache_budget_mib'] * 1024**2)


def initialize_from_train_measurements(model, dataset):
    """Set only final sigmoid bias from ALL166 train cases / ALL64 Pool A stations.

    Float64 pooled chi=sum(Y)/(1e6 sum(L)) is inverted to mass fraction, then
    transformed to a sigmoid logit. It is an acquisition-weighted scalar prior,
    not a volumetric estimate. Validation/test/GT and limited A/B subsets do not
    enter this statistic. Existing random final weights and hidden layers remain.
    """
    if dataset.split != 'train' or dataset.geometry_pool != 'train' or len(dataset.records) != 166 or dataset.station_count != 64:
        raise ValueError('Initialization requires full166 train cases and all64 Pool A stations')
    root = Path(dataset.root).resolve()
    valid = np.asarray(dataset.geometry['valid_ray'], dtype=bool)
    lengths = np.asarray(dataset.geometry['path_length_m'], dtype=np.float64)[valid]
    if valid.shape != (64,64,64) or not len(lengths) or not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError('Invalid train Pool A ray lengths for initialization')
    per_case_length = float(lengths.sum(dtype=np.float64))
    projection_sum = 0.
    files = {}
    for record in dataset.records:
        if record['split'] != 'train':
            raise ValueError('Initialization cannot consume nontrain measurements')
        path = (root / record['train_projection_path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Initialization projection escapes dataset root')
        projection = np.load(path, mmap_mode='r', allow_pickle=False)
        if projection.shape != valid.shape or projection.dtype != np.float32:
            raise ValueError('Invalid initialization projection schema')
        values = np.asarray(projection[valid], dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Initialization requires finite nonnegative valid measurements')
        projection_sum += float(values.sum(dtype=np.float64))
        files[str(path.relative_to(root))] = _sha(path)
    total_length = per_case_length * len(dataset.records)
    chi0 = projection_sum / (1e6 * total_length)
    if not 0 < chi0 < 1:
        raise ValueError('Pooled train mole fraction must lie strictly in (0,1)')
    concentration = (chi0 * 16.043) / (chi0 * 16.043 + (1-chi0)*28.965)
    logit = math.log(concentration / (1-concentration))
    final = model.decoder[-2]
    if not isinstance(final, torch.nn.Linear) or final.out_features != 1 or final.bias is None:
        raise ValueError('Expected the prescribed final64-to1 linear before sigmoid')
    with torch.no_grad():
        final.bias.fill_(logit)
    geometry_path = root / 'geometry/train_geometry.npz'
    return dict(applied=True, policy='train_full_pool_a_pooled_chi_bias_v1',
        formula='chi0=sum_valid_train_Y/(1e6*sum_valid_train_L); C0=chi0*M_CH4/(chi0*M_CH4+(1-chi0)*M_air); bias=logit(C0)',
        methane_molar_mass=16.043, air_molar_mass=28.965,
        case_count=len(dataset.records), station_count=dataset.station_count,
        valid_ray_count=int(valid.sum())*len(dataset.records),
        projection_sum_ppm_m=projection_sum, path_length_sum_m=total_length,
        pooled_mole_fraction=chi0, initial_mass_fraction=concentration,
        final_bias_logit=logit, pooling_dtype='float64',
        changed_parameters=['decoder.final_linear.bias'], random_weights_preserved=True,
        uses_validation=False, uses_gt=False, train_limit_ignored=True,
        case_ids=[r['case_id'] for r in dataset.records],
        projection_files_sha256=files, geometry_sha256=_sha(geometry_path),
        source_index_sha256=_sha(root/'source_index.jsonl'))


def objective(model, batch):
    assert_no_gt(batch)
    assert_disjoint(batch)
    contexts = dict(C_A=model.encode(batch['A']), C_B=model.encode(batch['B']))
    # These predictions query the MLP at the Q96 physical ray midpoints.
    # sample_grid is deliberately absent from this projection path.
    projections = {name: {key: model.integrate(context, batch[key]) for key in ('A', 'B', 'H')}
                   for name, context in contexts.items()}
    # Both contexts share identical 64^3 endpoint query locations; neither detached.
    ca, cb = (model.sample_grid(context, 64) for context in contexts.values())
    m = lambda name, key: measurement_discrepancy(projections[name][key], batch[key])
    parts = dict(hold=.5*(m('C_A', 'H')+m('C_B', 'H')),
                 cross=.5*(m('C_A', 'B')+m('C_B', 'A')),
                 observed=.5*(m('C_A', 'A')+m('C_B', 'B')),
                 consistency=consistency_discrepancy(ca, cb))
    total = sum(FROZEN_LOSS[key]*value for key, value in parts.items())
    if not all(bool(torch.isfinite(v).all()) for v in (ca, cb, total, *parts.values())):
        raise FloatingPointError('Nonfinite continuous field/measurement objective')
    return total, parts, (ca, cb), projections


def _provenance(config, train, val):
    # B lives outside src specifically to preserve the running A source identity.
    sources = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'research/plan_b').rglob('*.py')) + [ROOT / 'scripts/train_plan_b.py']
    code = {str(path.relative_to(ROOT)): _sha(path) for path in sources}
    data = dict(kind=config['dataset']['kind'])
    if data['kind'] == 'projection_pool':
        dataset_root = Path(config['dataset']['root']).resolve()
        files = {dataset_root / name for name in ('source_index.jsonl', 'train_statistics.json', 'geometry/train_geometry.npz')}
        for dataset in (train, val):
            base = getattr(dataset, 'dataset', dataset)
            # Initialization always consumes full train membership, even smoke/gate.
            indices = range(len(base)) if base.split == 'train' else getattr(dataset, 'indices', range(len(base)))
            for index in indices:
                path = (dataset_root / base.records[index]['train_projection_path']).resolve()
                if not path.is_relative_to(dataset_root):
                    raise ValueError('Projection path escapes dataset root')
                files.add(path)
        data['files'] = {str(path.relative_to(dataset_root)): _sha(path) for path in sorted(files)}
    else:
        data['generator'] = 'analytic_constant_mass_fraction_0.01_0.03_geometry_seed41_v1'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return dict(protocol=PROTOCOL, config_sha256=_canonical(config), data_sha256=_canonical(data),
                source_sha256=_canonical(code), git_commit=commit, code=code, data=data,
                train_case_ids=[r['case_id'] for r in _sampling_rows(train, 0, 'train')],
                val_case_ids=[r['case_id'] for r in _sampling_rows(val, 0, 'val')])


def _validate_acceptance(config, provenance):
    if config['stage'] != 'formal':
        return
    for kind in ('smoke', 'gate'):
        directory = Path(config.get('acceptance', {}).get(kind + '_run', ''))
        result = json.loads((directory / 'acceptance_result.json').read_text())
        prior_config = yaml.safe_load((directory / 'config_resolved.yaml').read_text())
        prior = json.loads((directory / 'provenance.json').read_text())
        audit = json.loads((directory / 'gt_read_audit.json').read_text())
        if not result.get('passed') or not result.get('completed') or result.get('synthetic') or result.get('stage') != kind:
            raise ValueError(f'Formal B requires completed, passed real-data {kind}')
        if prior_config['model'] != config['model'] or prior_config['dataset']['root'] != config['dataset']['root']:
            raise ValueError(f'Formal {kind} architecture/data differs')
        if prior['protocol'] != PROTOCOL or any(prior[k] != provenance[k] for k in ('source_sha256', 'git_commit')):
            raise ValueError(f'Formal {kind} used different source/protocol')
        files = provenance['data']['files']
        if any(files.get(path) != checksum for path, checksum in prior['data']['files'].items()):
            raise ValueError(f'Formal {kind} used changed data/statistics/geometry')
        if any(audit.get(k, 0) for k in ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts')):
            raise ValueError(f'Formal {kind} had forbidden read attempts')


def run_epoch(model, loader, config, optimizer=None, *, snapshot_path=None, max_updates=None, deadline=None):
    training = optimizer is not None
    model.train(training)
    totals = dict.fromkeys(('total', 'hold', 'cross', 'observed', 'consistency'), 0.)
    scores = {key: [] for key in ('heldout', 'cross', 'observed')}
    count = group_cases = updates = 0
    norms, moments, examples = [], [], {}
    device = config['training']['device']
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    began = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, raw in enumerate(loader):
            batch = move_dual_batch(raw, config['training']['device'])
            loss, parts, volumes, projections = objective(model, batch)
            if snapshot_path is not None and step == 0:
                path = Path(snapshot_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(case_id=batch['A']['case_id'][0], final=volumes[0][0].detach().cpu(),
                    heldout_prediction=projections['C_A']['H'][0].detach().cpu(),
                    heldout_reference=batch['H']['input_projection_raw'][0].detach().cpu()), path)
            n = len(batch['A']['case_id'])
            count += n
            if training:
                (loss * n).backward()
                group_cases += n
                if (step+1) % config['training']['gradient_accumulation'] == 0 or step+1 == len(loader):
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_cases)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    norms.append(float(norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    group_cases = 0
            for key, value in dict(total=loss, **parts).items():
                totals[key] += float(value.detach()) * n
            detached = {name: {key: value.detach() for key, value in preds.items()} for name, preds in projections.items()}
            for key, values in projection_scores(detached, batch).items():
                scores[key].extend(values)
            for volume in volumes:
                for prediction in volume.detach():
                    moments.append((float(prediction.min()), float(prediction.max()), float(prediction.mean()),
                        float(prediction.square().mean()), float((prediction<=1e-5).float().mean()), float((prediction>=.999).float().mean())))
            for i, case_id in enumerate(batch['A']['case_id']):
                if len(examples) < 8:
                    examples[case_id] = volumes[0][i].detach().cpu()
            if training and group_cases == 0 and ((max_updates is not None and updates >= max_updates)
                                                   or (deadline is not None and time.monotonic() >= deadline)):
                break
            if not training and deadline is not None and time.monotonic() >= deadline:
                break
    if not count:
        raise ValueError('An epoch cannot be empty')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-began
    mean = sum(v[2] for v in moments)/len(moments)
    ordered = [examples[k] for k in sorted(examples)]
    return dict(loss={k: v/count for k,v in totals.items()},
        measurement={k: {metric: sum(v[metric] for v in rows)/len(rows) for metric in ('relative_l1', 'relative_l2')} for k,rows in scores.items()},
        output=dict(min=min(v[0] for v in moments), max=max(v[1] for v in moments), mean=mean,
            std=math.sqrt(max(0., sum(v[3] for v in moments)/len(moments)-mean**2)),
            fraction_near_zero=sum(v[4] for v in moments)/len(moments), fraction_near_one=sum(v[5] for v in moments)/len(moments),
            case_to_case_mean_absolute_difference=sum(float((a-b).abs().mean()) for a,b in zip(ordered, ordered[1:]))/max(1,len(ordered)-1)),
        gradient_norm_max=max(norms,default=0.), gradient_norms=norms, sample_count=count, optimizer_updates=updates,
        gt_used=False, complete_epoch=count==len(loader.dataset), seconds=elapsed)


def measurement_dependence_probe(model, branch):
    altered = dict(branch, input_projection=branch['input_projection']*1.2,
                   input_projection_raw=branch['input_projection_raw']*1.2)
    model.eval()
    with torch.no_grad():
        original = model.sample_grid(model.encode(branch), 64)
        changed = model.sample_grid(model.encode(altered), 64)
    delta = float((original-changed).abs().mean())
    return dict(intervention='multiply measurements by1.2; identical geometry/masks',
                mean_absolute_output_change=delta, passed=bool(math.isfinite(delta) and delta>1e-8))


def train(config, *, resume=False, stop_after_epochs=None):
    """Train a new run or resume exactly at an epoch boundary.

    stop_after_epochs supports controlled interruption/replay checks. It never
    changes the resolved config or marks an unfinished run as frozen.
    """
    began_run = time.monotonic()
    validate_config(config)
    config = copy.deepcopy(config)
    budget = config.get('budget', {}) if config['stage'] != 'formal' else {}
    deadline = began_run + budget['max_seconds'] if budget else None
    run = Path(config['experiment']['output_root']) / config['experiment']['name']
    checkpoints = run / 'checkpoints'
    if run.exists() and not resume:
        raise FileExistsError('Run exists; use explicit resume or a new run name')
    if resume and not (checkpoints / 'last.pt').exists():
        raise FileNotFoundError('Resume requires checkpoints/last.pt')
    checkpoints.mkdir(parents=True, exist_ok=True)
    seed_everything(config['training']['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
            model = build_model(config).to(device)
            if not resume:
                initialization = (initialize_from_train_measurements(model, train_base)
                    if config['dataset']['kind'] == 'projection_pool' else
                    dict(applied=False, reason='Synthetic trainer engineering check; no real-data statistic'))
                write_json(run / 'initialization.json', initialization)
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
                    git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                    parameter_count=sum(p.numel() for p in model.parameters())))
                with (run / 'sampling_manifest.jsonl').open('w') as stream:
                    for row in _sampling_rows(val, 0, 'val'):
                        stream.write(json.dumps(row) + '\n')
                val_base.set_epoch(0)
                write_json(run / 'zero_volume_baseline.json', zero_volume_baseline(val_loader, config))
                initial = run_epoch(model, val_loader, config, deadline=deadline)
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
                training = run_epoch(model, train_loader, config, optimizer,
                    max_updates=budget['max_updates']-global_step if budget else None, deadline=deadline)
                executed_updates += training['optimizer_updates']
                if not training['complete_epoch']:
                    write_json(run / 'budget_partial_epoch.json', dict(epoch=epoch, result=training,
                        checkpoint_policy='Partial epoch never checkpointed; last.pt remains a completed boundary'))
                    budget_exhausted = True
                    break
                snapshot_path = run / 'diagnostics' / f'epoch_{epoch:03d}.pt' if epoch in (0, config['training']['max_epochs']-1) else None
                validation = run_epoch(model, val_loader, config, snapshot_path=snapshot_path, deadline=deadline)
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
                audit = guard.summary() if isinstance(guard, GTReadGuard) else dict(gt_open_attempts=0, synthetic=True)
                write_json(run / 'acceptance_result.json', dict(stage=config['stage'], synthetic=config['stage']=='synthetic_smoke', passed=False,
                    completed=False, completed_epochs=completed_epochs, stopping_reason='budget_exhausted',
                    budget=budget, elapsed_seconds=time.monotonic()-began_run, decision='NO-GO: incomplete bounded acceptance run'))
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
                    train_loss_decreases=rows[-1]['train']['loss']['total'] < rows[0]['train']['loss']['total'],
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
                    passed=passed, completed=True, criteria=criteria, best_heldout_relative_l1=best,
                    zero_heldout_relative_l1=baseline['relative_l1'], initial_heldout_relative_l1=selection_metric(initial)))
                frozen = dict(completed=True, completed_epochs=end, stage=config['stage'], protocol=PROTOCOL,
                    synthetic=config['dataset']['kind'] == 'synthetic', formal_gate_claim=config['stage']=='formal' and passed,
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
