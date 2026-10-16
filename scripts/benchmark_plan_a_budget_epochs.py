#!/usr/bin/env python3
"""Measure one cold, real Pool A training epoch and its full validation pass.

This is a performance_benchmark, not a formal run. It uses real AdamW updates
and the complete research epoch diagnostics, but saves no weights/checkpoints.
No warmup updates are run. Default membership is 166 train / 36 validation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from benchmark_plan_a_speed import LoadSampler, sha


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=ROOT/'configs/research/scheme_a_budget_v4_w16.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSON result path')
    parser.add_argument('--train-cases', type=int, default=166)
    parser.add_argument('--val-cases', type=int, default=36)
    parser.add_argument('--epoch', type=int, default=0, help='Train A/B/H sampling epoch; validation stays epoch 0')
    parser.add_argument('--microbatch', type=int, choices=(1, 2), default=1)
    parser.add_argument('--ray-chunk', type=int, default=1024)
    parser.add_argument('--checkpoint-prior', choices=('config', 'on', 'off'), default='off')
    parser.add_argument('--epochs', type=int, choices=(1, 3), default=3)
    parser.add_argument('--checkpoint-dir', type=Path, required=True, help='Fresh benchmark-only checkpoint directory')
    args = parser.parse_args()
    if args.checkpoint_dir.exists():
        parser.error('Checkpoint directory must be fresh')
    if args.output.exists():
        parser.error('Output already exists; choose a fresh JSON path')
    if not 1 <= args.train_cases <= 166 or not 1 <= args.val_cases <= 36:
        parser.error('Require 1..166 train cases and 1..36 validation cases')
    if args.epoch < 0 or args.ray_chunk < 1:
        parser.error('epoch must be nonnegative and ray-chunk must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


class RecordingLoader:
    """Record actual shuffle order using already loaded CPU batch identities."""
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.case_order = []

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.case_order.extend(batch['A']['case_id'])
            yield batch


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main():
    args = arguments()
    import torch
    from plan_a_budget_v4 import load_budget_config,build_budget_model,configure_backend,budget_provenance
    from geophy_set3d.models.set_unrolled3d import build_set_unrolled_model
    from geophy_set3d.training.gt_read_guard import GTReadGuard
    from geophy_set3d.training.research_selfsup import (
        _canonical, _limited, _provenance, _sampling_rows, load_research_config,
        make_research_dataset, make_research_loader, run_research_epoch,
    )
    from geophy_set3d.training.utils import seed_everything

    report = dict(protocol='plan_a_cold_full_epoch_speed_v2', stage='performance_benchmark',
                  status='started', device=args.device, config_path=str(args.config.resolve()),
                  gt_used=False, precision='fp32', tf32_matmul=False, tf32_cudnn=False,
                  cpu_threads=2, effective_batch=2, warmup_updates=0,
                  checkpoints_saved=True, formal_model=False, formal_acceptance=False,
                  timing_policy='One cold train epoch with fresh AdamW and initially empty geometry caches, then validation using the naturally populated cache. Includes disk loading, collation, transfers, complete A/B/H forward/backward, real updates, full diagnostics and metric aggregation. Setup/hashing reported separately. No repeated fixed-case hot-cache extrapolation.',
                  os_file_cache_policy='OS file caches are not cleared; provenance hashing before timing can warm measurement files. Cold refers to model geometry/kernel caches and fresh optimizer, not forced cold storage.',
                  limitations='Consecutive epoch speed/finite-gradient check, not convergence, 3D accuracy, formal training or acceptance evidence.',
                  phases={})
    began = time.perf_counter()
    phase = 'setup'
    sampler, guard, code = LoadSampler(), None, 1
    loaders = {}

    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)

    def memory():
        if not args.device.startswith('cuda'):
            return {}
        free, total = torch.cuda.mem_get_info(args.device)
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                    current_allocated_bytes=torch.cuda.memory_allocated(args.device),
                    current_reserved_bytes=torch.cuda.memory_reserved(args.device),
                    device_free_bytes=free, device_total_bytes=total)

    try:
        started = time.perf_counter()
        config = copy.deepcopy(load_budget_config(args.config))
        if config['dataset']['kind'] != 'projection_pool':
            raise ValueError('Only real projection_pool data are supported')
        required = dict(variant='A2', steps=4, samples=96,
                        lift_resolution=32, output_resolution=64)
        if any(config['model'].get(key) != value for key, value in required.items()):
            raise ValueError(f'Full-size benchmark requires {required}')
        config['training'].update(device=args.device, cpu_threads=2,
                                  batch_size=args.microbatch, gradient_accumulation=2//args.microbatch)
        config['model']['ray_chunk'] = args.ray_chunk
        if args.checkpoint_prior != 'config':
            config['model']['checkpoint_prior'] = args.checkpoint_prior == 'on'
        config['dataset'].update(train_limit=args.train_cases, val_limit=args.val_cases)
        # Keep the input protocol/config for provenance; this script never calls
        # train_research, formal acceptance, checkpoint loading or freeze logic.
        report['input_config_stage'] = config['stage']
        report['resolved_config'] = config
        report['config_file_sha256'] = sha(args.config)
        report['benchmark_script_sha256'] = sha(Path(__file__))
        torch.set_num_threads(2)
        seed_everything(config['training']['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        configure_backend(config['execution_runtime'])
        report.update(protocol='plan_a_budget_full_epoch_v4', runtime=config['execution_runtime'],
                      architecture=config['architecture'], precision='fp32_physics_'+config['execution_runtime']['network_precision']+'_network')
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA device is unavailable')
        report['environment'] = dict(python=platform.python_version(), torch=str(torch.__version__),
            cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(args.device)
            if args.device.startswith('cuda') else None)
        guard = GTReadGuard(config['dataset']['root'], args.checkpoint_dir)
        sampler.start()
        with guard:
            datasets = {name: make_research_dataset(config, validation=name == 'validation')
                        for name in ('train', 'validation')}
            if len(datasets['train']) != 166 or len(datasets['validation']) != 36:
                raise ValueError('Frozen real dataset membership must be 166 train / 36 validation')
            if any(dataset.y_ref != config['loss']['measurement_scale'] for dataset in datasets.values()):
                raise ValueError('Frozen measurement normalization scale changed')
            datasets['train'].set_epoch(args.epoch)
            datasets['validation'].set_epoch(0)
            selected = dict(train=_limited(datasets['train'], args.train_cases),
                            validation=_limited(datasets['validation'], args.val_cases))
            report['provenance'] = budget_provenance(config, selected['train'], selected['validation'], _provenance)
            report['sampling'] = {
                name: list(_sampling_rows(dataset, args.epoch if name == 'train' else 0,
                                           'train' if name == 'train' else 'val'))
                for name, dataset in selected.items()}
            report['sampling_sha256'] = _canonical(report['sampling'])
            for name, dataset in selected.items():
                generator_seed = config['training']['seed'] if name == 'train' else 0
                loaders[name] = RecordingLoader(make_research_loader(dataset, config,
                    shuffle=name == 'train', generator=torch.Generator().manual_seed(generator_seed)))
            model = build_budget_model(config).to(args.device)
            operator = model.operator
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'],
                                         weight_decay=config['optimizer']['weight_decay'])
            report['parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
            report['outer_operator_is_model_operator'] = operator is model.operator
            report['model_cache_bytes_before_train'] = operator.cache_bytes
            sync()
            report['phases']['setup'] = dict(seconds=time.perf_counter()-started, cuda=memory())
            from geophy_set3d.training.research_selfsup import atomic_checkpoint, rng_state
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
            report['epochs'] = []
            report['timing_policy'] = 'Consecutive real full training and validation epochs; includes first/last diagnostic snapshots, scheduler, full best+last checkpoints every epoch (conservative), and metric serialization. Initial geometry caches empty. Compiler disk/OS caches may be warm. No warmup updates, no formal freeze.'
            args.checkpoint_dir.mkdir(parents=True)
            for cycle in range(args.epochs):
                epoch_started = time.perf_counter()
                datasets['train'].set_epoch(args.epoch + cycle)
                epoch_sampling = list(_sampling_rows(selected['train'], args.epoch+cycle, 'train'))
                with (args.checkpoint_dir/'sampling_manifest.jsonl').open('a') as stream:
                    for item in epoch_sampling:
                        stream.write(json.dumps(item) + '\n')
                for loader in loaders.values():
                    loader.case_order.clear()
                report['phases'] = {}
                for phase in ('train', 'validation'):
                    sync()
                    if args.device.startswith('cuda'):
                        torch.cuda.reset_peak_memory_stats(args.device)
                    started = time.perf_counter()
                    result = run_research_epoch(model, loaders[phase], config, operator,
                        optimizer if phase == 'train' else None,
                        snapshot_path=args.checkpoint_dir.parent / 'diagnostics' / f'epoch_{cycle:03d}.pt'
                        if phase == 'validation' and cycle in (0, args.epochs-1) else None)
                    sync()
                    elapsed = time.perf_counter()-started
                    report['phases'][phase] = dict(seconds=elapsed, metrics=result, cuda=memory(),
                        case_order=loaders[phase].case_order,
                        cases_per_second=result['sample_count']/elapsed,
                        seconds_per_update=elapsed/result['optimizer_updates']
                        if result['optimizer_updates'] else None,
                        geometry_cpu_cache_bytes=operator.cache_bytes)
                    if not result['complete_epoch']:
                        raise RuntimeError(f'Incomplete {phase} epoch')
                    if phase == 'train':
                        norms = result['gradient_norms']
                        expected_updates = math.ceil(args.train_cases/2)
                        finite = len(norms) == expected_updates and all(math.isfinite(value) for value in norms)
                        report['gradient_check'] = dict(gradients_finite=finite,
                            expected_updates=expected_updates, actual_updates=result['optimizer_updates'],
                            check='Each update clip_grad_norm_(error_if_nonfinite=True), plus all recorded total norms finite')
                        if not finite or result['optimizer_updates'] != expected_updates:
                            raise FloatingPointError('Every expected AdamW update must have finite gradients')
                    print(json.dumps(dict(epoch=cycle+1, phase=phase, seconds=elapsed,
                        loss=result['loss'], optimizer_updates=result['optimizer_updates'])), flush=True)
                io_started = time.perf_counter()
                scheduler.step()
                payload = dict(boundary='completed_epoch', epoch=args.epoch+cycle,
                    sampling_epoch_next=args.epoch+cycle+1, global_step=math.ceil(args.train_cases/2)*(cycle+1),
                    model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    rng=rng_state(loaders['train'].loader.generator), config=config, provenance=report['provenance'],
                    benchmark_only=True, formal_training=False)
                atomic_checkpoint(args.checkpoint_dir/'best.pt', payload)
                atomic_checkpoint(args.checkpoint_dir/'last.pt', payload)
                (args.checkpoint_dir/'metrics.json').write_text(json.dumps(report['phases']))
                sync()
                row = dict(epoch=args.epoch+cycle, phases=copy.deepcopy(report['phases']),
                           train_sampling_sha256=_canonical(epoch_sampling),
                           checkpoint_and_metric_seconds=time.perf_counter()-io_started,
                           total_seconds=time.perf_counter()-epoch_started)
                report['epochs'].append(row)
                print(json.dumps(dict(epoch=cycle+1, total_seconds=row['total_seconds'], checkpoint_seconds=row['checkpoint_and_metric_seconds'])), flush=True)
            report['mean_epoch_seconds_including_checkpoints'] = sum(r['total_seconds'] for r in report['epochs']) / args.epochs
            report['max_epoch_seconds_including_checkpoints'] = max(r['total_seconds'] for r in report['epochs'])
            report['projected_300_epochs_hours'] = report['mean_epoch_seconds_including_checkpoints'] / 12
            report['final_losses'] = {name: report['phases'][name]['metrics']['loss']
                                      for name in ('train', 'validation')}
            report['train_plus_validation_seconds'] = sum(report['phases'][name]['seconds']
                                                          for name in ('train', 'validation'))
            report['status'], code = 'passed', 0
    except Exception as exc:
        report.update(status='oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'failed',
                      failed_phase=phase, error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        try:
            report['cuda_at_failure'] = memory()
        except Exception:
            pass
    finally:
        sampler.stop()
        report['gpu_load_samples'] = sampler.samples
        report['gt_read_audit'] = guard.summary() if guard is not None else dict(guard_not_started=True)
        report['elapsed_seconds'] = time.perf_counter()-began
        with args.output.open('x') as stream:
            json.dump(json_finite(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(status=report['status'], output=str(args.output),
                              elapsed_seconds=report['elapsed_seconds'])), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
