#!/usr/bin/env python3
"""Reproducible, guarded full-size Plan A update/validation speed comparison.

This engineering benchmark never changes a research checkpoint or acceptance
record. Each repeat starts with identical weights, inputs, seed and fresh AdamW;
only geometry/kernel caches are warm after the first repeat. Artifact copies and
weight/optimizer reset are outside the timed region. No GT is read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='Repository or snapshot containing src/geophy_set3d')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSON path')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--microbatch', type=int, choices=(1, 2), default=1)
    parser.add_argument('--checkpoint-prior', choices=('config', 'on', 'off'), default='config')
    parser.add_argument('--diagnostics', choices=('on', 'off'), default='on')
    parser.add_argument('--ray-chunk', type=int)
    parser.add_argument('--case-index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--profile', action='store_true', help='Extra untimed torch.profiler update')
    from plan_a_runtime_candidates import add_arguments
    add_arguments(parser)
    args = parser.parse_args()
    if not args.experimental_runtime and (args.network_layout != 'contiguous' or args.cudnn_autotune or args.compile_prior or args.network_precision != 'fp32'):
        parser.error('Runtime candidates require --experimental-runtime')
    if args.output.exists():
        parser.error('Output exists; use a fresh path')
    if args.warmup < 1 or args.repeats < 2 or args.case_index < 0 or args.epoch < 0:
        parser.error('Require warmup>=1, repeats>=2, and nonnegative case-index/epoch')
    source = args.source_root.resolve()
    if (source / 'src/geophy_set3d').is_dir():
        source = source / 'src'
    if not (source / 'geophy_set3d').is_dir():
        parser.error('source-root must contain src/geophy_set3d or geophy_set3d')
    args.source_path = source
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    return args


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command_snapshot():
    result = {'unix_time': time.time()}
    for key, query in (
        ('gpus', ['--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,power.draw', '--format=csv,noheader,nounits']),
        ('processes', ['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits']),
    ):
        try:
            out = subprocess.run(['nvidia-smi', *query], capture_output=True, text=True, timeout=8)
            result[key] = out.stdout.strip() if out.returncode == 0 else out.stderr.strip()
        except Exception as exc:
            result[key] = repr(exc)
    return result


class LoadSampler:
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.is_set():
            self.samples.append(command_snapshot())
            self.stop_event.wait(2)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)


def summary(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values), values=values)


def main():
    args = arguments()
    import numpy as np
    import torch
    from plan_a_runtime_candidates import configure_backend, configure_model, settings
    from geophy_set3d.data.measurement_only import dual_measurement_collate
    from geophy_set3d.models.set_unrolled3d import build_set_unrolled_model
    from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator
    from geophy_set3d.training.gt_read_guard import GTReadGuard
    from geophy_set3d.training.research_selfsup import load_research_config, make_research_dataset, research_objective
    from geophy_set3d.training.selfsup import move_dual_batch
    from geophy_set3d.training.utils import seed_everything

    began = time.perf_counter()
    report = dict(protocol='plan_a_full_update_speed_v2', status='started',
                  source_root=str(args.source_path), config_path=str(args.config.resolve()),
                  device=args.device, precision='fp32', tf32_matmul=False, tf32_cudnn=False,
                  effective_batch=2, microbatch=args.microbatch,
                  gradient_accumulation=2 // args.microbatch,
                  warmup=args.warmup, timed_repeats=args.repeats,
                  diagnostics=args.diagnostics, gt_used=False,
                  timings_include='CPU collation, CPU-to-device transfer, full A/B/H objective, backward, actual-case gradient averaging, clip and AdamW update',
                  timings_exclude='Dataset disk loading, model/optimizer reset, numerical artifact copies; validation excludes backward/optimizer',
                  cache_policy='First warmup is cold; model/operator geometry caches and CUDA kernels remain warm thereafter. Fixed two cases only, not full-epoch throughput.',
                  optimizer_policy='Fresh AdamW per repeat; timings include first-update moment initialization.',
                  train=[], validation=[])
    runtime = settings(args)
    if args.experimental_runtime:
        report.update(protocol='plan_a_network_runtime_speed_v3', runtime=runtime,
                      precision='fp32_physics_' + args.network_precision + '_network',
                      tf32_cudnn=args.network_precision == 'tf32',
                      runtime_script_sha256=sha(Path(__file__).with_name('plan_a_runtime_candidates.py')),
                      benchmark_script_sha256=sha(Path(__file__)),
                      formal_acceptance=False)
    guard = None
    sampler = LoadSampler()
    code = 1
    try:
        config = load_research_config(args.config)
        config = copy.deepcopy(config)
        if config['dataset']['kind'] != 'projection_pool':
            raise ValueError('Benchmark requires the real measurement-only projection pool')
        expected = dict(variant='A2', steps=4, samples=96, prior_width=32,
                        lift_resolution=32, output_resolution=64)
        if any(config['model'].get(key) != value for key, value in expected.items()):
            raise ValueError(f'Full-size benchmark requires {expected}')
        if args.checkpoint_prior != 'config':
            config['model']['checkpoint_prior'] = args.checkpoint_prior == 'on'
        if args.ray_chunk is not None:
            if args.ray_chunk <= 0:
                raise ValueError('ray-chunk must be positive')
            config['model']['ray_chunk'] = args.ray_chunk
        torch.set_num_threads(config['training'].get('cpu_threads', 2))
        seed = config['training']['seed']
        seed_everything(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['environment'] = dict(python=platform.python_version(), torch=str(torch.__version__),
            cuda_runtime=torch.version.cuda, numpy=np.__version__,
            gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None)
        report['resolved_model'] = config['model']
        report['config_sha256'] = sha(args.config)
        report['source_sha256'] = {str(path.relative_to(args.source_path)): sha(path)
                                   for path in sorted(args.source_path.rglob('*.py'))}
        report['training_seed'] = seed
        report['sampling_epoch'] = args.epoch
        guard = GTReadGuard(config['dataset']['root'], args.output.parent)
        sampler.start()

        def sync():
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)

        def memory():
            if not args.device.startswith('cuda'):
                return {}
            free, total = torch.cuda.mem_get_info(args.device)
            return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                        free_bytes=free, total_bytes=total)

        def digest_tree(value):
            digest = hashlib.sha256()
            def walk(item):
                if isinstance(item, torch.Tensor):
                    digest.update(str((item.dtype, tuple(item.shape))).encode())
                    digest.update(item.detach().cpu().contiguous().numpy().tobytes())
                elif isinstance(item, dict):
                    for key in sorted(item):
                        digest.update(key.encode())
                        walk(item[key])
                elif isinstance(item, (tuple, list)):
                    for child in item:
                        walk(child)
                else:
                    digest.update(repr(item).encode())
            walk(value)
            return digest.hexdigest()

        with guard:
            start = time.perf_counter()
            datasets = {name: make_research_dataset(config, validation=name == 'validation')
                        for name in ('train', 'validation')}
            samples = {}
            for name, dataset in datasets.items():
                dataset.set_epoch(args.epoch if name == 'train' else 0)
                samples[name] = [dataset[index] for index in range(args.case_index, args.case_index + 2)]
            report['dataset_load_seconds'] = time.perf_counter() - start
            report['input_sha256'] = {name: digest_tree(value) for name, value in samples.items()}
            report['case_ids'] = {name: [sample['A']['case_id'] for sample in rows] for name, rows in samples.items()}
            data_root = Path(config['dataset']['root'])
            paths = {'source_index.jsonl', 'train_statistics.json', 'geometry/train_geometry.npz'}
            for dataset in datasets.values():
                paths.update(dataset.records[index]['train_projection_path']
                             for index in range(args.case_index, args.case_index + 2))
            report['data_file_sha256'] = {str(path): sha(data_root / path) for path in sorted(paths)}
            model = build_set_unrolled_model(config).to(args.device)
            if args.experimental_runtime:
                model = configure_model(model, runtime)
            operator = ExplicitRayOperator(samples_per_ray=96,
                                          ray_chunk=config['model'].get('ray_chunk', 256))
            initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
            report['initial_weights_sha256'] = digest_tree(initial)
            report['parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
            artifact = dict(initial_state={name: value.cpu() for name, value in initial.items()},
                            parameter_names=[name for name, _ in model.named_parameters()])

            def iteration(name, repeat, capture=False):
                model.load_state_dict(initial)
                model.zero_grad(set_to_none=True)
                seed_everything(seed)
                if args.experimental_runtime:
                    configure_backend(runtime)
                training = name == 'train'
                model.train(training)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'],
                                             weight_decay=config['optimizer']['weight_decay']) if training else None
                sync()
                if args.device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats(args.device)
                started = time.perf_counter()
                losses, volumes_saved, h_predictions = [], [], []
                with torch.set_grad_enabled(training):
                    for offset in range(0, 2, args.microbatch):
                        batch = move_dual_batch(dual_measurement_collate(
                            samples[name][offset:offset + args.microbatch]), args.device)
                        loss, _, volumes, projections, _ = research_objective(
                            model, batch, operator, diagnostics=args.diagnostics == 'on')
                        losses.append(loss.detach())
                        if training:
                            (loss * args.microbatch).backward()
                        if capture:
                            volumes_saved.extend([value.detach() for value in volumes])
                            h_predictions.extend([projections[key]['H'].detach() for key in ('C_A', 'C_B')])
                    if training:
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(2)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                sync()
                elapsed = time.perf_counter() - started
                row = dict(repeat=repeat, seconds=elapsed,
                           loss=float(torch.stack(losses).mean()), memory=memory())
                if training:
                    row['preclip_gradient_norm'] = float(norm)
                if capture:
                    artifact[name] = dict(
                        volumes=[value.cpu() for value in volumes_saved],
                        heldout_predictions=[value.cpu() for value in h_predictions],
                        clipped_gradients={key: parameter.grad.detach().cpu()
                                           for key, parameter in model.named_parameters() if parameter.grad is not None},
                        post_update_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
                        loss=row['loss'])
                return row

            for name in ('train', 'validation'):
                report[name + '_warmup'] = [iteration(name, -(index + 1)) for index in range(args.warmup)]
                for index in range(args.repeats):
                    row = iteration(name, index, capture=index == 0)
                    report[name].append(row)
                    print(json.dumps(dict(phase=name, **row)), flush=True)
                report[name + '_seconds'] = summary([row['seconds'] for row in report[name]])
            artifact_path = args.output.with_suffix('.numerics.pt')
            if artifact_path.exists():
                raise FileExistsError(artifact_path)
            torch.save(artifact, artifact_path)
            report['numerical_artifact'] = dict(path=str(artifact_path.resolve()), sha256=sha(artifact_path),
                note='Only measurement-derived tensors, initial/post-update weights and clipped gradients; no GT. Compare keys/tensors with the same input/initial hashes.')
            if args.profile:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if args.device.startswith('cuda'):
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                    iteration('train', -100)
                trace = args.output.with_suffix('.trace.json')
                prof.export_chrome_trace(str(trace))
                report['profiler'] = dict(trace=str(trace),
                    cuda_table=prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=30),
                    cpu_table=prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=30))
            report['status'] = 'passed'
            code = 0
    except Exception as exc:
        report.update(status='oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'failed',
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        sampler.stop()
        report['gpu_load_samples'] = sampler.samples
        report['gt_read_audit'] = guard.summary() if guard is not None else {'guard_not_started': True}
        report['elapsed_seconds'] = time.perf_counter() - began
        with args.output.open('x') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(status=report['status'], output=str(args.output), elapsed_seconds=report['elapsed_seconds'])))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
