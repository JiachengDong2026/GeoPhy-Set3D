#!/usr/bin/env python3
"""Compare B runtimes on identical real training measurements, without saving weights.

This is a warmed, fixed-case kernel diagnostic; use a full epoch to forecast
training duration. Q96, all six projections and 64^3 consistency remain active.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
import yaml

from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.research_selfsup import make_research_dataset
from geophy_set3d.training.selfsup import move_dual_batch
from research.plan_b.model import SetConditionalField
from research.plan_b.training import objective


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=ROOT/'outputs/research/scheme_b_gate_gpu1_extended30')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--variants', nargs='+', default=['reference', 'no_checkpoint', 'chunk1024', 'chunk2048'])
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a fresh output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = yaml.safe_load((args.run_dir/'config_resolved.yaml').read_text())
    guard = GTReadGuard(config['dataset']['root'], args.run_dir/'checkpoints')
    report = dict(stage='performance_diagnostic', device=args.device, effective_batch=2,
                  samples=96, cases=[], variants=[], checkpoints_saved=False,
                  note='Fixed two training cases, one warmup update and timed updates; excludes loader and epoch diagnostics. No accuracy or convergence claim.')
    with guard:
        payload = torch.load(args.run_dir/'checkpoints/best.pt', map_location='cpu', weights_only=False)
        dataset = make_research_dataset(config)
        dataset.set_epoch(0)
        batches = [move_dual_batch(dual_measurement_collate([dataset[i]]), args.device) for i in (0, 1)]
        report['cases'] = [b['A']['case_id'] for b in batches]
        report['views'] = [{k: int(b[k]['view_mask'].sum()) for k in ('A', 'B', 'H')} for b in batches]
        reference_loss = reference_gradient = None
        for name in args.variants:
            torch.cuda.empty_cache()
            kwargs = dict(query_chunk=32768, ray_chunk=256, checkpoint_query=True)
            if name == 'no_checkpoint':
                kwargs['checkpoint_query'] = False
            elif name.startswith('chunk'):
                size = int(name.removeprefix('chunk'))
                kwargs.update(ray_chunk=size, query_chunk=max(32768, size*96), checkpoint_query=False)
            model = SetConditionalField(**kwargs).to(args.device)
            model.load_state_dict(payload['model'])
            model.train()
            torch.cuda.reset_peak_memory_stats(args.device)
            result = dict(name=name, settings=kwargs)
            try:
                # Compare loss and all attached parameter gradients at identical weights.
                value = 0.
                for b in batches:
                    loss = objective(model, b)[0] / 2
                    value += float(loss.detach())
                    loss.backward()
                    del loss
                gradient = torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters()])
                if reference_gradient is None:
                    reference_loss, reference_gradient = value, gradient
                result.update(loss=value, loss_abs_difference=abs(value-reference_loss),
                              gradient_max_abs=float((gradient-reference_gradient).abs().max()),
                              gradient_relative_l2=float((gradient-reference_gradient).norm()/reference_gradient.norm()))
                del gradient
                model.zero_grad(set_to_none=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
                times = []
                for step in range(args.steps + 1):
                    torch.cuda.synchronize(args.device)
                    started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    for b in batches:
                        (objective(model, b)[0] / 2).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize(args.device)
                    if step:
                        times.append(time.perf_counter()-started)
                result.update(update_seconds=times, median_seconds=statistics.median(times),
                              mean_seconds=statistics.mean(times),
                              peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/1024**3,
                              status='completed')
                if args.profile and name == 'reference':
                    model.zero_grad(set_to_none=True)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                        objective(model, batches[0])[0].backward()
                    table = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=25)
                    (args.output.parent/'reference_profile.txt').write_text(table)
                    print(table, flush=True)
            except torch.OutOfMemoryError as error:
                result.update(status='oom', error=str(error))
            report['variants'].append(result)
            print(json.dumps(result), flush=True)
            del model
            if 'optimizer' in locals():
                del optimizer
            torch.cuda.empty_cache()
            report['gt_access'] = guard.summary()
            args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
