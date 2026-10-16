#!/usr/bin/env python3
"""Measure a cold, full B train/validation epoch including real AdamW updates.

No warmup updates or weight exports. The two variants share seed, data order,
view sampling, initialization, FP32, Q96 and the complete objective/diagnostics.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import torch
import yaml
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.research_selfsup import _limited, make_research_dataset, make_research_loader, _sampling_rows
from geophy_set3d.training.utils import seed_everything
from research.plan_b.training import build_model, initialize_from_train_measurements, run_epoch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/research/scheme_b_gate_gpu1_extended30.yaml')
    p.add_argument('--variant', choices=('reference', 'fast'), required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-cases', type=int, default=166)
    p.add_argument('--val-cases', type=int, default=36)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.train_cases <= 166 or not 1 <= args.val_cases <= 36:
        p.error('Need a fresh output and valid case counts')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    config['stage'] = 'performance_benchmark'
    config['dataset'].update(train_limit=args.train_cases, val_limit=args.val_cases)
    config['training']['device'] = args.device
    if args.variant == 'fast':
        config['model'].update(query_chunk=196608, ray_chunk=2048, checkpoint_query=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(config['training']['seed'])
    report = dict(stage='performance_benchmark', variant=args.variant, config=config,
        checkpoints_saved=False, warmup_updates=0, cold_geometry_cache=True,
        timing_policy='Full run_epoch including loader, transfer, six continuous projections, 64^3 consistency, backward, AdamW, clipping and diagnostics. Validation included separately; source hashing, startup and checkpoint writes excluded.',
        accuracy_claim=False,
        hardware_before=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
    guard = GTReadGuard(config['dataset']['root'], args.output.parent/'unused_checkpoints')
    began = time.perf_counter()
    try:
        with guard:
            base_train = make_research_dataset(config)
            base_val = make_research_dataset(config, True)
            base_train.set_epoch(0)
            base_val.set_epoch(0)
            train, val = _limited(base_train, args.train_cases), _limited(base_val, args.val_cases)
            train_loader = make_research_loader(train, config, shuffle=True,
                generator=torch.Generator().manual_seed(config['training']['seed']))
            val_loader = make_research_loader(val, config, generator=torch.Generator().manual_seed(0))
            model = build_model(config).to(args.device)
            report['initialization'] = initialize_from_train_measurements(model, base_train)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            torch.cuda.reset_peak_memory_stats(args.device)
            report['setup_seconds'] = time.perf_counter()-began
            report['view_counts'] = [{key: row[key] for key in ('case_id','A','B','H')} for row in _sampling_rows(train,0,'train')]
            report['train'] = run_epoch(model, train_loader, config, optimizer)
            report['validation'] = run_epoch(model, val_loader, config)
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated(args.device)/1024**3
            report['peak_reserved_gib'] = torch.cuda.max_memory_reserved(args.device)/1024**3
            report['seconds_per_update'] = report['train']['seconds']/report['train']['optimizer_updates']
            report['epoch_seconds'] = report['train']['seconds']+report['validation']['seconds']
            report['extrapolated_300epoch_hours'] = report['epoch_seconds']*300/3600
            report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['gt_access'] = guard.summary()
        report['wall_seconds'] = time.perf_counter()-began
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('status','variant','seconds_per_update','epoch_seconds','extrapolated_300epoch_hours','peak_allocated_gib')}))


if __name__ == '__main__':
    main()
