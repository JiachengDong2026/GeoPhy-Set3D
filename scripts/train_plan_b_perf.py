#!/usr/bin/env python3
"""Train B's independently recorded FP32 performance runtime; formal gates apply."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from plan_b_perf_v2 import load_perf_config, runtime_engine, write_qualification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--device')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--stop-after-epochs', type=int)
    args = p.parse_args()
    config = load_perf_config(args.config)
    if args.device:
        config['training']['device'] = args.device
    if args.stop_after_epochs is not None and args.stop_after_epochs < 1:
        p.error('--stop-after-epochs must be positive')
    if args.validate_only:
        print('Valid isolated FP32 B performance config:', args.config)
        return
    run = runtime_engine().train(config, resume=args.resume,
                                 stop_after_epochs=args.stop_after_epochs)
    if config['stage'] in ('smoke', 'gate') and (run/'training_summary.json').exists():
        print(write_qualification(run))
    print(run)


if __name__ == '__main__':
    main()
