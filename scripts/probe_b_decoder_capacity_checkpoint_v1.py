#!/usr/bin/env python3
"""Fresh matched runtime amendment after width128 exceeded 24GB in v1.

Both arms enable identical activation checkpointing. Earlier v1 weights,
metrics and OOM evidence are preserved; neither arm reuses those weights.
"""
import argparse
import json
from pathlib import Path

import probe_b_decoder_capacity_v1 as probe

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'outputs/research/b_decoder_capacity_v1'
probe.BASE = ROOT / 'outputs/research/b_decoder_capacity_checkpoint_v1'
original_config = probe.config


def config():
    c = original_config()
    c['protocol'] = 'b_decoder_capacity_checkpoint_diagnostic_v1'
    c['model']['checkpoint_query'] = True
    return c


probe.config = config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'benchmark', 'train'))
    parser.add_argument('--arm', choices=probe.ARMS, default=probe.ARMS[0])
    args = parser.parse_args()
    if args.mode == 'register':
        probe.register()
        path = probe.BASE / 'registration.json'
        registration = json.loads(path.read_text())
        registration['code']['scripts/probe_b_decoder_capacity_checkpoint_v1.py'] = probe.sha(__file__)
        registration['runtime_amendment'] = dict(
            before_optimizer_updates_in_this_experiment=True,
            checkpoint_query=True, identical_for_both_arms=True,
            reason='Width128 v1 benchmark OOM: 21.60GiB allocated, 22.48GiB process total plus ~1GiB other process on 24GB GPU.',
            previous_registration_sha256=probe.sha(ORIGINAL / 'registration.json'),
            previous_oom_log_sha256=probe.sha(ORIGINAL / 'benchmark_wide128.log'),
            previous_reference_preserved=True, previous_weights_reused=False,
            interpretation='Fresh matched capacity comparison; timing is checkpoint-enabled runtime only, not perf_v2 default.')
        probe.save(path, registration)
    else:
        probe.train(args.arm, args.mode == 'benchmark')


if __name__ == '__main__':
    main()
