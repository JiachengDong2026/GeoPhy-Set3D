#!/usr/bin/env python3
"""Continue the already authorized frozen batch after both formal runs finish.

This only orchestrates the registered evaluation; it never changes or selects
training configurations. Any failed stage stops, retaining logs and outputs.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'outputs/research/mrgr_autonomous_optimization_v2'
OUT = RECORD / 'frozen_formal_evaluation'
RUNS = [ROOT / 'outputs/research' / name for name in (
    'mrgr_optimization_v2_control_seed20261011',
    'mrgr_optimization_v2_lr4e4_seed20261011')]
DECISION = RECORD / 'promotion_decision.json'
EVALUATOR = ROOT / 'scripts/evaluate_mrgr_optimization_frozen_v2.py'


def record(state, **extra):
    value = dict(state=state, at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    temp = RECORD / 'completion_pipeline_status.tmp'
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(RECORD / 'completion_pipeline_status.json')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def start(label, args, gpu=''):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OPENBLAS_NUM_THREADS='2',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    logfile = RECORD / f'completion_{label}.log'
    assert not logfile.exists(), f'Refuse to overwrite {logfile}'
    handle = logfile.open('x')
    process = subprocess.Popen([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                               stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    return process


def finish(process, label):
    code = process.wait()
    assert code == 0, f'{label} failed ({code}); see completion_{label}.log'


def available_pair():
    while True:
        inventory = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        free = []
        for line in inventory.splitlines():
            index, memory, utilization = [int(x.strip()) for x in line.split(',')]
            if index in (1, 2, 3) and memory < 2048 and utilization < 20:
                free.append(str(index))
        if len(free) >= 2:
            return free[:2]
        record('waiting_free_gpus', inventory=inventory)
        time.sleep(30)


def main():
    assert not OUT.exists(), 'Evaluation already exists; inspect rather than replace'
    last = None
    while True:
        states = [json.loads((run / 'status.json').read_text()) for run in RUNS]
        epochs = tuple(s['epoch'] for s in states)
        if epochs != last:
            record('waiting_formal300', epochs=dict(zip([p.name for p in RUNS], epochs)))
            last = epochs
        if all((run / 'summary.json').exists() for run in RUNS):
            summaries = [json.loads((run / 'summary.json').read_text()) for run in RUNS]
            assert all(s['completed'] and s['epochs'] == 300 and s['updates'] == 24900 for s in summaries)
            break
        assert all(s['state'] not in ('failed', 'error') for s in states), 'Training failure; inspect first'
        time.sleep(30)
    record('registering')
    finish(start('register', [EVALUATOR, 'register', '--out', OUT,
                             '--runs', *RUNS, '--decision', DECISION]), 'register')
    devices = available_pair()
    record('predicting_all_models', physical_gpus=devices)
    processes = [(start(f'predict_{i}', [EVALUATOR, 'predict', '--out', OUT,
                  '--key', run.name, '--device', 'cuda:0'], gpu), f'predict_{i}')
                 for i, (run, gpu) in enumerate(zip(RUNS, devices))]
    # Wait for both even if either fails; never score a partial batch.
    codes = [(label, process.wait()) for process, label in processes]
    assert all(code == 0 for _, code in codes), f'Prediction failed: {codes}'
    for label, args in (
        ('seal', [EVALUATOR, 'seal', '--out', OUT]),
        ('score', [EVALUATOR, 'score', '--out', OUT]),
        ('independent_verification', [ROOT / 'scripts/verify_mrgr_optimization_frozen_v2.py', '--out', OUT]),
        ('report', [ROOT / 'scripts/report_mrgr_optimization_frozen_v2.py', '--out', OUT]),
    ):
        record(label)
        finish(start(label, args), label)
    record('evaluation_complete_pending_root_interpretation', evaluation=str(OUT))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        record('failed', error=repr(error))
        raise
