#!/usr/bin/env python3
"""Pre-update resource amendment for the registered, unchanged MRGR probe."""
import argparse
import copy
import gc
import json
import subprocess
import time
from pathlib import Path
import torch
import probe_mrgr_multiscale_v1 as probe


def verify_amendment():
    original = probe.verify()
    path = probe.OUT / 'resource_amendment.json'
    amended = json.loads(path.read_text())
    assert amended['original_protocol_sha256'] == probe.digest(probe.OUT/'protocol.json')
    assert amended['runner_sha256'] == probe.digest(Path(__file__))
    assert probe.digest(path) == (probe.OUT/'resource_amendment.sha256').read_text().strip()
    return original, amended


def prepare():
    original = probe.verify()
    assert not any((probe.OUT/a/'metrics.jsonl').exists() for a in probe.ARMS)
    if (probe.OUT/'resource_amendment.json').exists():
        return verify_amendment()[1]
    config = copy.deepcopy(original['config']); config['training']['device'] = 'cuda:3'
    guard = probe.GTReadGuard(config['dataset']['root'], probe.OUT/'checkpoints')
    with guard:
        model = probe.model_for(config, 'multiscale', 'cuda:3')
        ds = probe.selfsup.make_dataset(config)
        batch = probe.selfsup.move_dual_batch(probe.dual_measurement_collate([ds[0], ds[1]]), 'cuda:3')
        before = probe.state_hash(model.state_dict()); times = []
        model.train(); torch.cuda.reset_peak_memory_stats('cuda:3')
        for _ in range(3):
            model.zero_grad(set_to_none=True); torch.cuda.synchronize('cuda:3'); began=time.perf_counter()
            loss, parts, volumes, projections = probe.selfsup.objective(model, batch)
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            torch.cuda.synchronize('cuda:3'); times.append(time.perf_counter()-began)
            del loss, parts, volumes, projections
        assert probe.state_hash(model.state_dict()) == before
        timing = sum(times[1:])/2
        benchmark = dict(device='cuda:3',backward_seconds=times,timed_update_equivalent_seconds=timing,
            peak_allocated_bytes=torch.cuda.max_memory_allocated('cuda:3'),gradient_norm=float(norm),
            weights_unchanged=True,optimizer_steps=0,gt_access=guard.summary())
        probe.write_json(probe.OUT/'gpu3_benchmark.json',benchmark)
        del model,batch; gc.collect(); torch.cuda.empty_cache()
    parallel = timing < 1.1
    epochs = 20 if parallel else 12
    devices = dict(reference='cuda:0',multiscale='cuda:3' if parallel else 'cuda:0')
    amended = dict(registered_at=probe.now(),status='REGISTERED BEFORE ANY OPTIMIZER UPDATES',
        original_protocol_sha256=probe.digest(probe.OUT/'protocol.json'),
        runner_sha256=probe.digest(Path(__file__)),
        reason='GPU0 shared load predicts 57.7 aggregate minutes for original 20-epoch pair; parent authorizes parallel device placement or 12-epoch fallback',
        overrides=dict(epochs=epochs,devices=devices,execution='parallel' if parallel else 'sequential'),
        original_numerical_sources_unchanged=True,
        gpu3_benchmark=benchmark,
        gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv'],text=True),
        precision='same original fp32; TF32 disabled; cuDNN benchmark disabled',
        experimental_factor='multiscale context only; physical GPU differs in parallel arm, runtime comparison descriptive',
        wall_clock_budget_minutes=40,aggregate_gpu_minutes_estimate=58 if parallel else 35,
        no_gt=True,no_formal300=True)
    probe.write_json(probe.OUT/'resource_amendment.json',amended)
    (probe.OUT/'resource_amendment.sha256').write_text(probe.digest(probe.OUT/'resource_amendment.json')+'\n')
    print(json.dumps(amended),flush=True)
    return amended


def train(arm):
    original, amended = verify_amendment()
    runtime = copy.deepcopy(original)
    runtime['epochs'] = amended['overrides']['epochs']
    runtime['config']['training']['device'] = amended['overrides']['devices'][arm]
    original_save = probe.atomic_checkpoint
    def save(path,payload):
        payload['resource_amendment_sha256'] = probe.digest(probe.OUT/'resource_amendment.json')
        payload['resource_amendment_path'] = str(probe.OUT/'resource_amendment.json')
        original_save(path,payload)
    probe.atomic_checkpoint = save
    result = probe.train_arm(runtime,arm)
    result['resource_amendment_sha256'] = probe.digest(probe.OUT/'resource_amendment.json')
    result['runtime_device'] = runtime['config']['training']['device']
    probe.write_json(probe.OUT/arm/'summary.json',result)


def compare():
    original, amended = verify_amendment()
    result = probe.compare(original)
    result['resource_amendment_sha256'] = probe.digest(probe.OUT/'resource_amendment.json')
    result['runtime_devices'] = amended['overrides']['devices']
    result['execution'] = amended['overrides']['execution']
    result['completed_epochs_per_arm'] = amended['overrides']['epochs']
    probe.write_json(probe.OUT/'comparison.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','compare'))
    parser.add_argument('--arm',choices=probe.ARMS)
    args=parser.parse_args(); torch.set_num_threads(2)
    if args.action=='prepare': prepare()
    elif args.action=='train':
        assert args.arm; train(args.arm)
    else: compare()


if __name__=='__main__': main()
