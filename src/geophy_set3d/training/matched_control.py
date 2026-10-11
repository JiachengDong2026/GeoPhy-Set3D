"""One matched supervised-only continuation; existing B3 candidates are read-only."""
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml

from geophy_set3d.data.measurement_sampling import measurement_dataset
from geophy_set3d.models import build_model
from .engine import make_loader, run_epoch
from .measurement_engine import run_measurement_epoch
from .utils import (seed_everything, environment, write_json, rng_state, restore_rng,
                    atomic_checkpoint, load_config)

B2_CHECKPOINT = Path('/data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_b2_gcsi_300/best.pt')
ROOT = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/b2_vs_b3_phys_v1')
CONTROL_NAME = 'b3_lambda_gate_control'
CANDIDATES = ('weak', 'medium', 'strong')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def control_config(candidate):
    result = copy.deepcopy(candidate)
    result['experiment']['name'] = CONTROL_NAME
    result['physics'].update(lambda_obs=0., lambda_hold=0.)
    return result


def validate_control(config, candidates, protocols):
    if config['experiment']['name'] != CONTROL_NAME:
        raise ValueError('A new, separate control experiment is required')
    if len(candidates) != 3 or len(protocols) != 3:
        raise ValueError('Match all three frozen candidates')
    for candidate, protocol in zip(candidates, protocols):
        if config != control_config(candidate):
            raise ValueError('Only experiment name and physics weights may differ')
        if protocol['initial_checkpoint'] != str(B2_CHECKPOINT):
            raise ValueError('Control and candidate initial checkpoints differ')
    if (config['physics']['lambda_obs'], config['physics']['lambda_hold']) != (0., 0.):
        raise ValueError('Supervised-only control requires zero physics weights')
    if config['train']['hold_views'] != 1 or config['validation']['hold_views'] != 1:
        raise ValueError('Preserve held-out sampling even for zero weights')
    return B2_CHECKPOINT


def candidate_protocols(config, repo):
    base = Path(config['experiment']['output_root'])/'logs'
    candidates = [load_config(Path(repo)/f'configs/train_b3_gate_{name}.yaml') for name in CANDIDATES]
    protocols = [json.loads((base/f'b3_lambda_gate_{name}'/'data_protocol.json').read_text()) for name in CANDIDATES]
    return candidates, protocols


def verify_station_sampling(config, candidates):
    datasets = [measurement_dataset(c) for c in [config, *candidates]]
    memberships = [[(r['case_id'], r['sample_id']) for r in d.records] for d in datasets]
    if any(m != memberships[0] for m in memberships[1:]):
        raise AssertionError('Train membership/order differs')
    hashes = dict(input=hashlib.sha256(), hold=hashlib.sha256())
    for epoch in range(20):
        for i, record in enumerate(datasets[0].records):
            pairs = [d.sampler.sample(d.station_count, epoch=epoch, sample_index=i,
                                      case_id=record['case_id']) for d in datasets]
            for inputs, holds in pairs:
                if len(holds) != 1 or set(inputs) & set(holds):
                    raise AssertionError('Held-out leakage')
            for inputs, holds in pairs[1:]:
                if not (inputs == pairs[0][0]).all() or not (holds == pairs[0][1]).all():
                    raise AssertionError('Control/candidate station mismatch')
            identity = dict(case_id=record['case_id'], sample_index=i, epoch=epoch)
            for key, values in zip(('input', 'hold'), pairs[0]):
                hashes[key].update(json.dumps(dict(identity, stations=values.tolist()), sort_keys=True).encode())
    return dict(passed=True, epochs=20, train_cases=len(datasets[0]), comparisons=3*20*len(datasets[0]),
                input_ids_identical=True, hold_ids_identical=True, overlap=0,
                method='replay frozen deterministic sampler for every case/index/epoch and candidate',
                station_signature_sha256={k: h.hexdigest() for k,h in hashes.items()})


def run_control_epoch(model, loader, config, optimizer):
    # The frozen engine skips both projection calls at zero weights, retaining identical
    # supervised arithmetic, clipping, sample order and optimizer updates.
    if config['physics']['lambda_obs'] or config['physics']['lambda_hold']:
        raise ValueError('Control must not compute or backpropagate projection loss')
    return run_measurement_epoch(model, loader, config, optimizer, (0., 0.))


def train_control(config, repo, resume=False):
    candidates, protocols = candidate_protocols(config, repo)
    initial_path = validate_control(config, candidates, protocols)
    output = ROOT/'lambda_gate/control'; output.mkdir(parents=True, exist_ok=True)
    write_json(output/'station_sampling_verification.json', verify_station_sampling(config, candidates))
    seed_everything(config['training']['seed']); torch.set_num_threads(2)
    env = environment(config)
    if env['git_dirty']:
        raise RuntimeError('Commit control source and config before training')
    device = config['training']['device']
    model = build_model(config).to(device)
    initial = torch.load(initial_path, map_location=device, weights_only=False)
    model.load_state_dict(initial['model'])
    if sum(p.numel() for p in model.parameters()) != 319042:
        raise AssertionError('Architecture changed')
    if any(not torch.equal(value, initial['model'][key]) for key,value in model.state_dict().items()):
        raise AssertionError('Control did not load the exact B2 weights')
    del initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'],
                                   weight_decay=config['optimizer']['weight_decay'])
    generator = torch.Generator().manual_seed(config['training']['seed'])
    train, val = measurement_dataset(config), measurement_dataset(config, True)
    if len(train) != 166 or len(val) != 36:
        raise AssertionError('Unexpected membership')
    train_loader = make_loader(train, config, True, generator); val_loader = make_loader(val, config)
    checkpoint_dir = output/'checkpoint'; checkpoint_dir.mkdir(exist_ok=True)
    last = checkpoint_dir/'last.pt'; start=0; rows=[]
    if last.exists():
        if not resume:
            raise FileExistsError('Control exists; resume explicitly')
        checkpoint=torch.load(last,map_location=device,weights_only=False)
        if checkpoint['config'] != config or checkpoint['initial_checkpoint_sha256'] != digest(initial_path):
            raise ValueError('Resume config/initial reference differs')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        restore_rng(checkpoint['rng'], generator); start=checkpoint['epoch']+1
        rows=[json.loads(l) for l in (output/'training_metrics.jsonl').read_text().splitlines()
              if json.loads(l)['epoch'] < start]
    initial_hash=digest(initial_path)
    write_json(output/'environment.json',env)
    (output/'effective_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(output/'data_protocol.json',dict(initial_checkpoint=str(initial_path),initial_checkpoint_sha256=initial_hash,
        exact_initial_weights_verified=True,train_cases=len(train),val_cases=len(val),parameter_count=319042,
        scheduler='constant',projection_training_calls=0,hold_views=1,pool='A only',pool_b_used=False,test_used=False,
        checkpoint_selection='final epoch20 only',sampler='same frozen IndependentHoldSampler as candidates'))
    validation_ids={val[i]['case_id']:dict(input=val[i]['input_station_indices'].tolist(),hold=val[i]['hold_station_indices'].tolist()) for i in range(len(val))}
    write_json(output/'validation_station_ids.json',validation_ids)
    for name in CANDIDATES:
        expected=json.loads((Path(config['experiment']['output_root'])/'logs'/f'b3_lambda_gate_{name}'/'validation_station_ids.json').read_text())
        if validation_ids != expected:
            raise AssertionError('Fixed validation station IDs differ')
    (output/'training_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for epoch in range(start,20):
        train.set_epoch(epoch); val.set_epoch(0); began=time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        tr=run_control_epoch(model,train_loader,config,optimizer)
        va=run_epoch(model,val_loader,config)
        row=dict(epoch=epoch,train=tr,val=va,learning_rate=optimizer.param_groups[0]['lr'],
                 seconds=time.perf_counter()-began,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        rows.append(row)
        with (output/'training_metrics.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(dict(epoch=epoch+1,train_loss=tr['loss'],val_loss=va['loss']['total'],
            val_l1=va['metrics']['mean_per_case']['relative_l1'],seconds=row['seconds'])),flush=True)
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=None,
            epoch=epoch,config=config,git_commit=env['git_commit'],git_dirty=False,seed=config['training']['seed'],
            initial_checkpoint=str(initial_path),initial_checkpoint_sha256=initial_hash,rng=rng_state(generator))
        atomic_checkpoint(last,payload)
    write_json(output/'training_summary.json',dict(completed=True,completed_epochs=len(rows),
        evaluated_epoch=20,checkpoint_selection='final epoch20',checkpoint=str(last),
        final_val_loss=rows[-1]['val']['loss']['total'],final_val_metrics=rows[-1]['val']['metrics'],
        stopping_reason='fixed_20_epoch_budget',physics_training_calls=0))
    return last
