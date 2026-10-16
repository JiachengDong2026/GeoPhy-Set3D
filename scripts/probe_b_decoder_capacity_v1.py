#!/usr/bin/env python3
"""Independent, measurement-only B capacity probe; never a formal qualification.

Fixed 16 training cases and fixed disjoint A/B/H stations diagnose fitting.
Validation is eight different cases. No volumetric labels/checkpoints are read.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
import torch
from torch import nn
import yaml

from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.research_selfsup import (
    _limited, _sampling_rows, make_research_dataset, make_research_loader,
)
from geophy_set3d.training.selfsup import move_dual_batch
from geophy_set3d.training.utils import seed_everything
from research.plan_b import training as engine

BASE = ROOT / 'outputs/research/b_decoder_capacity_v1'
TEMPLATE = ROOT / 'outputs/research/scheme_b_perf_v2_formal_seed20261011/config_resolved.yaml'
ARMS = ('reference64', 'wide128')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def widen(reference):
    """Embed the reference function exactly into width128; extras can learn.

    All old neurons and their connections retain their initial values. Extra
    neurons have random inputs but zero outgoing connections into old neurons
    and the output, preserving the original function at initialization.
    """
    model = copy.deepcopy(reference)
    old = reference.decoder
    layers = []
    for index, layer in enumerate(old):
        if not isinstance(layer, nn.Linear):
            layers.append(copy.deepcopy(layer))
            continue
        first, final = index == 0, layer.out_features == 1
        new = nn.Linear(layer.in_features if first else 128, 1 if final else 128)
        with torch.no_grad():
            if final:
                new.weight.zero_()
                new.weight[:, :64].copy_(layer.weight)
                new.bias.copy_(layer.bias)
            else:
                new.weight[:64].zero_()
                new.weight[:64, :layer.in_features].copy_(layer.weight)
                new.bias[:64].copy_(layer.bias)
        layers.append(new)
    model.decoder = nn.Sequential(*layers)
    return model


def config():
    c = yaml.safe_load(TEMPLATE.read_text())
    c.update(protocol='b_decoder_capacity_diagnostic_v1', stage='development_probe')
    c['dataset'].update(train_limit=16, val_limit=8)
    c['training'].update(device='cuda:0', max_epochs=20)
    c['scheduler']['T_max'] = 300
    c['loader'].update(num_workers=0, persistent_workers=False)
    return c


def setup(c, arm):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(c['training']['seed'])
    train_base = make_research_dataset(c)
    val_base = make_research_dataset(c, validation=True)
    # Deliberately fixed layouts over all probe epochs, unlike formal training.
    train_base.set_epoch(0)
    val_base.set_epoch(0)
    train, val = _limited(train_base, 16), _limited(val_base, 8)
    reference = engine.build_model(c)
    initialization = engine.initialize_from_train_measurements(reference, train_base)
    common_sha = state_sha(reference.state_dict())
    model = reference if arm == 'reference64' else widen(reference)
    return model, train, val, initialization, common_sha


def register():
    BASE.mkdir(parents=True, exist_ok=True)
    if (BASE / 'registration.json').exists():
        raise FileExistsError('This probe has already been registered')
    c = config()
    code = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'src').rglob('*.py'))}
    code.update({str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'research/plan_b').rglob('*.py'))})
    code['scripts/probe_b_decoder_capacity_v1.py'] = sha(__file__)
    with GTReadGuard(c['dataset']['root'], BASE / 'checkpoints') as guard:
        reference, train, val, initialization, common_sha = setup(c, 'reference64')
        treatment = widen(reference)
        generator = torch.Generator().manual_seed(1234)
        test_vector = torch.randn(1, 257, 61, generator=generator)
        with torch.no_grad():
            difference = float((reference.decoder(test_vector) - treatment.decoder(test_vector)).abs().max())
        assert difference < 1e-6, difference
        shared = ('encoder.', 'refiner.')
        assert state_sha({k:v for k,v in reference.state_dict().items() if k.startswith(shared)}) == state_sha({k:v for k,v in treatment.state_dict().items() if k.startswith(shared)})
        (BASE / 'config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
        save(BASE / 'registration.json', dict(
            registered_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            code=code, config_sha256=sha(BASE / 'config.yaml'),
            arms=list(ARMS), epochs=20, updates_per_arm=160,
            max_seconds_per_arm=1200, scheduler='first20 epochs of cosine T_max300',
            fixed_layout_epoch=0, train_sampling=list(_sampling_rows(train, 0, 'train')),
            val_sampling=list(_sampling_rows(val, 0, 'val')),
            initialization=initialization, reference_initialization_sha256=common_sha,
            initial_decoder_max_abs_difference=difference,
            parameters={a:sum(p.numel() for p in m.parameters()) for a,m in zip(ARMS,(reference,treatment))},
            screening_only=True, volumetric_gt_allowed=False, final_test_allowed=False,
            question='Does width128 improve early fixed-case measurement fitting and independent-case H versus function-matched width64?',
            decision='No full training authorized by this probe; no volumetric accuracy claim.',
            gt_audit=guard.summary()))
    print('registered', BASE, flush=True)


def train(arm, benchmark=False):
    c = yaml.safe_load((BASE / 'config.yaml').read_text())
    registered = json.loads((BASE / 'registration.json').read_text())
    assert sha(BASE / 'config.yaml') == registered['config_sha256']
    for name, expected in registered['code'].items():
        assert sha(ROOT / name) == expected, name
    directory = BASE / ('benchmark_' + arm if benchmark else arm)
    directory.mkdir(exist_ok=False)
    began = time.monotonic()
    with GTReadGuard(c['dataset']['root'], directory / 'checkpoints') as guard:
        model, train_data, val_data, initialization, common_sha = setup(c, arm)
        assert common_sha == registered['reference_initialization_sha256']
        assert list(_sampling_rows(train_data, 0, 'train')) == registered['train_sampling']
        assert list(_sampling_rows(val_data, 0, 'val')) == registered['val_sampling']
        model.to(c['training']['device'])
        generator = torch.Generator().manual_seed(c['training']['seed'])
        loader = make_research_loader(train_data, c, shuffle=True, generator=generator)
        val_loader = make_research_loader(val_data, c, generator=torch.Generator().manual_seed(0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        first = move_dual_batch(next(iter(val_loader)), c['training']['device'])
        with torch.no_grad():
            baseline = engine.objective(model, first)
        assert torch.isfinite(baseline[0])
        initial = engine.run_epoch(model, val_loader, c)
        save(directory / 'initial.json', initial)
        records = []
        epochs = 1 if benchmark else registered['epochs']
        deadline = began + registered['max_seconds_per_arm']
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']
            training = engine.run_epoch(model, loader, c, optimizer, deadline=deadline)
            validation = engine.run_epoch(model, val_loader, c, deadline=deadline)
            row = dict(epoch=epoch + 1, train=training, val=validation, learning_rate=lr)
            records.append(row)
            with (directory / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(dict(arm=arm, epoch=epoch+1, train_H=training['measurement']['heldout']['relative_l1'], val_H=validation['measurement']['heldout']['relative_l1'], seconds=training['seconds']+validation['seconds'])), flush=True)
            scheduler.step()
            if time.monotonic() >= deadline:
                break
        save(directory / 'summary.json', dict(
            arm=arm, benchmark=benchmark, epochs=len(records),
            complete=bool(len(records)==epochs and all(r[k]['complete_epoch'] for r in records for k in ('train','val'))),
            optimizer_updates=sum(r['train']['optimizer_updates'] for r in records),
            initial_val=initial, final=records[-1],
            best_val_H=min(r['val']['measurement']['heldout']['relative_l1'] for r in records),
            parameters=sum(p.numel() for p in model.parameters()),
            reference_initialization_sha256=common_sha,
            final_loader_generator_sha256=state_sha({'state':generator.get_state()}),
            elapsed_seconds=time.monotonic()-began, gt_audit=guard.summary(),
            source_preserved=all(sha(ROOT/n)==h for n,h in registered['code'].items()),
            formal_qualification=False, gt_evaluated=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('register', 'benchmark', 'train'))
    p.add_argument('--arm', choices=ARMS, default=ARMS[0])
    args = p.parse_args()
    if args.mode == 'register':
        register()
    else:
        train(args.arm, args.mode == 'benchmark')
