#!/usr/bin/env python3
"""Finite, fresh matched MRGR spatial-context probe; no GT and no formal training."""
import argparse
import copy
import datetime
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.models import build_model
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training import selfsup
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.utils import (atomic_checkpoint, environment, load_config,
    restore_rng, rng_state, seed_everything, write_json)

OUT = ROOT / 'outputs/research/mrgr_multiscale_v1'
ARMS = ('reference', 'multiscale')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class MultiscaleContext(nn.Module):
    """Low-channel context at 16/8 cubed, with exactly zero initial residual."""
    def __init__(self, width=12):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv3d(32, width, 1), nn.GroupNorm(4, width), nn.SiLU())
        def block():
            return nn.Sequential(nn.Conv3d(width, width, 3, padding=1),
                                 nn.GroupNorm(4, width), nn.SiLU())
        self.local16, self.context8, self.fuse16 = block(), block(), block()
        self.output = nn.Conv3d(width, 32, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x16 = self.local16(self.reduce(F.avg_pool3d(x, 2)))
        x8 = self.context8(F.avg_pool3d(x16, 2))
        y16 = self.fuse16(x16 + F.interpolate(x8, size=x16.shape[-3:],
                                            mode='trilinear', align_corners=True))
        return self.output(F.interpolate(y16, size=x.shape[-3:],
                                         mode='trilinear', align_corners=True))


class ContextDecoder(nn.Sequential):
    """Original decoder keys/weights preserved; context follows its three 32³ blocks."""
    def __init__(self, original):
        super().__init__(*list(original.children()))
        self.context = MultiscaleContext()

    def forward(self, x):
        for name, module in self._modules.items():
            if name == 'context':
                continue
            x = module(x)
            if name == '2':
                x = x + self.context(x)
        return x


def model_for(config, arm, device):
    # Same fresh base construction. Branch RNG is isolated from data/training RNG.
    seed_everything(config['training']['seed'])
    model = build_model(config)
    before = torch.get_rng_state()
    if arm == 'multiscale':
        torch.manual_seed(config['training']['seed'] + 1)
        model.decoder = ContextDecoder(model.decoder)
    torch.set_rng_state(before)
    return model.to(device)


def source_hashes():
    files = [*sorted((ROOT / 'src').rglob('*.py')), *sorted((ROOT / 'src').rglob('*.cpp')),
             Path(__file__), ROOT / 'configs/train_selfsup_oari_mrgr_v1.yaml']
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def verify():
    p = json.loads((OUT / 'protocol.json').read_text())
    assert digest(OUT / 'protocol.json') == (OUT / 'protocol.sha256').read_text().strip()
    assert p['sources'] == source_hashes(), 'Registered numerical sources changed'
    return p


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def register(args):
    if (OUT / 'protocol.json').exists():
        return verify()
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError('Refuse to overwrite an unregistered experiment directory')
    config = load_config(ROOT / 'configs/train_selfsup_oari_mrgr_v1.yaml')
    selfsup.validate_selfsup_config(config)
    config['training']['device'] = args.device
    guard = GTReadGuard(config['dataset']['root'], OUT / 'checkpoints')
    with guard:
        ref = model_for(config, 'reference', 'cpu')
        treatment = model_for(config, 'multiscale', 'cpu')
        s0, s1 = ref.state_dict(), treatment.state_dict()
        assert all(torch.equal(s0[k], s1[k]) for k in s0)
        extra = {k: v for k, v in s1.items() if k not in s0}
        assert extra and all(k.startswith('decoder.context.') for k in extra)
        counts = {arm: sum(v.numel() for v in m.parameters())
                  for arm, m in zip(ARMS, (ref, treatment))}
        assert counts['reference'] == 332070
        assert counts['multiscale'] - counts['reference'] <= 33207
        seed_everything(20261011)
        features = torch.randn(1, 32, 32, 32, 32)
        with torch.no_grad():
            r, t = ref.decoder(features), treatment.decoder(features)
        assert torch.equal(r, t), 'Initial decoder equivalence failed'
        train, val = selfsup.make_dataset(config), selfsup.make_dataset(config, validation=True)
        assert (len(train), len(val)) == (166, 36)
        stations = {train.records[i]['case_id']: {k: train[i][k]['input_station_indices'].tolist()
                    for k in ('A', 'B', 'H')} for i in range(len(train))}
        val_stations = {val.records[i]['case_id']: {k: val[i][k]['input_station_indices'].tolist()
                        for k in ('A', 'B', 'H')} for i in range(len(val))}
        write_json(OUT / 'station_epoch0.json', dict(train=stations, val=val_stations))
        protocol = dict(registered_at=now(), config=config, epochs=args.epochs,
            sources=source_hashes(), parameter_count=counts,
            added_parameters=counts['multiscale']-counts['reference'],
            base_initial_sha256=state_hash(s0), shared_initial_tensors_exact=True,
            initial_decoder_output_bitwise_equal=True,
            architecture='original OARI/GCA/MRGR unchanged; shared decoder 32³ features after block3 + zero-output 12-channel 16³/8³ context residual',
            seed=20261011, initialization='fresh random, no historical checkpoints',
            sampler='frozen selfsup v1 disjoint A3–5/B3–5/H2; epoch varies; val A4/B4/H4 fixed epoch0',
            train_cases=166, val_cases=36, updates_per_epoch=83,
            planned_updates_per_arm=83*args.epochs,
            precision='fp32', optimizer='frozen AdamW lr2e-4 wd1e-4 clip1 batch2',
            schedule='first epochs of cosine T_max=300; NOT a short cosine restart',
            loss='unmodified selfsup v1 heldout1 cross0.5 observed0.1 consistency0.05',
            primary='fixed final epoch Val-A heldout measurement relative L1; per-case mean of A/B reconstruction errors',
            secondary='best Val-A H checkpoint within prefix; not a full-convergence comparison',
            no_gt=True, no_formal300=True,
            benchmark_gate_seconds_two_arms=args.max_minutes*60,
            limitations='single seed, known development Val-A, incomplete training; no 3D or extreme-geometry conclusion')
        write_json(OUT / 'protocol.json', protocol)
        (OUT / 'protocol.sha256').write_text(digest(OUT / 'protocol.json') + '\n')
        for arm, state in zip(ARMS, (s0, s1)):
            p = OUT / 'checkpoints' / arm; p.mkdir(parents=True)
            atomic_checkpoint(p / 'initial.pt', dict(model=state, arm=arm,
                initialization='fresh random', epoch=0, updates=0,
                protocol_sha256=digest(OUT / 'protocol.json')))
        write_json(OUT / 'registration_audit.json', guard.summary())
        print(json.dumps(dict(stage='registered', parameters=counts,
                              added=protocol['added_parameters']), allow_nan=False), flush=True)
        return protocol


def smoke(protocol):
    target = OUT / 'smoke.json'
    if target.exists():
        return json.loads(target.read_text())
    config = protocol['config']; device = config['training']['device']
    guard = GTReadGuard(config['dataset']['root'], OUT / 'checkpoints')
    with guard:
        ds = selfsup.make_dataset(config)
        batch = selfsup.move_dual_batch(dual_measurement_collate([ds[0], ds[1]]), device)
        predictions = {}; records = {}
        for arm in ARMS:
            model = model_for(config, arm, device)
            model.eval()
            with torch.no_grad():
                predictions[arm] = model(batch['A']).cpu()
            model.train(); torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device); start = time.perf_counter()
            # Warmup and two timed full-objective backwards; no optimizer step.
            times = []
            before = state_hash(model.state_dict())
            for iteration in range(3):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device); began = time.perf_counter()
                loss, parts, volumes, projections = selfsup.objective(model, batch)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                torch.cuda.synchronize(device); times.append(time.perf_counter() - began)
                del loss, parts, volumes, projections
            if arm == 'multiscale':
                gradient = model.decoder.context.output.weight.grad
                assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                context_gradient = float(gradient.norm())
            else:
                context_gradient = None
            assert before == state_hash(model.state_dict()), 'Smoke changed model weights'
            records[arm] = dict(backward_seconds=times, timed_update_equivalent_seconds=sum(times[1:])/2,
                gradient_norm=float(norm), context_output_gradient_norm=context_gradient,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device), optimizer_steps=0,
                weights_unchanged=True, total_seconds=time.perf_counter()-start)
            del model; gc.collect(); torch.cuda.empty_cache()
        assert torch.equal(predictions['reference'], predictions['multiscale']), 'Initial full-model outputs differ'
        # Conservative allowance for validation plus loader/optimizer overhead.
        estimated = sum(r['timed_update_equivalent_seconds'] for r in records.values()) * (83+18*.6) * protocol['epochs'] * 1.15
        result = dict(passed=True, full_model_initial_bitwise_equal=True, arms=records,
            estimated_two_arm_seconds=estimated, budget_seconds=protocol['benchmark_gate_seconds_two_arms'],
            within_budget=estimated <= protocol['benchmark_gate_seconds_two_arms'], gt_access=guard.summary())
        write_json(target, result); print(json.dumps(dict(stage='smoke', **result)), flush=True)
        return result


def final_cases(model, loader, config):
    model.eval(); rows = []
    with torch.no_grad():
        for raw in loader:
            b = selfsup.move_dual_batch(raw, config['training']['device'])
            values = []
            for key in ('A', 'B'):
                volume = model(b[key]); predicted = project_input(volume, b['H'])
                target = b['H']; valid = target['input_valid_ray'].bool() & target['view_mask'][:, :, None, None, None]
                values.append([relative_projection_metrics(predicted[i], target['input_projection_raw'][i], valid[i])
                               for i in range(len(volume))])
            for i, case in enumerate(b['A']['case_id']):
                rows.append(dict(case_id=case, heldout_relative_l1=.5*(values[0][i]['relative_l1']+values[1][i]['relative_l1']),
                    heldout_relative_l2=.5*(values[0][i]['relative_l2']+values[1][i]['relative_l2'])))
    return rows


def train_arm(protocol, arm):
    config = protocol['config']; device = config['training']['device']
    p = OUT / arm; p.mkdir(exist_ok=True)
    if (p / 'summary.json').exists():
        return json.loads((p / 'summary.json').read_text())
    if (p / 'metrics.jsonl').exists():
        raise RuntimeError('Partial arm exists; explicit exact-resume implementation required, do not silently restart')
    guard = GTReadGuard(config['dataset']['root'], OUT / 'checkpoints')
    with guard:
        seed_everything(20261011)
        model = model_for(config, arm, device)
        initial = torch.load(OUT / 'checkpoints' / arm / 'initial.pt', map_location='cpu', weights_only=False)
        assert state_hash(initial['model']) == state_hash(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        generator = torch.Generator().manual_seed(20261011)
        train, val = selfsup.make_dataset(config), selfsup.make_dataset(config, validation=True)
        tr_loader = selfsup.make_loader(train, config, True, generator)
        va_loader = selfsup.make_loader(val, config)
        write_json(p / 'environment.json', environment(config))
        began = time.perf_counter(); started_at = now(); best = float('inf'); rows = []
        for epoch in range(protocol['epochs']):
            train.set_epoch(epoch); val.set_epoch(0)
            torch.cuda.reset_peak_memory_stats(device); start = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            tr = selfsup.run_epoch(model, tr_loader, config, optimizer)
            va = selfsup.run_epoch(model, va_loader, config)
            scheduler.step(); score = selfsup.selection_metric(va); improved = score < best; best = min(best, score)
            row = dict(epoch=epoch+1, train=tr, val=va, learning_rate=lr,
                       seconds=time.perf_counter()-start,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       loader_rng_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
            append(p / 'metrics.jsonl', row); rows.append(row)
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                epoch=epoch+1, updates=(epoch+1)*83, rng=rng_state(generator), config=config,
                arm=arm, protocol_sha256=digest(OUT/'protocol.json'), volumetric_gt_used=False)
            atomic_checkpoint(OUT/'checkpoints'/arm/'last.pt', payload)
            if improved:
                atomic_checkpoint(OUT/'checkpoints'/arm/'best.pt', payload)
            write_json(p/'gt_access_audit.json', guard.summary())
            print(f"{arm} {epoch+1}/{protocol['epochs']} trainH={tr['measurement']['heldout']['relative_l1']:.6f} valH={score:.6f} seconds={row['seconds']:.2f}", flush=True)
        per_case = final_cases(model, va_loader, config)
        for row in per_case:
            append(p/'final_per_case.jsonl', row)
        summary = dict(completed_short_probe=True, completed_formal300=False,
            epochs=protocol['epochs'], updates=protocol['epochs']*83, started_at=started_at, ended_at=now(),
            seconds=time.perf_counter()-began, final=rows[-1], best_epoch=min(rows, key=lambda r:selfsup.selection_metric(r['val']))['epoch'],
            best_heldout_relative_l1=best, final_per_case_mean=float(np.mean([r['heldout_relative_l1'] for r in per_case])),
            gt_access=guard.summary(), final_checkpoint_sha256=digest(OUT/'checkpoints'/arm/'last.pt'),
            best_checkpoint_sha256=digest(OUT/'checkpoints'/arm/'best.pt'))
        assert not guard.summary()['gt_file_opened'] and not guard.summary()['forbidden_npy_attempts']
        write_json(p/'summary.json', summary)
        del tr_loader, va_loader, model, optimizer; gc.collect(); torch.cuda.empty_cache()
        return summary


def compare(protocol):
    summaries = {arm:json.loads((OUT/arm/'summary.json').read_text()) for arm in ARMS}
    logs = {arm:[json.loads(l) for l in (OUT/arm/'metrics.jsonl').read_text().splitlines()] for arm in ARMS}
    for a, b in zip(logs['reference'], logs['multiscale']):
        assert a['epoch'] == b['epoch'] and a['learning_rate'] == b['learning_rate']
        assert a['loader_rng_sha256'] == b['loader_rng_sha256']
        for split in ('train', 'val'):
            assert a[split]['case_station_digest'] == b[split]['case_station_digest']
            assert a[split]['sample_count'] == b[split]['sample_count']
    cases = {arm:[json.loads(l) for l in (OUT/arm/'final_per_case.jsonl').read_text().splitlines()] for arm in ARMS}
    assert [r['case_id'] for r in cases['reference']] == [r['case_id'] for r in cases['multiscale']]
    x = np.array([r['heldout_relative_l1'] for r in cases['reference']])
    y = np.array([r['heldout_relative_l1'] for r in cases['multiscale']]); delta = y-x
    rng = np.random.default_rng(20261016)
    boot = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(1)
    result = dict(completed=True, formal_training_completed=False, summaries=summaries,
        matched_shared_initial_weights=True, matched_initial_full_forward=True,
        matched_all_epoch_case_station_digests=True, matched_all_epoch_loader_rng=True,
        matched_lr_schedule=True, final_reference=float(x.mean()), final_multiscale=float(y.mean()),
        treatment_minus_reference=float(delta.mean()), relative_improvement=float(1-y.mean()/x.mean()),
        paired_case_bootstrap_95ci=np.quantile(boot, [.025,.975]).tolist(),
        wins=int((delta<0).sum()), losses=int((delta>0).sum()), ties=int((delta==0).sum()),
        gt_used=False, interpretation='measurement-only short-prefix screen, not 3D or converged superiority',
        sources_unchanged=protocol['sources']==source_hashes())
    write_json(OUT/'comparison.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for arm in ARMS:
        epochs=[r['epoch'] for r in logs[arm]]
        axes[0].plot(epochs,[r['train']['loss']['total'] for r in logs[arm]],label=arm)
        axes[1].plot(epochs,[r['val']['measurement']['heldout']['relative_l1'] for r in logs[arm]],label=arm)
    for ax,title in zip(axes,('Training objective','Val-A heldout relative L1')):
        ax.set_title(title); ax.set_xlabel('Epoch (300-epoch schedule prefix)'); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(OUT/'curves.png',dpi=160); plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('register','smoke','train','compare','all'), default='all')
    parser.add_argument('--device',default='cuda:0'); parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--max-minutes',type=float,default=40)
    args=parser.parse_args(); torch.set_num_threads(2)
    if args.epochs < 1 or args.epochs > 20:
        raise ValueError('Finite probe permits 1–20 epochs only')
    protocol=register(args) if args.action in ('register','all') else verify()
    if args.action=='register': return
    if args.action in ('smoke','all'):
        result=smoke(protocol)
        if args.action=='smoke': return
        if not result['within_budget']:
            raise RuntimeError('Measured two-arm budget exceeds preregistered ceiling; stop before training')
    if args.action in ('train','all'):
        if not json.loads((OUT/'smoke.json').read_text())['within_budget']:
            raise RuntimeError('Budget gate failed')
        for arm in ARMS:
            train_arm(protocol,arm)
    if args.action in ('compare','train','all'):
        compare(protocol)


if __name__=='__main__':
    main()
