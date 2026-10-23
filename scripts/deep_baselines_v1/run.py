"""Common engineering smoke, fixed small pilot, canonical formal training entry."""
from baseline_context import *
import argparse
import gc
import math
import time
import torch
from torch.utils.data import Subset
from geophy_set3d.training import selfsup


def cpu():
    c.backend()
    result = regression()
    c.freeze(ROOT / 'checks/main_regression.json', dict(at=c.now(), **result))
    print('Frozen main regression bitwise PASS', flush=True)


def smoke(arm):
    c.backend()
    target = ROOT / 'smoke' / arm
    assert not (target / 'result.json').exists()
    target.mkdir(parents=True, exist_ok=True)
    config = training_config(arm, SEEDS[0], target)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with c.AccessGuard(splits=['train', 'val'], output=ROOT) as guard:
        ds = selfsup.make_dataset(config)
        loader = selfsup.make_loader(Subset(ds, range(2)), dict(config, loader=dict(num_workers=0, pin_memory=False, persistent_workers=False)))
        batch = selfsup.move_dual_batch(next(iter(loader)), 'cuda:0')
        model = build_model(arm, 'cuda:0').train()
        initial = c.canonical_worker.probe.state_hash(model.state_dict())
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        loss, parts, volumes, _ = selfsup.objective(model, batch)
        for x in volumes:
            assert x.shape == (2, 1, 64, 64, 64) and x.dtype == torch.float32
            assert torch.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        assert float(norm) > 0
        opt.step(); opt.zero_grad(set_to_none=True)
        assert c.canonical_worker.probe.state_hash(model.state_dict()) != initial
        del volumes
        model.eval()
        with torch.inference_mode():
            torch.cuda.synchronize(); t = time.perf_counter()
            pred = model(batch['A']); torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - t
        ckpath = target / 'roundtrip.pt'
        c.canonical_worker.probe.atomic_checkpoint(ckpath, dict(model=model.state_dict(), optimizer=opt.state_dict(), seed=SEEDS[0], config=config, volumetric_gt_used=False))
        state = torch.load(ckpath, map_location='cpu', weights_only=False)
        clone = build_model(arm, 'cuda:0').eval()
        clone.load_state_dict(state['model'], strict=True)
        with torch.inference_mode():
            again = clone(batch['A'])
            torch.testing.assert_close(again, pred, atol=0, rtol=0)
            altered = dict(batch['A']); altered['input_projection_raw'] = altered['input_projection_raw'] * .5
            sensitivity = float((clone(altered) - pred).abs().mean())
        assert sensitivity > 0 and math.isfinite(sensitivity)
        torch.cuda.synchronize()
        c.freeze(target / 'result.json', dict(at=c.now(), passed=True, seed=SEEDS[0], parameters=sum(p.numel() for p in model.parameters()),
            loss=float(loss.detach()), loss_parts={k: float(v.detach()) for k, v in parts.items()}, gradient_norm=float(norm),
            initial_state_sha256=initial, checkpoint_sha256=sha(ckpath), checkpoint_bytes=ckpath.stat().st_size,
            inference_seconds_batch2=inference_seconds, seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            output=dict(min=float(pred.min()), max=float(pred.max()), mean=float(pred.mean()), std=float(pred.std())),
            measurement_halving_mean_absolute_change=sensitivity, roundtrip_bitwise=True, audit=guard.summary(),
            sources=source_identity(), gpu_uuid=c.os.environ.get('CUDA_VISIBLE_DEVICES')))
    print('SMOKE PASS', arm, flush=True)


def register(arm, seed, pilot=False):
    target = ROOT / ('pilots' if pilot else 'training') / f'{arm}_seed{seed}'
    if (target / 'protocol.json').exists():
        c.verified(target / 'protocol.json')
        return target
    config = training_config(arm, seed, target)
    model = build_model(arm, 'cpu', seed)
    prior = c.verified(c.HISTORY / 'model_manifest.json')
    cases = c.verified(c.HISTORY / 'registration.json')['cases']
    c.freeze(target / 'protocol.json', dict(registered_at=c.now(), arm=arm, seed=seed, config=config, epochs=300,
        parameters=sum(p.numel() for p in model.parameters()), sources=source_identity(),
        train_cases=cases['train'], validation_cases=cases['val'], data_files=prior['training_data_sha256'],
        subset_smoke=pilot, planned_updates=10 if pilot else 24900,
        phase='engineering pilot 5 epochs x 2updates, 4train/2val; NOT formal result' if pilot else 'formal 300 epochs 166train/36val',
        pilot_stop_after=5 if pilot else None, fresh_initialization=True,
        initial_state_sha256=c.canonical_worker.probe.state_hash(model.state_dict()),
        selection='strict minimum mean validation heldout H4 relative L1; no volume GT',
        resume='canonical authoritative last.pt complete epoch with optimizer/scheduler/all RNG/loader/best/metrics',
        source_identity_note='independent model factory only; existing selfsup.objective/run_epoch and canonical train engine unchanged'))
    return target


def train(target, stop):
    target = Path(target)
    p = c.verified(target / 'protocol.json')
    validate_config(p['config'], p['arm'])
    assert c.verified(ROOT / 'smoke' / p['arm'] / 'result.json')['passed']
    if not p['subset_smoke']:
        gate = c.verified(ROOT / 'gates' / (p['arm'] + '.json'))
        assert gate['eligible_for_formal'], gate
        assert stop == 300, 'No shortened formal training budget'
    else:
        assert stop == 5
    engine = c.canonical_worker.engine
    engine.OUT = ROOT
    def dispatcher(config, arm, device):
        assert arm == p['arm'] and config['training']['seed'] == p['seed']
        model = build_model(arm, device, config['training']['seed'])
        assert c.canonical_worker.probe.state_hash(model.state_dict()) == p['initial_state_sha256']
        return model
    engine.model_for = dispatcher
    with c.AccessGuard(splits=['train', 'val'], output=ROOT) as guard:
        engine.train(target, stop_after=stop)
        c.write(target / 'boundary_audit.json', guard.summary())
    if p['subset_smoke']:
        last = torch.load(target / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        assert last['epoch'] == 5 and last['updates'] == 10
        c.freeze(target / 'pilot_complete.json', dict(at=c.now(), completed=True, formal=False, epochs=5, updates=10,
            checkpoint_sha256=sha(target / 'checkpoints/best.pt'), best_H4=last['best_score'], best_epoch=last['best_epoch'],
            elapsed_seconds=last['elapsed_seconds'], zero_volume_H4_relative_l1=1.,
            caution='Engineering learning-curve probe only: 2 validation cases, insufficient to establish scientific eligibility'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['cpu', 'smoke', 'register', 'train'])
    ap.add_argument('--arm', choices=ARMS)
    ap.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--target')
    ap.add_argument('--stop-after', type=int, default=300)
    args = ap.parse_args()
    if args.action == 'cpu': cpu()
    elif args.action == 'smoke': smoke(args.arm)
    elif args.action == 'register': print(register(args.arm, args.seed, args.pilot))
    else: train(args.target, args.stop_after)
