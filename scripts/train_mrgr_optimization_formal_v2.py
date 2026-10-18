#!/usr/bin/env python3
"""Fresh, registered 300-epoch training of independently screened MRGR variants."""
import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time

import torch
import train_mrgr_multiscale_formal_v1 as old

base = old.probe
ROOT = base.ROOT
FACTORIES = {
    'baseline': ('probe_mrgr_multiscale_v1', 'multiscale'),
    'content': ('mrgr_scale_fusion_v2', 'C_content'),
    'global': ('mrgr_scale_fusion_v2', 'E_global'),
    'geometry': ('mrgr_adaptive_geometry_v2', 'D_geometry'),
    'geometry_control': ('mrgr_adaptive_geometry_v2', 'C_content'),
    'two_step': ('mrgr_two_step_v2', 'T_two'),
    'one_step_control': ('mrgr_two_step_v2', 'B_one'),
}
ORIGINAL_OBJECTIVE = base.selfsup.objective


def read(path):
    return json.loads(Path(path).read_text())


def model_for(config, candidate, device):
    module, arm = FACTORIES[candidate]
    return importlib.import_module(module).model_for(config, arm, device)


def install_objective(config):
    """Process-local recombination; never change the frozen source or loss dictionary."""
    weight = config['loss']['consistency']
    assert set(config['loss']) == set(base.selfsup.FROZEN_LOSS)
    assert all(config['loss'][k] == v for k, v in base.selfsup.FROZEN_LOSS.items() if k != 'consistency')
    if weight == base.selfsup.FROZEN_LOSS['consistency']:
        base.selfsup.objective = ORIGINAL_OBJECTIVE
    else:
        def objective(model, batch):
            total, parts, fields, projections = ORIGINAL_OBJECTIVE(model, batch)
            total = total + (weight - base.selfsup.FROZEN_LOSS['consistency']) * parts['consistency']
            assert bool(torch.isfinite(total))
            return total, parts, fields, projections
        base.selfsup.objective = objective


def sources(candidate):
    result = old.sources()
    names = {Path(__file__), ROOT / 'scripts/mrgr_ocmsf_v1.py'}
    module, _ = FACTORIES[candidate]
    names.add(ROOT / 'scripts' / f'{module}.py')
    for path in sorted(names):
        result[str(path.relative_to(ROOT))] = base.digest(path)
    return result


def verify(out):
    out = Path(out)
    p = read(out / 'protocol.json')
    assert base.digest(out / 'protocol.json') == (out / 'protocol.sha256').read_text().strip()
    assert p['sources'] == sources(p['candidate']), 'Registered numerical sources changed'
    for name, sha in p['screening_evidence_sha256'].items():
        assert base.digest(Path(name)) == sha, name
    old.verify(old.DEFAULT_OUT)
    old.verify_data(p)
    return p


def register(args):
    out = Path(args.out).resolve()
    reference = old.verify(old.DEFAULT_OUT)
    assert not out.exists(), f'Refuse to overwrite run: {out}'
    evidence = {str(Path(p).resolve()): base.digest(Path(p)) for p in args.evidence}
    assert evidence, 'Register the completed screening and promotion decision first'
    config = copy.deepcopy(reference['config'])
    config['training']['device'] = args.device
    config['training']['seed'] = args.seed
    config['optimizer']['lr'] = args.lr
    config['loss']['consistency'] = args.consistency
    config['experiment'] = dict(name=out.name, output_root=str(out))
    assert args.lr > 0 and args.consistency >= 0
    out.mkdir(parents=True)
    (out / 'checkpoints').mkdir()
    with base.GTReadGuard(config['dataset']['root'], out / 'checkpoints') as guard:
        model = model_for(config, args.candidate, 'cpu')
        train = base.selfsup.make_dataset(config)
        val = base.selfsup.make_dataset(config, validation=True)
        assert (len(train), len(val)) == (166, 36)
        p = dict(registered_at=base.now(), candidate=args.candidate, config=config,
            epochs=300, updates_per_epoch=83, planned_updates=24900, seed=args.seed,
            parameters=sum(t.numel() for t in model.parameters()), sources=sources(args.candidate),
            data_files=reference['data_files'], screening_evidence_sha256=evidence,
            initial_model_sha256=base.state_hash(model.state_dict()),
            train_case_ids=[x['case_id'] for x in train.records],
            validation_case_ids=[x['case_id'] for x in val.records],
            initialization='fresh factory construction; no historical/screened checkpoints loaded',
            selector='strict lowest mean ordinary validation heldout projection relative L1; all300 epochs; no early stopping',
            precision='FP32 TF32off cudnn benchmarkoff; nondeterministic CUDA backward may remain',
            persistence='last.pt authoritative atomic epoch commit; model/optimizer/scheduler/best/RNG/loader/metrics restored on explicit resume',
            gt_allowed=False, final_test_allowed=False,
            evaluation_plan='freeze best checkpoint and every prediction before separate validation GT64 scoring; original17 matched layouts',
            registration_audit=guard.summary())
        old.verify_data(p)
        old.write(out / 'protocol.json', p)
        (out / 'protocol.sha256').write_text(base.digest(out / 'protocol.json') + '\n')
        base.atomic_checkpoint(out / 'checkpoints/initial.pt', dict(model=model.state_dict(),
            config=config, epoch=0, candidate=args.candidate,
            protocol_sha256=base.digest(out / 'protocol.json'), volumetric_gt_used=False))
    old.write(out / 'status.json', dict(state='registered', epoch=0, max_epochs=300))
    print(json.dumps(dict(registered=str(out), candidate=args.candidate, parameters=p['parameters'])), flush=True)


def publish_best(out, payload):
    base.atomic_checkpoint(out / 'checkpoints/best.pt', dict(model=payload['best_model'],
        config=payload['config'], epoch=payload['best_epoch'], selection_metric=payload['best_score'],
        candidate=payload['candidate'], protocol_sha256=payload['protocol_sha256'], volumetric_gt_used=False))


def train(out, resume=False):
    out = Path(out).resolve()
    p = verify(out)
    config = p['config']
    install_objective(config)
    device = config['training']['device']
    last = out / 'checkpoints/last.pt'
    assert resume == last.exists(), 'Existing epoch commit requires --resume; new run must not request it'
    assert resume or not (out / 'metrics.jsonl').exists()
    with base.GTReadGuard(config['dataset']['root'], out / 'checkpoints') as guard:
        old.verify_data(p)
        model = model_for(config, p['candidate'], device)
        assert base.state_hash(model.state_dict()) == p['initial_model_sha256']
        optimizer = torch.optim.AdamW(model.parameters(), **{k: config['optimizer'][k] for k in ('lr', 'weight_decay')})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        generator = torch.Generator().manual_seed(p['seed'])
        ds = base.selfsup.make_dataset(config)
        vd = base.selfsup.make_dataset(config, validation=True)
        loader = base.selfsup.make_loader(ds, config, True, generator)
        val_loader = base.selfsup.make_loader(vd, config)
        rows = []
        best, best_epoch, best_model, previous = float('inf'), 0, None, 0.
        if resume:
            saved = torch.load(last, map_location='cpu', weights_only=False)
            assert saved['protocol_sha256'] == base.digest(out / 'protocol.json') and saved['config'] == config
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            rows = saved['metrics']
            best, best_epoch, best_model = saved['best_score'], saved['best_epoch'], saved['best_model']
            previous = saved['elapsed_seconds']
            assert len(rows) == saved['epoch'] and saved['updates'] == 83 * len(rows)
            old.prime_for_resume(loader, val_loader)
            base.restore_rng(saved['rng'], generator)
            old.write_metrics(out, rows)
            publish_best(out, saved)
            base.append(out / 'events.jsonl', dict(event='resume', epoch=len(rows), time=base.now()))
            del saved
        old.write(out / 'environment.json', dict(base.environment(config), pid=os.getpid(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), resumed=resume,
            physical_gpu_uuid=str(torch.cuda.get_device_properties(device).uuid)))
        began = time.perf_counter()
        for epoch in range(len(rows), 300):
            ds.set_epoch(epoch)
            vd.set_epoch(0)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            tr = base.selfsup.run_epoch(model, loader, config, optimizer)
            va = base.selfsup.run_epoch(model, val_loader, config)
            torch.cuda.synchronize(device)
            assert (tr['sample_count'], va['sample_count']) == (166, 36)
            scheduler.step()
            score = base.selfsup.selection_metric(va)
            assert math.isfinite(score)
            improved = score < best
            if improved:
                best, best_epoch = score, epoch + 1
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # Standard total allows comparison when the consistency coefficient differs.
            for split in (tr, va):
                split['standard_total'] = sum(base.selfsup.FROZEN_LOSS[k] * split['loss'][k]
                    for k in ('hold', 'cross', 'observed', 'consistency'))
            row = dict(epoch=epoch+1, train=tr, val=va, learning_rate=lr,
                seconds=time.perf_counter()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                loader_rng_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
            rows.append(row)
            elapsed = previous + time.perf_counter() - began
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                epoch=epoch+1, updates=(epoch+1)*83, rng=base.rng_state(generator), config=config,
                candidate=p['candidate'], protocol_sha256=base.digest(out / 'protocol.json'), volumetric_gt_used=False,
                best_model=best_model, best_epoch=best_epoch, best_score=best, metrics=rows, elapsed_seconds=elapsed)
            base.atomic_checkpoint(last, payload)
            if improved:
                publish_best(out, payload)
            if (epoch+1) % 25 == 0:
                base.atomic_checkpoint(out / f'checkpoints/epoch_{epoch+1:03d}.pt',
                    dict(model=model.state_dict(), config=config, epoch=epoch+1, candidate=p['candidate'],
                        protocol_sha256=payload['protocol_sha256'], volumetric_gt_used=False))
            old.write_metrics(out, rows)
            old.write(out / 'gt_access_audit.json', guard.summary())
            old.write(out / 'status.json', dict(state='running' if epoch < 299 else 'finalizing',
                epoch=epoch+1, max_epochs=300, candidate=p['candidate'], updates=(epoch+1)*83,
                val_H=score, best_val_H=best, best_epoch=best_epoch, elapsed_seconds=elapsed,
                last_epoch_seconds=row['seconds'], estimated_remaining_hours=sum(r['seconds'] for r in rows[-10:])/len(rows[-10:])*(299-epoch)/3600,
                pid=os.getpid(), updated_at=base.now()))
            print(f"epoch={epoch+1}/300 H={score:.7f} best={best:.7f}@{best_epoch} seconds={row['seconds']:.2f}", flush=True)
            if epoch == 0 or (epoch+1) % 10 == 0:
                old.plot(out, rows)
            del payload
        verify(out)
        audit = guard.summary()
        assert all(audit[k] == 0 for k in ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts'))
        summary = dict(completed=True, epochs=300, updates=24900, candidate=p['candidate'], seed=p['seed'],
            best_epoch=best_epoch, best_val_H=best,
            final_val_H=base.selfsup.selection_metric(rows[-1]['val']),
            elapsed_seconds=previous+time.perf_counter()-began, finished_at=base.now(),
            best_checkpoint_sha256=base.digest(out / 'checkpoints/best.pt'), last_checkpoint_sha256=base.digest(last),
            sources_preserved=True, gt_audit=audit, volumetric_evaluation_completed=False, final_test_evaluated=False)
        old.write(out / 'summary.json', summary)
        old.write(out / 'status.json', dict(summary, state='complete', epoch=300, max_epochs=300))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'train', 'verify'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--candidate', choices=tuple(FACTORIES), default='baseline')
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--consistency', type=float, default=.05)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--evidence', nargs='*', default=[])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.action == 'register':
        register(args)
    elif args.action == 'verify':
        p = verify(Path(args.out)); old.verify_data(p)
        print('Registered numerical sources, evidence and data verified', flush=True)
    else:
        # Separate instances cannot write the same run concurrently.
        with (Path(args.out) / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(args.out, args.resume)


if __name__ == '__main__':
    main()
