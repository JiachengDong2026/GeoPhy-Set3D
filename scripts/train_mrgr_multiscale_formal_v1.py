#!/usr/bin/env python3
"""Registered 300-epoch MRGR multiscale run, independent of frozen screening."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch
import probe_mrgr_multiscale_v1 as probe

ROOT = probe.ROOT
DEFAULT_OUT = ROOT / 'outputs/research/mrgr_multiscale_formal_v1_seed20261011'


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_metrics(out, rows):
    path = out / 'metrics.jsonl'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    temporary.replace(path)


def sources():
    snapshot = json.loads((ROOT / 'outputs/research/optimization_screening_20261016/frozen_source_snapshot.json').read_text())
    result = {name: probe.digest(ROOT / name) for name in snapshot['files']}
    assert result == snapshot['files'], 'Historical source snapshot changed'
    result.update(probe.source_hashes())
    result[str(Path(__file__).relative_to(ROOT))] = probe.digest(Path(__file__))
    return result


def verify(out):
    p = json.loads((out / 'protocol.json').read_text())
    assert probe.digest(out / 'protocol.json') == (out / 'protocol.sha256').read_text().strip()
    assert p['sources'] == sources(), 'Registered sources changed; stop rather than silently continue'
    probe.verify()
    return p


def verify_data(protocol):
    root = Path(protocol['config']['dataset']['root'])
    for name, expected in protocol['data_files'].items():
        assert probe.digest(root / name) == expected, f'Data changed: {name}'


def register(out, device):
    original = probe.verify()
    if out.exists():
        raise FileExistsError(f'Refuse to reuse existing run: {out}')
    out.mkdir(parents=True)
    (out / 'checkpoints').mkdir()
    config = copy.deepcopy(original['config'])
    config['training']['device'] = device
    config['experiment'] = dict(name=out.name, output_root=str(out))
    probe.selfsup.validate_selfsup_config(config)
    frozen_data = json.loads((ROOT / 'outputs/research/optimization_screening_20261016/data_identity_postflight.json').read_text())['files']
    p = dict(registered_at=probe.now(), stage='formal300', config=config,
        epochs=300, updates_per_epoch=83, planned_updates=24900,
        seed=20261011, parameters=344678, context_width=12,
        architecture=original['architecture'], sources=sources(), data_files=frozen_data,
        screening_protocol_sha256=probe.digest(probe.OUT / 'protocol.json'),
        screening_result_sha256=probe.digest(probe.OUT / 'comparison.json'),
        initialization='fresh random using the unchanged screened model_for; no screening/historical weights loaded',
        selector='lowest mean Val-A heldout projection relative L1; strict improvement; no early stopping',
        optimizer='AdamW lr2e-4 wd1e-4 clip1 batch2', schedule='cosine T_max=300, all 300 epochs',
        precision='FP32, TF32 off, cudnn benchmark off',
        gt_allowed=False, final_test_allowed=False,
        evaluation_plan='After training freeze best/last, then independently freeze predictions and score ordinary/multi-layout validation 3D; final test untouched',
        persistence='last.pt atomic authoritative epoch commit with optimizer, scheduler, RNG, loader state, full metrics and best weights; best.pt inference checkpoint; snapshot every25 epochs',
        resume='explicit --resume only; restore authoritative checkpoint, prime persistent workers before restoring RNG; deterministic dataset sampling',
        gpu_status_at_registration=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True))
    with probe.GTReadGuard(config['dataset']['root'], out / 'checkpoints') as guard:
        verify_data(p)
        train = probe.selfsup.make_dataset(config)
        val = probe.selfsup.make_dataset(config, validation=True)
        assert (len(train), len(val)) == (166, 36)
        model = probe.model_for(config, 'multiscale', 'cpu')
        assert sum(t.numel() for t in model.parameters()) == p['parameters']
        shared = {k:v for k,v in model.state_dict().items() if not k.startswith('decoder.context.')}
        assert probe.state_hash(shared) == original['base_initial_sha256']
        assert torch.count_nonzero(model.decoder.context.output.weight) == 0
        p['initial_model_sha256'] = probe.state_hash(model.state_dict())
        p['train_case_ids'] = [r['case_id'] for r in train.records]
        p['validation_case_ids'] = [r['case_id'] for r in val.records]
        p['registration_audit'] = guard.summary()
        write(out / 'protocol.json', p)
        (out / 'protocol.sha256').write_text(probe.digest(out / 'protocol.json') + '\n')
        probe.atomic_checkpoint(out / 'checkpoints/initial.pt', dict(model=model.state_dict(),
            epoch=0, config=config, protocol_sha256=probe.digest(out / 'protocol.json'), volumetric_gt_used=False))
    write(out / 'status.json', dict(state='registered', epoch=0, max_epochs=300, updated_at=probe.now()))
    print(json.dumps(dict(stage='registered', out=str(out), parameters=p['parameters'], epochs=300)), flush=True)


def plot(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for split in ('train','val'):
        axes[0].plot([r['epoch'] for r in rows], [r[split]['loss']['total'] for r in rows], label=split)
        axes[1].plot([r['epoch'] for r in rows], [100*r[split]['measurement']['heldout']['relative_l1'] for r in rows], label=split)
    for ax, title in zip(axes, ('Self-supervised total loss','Heldout projection relative L1 (%)')):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_yscale('log'); ax.grid(alpha=.2); ax.legend()
    temporary = out / 'curves.tmp.png'
    fig.savefig(temporary, dpi=160); plt.close(fig)
    temporary.replace(out / 'curves.png')


def prime_for_resume(train_loader, val_loader):
    # Persistent workers are already alive after epoch1 in uninterrupted training.
    # Start the replacement workers BEFORE restoring saved sampler/global RNG.
    # Iteration is deliberately discarded; the following run_epoch resets them.
    for loader in (train_loader, val_loader):
        if loader.num_workers and loader.persistent_workers:
            iter(loader)


def publish_best(out, payload):
    probe.atomic_checkpoint(out / 'checkpoints/best.pt', dict(model=payload['best_model'],
        config=payload['config'], epoch=payload['best_epoch'], selection_metric=payload['best_score'],
        arm='multiscale', protocol_sha256=payload['protocol_sha256'], volumetric_gt_used=False))


def train(out, resume):
    protocol = verify(out)
    config = protocol['config']; device = config['training']['device']
    last = out / 'checkpoints/last.pt'
    if last.exists() and not resume:
        raise RuntimeError('Existing checkpoint requires explicit --resume')
    if resume and not last.exists():
        raise FileNotFoundError('No committed epoch to resume')
    if (out / 'metrics.jsonl').exists() and not resume:
        raise RuntimeError('Refuse to overwrite existing metrics')
    with probe.GTReadGuard(config['dataset']['root'], out / 'checkpoints') as guard:
        verify_data(protocol)
        model = probe.model_for(config, 'multiscale', device)
        assert probe.state_hash(model.state_dict()) == protocol['initial_model_sha256']
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        generator = torch.Generator().manual_seed(20261011)
        ds = probe.selfsup.make_dataset(config)
        vd = probe.selfsup.make_dataset(config, validation=True)
        loader = probe.selfsup.make_loader(ds, config, True, generator)
        val_loader = probe.selfsup.make_loader(vd, config)
        rows = []; best = float('inf'); best_epoch = 0; best_model = None; previous_seconds = 0.
        if resume:
            saved = torch.load(last, map_location='cpu', weights_only=False)
            assert saved['protocol_sha256'] == probe.digest(out / 'protocol.json')
            assert saved['config'] == config
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            rows = saved['metrics']; best = saved['best_score']; best_epoch = saved['best_epoch']
            best_model = saved['best_model']; previous_seconds = saved['elapsed_seconds']
            assert len(rows) == saved['epoch'] and saved['updates'] == len(rows)*83
            prime_for_resume(loader, val_loader)
            probe.restore_rng(saved['rng'], generator)
            write_metrics(out, rows); publish_best(out, saved)
            probe.append(out / 'events.jsonl', dict(event='resume', epoch=len(rows), timestamp=probe.now(), pid=os.getpid()))
            del saved
        write(out / 'environment.json', dict(probe.environment(config), pid=os.getpid(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), resumed=resume))
        began = time.perf_counter()
        write(out / 'status.json', dict(state='running', epoch=len(rows), max_epochs=300, pid=os.getpid(), updated_at=probe.now()))
        for epoch in range(len(rows), 300):
            ds.set_epoch(epoch); vd.set_epoch(0)
            torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
            start = time.perf_counter(); lr = optimizer.param_groups[0]['lr']
            tr = probe.selfsup.run_epoch(model, loader, config, optimizer)
            va = probe.selfsup.run_epoch(model, val_loader, config)
            torch.cuda.synchronize(device)
            assert (tr['sample_count'], va['sample_count']) == (166, 36)
            scheduler.step(); score = probe.selfsup.selection_metric(va)
            assert math.isfinite(score)
            improved = score < best
            if improved:
                best = score; best_epoch = epoch+1
                best_model = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            row = dict(epoch=epoch+1, train=tr, val=va, learning_rate=lr,
                seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                loader_rng_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
            rows.append(row)
            elapsed = previous_seconds + time.perf_counter()-began
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                epoch=epoch+1, updates=(epoch+1)*83, rng=probe.rng_state(generator), config=config,
                arm='multiscale', protocol_sha256=probe.digest(out / 'protocol.json'), volumetric_gt_used=False,
                best_model=best_model, best_epoch=best_epoch, best_score=best, metrics=rows, elapsed_seconds=elapsed)
            # last.pt is the commit. On resume it reconstructs metrics and best.pt.
            probe.atomic_checkpoint(last, payload)
            if improved:
                publish_best(out, payload)
            if (epoch+1) % 25 == 0:
                probe.atomic_checkpoint(out / f'checkpoints/epoch_{epoch+1:03d}.pt', dict(model=model.state_dict(),
                    epoch=epoch+1, config=config, arm='multiscale', protocol_sha256=payload['protocol_sha256'], volumetric_gt_used=False))
            write_metrics(out, rows)
            write(out / 'gt_access_audit.json', guard.summary())
            warm = rows[-10:] if len(rows)>1 else rows
            remaining = sum(r['seconds'] for r in warm)/len(warm)*(300-epoch-1)
            write(out / 'status.json', dict(state='running' if epoch<299 else 'finalizing',
                epoch=epoch+1, max_epochs=300, updates=(epoch+1)*83, val_H=score,
                best_val_H=best, best_epoch=best_epoch, last_epoch_seconds=row['seconds'],
                elapsed_seconds=elapsed, estimated_remaining_hours=remaining/3600,
                pid=os.getpid(), updated_at=probe.now()))
            print(f"epoch={epoch+1}/300 trainLoss={tr['loss']['total']:.7f} valH={score:.7f} bestH={best:.7f}@{best_epoch} seconds={row['seconds']:.2f} ETA_hours={remaining/3600:.2f}", flush=True)
            if epoch == 0 or (epoch+1) % 5 == 0:
                plot(out, rows)
            del payload
        verify(out)
        audit = guard.summary()
        assert all(audit[k] == 0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        summary = dict(completed=True, epochs=300, updates=24900, best_epoch=best_epoch,
            best_val_H=best, final_val_H=rows[-1]['val']['measurement']['heldout']['relative_l1'],
            elapsed_seconds=previous_seconds+time.perf_counter()-began, finished_at=probe.now(),
            best_checkpoint_sha256=probe.digest(out / 'checkpoints/best.pt'),
            last_checkpoint_sha256=probe.digest(last), sources_preserved=True,
            gt_audit=audit, volumetric_evaluation_completed=False, final_test_evaluated=False)
        write(out / 'summary.json', summary)
        write(out / 'status.json', dict(summary, state='complete', epoch=300, max_epochs=300))
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register','train','verify'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    out = args.out.resolve()
    if args.action == 'register':
        register(out, args.device)
    elif args.action == 'verify':
        p = verify(out); verify_data(p); print('Sources and data verified', flush=True)
    else:
        # Prevent a second invocation from sharing the same output/checkpoint files.
        with (out / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                train(out, args.resume)
            except BaseException as error:
                write(out / 'failure.json', dict(timestamp=probe.now(), error=repr(error), pid=os.getpid()))
                write(out / 'status.json', dict(state='failed', error=repr(error), pid=os.getpid(), updated_at=probe.now()))
                raise


if __name__ == '__main__':
    main()
