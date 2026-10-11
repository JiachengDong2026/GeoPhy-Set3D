#!/usr/bin/env python3
"""Train supervised B0/B1/B2; select checkpoints exclusively using fixed val views."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import torch
import yaml
from geophy_set3d.models import build_model
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.engine import make_dataset, make_loader, run_epoch
from geophy_set3d.training.comparison import validate_comparison_config
from geophy_set3d.training.utils import (load_config, seed_everything, directories, environment,
    write_json, rng_state, restore_rng, atomic_checkpoint, save_slices, save_curves, move_batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train_supervised_b0_300.yaml')
    parser.add_argument('--resume')
    parser.add_argument('--gate-config', help='Reuse an existing matching B0 overfit gate')
    args = parser.parse_args()
    config = load_config(args.config); paths = directories(config)
    validate_comparison_config(config)
    gate_path = directories(load_config(args.gate_config) if args.gate_config else config, '_overfit')['logs']/'overfit_summary.json'
    if not gate_path.exists() or not json.loads(gate_path.read_text())['passed']:
        raise RuntimeError('Run and pass the 4-8-case FP32 overfit gate before full training')
    gate_config = load_config(gate_path.parent/'effective_config.yaml')
    for key in ('dataset', 'model', 'normalization', 'loss', 'optimizer'):
        if gate_config[key] != config[key]:
            raise ValueError(f'Overfit gate used a different {key} configuration')
    if gate_config['training']['seed'] != config['training']['seed']:
        raise ValueError('Overfit seed differs from full training')
    patience = config['training']['early_stopping_patience']
    seed_everything(config['training']['seed']); torch.set_num_threads(2)
    device = config['training']['device']
    if not torch.cuda.is_available() and device.startswith('cuda'):
        raise RuntimeError('CUDA requested but unavailable')
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('scheduler', {}).get('T_max', config['training']['max_epochs']))
    generator = torch.Generator().manual_seed(config['training']['seed'])
    train, val = make_dataset(config), make_dataset(config, validation=True)
    train_loader = make_loader(train, config, True, generator)
    val_loader = make_loader(val, config)
    start, best, bad, rows = 0, float('inf'), 0, []
    env = environment(config)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt['config'] != config:
            raise ValueError('Resume config differs from saved effective config')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['scheduler'])
        restore_rng(ckpt['rng'], generator)
        start, best, bad = ckpt['epoch']+1, ckpt['best_val_metric'], ckpt['bad_epochs']
        metrics_path = paths['logs']/'metrics.jsonl'
        if metrics_path.exists():
            rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if json.loads(l)['epoch'] < start]
            metrics_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif (paths['checkpoints']/'last.pt').exists():
        raise FileExistsError('Experiment already exists; use --resume or a new experiment output name')
    (paths['logs']/'effective_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(paths['logs']/'environment.json', env)
    write_json(paths['logs']/'data_protocol.json', dict(train_cases=len(train), val_cases=len(val),
               normalization_scale_ppm_m=train.y_ref, validation_epoch=0, validation_views=4,
               test_used=False, pool='A', parameter_count=sum(p.numel() for p in model.parameters())))
    write_json(paths['logs']/'validation_station_ids.json', {val[i]['case_id']: val[i]['input_station_indices'].tolist() for i in range(len(val))})
    fixed = move_batch(projection_pool_collate([val[0]]), device)
    with (paths['logs']/'train.log').open('a' if args.resume else 'w') as log:
        for epoch in range(start, config['training']['max_epochs']):
            if patience is not None and bad >= patience:
                break
            train.set_epoch(epoch); val.set_epoch(0)
            torch.cuda.reset_peak_memory_stats(device)
            lr = optimizer.param_groups[0]['lr']; began = time.perf_counter()
            tr = run_epoch(model, train_loader, config, optimizer)
            va = run_epoch(model, val_loader, config)
            scheduler.step()
            score = va['loss']['total']; improved = score < best
            best = min(best, score); bad = 0 if improved else bad+1
            row = dict(epoch=epoch, train=tr, val=va, learning_rate=lr,
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device), seconds=time.perf_counter()-began)
            rows.append(row)
            line = json.dumps(row, allow_nan=False)
            print(line, flush=True); log.write(line+'\n'); log.flush()
            with (paths['logs']/'metrics.jsonl').open('a') as f:
                f.write(line+'\n')
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           epoch=epoch, best_val_metric=best, bad_epochs=bad, config=config,
                           git_commit=env['git_commit'], git_dirty=env['git_dirty'], seed=config['training']['seed'],
                           rng=rng_state(generator))
            atomic_checkpoint(paths['checkpoints']/'last.pt', payload)
            if improved:
                atomic_checkpoint(paths['checkpoints']/'best.pt', payload)
                model.eval()
                with torch.no_grad():
                    save_slices(paths['figures']/'best_fixed_val.png', fixed['gt_volume_64'], model(fixed), f"Epoch {epoch}: {fixed['case_id'][0]}")
            if epoch % 5 == 0 or patience is not None and bad >= patience:
                save_curves(paths['figures']/'training_curves.png', rows)
        save_curves(paths['figures']/'training_curves.png', rows)
    write_json(paths['logs']/'training_summary.json', dict(completed=True, completed_epochs=len(rows),
        best_epoch=min(rows, key=lambda r:r['val']['loss']['total'])['epoch'],
        best_val_loss=best, best_val_metrics=min(rows, key=lambda r:r['val']['loss']['total'])['val']['metrics'], stopping_reason='early_stopping' if patience is not None and bad >= patience else 'max_epochs'))


if __name__ == '__main__':
    main()
