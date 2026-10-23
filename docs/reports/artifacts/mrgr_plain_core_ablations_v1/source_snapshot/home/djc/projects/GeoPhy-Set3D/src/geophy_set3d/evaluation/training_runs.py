"""Additional seeds and a fresh-horizon diagnostic using the unchanged B0 training engine."""
import copy
import json
import time
from pathlib import Path

import torch
import yaml

from geophy_set3d.models import GeoLiftMeanRes3D
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.engine import make_dataset, make_loader, run_epoch
from geophy_set3d.training.utils import (seed_everything, environment, write_json, rng_state,
    restore_rng, atomic_checkpoint, save_slices, save_curves, move_batch)
from .protocols import load_protocol, require_reference, freeze_reference


def run_training(protocol_path, seed, horizon=False, resume=False):
    protocol, reference, root = load_protocol(protocol_path)
    require_reference(root); freeze_reference(protocol, reference, root)
    if horizon:
        if seed != protocol['horizon_probe']['seed']:
            raise ValueError('Horizon probe only uses the original seed')
        name = 'supervised_b0_horizon_probe'
        directory = root/'horizon_probe'/name
    else:
        if seed not in protocol['training_seeds'][1:]:
            raise ValueError('Original seed must be reused, not retrained')
        name = f'supervised_b0_seed{seed}'
        directory = root/'multiseed'/name
    config = copy.deepcopy(reference)
    config['experiment'] = dict(name=name, output_root=str(directory))
    config['training']['seed'] = seed
    if horizon:
        config['training']['max_epochs'] = protocol['horizon_probe']['max_epochs']
    paths = {kind: directory/kind for kind in ('logs','checkpoints','figures')}
    for p in paths.values(): p.mkdir(parents=True, exist_ok=True)
    last = paths['checkpoints']/'last.pt'
    if last.exists() and not resume:
        raise FileExistsError('Run already exists; use --resume')
    seed_everything(seed); torch.set_num_threads(2)
    device = config['training']['device']; model = GeoLiftMeanRes3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['max_epochs'])
    generator = torch.Generator().manual_seed(seed)
    train = make_dataset(config)
    # All seeds and the diagnostic use the exact reference val views, not seed-specific val subsets.
    val_config = copy.deepcopy(reference)
    val_config['training']['seed'] = protocol['training_validation_selection_seed']
    val = make_dataset(val_config, validation=True)
    train_loader = make_loader(train, config, True, generator)
    val_loader = make_loader(val, config)
    start, best, bad, rows = 0, float('inf'), 0, []
    env = environment(config)
    env.update(validation_selection_seed=protocol['training_validation_selection_seed'],
               reference_source_commit='e30bbcd5f2962a509bffe7a24ebdd1db709fc231',
               scheduler_t_max=config['training']['max_epochs'], fresh_initialization=not resume,
               horizon_strategy=protocol['horizon_probe']['strategy'] if horizon else None)
    if resume:
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        if checkpoint['config'] != config: raise ValueError('Resume config differs')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng'], generator)
        start, best, bad = checkpoint['epoch']+1, checkpoint['best_val_metric'], checkpoint['bad_epochs']
        rows = [json.loads(l) for l in (paths['logs']/'metrics.jsonl').read_text().splitlines() if json.loads(l)['epoch'] < start]
        (paths['logs']/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (paths['logs']/'effective_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(paths['logs']/'environment.json', env)
    write_json(paths['logs']/'validation_protocol.json',dict(split='val', pool='train', views=4, epoch=0,
               selection_seed=protocol['training_validation_selection_seed'], test_used=False))
    fixed = move_batch(projection_pool_collate([val[0]]), device)
    with (paths['logs']/'train.log').open('a' if resume else 'w') as log:
        for epoch in range(start, config['training']['max_epochs']):
            if bad >= config['training']['early_stopping_patience']: break
            train.set_epoch(epoch); val.set_epoch(0)
            torch.cuda.reset_peak_memory_stats(device); lr = optimizer.param_groups[0]['lr']; began = time.perf_counter()
            tr = run_epoch(model, train_loader, config, optimizer)
            va = run_epoch(model, val_loader, config); scheduler.step()
            score = va['loss']['total']; improved = score < best
            best = min(best, score); bad = 0 if improved else bad+1
            row = dict(epoch=epoch, train=tr, val=va, learning_rate=lr, seconds=time.perf_counter()-began,
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device), gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            rows.append(row); line = json.dumps(row, allow_nan=False)
            log.write(line+'\n'); log.flush()
            with (paths['logs']/'metrics.jsonl').open('a') as f: f.write(line+'\n')
            print(f"{name}: epoch={epoch+1} train={tr['loss']['total']:.6g} val={score:.6g} relL1={va['metrics']['mean_per_case']['relative_l1']:.6g}", flush=True)
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           epoch=epoch, best_val_metric=best, bad_epochs=bad, config=config, seed=seed,
                           git_commit=env['git_commit'], git_dirty=env['git_dirty'], rng=rng_state(generator),
                           validation_selection_seed=protocol['training_validation_selection_seed'])
            atomic_checkpoint(last, payload)
            if improved: atomic_checkpoint(paths['checkpoints']/'best.pt', payload)
            # Plotting frequency does not change optimization or checkpoint selection.
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad(): save_slices(paths['figures']/f'fixed_val_epoch_{epoch+1:03d}.png', fixed['gt_volume_64'], model(fixed), f'{name}, epoch {epoch+1}')
                save_curves(paths['figures']/'training_curves.png', rows)
        save_curves(paths['figures']/'training_curves.png', rows)
    best_row = min(rows, key=lambda r:r['val']['loss']['total'])
    write_json(paths['logs']/'training_summary.json', dict(completed=True, completed_epochs=len(rows), best_epoch=best_row['epoch'],
        best_val_loss=best, best_val_metrics=best_row['val']['metrics'], stopping_reason='early_stopping' if bad>=config['training']['early_stopping_patience'] else 'max_epochs',
        validation_selection_seed=protocol['training_validation_selection_seed'], fresh_horizon_diagnostic=horizon))
    return directory
