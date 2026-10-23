"""Bounded B3 experiments with frozen supervised checkpoint selection."""
import json
import time
from pathlib import Path
import torch
import yaml
from geophy_set3d.data.measurement_sampling import measurement_dataset
from geophy_set3d.models import build_model
from .engine import make_loader, run_epoch
from .measurement_engine import run_measurement_epoch, gradient_diagnostics
from .projection_losses import physics_weights
from .utils import (seed_everything, directories, environment, move_batch, write_json,
                    rng_state, restore_rng, atomic_checkpoint)


def train_run(config, *, initial_checkpoint=None, resume=False):
    paths = directories(config)
    seed_everything(config['training']['seed']); torch.set_num_threads(2)
    device = config['training']['device']
    env = environment(config)
    if env['git_dirty']:
        raise RuntimeError('Commit implementation/config before training')
    model = build_model(config).to(device)
    if sum(p.numel() for p in model.parameters()) != 319042:
        raise AssertionError('B2 architecture changed')
    if initial_checkpoint:
        model.load_state_dict(torch.load(initial_checkpoint, map_location=device, weights_only=False)['model'])
    optimizer = torch.optim.AdamW(model.parameters(), **{k: config['optimizer'][k] for k in ('lr', 'weight_decay')})
    scheduler = None if initial_checkpoint else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    generator = torch.Generator().manual_seed(config['training']['seed'])
    train, val = measurement_dataset(config), measurement_dataset(config, True)
    if len(train) != 166 or len(val) != 36:
        raise AssertionError('Unexpected frozen membership')
    if abs(train.y_ref - 32284.894580078173) > 1e-9:
        raise AssertionError('Unexpected frozen normalization')
    train_loader = make_loader(train, config, True, generator)
    val_loader = make_loader(val, config)
    last = paths['checkpoints']/'last.pt'
    start, best = 0, float('inf')
    rows = []
    if last.exists():
        if not resume:
            raise FileExistsError('Run exists; explicit resume required')
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Resume config differs')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng'], generator)
        start, best = checkpoint['epoch']+1, checkpoint['best_val_metric']
        rows = [json.loads(l) for l in (paths['logs']/'metrics.jsonl').read_text().splitlines()
                if json.loads(l)['epoch'] < start]
    elif initial_checkpoint:
        # A fixed, representative batch2, including a V6 sample; does not consume loader RNG.
        from geophy_set3d.data.collate import projection_pool_collate
        indices = [i for i in range(len(train)) if len(train[i]['input_station_indices']) == 6][:2]
        batch = move_batch(projection_pool_collate([train[i] for i in indices]), device)
        torch.cuda.reset_peak_memory_stats(device)
        diagnostic = gradient_diagnostics(model, batch, config)
        diagnostic.update(case_ids=batch['case_id'], input_station_ids=batch['input_station_indices'].tolist(),
                          hold_station_ids=batch['hold_station_indices'].tolist(),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        write_json(paths['logs']/'gradient_diagnostics.json', diagnostic)
        del batch
    (paths['logs']/'effective_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(paths['logs']/'environment.json', env)
    write_json(paths['logs']/'data_protocol.json', dict(train_cases=len(train), val_cases=len(val),
               normalization_scale_ppm_m=train.y_ref, validation_epoch=0, validation_views=4,
               test_used=False, pool='A', initial_checkpoint=str(initial_checkpoint) if initial_checkpoint else None,
               random_initialization=not bool(initial_checkpoint), parameter_count=319042))
    write_json(paths['logs']/'validation_station_ids.json', {
        val[i]['case_id']: dict(input=val[i]['input_station_indices'].tolist(), hold=val[i]['hold_station_indices'].tolist())
        for i in range(len(val))})
    (paths['logs']/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for epoch in range(start, config['training']['max_epochs']):
        train.set_epoch(epoch); val.set_epoch(0)
        torch.cuda.reset_peak_memory_stats(device)
        lr = optimizer.param_groups[0]['lr']; began = time.perf_counter()
        tr = run_measurement_epoch(model, train_loader, config, optimizer, physics_weights(epoch+1, config['physics']))
        va = run_epoch(model, val_loader, config)
        if scheduler:
            scheduler.step()
        score = va['loss']['total']; improved = score < best; best = min(best, score)
        row = dict(epoch=epoch, train=tr, val=va, learning_rate=lr, seconds=time.perf_counter()-began,
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        rows.append(row)
        with (paths['logs']/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(dict(experiment=config['experiment']['name'], epoch=epoch+1,
                             train_loss=tr['loss'], val_loss=score,
                             val_l1=va['metrics']['mean_per_case']['relative_l1'], seconds=row['seconds'])), flush=True)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict() if scheduler else None, epoch=epoch,
                       best_val_metric=best, config=config, git_commit=env['git_commit'], git_dirty=False,
                       seed=config['training']['seed'], rng=rng_state(generator))
        atomic_checkpoint(last, payload)
        if improved:
            atomic_checkpoint(paths['checkpoints']/'best.pt', payload)
    best_row = min(rows, key=lambda r: r['val']['loss']['total'])
    write_json(paths['logs']/'training_summary.json', dict(completed=True, completed_epochs=len(rows),
               best_epoch=best_row['epoch']+1, best_val_loss=best,
               best_val_metrics=best_row['val']['metrics'], stopping_reason='max_epochs'))
    return paths
