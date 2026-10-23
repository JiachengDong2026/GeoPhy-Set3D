"""FP32 supervised loops; validation uses no gradients and a fixed station subset."""
import math
import time

import torch
from torch.utils.data import DataLoader

from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.data.collate import projection_pool_collate
from .losses import supervised_loss
from .metrics import VolumeMetrics
from .utils import move_batch


def make_dataset(config, validation=False, fixed_views=None):
    settings = config['validation'] if validation else config['train']
    views = settings.get('input_views', fixed_views)
    options = dict(min_input_views=views, max_input_views=views) if views else dict(
        min_input_views=settings['min_input_views'], max_input_views=settings['max_input_views'])
    return ProjectionPoolDataset(config['dataset']['root'], split=settings['split'],
                                 geometry_pool=settings['geometry_pool'], **options,
                                 hold_views=settings['hold_views'], base_seed=config['training']['seed'],
                                 normalization=config['normalization']['method'],
                                 statistics_file=config['normalization']['statistics_file'])


def make_loader(dataset, config, shuffle=False, generator=None):
    workers = config['loader']['num_workers']
    return DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=shuffle,
                      num_workers=workers, pin_memory=config['loader']['pin_memory'],
                      persistent_workers=config['loader']['persistent_workers'] and workers > 0,
                      collate_fn=projection_pool_collate, generator=generator)


def run_epoch(model, loader, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    device = config['training']['device']
    metrics = VolumeMetrics()
    totals = dict(total=0., smooth_l1=0., relative_l1=0.)
    count, started = 0, time.perf_counter()
    accumulation = config['training']['gradient_accumulation']
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, raw in enumerate(loader):
            batch = move_batch(raw, device)
            prediction = model(batch)
            loss, parts = supervised_loss(prediction, batch['gt_volume_64'], **config['loss'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite supervised loss')
            if training:
                # Weight micro-batches by actual sample counts, including the final partial group.
                start = (step // accumulation)*accumulation
                stop = min(start + accumulation, len(loader))
                bs = loader.batch_size
                group_samples = min(stop*bs, len(loader.dataset)) - start*bs
                (loss * len(prediction)/group_samples).backward()
                if (step+1) % accumulation == 0 or step+1 == len(loader):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'], error_if_nonfinite=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
            n = len(prediction); count += n
            for key in totals:
                totals[key] += float(parts[key].detach())*n
            metrics.update(prediction, batch['gt_volume_64'], batch['case_id'])
    return dict(loss={k: v/count for k, v in totals.items()}, metrics=metrics.compute(),
                seconds=time.perf_counter()-started)
