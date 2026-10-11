"""Paired voxel/projection evaluation; never selects checkpoints using measurements."""
import math
import torch
from geophy_set3d.data.measurement_sampling import attach_hold
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.physics.projectors import differentiable_project
from geophy_set3d.training.projection_losses import project_measurements, geometry_for
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.losses import supervised_loss
from geophy_set3d.training.utils import move_batch, permute_views
from .protocols import evaluation_dataset


def relative_projection_metrics(prediction, reference, valid):
    p, y = prediction[valid].double(), reference[valid].double()
    d = p-y
    return dict(relative_l1=float(d.abs().sum()/y.abs().sum().clamp_min(1e-6)),
                relative_l2=float(torch.sqrt(d.square().sum()/y.square().sum().clamp_min(1e-12))),
                valid_rays=int(valid.sum()))


def evaluate_measurements(model, config, *, pool='train', views=4, selection='reference',
                          draw=0, shortcut='normal', wrong_hold=False):
    dataset = attach_hold(evaluation_dataset(config, pool=pool, views=views, selection=selection, draw=draw))
    model.eval(); voxel = VolumeMetrics(); rows = []; sup_sum = 0.
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            batch = move_batch(projection_pool_collate([sample]), config['training']['device'])
            if shortcut == 'zero_measurement':
                batch['input_projection'] = torch.zeros_like(batch['input_projection'])
                batch['input_projection_raw'] = torch.zeros_like(batch['input_projection_raw'])
            elif shortcut == 'wrong_geometry':
                batch = permute_views(batch, torch.arange(views, device=config['training']['device']).roll(1), geometry_only=True)
            prediction = model(batch)
            voxel.update(prediction, batch['gt_volume_64'], batch['case_id'])
            sup, _ = supervised_loss(prediction, batch['gt_volume_64'], **config['loss']); sup_sum += float(sup)
            row = dict(voxel.rows[-1], input_station_ids=sample['input_station_indices'].tolist(),
                       hold_station_ids=sample['hold_station_indices'].tolist(), pool=pool, draw=draw)
            if set(row['input_station_ids']) & set(row['hold_station_ids']):
                raise AssertionError('Held-out leakage')
            if shortcut == 'normal':
                for prefix, label in [('input', 'observed'), ('hold', 'heldout')]:
                    p, y, mask = project_measurements(prediction, batch, prefix)[0]
                    row[label] = relative_projection_metrics(p, y, mask)
                if wrong_hold:
                    # Fixed volume AND reference/valid mask, geometry from the next held-out case.
                    other = dataset[(i+1) % len(dataset)]
                    wrong_batch = move_batch(projection_pool_collate([other]), config['training']['device'])
                    geometry, _ = geometry_for(wrong_batch, 0, 'hold')
                    actual, _ = geometry_for(batch, 0, 'hold')
                    # If two consecutive hold IDs coincide, choose the first other station.
                    if other['hold_station_indices'].tolist() == sample['hold_station_indices'].tolist():
                        station = (int(sample['hold_station_indices'][0])+1) % dataset.station_count
                        for key in geometry:
                            geometry[key] = torch.as_tensor(dataset.geometry[key][station:station+1], device=prediction.device)
                    p = differentiable_project(prediction[0, 0], geometry, samples=96)
                    y = batch['hold_projection_raw'][0, :, 0]
                    row['wrong_heldout_geometry'] = relative_projection_metrics(p, y, actual['valid_ray'])
                    row['wrong_hold_station_id'] = int(other['hold_station_indices'][0]) if other['hold_station_indices'].tolist() != sample['hold_station_indices'].tolist() else station
            rows.append(row)
    result = dict(voxel=voxel.compute(), supervised_loss=sup_sum/len(dataset), pool=pool,
                  selection=selection, draw=draw, views=views, shortcut=shortcut)
    if shortcut == 'normal':
        for label in ('observed', 'heldout') + (('wrong_heldout_geometry',) if wrong_hold else ()):
            result[label] = {k: sum(r[label][k] for r in rows)/len(rows) for k in ('relative_l1', 'relative_l2')}
    return result, rows
