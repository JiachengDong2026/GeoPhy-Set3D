"""Paired MRGR stages, raw projection metrics and evaluation-only shortcuts."""
import json
from pathlib import Path
import numpy as np
import torch
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training.projection_losses import project_measurements
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.losses import supervised_loss
from geophy_set3d.training.utils import move_batch, write_json, permute_views
from .measurement_consistency import relative_projection_metrics
from .view_selection import coverage_descriptor


def evaluate_stages(model, dataset, config, output, *, condition, draw=0, mode='normal', heldout=False, diagnostics=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model.eval(); metrics = {k: VolumeMetrics() for k in ('C0', 'C1')}
    rows = {k: [] for k in metrics}; losses = {k: 0. for k in metrics}
    projection = {k: [] for k in metrics}; hold = {k: [] for k in metrics}
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]; batch = move_batch(projection_pool_collate([sample]), config['training']['device'])
            s = model(batch, return_stages=True, residual_mode=mode)
            ids = sample['input_station_indices'].tolist()
            az = dataset.geometry['station_azimuth_deg'][ids]
            pp = {'C0': s['measurement_prediction_Y0'], 'C1': project_input(s['C1'], batch)}
            valid = batch['input_valid_ray'][0, :, 0].bool()
            for stage in metrics:
                if not torch.isfinite(s[stage]).all():
                    raise FloatingPointError('Nonfinite MRGR evaluation')
                metrics[stage].update(s[stage], batch['gt_volume_64'], batch['case_id'])
                loss, _ = supervised_loss(s[stage], batch['gt_volume_64'], **config['loss'])
                losses[stage] += float(loss)
                projection[stage].append(relative_projection_metrics(pp[stage][0, :, 0], batch['input_projection_raw'][0, :, 0], valid))
                row = dict(metrics[stage].rows[-1], condition=condition, draw_id=draw,
                           selected_station_ids=ids, station_azimuth_deg=az.tolist(),
                           pool=dataset.geometry_pool, views=len(ids), observed=projection[stage][-1],
                           **coverage_descriptor(az))
                if heldout:
                    hids = sample['hold_station_indices'].tolist()
                    assert not set(ids) & set(hids)
                    p, y, mask = project_measurements(s[stage], batch, 'hold')[0]
                    hold[stage].append(relative_projection_metrics(p, y, mask))
                    row['hold_station_ids'] = hids; row['heldout'] = hold[stage][-1]
                rows[stage].append(row)
            if diagnostics is not None and i in (0, 18, 35):
                d = Path(diagnostics); d.mkdir(exist_ok=True)
                name = f'{condition}_draw{draw}_val{i}'
                arrays = {k: v.detach().cpu().numpy() for k, v in
                    dict(Y_raw=batch['input_projection_raw'], P_C0=pp['C0'],
                         residual=s['measurement_residual'], P_C1=pp['C1'], valid_ray=valid).items()}
                np.savez_compressed(d/(name+'.npz'), **arrays)
                write_json(d/(name+'.json'), dict(case_index=i, case_id=sample['case_id'], station_ids=ids,
                    C0=metrics['C0'].rows[-1], C1=metrics['C1'].rows[-1],
                    residual_feature_abs_mean=float(s['residual_features'].abs().mean()),
                    residual_lifted_abs_mean=float(s['residual_lifted'].abs().mean()),
                    delta_context_abs_mean=float(s['residual_context'].abs().mean()),
                    delta_context_abs_max=float(s['residual_context'].abs().max())))
    summaries = {}
    for stage in metrics:
        summary = dict(**metrics[stage].compute(), supervised_loss=losses[stage]/len(dataset),
                       condition=condition, draw_id=draw, residual_mode=mode,
                       observed={k: float(np.mean([r[k] for r in projection[stage]])) for k in ('relative_l1', 'relative_l2')})
        if heldout:
            summary['heldout'] = {k: float(np.mean([r[k] for r in hold[stage]])) for k in ('relative_l1', 'relative_l2')}
        summaries[stage] = summary
        directory = output/stage; directory.mkdir(exist_ok=True)
        write_json(directory/'metrics.json', summary)
        (directory/'per_case_metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows[stage]))
    return summaries, rows


def connectivity_diagnostics(model, dataset, config):
    model.eval(); batch = move_batch(projection_pool_collate([dataset[0]]), config['training']['device'])
    with torch.no_grad():
        original = model(batch, return_stages=True)
        shuffled = model(permute_views(batch, torch.arange(batch['view_mask'].shape[1], device=config['training']['device']).flip(0)), return_stages=True)
        padded = dict(batch)
        for k, v in batch.items():
            if (k.startswith('input_') or k == 'view_mask') and isinstance(v, torch.Tensor):
                padded[k] = torch.cat((v, torch.zeros_like(v[:, :1])), 1)
        padded['input_projection'][:, -1] = float('nan'); padded['input_projection_raw'][:, -1] = float('nan')
        pad = model(padded, return_stages=True)
        all_invalid = dict(batch); all_invalid['view_mask'] = torch.zeros_like(batch['view_mask'])
        all_invalid['input_valid_ray'] = torch.zeros_like(batch['input_valid_ray'])
        inv = model(all_invalid, return_stages=True)
        result = dict(permutation_max_diff=max(float((original[k]-shuffled[k]).abs().max()) for k in ('C0', 'C1')),
                      padding_max_diff=max(float((original[k]-pad[k]).abs().max()) for k in ('C0', 'C1')),
                      visibility_identical=bool(torch.equal(original['visibility_base'], original['visibility_residual'])),
                      all_invalid_safe=bool(torch.isfinite(inv['C1']).all() and (inv['context1']==0).all()),
                      prediction_std=float(original['C1'].std()))
    del original, shuffled, pad, inv
    model.zero_grad(set_to_none=True)
    s = model(batch, return_stages=True)
    loss, _ = supervised_loss(s['C1'], batch['gt_volume_64'], **config['loss'])
    gy, = torch.autograd.grad(loss, s['measurement_prediction_Y0'], retain_graph=True)
    gc, = torch.autograd.grad(s['measurement_prediction_Y0'], s['C0'], grad_outputs=gy, retain_graph=True)
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if n.startswith('residual_')]
    result.update(residual_gradient_finite=all(g is not None and bool(torch.isfinite(g).all()) for g in grads),
                  residual_gradient_abs_sum=sum(float(g.abs().sum()) for g in grads if g is not None),
                  projector_feedback_gradient_finite=bool(torch.isfinite(gy).all() and torch.isfinite(gc).all()),
                  projector_feedback_gradient_abs_sum=float(gc.abs().sum()))
    result['passed'] = (result['permutation_max_diff'] <= 1e-5 and result['padding_max_diff'] <= 1e-5 and
        result['visibility_identical'] and result['all_invalid_safe'] and result['residual_gradient_finite'] and
        result['residual_gradient_abs_sum'] > 0 and result['projector_feedback_gradient_finite'] and
        result['projector_feedback_gradient_abs_sum'] > 0)
    model.zero_grad(set_to_none=True)
    return result
