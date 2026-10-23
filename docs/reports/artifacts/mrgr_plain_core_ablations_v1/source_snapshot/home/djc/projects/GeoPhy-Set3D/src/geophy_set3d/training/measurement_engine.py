"""B3 training only; the frozen B2 supervised validation engine is reused."""
import time
import torch
from .losses import supervised_loss
from .metrics import VolumeMetrics
from .projection_losses import compute_observed_consistency, compute_heldout_consistency
from .utils import move_batch


def gradient_diagnostics(model, batch, config):
    model.train()
    prediction = model(batch)
    sup, _ = supervised_loss(prediction, batch['gt_volume_64'], **config['loss'])
    obs = compute_observed_consistency(prediction, batch)
    hold = compute_heldout_consistency(prediction, batch)
    wo, wh = config['physics']['lambda_obs'], config['physics']['lambda_hold']
    result = {}
    parameters = [p for p in model.parameters() if p.requires_grad]
    for key, loss in [('supervised', sup), ('observed_weighted', wo*obs),
                      ('heldout_weighted', wh*hold), ('physics_combined', wo*obs+wh*hold)]:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        active = [g for g in gradients if g is not None]
        result[key] = float(torch.sqrt(sum(g.square().sum() for g in active)))
        if not all(torch.isfinite(g).all() for g in active) or result[key] <= 0:
            raise AssertionError('Nonfinite or zero network gradient: '+key)
    for key, loss in [('observed_volume', obs), ('heldout_volume', hold)]:
        grad = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
        result[key] = float(grad.norm())
        if not torch.isfinite(grad).all() or result[key] <= 0:
            raise AssertionError('Nonfinite or zero volume gradient: '+key)
    result['physics_to_supervised_ratio'] = result['physics_combined']/result['supervised']
    result['reference_and_geometry_detached'] = True
    return result


def run_measurement_epoch(model, loader, config, optimizer, weights):
    model.train(); metrics = VolumeMetrics(); count = 0; began = time.perf_counter()
    totals = {k: 0. for k in ('total', 'supervised', 'observed', 'heldout', 'observed_weighted', 'heldout_weighted')}
    wo, wh = weights
    if config['training']['gradient_accumulation'] != 1:
        raise ValueError('B3 v1 uses batch2 without gradient accumulation')
    for raw in loader:
        batch = move_batch(raw, config['training']['device'])
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        sup, _ = supervised_loss(prediction, batch['gt_volume_64'], **config['loss'])
        obs = compute_observed_consistency(prediction, batch) if wo else sup.new_zeros(())
        hold = compute_heldout_consistency(prediction, batch) if wh else sup.new_zeros(())
        total = sup + wo*obs + wh*hold
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite B3 loss')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        parts = dict(total=total, supervised=sup, observed=obs, heldout=hold,
                     observed_weighted=wo*obs, heldout_weighted=wh*hold)
        n = len(prediction); count += n
        for k, value in parts.items():
            totals[k] += float(value.detach())*n
        metrics.update(prediction, batch['gt_volume_64'], batch['case_id'])
    return dict(loss={k: v/count for k, v in totals.items()}, metrics=metrics.compute(),
                seconds=time.perf_counter()-began, lambda_obs=wo, lambda_hold=wh)
