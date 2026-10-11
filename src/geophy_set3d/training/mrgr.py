"""Stage-aware MRGR loop; the objective is exclusively the frozen C1 supervision."""
import time
import torch
from .metrics import VolumeMetrics
from .losses import supervised_loss
from .utils import move_batch


def run_epoch(model, loader, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metrics = {k: VolumeMetrics() for k in ('C0', 'C1')}
    totals = dict(total=0., smooth_l1=0., relative_l1=0.)
    residuals, count, started = [], 0, time.perf_counter()
    branch_grad_max = 0.
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for raw in loader:
            batch = move_batch(raw, config['training']['device'])
            s = model(batch, return_stages=True)
            loss, parts = supervised_loss(s['C1'], batch['gt_volume_64'], **config['loss'])
            if not torch.isfinite(loss) or not torch.isfinite(s['C0']).all():
                raise FloatingPointError('Nonfinite MRGR stage/loss')
            if training:
                loss.backward()
                grads = [p.grad for name, p in model.named_parameters() if name.startswith('residual_') and p.grad is not None]
                if not all(torch.isfinite(g).all() for g in grads):
                    raise FloatingPointError('Nonfinite residual gradient')
                branch_grad_max = max(branch_grad_max, sum(float(g.abs().sum()) for g in grads))
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            n = len(s['C1']); count += n
            for k in totals:
                totals[k] += float(parts[k].detach())*n
            for k in metrics:
                metrics[k].update(s[k], batch['gt_volume_64'], batch['case_id'])
            valid = batch['input_valid_ray'].bool() & batch['view_mask'][:, :, None, None, None]
            residuals.append(s['residual_scaled'][valid].detach().float().cpu())
    r = torch.cat(residuals)
    if len(r):
        stats = dict(min=float(r.min()), max=float(r.max()), mean=float(r.mean()), std=float(r.std(unbiased=False)),
                     p95=float(torch.quantile(r, .95)), p99=float(torch.quantile(r, .99)), valid_ray_count=len(r))
    else:
        stats = dict(min=0., max=0., mean=0., std=0., p95=0., p99=0., valid_ray_count=0)
    stages = {k: m.compute() for k, m in metrics.items()}
    return dict(loss={k: v/count for k, v in totals.items()}, metrics=stages['C1'], stages=stages,
                signed_scaled_residual=stats, residual_gradient_abs_sum_max=branch_grad_max,
                seconds=time.perf_counter()-started)
