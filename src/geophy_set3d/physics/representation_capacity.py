"""Direct bounded-volume measurement fitting with the frozen K96 projector."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.projection_audit import ErrorTotals
from geophy_set3d.data.projection_pool_dataset import artifact_path
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler
from geophy_set3d.physics.projectors import differentiable_project

Y_SCALE = 32284.894580078173
FROZEN_PROJECTOR_HASHES = {
    'projectors.py': 'fe3475e29abbba0d3495be9fc69909f3b8b1d7e8d40c25aed683094198f6e776',
    'reference_projector.cpp': '3773bb832aeb04928e45d1baac4fcf0a9701502727ee6046a05fa85811881a0c',
}


def assert_projector_unchanged():
    for name, expected in FROZEN_PROJECTOR_HASHES.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise AssertionError(f'Frozen projector source changed: {name}')


def select_train_cases(rows, count=12):
    train = [row for row in rows if row['split'] == 'train']
    if not 1 <= count <= len(train):
        raise ValueError('Invalid train case count')
    indices = np.rint(np.linspace(0, len(train)-1, count)).astype(int)
    return indices.tolist(), [train[i] for i in indices]


def select_stations(azimuths, fit_count=16, heldout_count=16):
    azimuths = np.asarray(azimuths)
    if fit_count + heldout_count > len(azimuths):
        raise ValueError('Not enough disjoint Pool A stations')
    fit = EvaluationViewSampler(azimuths, fit_count, 0, 'train',
                                protocol='spread').select('capacity-audit')
    remaining = np.setdiff1d(np.arange(len(azimuths)), fit)
    local = EvaluationViewSampler(azimuths[remaining], heldout_count, 0, 'train',
                                  protocol='spread').select('capacity-audit')
    heldout = remaining[local]
    if set(fit) & set(heldout):
        raise AssertionError('Fit/held-out stations overlap')
    return fit, heldout


def load_audit_case(root, row):
    """Load saved P_ref(C128,256); never call or regenerate P_ref."""
    if row['split'] != 'train':
        raise ValueError('Capacity audit is train-only')
    root = Path(root)
    projection_path = artifact_path(root, row['train_projection_path'])
    gt64_path = artifact_path(root, row['gt_volume_64_path'])
    meta = json.loads((projection_path.parent/'sample_meta.json').read_text())
    if meta['split'] != 'train' or meta['reference_samples'] != 256:
        raise ValueError('Expected train saved K256 reference')
    for path in (projection_path, gt64_path):
        if digest(path) != meta['hashes'][path.name]:
            raise AssertionError(f'Frozen array changed: {path}')
    source = Path(row['source_target_path'])
    if digest(source) != row['source_gt_sha256']:
        raise AssertionError('Source GT128 changed')
    reference = np.load(projection_path, mmap_mode='r', allow_pickle=False)
    gt64 = np.load(gt64_path, mmap_mode='r', allow_pickle=False)
    gt128 = np.load(source, mmap_mode='r', allow_pickle=False)
    if gt64.shape != (64,)*3 or gt128.shape != (128,)*3:
        raise ValueError('Incorrect frozen volume resolution')
    if not all(np.isfinite(array).all() for array in (reference, gt64, gt128)):
        raise ValueError('Nonfinite audit input')
    if np.any(reference < 0) or any(np.any((a < 0) | (a > 1)) for a in (gt64, gt128)):
        raise ValueError('Invalid measurement/concentration range')
    return reference, gt64, gt128


def measurement_metrics(prediction, reference, valid):
    totals = ErrorTotals()
    totals.update(prediction, reference, np.asarray(valid, dtype=bool))
    return totals.metrics()


def measurement_loss(prediction, reference, valid, scale=Y_SCALE):
    if prediction.shape != reference.shape or valid.shape != reference.shape:
        raise ValueError('Measurement/mask shapes differ')
    if not bool(valid.any()) or scale <= 0:
        raise ValueError('Valid rays and a positive frozen scale required')
    return F.smooth_l1_loss(prediction[valid]/scale, reference[valid]/scale, beta=1.)


def optimize_volume(initial, geometry, reference, *, device='cpu', scale=Y_SCALE,
                    lr=.01, adam_epsilon=1e-12, max_steps=2000,
                    patience_steps=200, improvement=1e-5, eps=1e-6,
                    log_every=25, trace_path=None, label=''):
    """Return the lowest fit-loss iterate; held-out rays never enter this function."""
    if np.shape(initial) != (64,)*3 or max_steps < 1 or patience_steps < 1:
        raise ValueError('Expected 64³ initial volume and positive step limits')
    initial = torch.as_tensor(np.array(initial, copy=True), dtype=torch.float32, device=device)
    latent = torch.nn.Parameter(torch.logit(initial.clamp(eps, 1-eps)))
    reference = torch.as_tensor(np.array(reference, copy=True), dtype=torch.float32, device=device)
    # Reuse the EXACT projector. Tensor conversion only avoids repeated host copies.
    geometry = {k: torch.as_tensor(np.array(v, copy=True), device=device,
                                   dtype=torch.bool if k == 'valid_ray' else torch.float32)
                for k, v in geometry.items()}
    valid = geometry['valid_ray']
    if reference.shape != valid.shape or not bool(torch.isfinite(reference[valid]).all()):
        raise ValueError('Invalid saved reference subset')
    optimizer = torch.optim.Adam([latent], lr=lr, eps=adam_epsilon)
    best_loss, best_l1, anchor_l1 = float('inf'), float('inf'), float('inf')
    best_volume, best_step, anchor_step = None, 0, 0
    trace, gradient_max = [], 0.
    started = time.monotonic()
    stop_reason = 'max_steps'
    for step in range(max_steps+1):
        volume = torch.sigmoid(latent)
        prediction = differentiable_project(volume, geometry, samples=96)
        loss = measurement_loss(prediction, reference, valid, scale)
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            raise FloatingPointError('Nonfinite fitting loss')
        with torch.no_grad():
            l1 = float((prediction[valid]-reference[valid]).abs().double().sum()
                       / reference[valid].abs().double().sum().clamp_min(1e-20))
            if loss_value < best_loss:
                best_loss, best_step = loss_value, step
                best_volume = volume.detach().clone()
            best_l1 = min(best_l1, l1)
            if anchor_l1-best_l1 >= improvement:
                anchor_l1, anchor_step = best_l1, step
        if step % log_every == 0 or step == max_steps:
            trace.append(dict(step=step,loss=loss_value,relative_l1=l1,
                              elapsed_seconds=time.monotonic()-started))
        should_stop = step-anchor_step >= patience_steps
        if step == max_steps or should_stop:
            if should_stop:
                stop_reason = 'fit_relative_l1_plateau'
                if trace[-1]['step'] != step:
                    trace.append(dict(step=step,loss=loss_value,relative_l1=l1,
                                      elapsed_seconds=time.monotonic()-started))
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if latent.grad is None or not bool(torch.isfinite(latent.grad).all()):
            raise FloatingPointError('Nonfinite latent gradient')
        gradient_max = max(gradient_max, float(latent.grad.abs().max()))
        optimizer.step()
        if step % 200 == 0:
            print(f'{label} step={step} fit L1={l1:.6%} loss={loss_value:.8g}', flush=True)
    if gradient_max == 0:
        raise AssertionError('No optimization gradient')
    if trace_path is not None:
        Path(trace_path).write_text(''.join(json.dumps(item)+'\n' for item in trace))
    return best_volume.cpu().numpy(), dict(
        completed_steps=step,best_step=best_step,best_fit_loss=best_loss,
        initial_fit_relative_l1=trace[0]['relative_l1'],
        minimum_observed_fit_relative_l1=best_l1,
        stopping_reason=stop_reason,all_gradients_finite=True,
        maximum_latent_gradient=gradient_max,elapsed_seconds=time.monotonic()-started)


def volume_diagnostics(volume, gt64, active_threshold=1e-4):
    volume, gt64 = np.asarray(volume, dtype=np.float64), np.asarray(gt64, dtype=np.float64)
    difference = volume-gt64
    def stats(array):
        return dict(min=float(array.min()),max=float(array.max()),mean=float(array.mean()),
                    active_fraction=float((array > active_threshold).mean()),
                    total_concentration_proxy=float(array.sum()))
    gt, opt = stats(gt64), stats(volume)
    relative_l1 = float(np.abs(difference).sum()/max(np.abs(gt64).sum(),1e-20))
    relative_l2 = float(np.linalg.norm(difference.ravel())/max(np.linalg.norm(gt64.ravel()),1e-20))
    background = gt64 <= active_threshold
    # Supplementary plausibility screens, NOT capacity PASS thresholds or loss terms.
    flags=[]
    if opt['max'] > .9 and opt['max'] > 5*gt['max']:
        flags.append('max_near_one_far_above_GT')
    background_active = float((volume[background] > active_threshold).mean()) if background.any() else 0.
    if background_active > .25:
        flags.append('widespread_background_above_1e-4')
    tv = lambda a: float(np.mean([np.abs(np.diff(a,axis=i)).mean() for i in range(3)]))
    opt_tv, gt_tv = tv(volume), tv(gt64)
    if opt_tv > 3*max(gt_tv,1e-20) and opt_tv > .01*max(gt['max'],1e-6):
        flags.append('large_neighbor_variation')
    return dict(optimized=opt,gt64=gt,relative_l1_vs_gt64=relative_l1,
                relative_l2_vs_gt64=relative_l2,active_threshold=active_threshold,
                background_active_fraction=background_active,
                neighbor_absolute_difference=opt_tv,gt_neighbor_absolute_difference=gt_tv,
                plausibility_flags=flags,measurement_compatible_but_implausible=bool(flags))
