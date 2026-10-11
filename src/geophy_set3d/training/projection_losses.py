"""Soft, robust measurement regularization against frozen raw 128³ references."""
import torch
from torch import nn
from torch.nn import functional as F
from geophy_set3d.physics.projectors import differentiable_project

Y_SCALE = 32284.894580078173
PROJECTOR_SAMPLES = 96


class MaskedProjectionSmoothL1(nn.Module):
    def __init__(self, scale=Y_SCALE, beta=.01):
        super().__init__()
        if scale != Y_SCALE or beta != .01:
            raise ValueError('B3 v1 fixes the raw ppm·m scale and robust beta')
        self.scale, self.beta = scale, beta

    def forward(self, prediction, reference_raw, valid):
        valid = valid.detach().bool()
        # Mask before arithmetic, so invalid placeholders (including NaNs) contribute zero.
        p = torch.where(valid, prediction, torch.zeros_like(prediction)) / self.scale
        y = torch.where(valid, reference_raw.detach(), torch.zeros_like(reference_raw)) / self.scale
        residual = F.smooth_l1_loss(p, y, reduction='none', beta=self.beta)
        return residual.sum() / valid.sum().clamp_min(1)


def geometry_for(batch, index, prefix):
    mask_name = 'view_mask' if prefix == 'input' else 'hold_view_mask'
    selected = batch[mask_name][index].detach().bool()
    result = {}
    for key in ('station_origin_m', 'ray_direction', 't_near', 'path_length_m', 'valid_ray'):
        value = batch[f'{prefix}_{key}'][index, selected].detach()
        if key in ('t_near', 'path_length_m', 'valid_ray'):
            value = value[:, 0]
        result[key] = value
    return result, selected


def project_measurements(prediction, batch, prefix):
    values = []
    for i in range(len(prediction)):
        geometry, selected = geometry_for(batch, i, prefix)
        projected = differentiable_project(prediction[i, 0], geometry, samples=PROJECTOR_SAMPLES)
        reference = batch[f'{prefix}_projection_raw'][i, selected, 0].detach()
        values.append((projected, reference, geometry['valid_ray']))
    return values


def consistency_loss(prediction, batch, prefix):
    helper = MaskedProjectionSmoothL1()
    return torch.stack([helper(p, y, mask) for p, y, mask in
                        project_measurements(prediction, batch, prefix)]).mean()


def compute_observed_consistency(prediction, batch):
    return consistency_loss(prediction, batch, 'input')


def compute_heldout_consistency(prediction, batch):
    return consistency_loss(prediction, batch, 'hold')


def physics_weights(epoch_one_based, physics):
    if physics.get('schedule', 'warmup_ramp') == 'constant':
        factor = 1.
    elif epoch_one_based <= 20:
        factor = 0.
    elif epoch_one_based <= 40:
        # Epoch 21 is the zero endpoint, epoch 40 reaches the fixed target.
        factor = (epoch_one_based - 21) / 19
    else:
        factor = 1.
    return physics['lambda_obs']*factor, physics['lambda_hold']*factor
