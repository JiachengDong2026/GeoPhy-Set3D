"""Plan A: sum/4 set initialization and exact, differentiable physical updates.

The inner objective is squared normalized measurement error; the external
self-supervised training objective is deliberately defined by the trainer.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import ResidualBlock
from .geolift_deepsets_sum import GeoLiftDeepSetsSum, masked_scaled_sum
from .geolift_oari_res3d import GeoLiftOARIRes3D
from .lifting import GeometryLifting
from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator


Y_SCALE = 32284.894580078173


def _clean_inputs(batch):
    """Padding may contain NaNs; valid zero measurements remain measurements."""
    result = dict(batch)
    view = batch['view_mask'].bool()
    valid = batch['input_valid_ray'].bool() & view[:, :, None, None, None]
    result['input_valid_ray'] = valid
    for key in ('input_projection', 'input_projection_raw', 'input_path_length_m',
                'input_t_near'):
        if key in batch:
            result[key] = torch.where(valid, batch[key], 0)
    if 'input_ray_direction' in batch:
        result['input_ray_direction'] = torch.where(valid[:, :, 0, :, :, None],
                                                   batch['input_ray_direction'], 0)
    for suffix in ('station_origin_m', 'center_direction', 'right_direction',
                   'up_direction', 'station_azimuth_deg'):
        key = 'input_' + suffix
        if key in batch:
            mask = view.reshape(*view.shape, *([1] * (batch[key].ndim - 2)))
            result[key] = torch.where(mask, batch[key], 0)
    return result


class _SetInitializer(GeoLiftDeepSetsSum):
    """Identical production initializer, with explicit small-grid debug support."""
    def __init__(self, lift_resolution, output_resolution):
        super().__init__(v_ref=4)
        if lift_resolution != 32:
            self.lifting = GeometryLifting(lift_resolution)
        if output_resolution != 64:
            self.decoder[3] = nn.Upsample(size=(output_resolution,) * 3,
                                         mode='trilinear', align_corners=True)

    def forward(self, batch):
        y = batch['input_projection']
        view, valid = batch['view_mask'].bool(), batch['input_valid_ray'].bool()
        inputs = torch.cat((torch.where(valid, y, 0), valid.to(y.dtype),
                            torch.where(valid, batch['input_path_length_m'], 0)
                            / (math.sqrt(3) * .6)), 2)
        b, v = inputs.shape[:2]
        features = self.encoder(inputs.flatten(0, 1))
        features = features.reshape(b, v, *features.shape[1:])
        lifted, visibility = self.lifting(features, batch)
        return self.decoder(masked_scaled_sum(lifted, visibility, view, 4))


class SharedVolumePrior(nn.Sequential):
    def __init__(self, width=32):
        if width < 8 or width % 8:
            raise ValueError('prior_width must be a positive multiple of 8')
        super().__init__(nn.Conv3d(9, width, 3, padding=1),
                         nn.GroupNorm(8, width), nn.SiLU(),
                         *[ResidualBlock(width, 3) for _ in range(3)],
                         nn.Conv3d(width, 1, 1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class SetUnrolled3D(nn.Module):
    """A0/A1/A2/A3/A-control with independent, reproducible initialization.

    Nondefault resolutions and prior widths are explicit debugging/ablation
    settings. They must not be reported as the default 32³/64³, width-32 method.
    No checkpoint warm start or truncated concentration gradient is performed.
    """
    def __init__(self, variant='A2', steps=None, y_scale=Y_SCALE, samples=96,
                 ray_chunk=256, prior_width=32, prior_amplitude=.02,
                 lift_resolution=32, output_resolution=64, oari_settings=None,
                 checkpoint_prior=False, preconditioner='geometry',
                 unpreconditioned_scale=1., operator=None, initializer=None):
        super().__init__()
        variant = variant.upper()
        if variant not in ('A0', 'A1', 'A2', 'A3', 'A-CONTROL'):
            raise ValueError('variant must be A0, A1, A2, A3, or A-control')
        steps = (0 if variant == 'A0' else 4) if steps is None else steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        if variant == 'A0' and steps != 0:
            raise ValueError('A0 fixes steps=0')
        if variant != 'A0' and steps == 0:
            raise ValueError('A1/A2/A3 require at least one physical update')
        if y_scale <= 0 or not math.isfinite(y_scale):
            raise ValueError('y_scale must be finite and positive')
        if not (0 < prior_amplitude <= 1):
            raise ValueError('prior_amplitude must be in (0,1]')
        if min(lift_resolution, output_resolution) < 2:
            raise ValueError('grid resolutions must be at least two')
        if preconditioner not in ('geometry', 'none'):
            raise ValueError('preconditioner must be geometry or none')
        if unpreconditioned_scale <= 0 or not math.isfinite(unpreconditioned_scale):
            raise ValueError('unpreconditioned_scale must be finite and positive')
        self.variant, self.steps, self.y_scale = variant, steps, float(y_scale)
        self.prior_amplitude = float(prior_amplitude)
        self.checkpoint_prior = bool(checkpoint_prior)
        self.preconditioner = preconditioner
        self.unpreconditioned_scale = float(unpreconditioned_scale)
        # Initialize shared modules BEFORE added modules, preserving same-seed
        # DeepSets weights across A0/A1/A2 and historical random initialization.
        if initializer is not None:
            self.initializer = initializer
        elif variant == 'A3':
            if (lift_resolution, output_resolution) != (32, 64):
                raise ValueError('OARI A3 uses its historical 32³/64³ architecture')
            self.initializer = GeoLiftOARIRes3D(**(oari_settings or {}))
        else:
            self.initializer = _SetInitializer(lift_resolution, output_resolution)
        self.operator = operator if operator is not None else ExplicitRayOperator(
            samples_per_ray=samples, projection_scale_ppm_m=y_scale, ray_chunk=ray_chunk)
        self.step_logits = nn.Parameter(torch.zeros(steps))
        self.prior = SharedVolumePrior(prior_width) if variant in ('A2', 'A3', 'A-CONTROL') else None
        self.step_embedding = nn.Embedding(steps, 4) if self.prior is not None else None
        self.visibility_lifting = GeometryLifting(output_resolution)

    def _visible_fraction(self, batch, c):
        lifting = self.visibility_lifting
        if lifting.resolution != c.shape[-1]:
            lifting = GeometryLifting(c.shape[-1]).to(device=c.device, dtype=c.dtype)
        _, visible = lifting.angular_grid(batch)
        count = batch['view_mask'].sum(1).clamp_min(1).to(c.dtype)
        return (visible.sum(1).to(c.dtype) / count[:, None]).reshape_as(c)

    @staticmethod
    def _norm(x):
        return torch.linalg.vector_norm(x.flatten(1), dim=1)

    def forward(self, batch, return_diagnostics=False):
        batch = _clean_inputs(batch)
        if bool((batch['input_valid_ray'].flatten(1).sum(1) == 0).any()):
            raise ValueError('Every case must have at least one usable input measurement')
        c = c0 = self.initializer(batch)
        if self.steps == 0:
            if return_diagnostics:
                return dict(volume=c, C0=c0, steps=[], tau=c.new_empty((0,)),
                            visible_fraction=self._visible_fraction(batch, c))
            return c
        stencil = self.operator.prepare(batch, c.shape[-1])
        valid = stencil.valid_ray.to(device=c.device)
        counts = stencil.valid_counts.to(device=c.device, dtype=c.dtype)
        if bool((counts <= 0).any()):
            raise ValueError('Every case must have at least one usable input measurement')
        source = batch.get('input_projection_raw')
        y = batch['input_projection'] if source is None else source / self.y_scale
        y = torch.where(valid, y, 0)
        if not bool(torch.isfinite(y).all()):
            raise ValueError('Valid input measurements must be finite')
        tt1, gram_row_sum = self.operator.coverage(stencil, device=c.device, dtype=c.dtype)
        d0 = gram_row_sum * (28.965 / 16.043) ** 2 / counts[:, None, None, None, None]
        positive = d0 > 0
        n_positive = positive.flatten(1).sum(1)
        if bool((n_positive == 0).any()):
            raise ValueError('Case has no covered nodes / usable input measurements')
        eps = (1e-3 * d0.flatten(1).sum(1) / n_positive).clamp_min(1e-12)
        denominator = torch.maximum(d0, eps[:, None, None, None, None])
        log_coverage = torch.log1p(tt1)
        maximum = log_coverage.flatten(1).amax(1).reshape(-1, 1, 1, 1, 1)
        coverage = torch.where(maximum > 0, log_coverage / maximum.clamp_min(1e-30), 0)
        visible_fraction = self._visible_fraction(batch, c) if self.prior is not None or return_diagnostics else None
        records = []
        tau = self.step_logits.sigmoid()
        for k in range(self.steps):
            if self.variant == 'A-CONTROL':
                prediction = self.operator.project_normalized(c, stencil)
            else:
                prediction, gradient = self.operator.residual_gradient(c, y, stencil)
            residual = torch.where(valid, prediction - y, 0)
            feedback = None
            if self.variant == 'A-CONTROL':
                # Same trainable tensors as A2, but learned residual feedback
                # replaces its exact physical step. No Jacobian action occurs.
                lifting = self.visibility_lifting
                if lifting.resolution != c.shape[-1]:
                    lifting = GeometryLifting(c.shape[-1]).to(device=c.device, dtype=c.dtype)
                lifted, visibility = lifting(-residual, batch)
                feedback = masked_scaled_sum(lifted, visibility, batch['view_mask'], 4)
                update = torch.zeros_like(c)
                prior_signal = torch.tanh(tau[k] * feedback / .02)
                z_unclipped = c
            else:
                # Full differentiable J(C)^T r(C): neither r nor chi'(C)
                # is detached. Outer differentiation retains both paths.
                g = gradient / counts[:, None, None, None, None]
                update = (torch.where(positive, g / denominator, 0)
                          if self.preconditioner == 'geometry'
                          else g * self.unpreconditioned_scale)
                prior_signal = torch.tanh(update / .02)
                z_unclipped = c - tau[k] * update
            z = z_unclipped.clamp(0, 1)
            correction = torch.zeros_like(c)
            if self.prior is not None:
                embedding = self.step_embedding.weight[k].reshape(1, 4, 1, 1, 1)
                embedding = embedding.expand(c.shape[0], -1, *c.shape[2:])
                inputs = torch.cat((c, z, prior_signal, coverage,
                                    visible_fraction, embedding), 1)
                raw_correction = (checkpoint(self.prior, inputs, use_reentrant=False)
                    if self.checkpoint_prior and self.training and torch.is_grad_enabled()
                    else self.prior(inputs))
                correction = self.prior_amplitude * raw_correction.tanh()
            next_unclipped = z + correction
            next_c = next_unclipped.clamp(0, 1)
            if return_diagnostics:
                records.append(dict(E=residual.square().flatten(1).sum(1) / (2 * counts),
                    prediction=prediction, residual=residual, volume=next_c,
                    update_norm=self._norm(update), physical_step_norm=self._norm(z-c),
                    feedback_norm=self._norm(feedback) if feedback is not None else self._norm(torch.zeros_like(c)),
                    signal_saturation_fraction=(prior_signal.abs() >= 1-1e-6).to(c.dtype).flatten(1).mean(1),
                    prior_norm=self._norm(correction), tau=tau[k],
                    clip_fraction=((z_unclipped < 0) | (z_unclipped > 1)
                        | (next_unclipped < 0) | (next_unclipped > 1)).to(c.dtype).flatten(1).mean(1),
                    physical_clip_fraction=((z_unclipped < 0) | (z_unclipped > 1)).to(c.dtype).flatten(1).mean(1),
                    prior_clip_fraction=((next_unclipped < 0) | (next_unclipped > 1)).to(c.dtype).flatten(1).mean(1),
                    near_zero_fraction=(next_c <= 1e-6).to(c.dtype).flatten(1).mean(1),
                    near_one_fraction=(next_c >= 1-1e-6).to(c.dtype).flatten(1).mean(1)))
            c = next_c
        if return_diagnostics:
            return dict(volume=c, C0=c0, steps=records, tau=tau, coverage=coverage,
                        linear_coverage=tt1, visible_fraction=visible_fraction,
                        D0=d0, eps_D=eps, valid_counts=counts, stencil=stencil,
                        zero_coverage_fraction=(~positive).to(c.dtype).flatten(1).mean(1))
        return c


def build_set_unrolled_model(config):
    """Build from a research config; legacy model registry remains untouched."""
    settings = dict(config.get('model', config))
    settings.pop('name', None)
    return SetUnrolled3D(**settings)
