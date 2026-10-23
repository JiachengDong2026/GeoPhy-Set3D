"""One differentiable signed-measurement feedback step on the unchanged OARI."""
import math
import torch
from torch import nn
from geophy_set3d.physics.projectors import differentiable_project
from .geolift_oari_res3d import GeoLiftOARIRes3D

Y_SCALE = 32284.894580078173
PROJECTOR_SAMPLES = 96


class ResidualBlock2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.GroupNorm(4, 16),
            nn.SiLU(), nn.Conv2d(16, 16, 3, padding=1), nn.GroupNorm(4, 16))
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.body(x))


class ResidualEncoder2D(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
                         ResidualBlock2D(), ResidualBlock2D())


def project_input(c0, batch, view_chunk=2):
    """Keep every ray/K96; chunk only views, scatter padded views as exact zeros."""
    outputs = []
    for i in range(len(c0)):
        mask = batch['view_mask'][i].detach().bool()
        indices = mask.nonzero().flatten()
        chunks = []
        for selected in indices.split(view_chunk):
            if not len(selected):
                continue
            valid = batch['input_valid_ray'][i, selected, 0].detach().bool()
            geometry = {'valid_ray': valid,
                        'station_origin_m': batch['input_station_origin_m'][i, selected].detach()}
            for key in ('ray_direction', 't_near', 'path_length_m'):
                value = batch['input_'+key][i, selected].detach()
                if key != 'ray_direction':
                    value = value[:, 0]
                geometry[key] = torch.where(valid[..., None] if key == 'ray_direction' else valid, value, 0)
            chunks.append(differentiable_project(c0[i, 0], geometry, samples=PROJECTOR_SAMPLES))
        empty = c0[i].sum()*0 + c0.new_zeros((len(mask), 64, 64))
        if chunks:
            empty = empty.index_copy(0, indices, torch.cat(chunks))
        outputs.append(empty[:, None])
    return torch.stack(outputs)


class GeoLiftOARIMRGRRes3D(GeoLiftOARIRes3D):
    def __init__(self, **settings):
        # All shared initialization precedes the added branch, preserving same-seed OARI.
        super().__init__(**settings)
        self.residual_encoder = ResidualEncoder2D()
        self.residual_adapter = nn.Sequential(nn.Conv3d(48, 32, 1), nn.SiLU(), nn.Conv3d(32, 32, 1))
        nn.init.zeros_(self.residual_adapter[-1].weight)
        nn.init.zeros_(self.residual_adapter[-1].bias)

    def forward(self, batch, return_stages=False, residual_mode='normal'):
        if residual_mode != 'normal' and self.training:
            raise ValueError('Residual shortcuts are evaluation-only')
        if residual_mode not in ('normal', 'zero', 'wrong_pairing', 'signflip'):
            raise ValueError('Unknown residual shortcut')
        batch = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        c0, base = super().forward(batch, return_attention=True)
        context0, visibility = base['context_features'], base['visibility']
        del base
        y0 = project_input(c0, batch)
        valid = batch['input_valid_ray'].bool() & batch['view_mask'][:, :, None, None, None]
        residual = torch.where(valid, batch['input_projection_raw'] - y0, 0)
        scaled = residual / Y_SCALE
        if residual_mode == 'zero':
            scaled = torch.zeros_like(scaled)
        elif residual_mode == 'signflip':
            scaled = -scaled
        elif residual_mode == 'wrong_pairing':
            # Roll real residual maps only; retain all geometry, path and mask features.
            scaled = scaled.clone()
            for i in range(len(c0)):
                ids = batch['view_mask'][i].nonzero().flatten()
                scaled[i, ids] = scaled[i, ids.roll(1)]
        scaled = torch.where(valid, scaled, 0)
        path = torch.where(valid, batch['input_path_length_m']/(math.sqrt(3)*.6), 0)
        inputs = torch.cat((scaled, valid.to(c0.dtype), path), 2)
        b, v = inputs.shape[:2]
        features = self.residual_encoder(inputs.flatten(0, 1)).reshape(b, v, 16, 64, 64)
        lifted, residual_visibility = self.lifting(features, batch)
        delta = self.residual_adapter(torch.cat((context0, lifted), 2).flatten(0, 1))
        delta = delta.reshape_as(context0)
        visible = visibility & batch['view_mask'][:, :, None, None, None]
        delta = torch.where(visible[:, :, None], delta, 0)
        context1 = torch.where(visible[:, :, None], context0 + delta, 0)
        fused, _ = self.fusion(context1, self.geometry_descriptor(batch), visibility, batch['view_mask'])
        c1 = self.decoder(fused)
        if return_stages:
            return dict(C0=c0, C1=c1, measurement_prediction_Y0=y0,
                measurement_residual=residual, residual_scaled=residual/Y_SCALE,
                residual_input=inputs, residual_context=delta,
                residual_features=features, residual_lifted=lifted,
                visibility_base=visibility, visibility_residual=residual_visibility,
                context1=context1)
        return c1
