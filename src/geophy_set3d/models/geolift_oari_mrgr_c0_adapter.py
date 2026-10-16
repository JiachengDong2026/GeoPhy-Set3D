"""An isolated C0 experiment: condition fused features on existing ray descriptors.

The acquisition, original OARI/MRGR modules and projector are unchanged. The
zero-initialized adapter adds 1,152 parameters and is applied only before C0.
Directional diversity is a descriptor, not a guarantee of unique recovery.
"""
import math

import torch
from torch import nn

from .geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D, project_input, Y_SCALE


class C0ObservabilityAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(37, 16, 1), nn.SiLU(), nn.Conv3d(16, 32, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, fused, descriptor, visible):
        mask = visible[:, None]
        inputs = torch.cat((torch.where(mask, fused, 0),
                            torch.where(mask, descriptor, 0)), 1)
        correction = torch.where(mask, self.body(inputs), 0)
        return fused + correction, correction


class GeoLiftOARIMRGRC0Adapter(GeoLiftOARIMRGRRes3D):
    def __init__(self, **settings):
        super().__init__(**settings)
        self.c0_adapter = C0ObservabilityAdapter()

    def _initial_reconstruction(self, batch):
        y = batch['input_projection']
        inputs = torch.cat((y, batch['input_valid_ray'].to(y.dtype),
                            batch['input_path_length_m'] / (math.sqrt(3) * .6)), 2)
        inputs = torch.where(batch['view_mask'][:, :, None, None, None], inputs, 0)
        b, v = inputs.shape[:2]
        features = self.encoder(inputs.flatten(0, 1)).reshape(b, v, 32, 64, 64)
        lifted, visibility = self.lifting(features, batch)
        geometry = self.geometry_descriptor(batch)
        valid = visibility & batch['view_mask'][:, :, None, None, None]
        mask = valid.flatten(2).transpose(1, 2).reshape(-1, v)
        directions, relations = self.ray_geometry(batch['input_station_origin_m'],
            self.lifting.voxel_xyz, batch['input_station_azimuth_deg'], valid)
        count = batch['view_mask'].sum(1)[:, None].expand(-1, 32**3).reshape(-1)
        observability = self.observability(directions, mask, count)
        context = self.interaction(lifted, geometry, visibility, batch['view_mask'],
                                   relations, observability)
        fused, _ = self.fusion(context, geometry, visibility, batch['view_mask'])
        descriptor = observability.reshape(b, 32, 32, 32, 5).permute(0, 4, 1, 2, 3)
        adapted, correction = self.c0_adapter(fused, descriptor, valid.any(1))
        return self.decoder(adapted), context, visibility, correction

    def forward(self, batch, return_stages=False, residual_mode='normal'):
        if residual_mode != 'normal' and self.training:
            raise ValueError('Residual shortcuts are evaluation-only')
        if residual_mode not in ('normal', 'zero', 'wrong_pairing', 'signflip'):
            raise ValueError('Unknown residual shortcut')
        batch = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        c0, context0, visibility, correction = self._initial_reconstruction(batch)
        y0 = project_input(c0, batch)
        valid = batch['input_valid_ray'].bool() & batch['view_mask'][:, :, None, None, None]
        residual = torch.where(valid, batch['input_projection_raw'] - y0, 0)
        scaled = residual / Y_SCALE
        if residual_mode == 'zero':
            scaled = torch.zeros_like(scaled)
        elif residual_mode == 'signflip':
            scaled = -scaled
        elif residual_mode == 'wrong_pairing':
            scaled = scaled.clone()
            for i in range(len(c0)):
                ids = batch['view_mask'][i].nonzero().flatten()
                scaled[i, ids] = scaled[i, ids.roll(1)]
        scaled = torch.where(valid, scaled, 0)
        path = torch.where(valid, batch['input_path_length_m'] / (math.sqrt(3) * .6), 0)
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
                measurement_residual=residual, residual_scaled=residual / Y_SCALE,
                residual_input=inputs, residual_context=delta, residual_features=features,
                residual_lifted=lifted, visibility_base=visibility,
                visibility_residual=residual_visibility, context1=context1,
                c0_adapter_correction=correction)
        return c1
