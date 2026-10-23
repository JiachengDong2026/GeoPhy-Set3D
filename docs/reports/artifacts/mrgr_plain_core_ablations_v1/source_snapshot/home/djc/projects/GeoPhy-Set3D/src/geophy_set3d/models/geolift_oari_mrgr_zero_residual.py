"""Matched MRGR capacity control: the scalar feedback channel is always zero."""
import math
import torch
from .geolift_oari_res3d import GeoLiftOARIRes3D
from .geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D


class GeoLiftOARIMRGRZeroResidualControl(GeoLiftOARIMRGRRes3D):
    """Same modules, parameter names/order and initialization; no projector call."""
    def forward(self, batch, return_stages=False):
        batch = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        c0, base = GeoLiftOARIRes3D.forward(self, batch, return_attention=True)
        context0, visibility = base['context_features'], base['visibility']
        del base
        valid = batch['input_valid_ray'].bool() & batch['view_mask'][:, :, None, None, None]
        zero = torch.zeros_like(batch['input_projection'])
        path = torch.where(valid, batch['input_path_length_m']/(math.sqrt(3)*.6), 0)
        inputs = torch.cat((zero, valid.to(c0.dtype), path), 2)
        b, v = inputs.shape[:2]
        features = self.residual_encoder(inputs.flatten(0, 1)).reshape(b, v, 16, 64, 64)
        lifted, residual_visibility = self.lifting(features, batch)
        delta = self.residual_adapter(torch.cat((context0, lifted), 2).flatten(0, 1)).reshape_as(context0)
        visible = visibility & batch['view_mask'][:, :, None, None, None]
        delta = torch.where(visible[:, :, None], delta, 0)
        context1 = torch.where(visible[:, :, None], context0+delta, 0)
        fused, _ = self.fusion(context1, self.geometry_descriptor(batch), visibility, batch['view_mask'])
        c1 = self.decoder(fused)
        if return_stages:
            return dict(C0=c0, C1=c1, residual_scaled=zero, residual_input=inputs,
                        residual_context=delta, residual_features=features, residual_lifted=lifted,
                        visibility_base=visibility, visibility_residual=residual_visibility,
                        context1=context1, measurement_prediction_Y0=None, measurement_residual=None)
        return c1
