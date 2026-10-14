"""Task-adapted Deep Sets: shared station mapping, visibility sum/4, decoder."""
import math
import torch
from torch import nn
from .encoder2d import SharedEncoder2D
from .lifting import GeometryLifting
from .decoder3d import ResidualDecoder3D


def masked_scaled_sum(lifted, visibility, view_mask, v_ref=4):
    if v_ref != 4:
        raise ValueError('DeepSets v1 fixes V_ref=4, independent of actual V')
    active = visibility.bool() & view_mask[:, :, None, None, None]
    return torch.where(active[:, :, None], lifted, 0).sum(1) / v_ref


class GeoLiftDeepSetsSum(nn.Module):
    def __init__(self, v_ref=4):
        super().__init__()
        if v_ref != 4:
            raise ValueError('DeepSets v1 fixes V_ref=4')
        self.v_ref = v_ref
        self.encoder = SharedEncoder2D()
        self.lifting = GeometryLifting()
        self.decoder = ResidualDecoder3D()

    def forward(self, batch):
        y = batch['input_projection']
        view = batch['view_mask'].bool()
        valid = batch['input_valid_ray'].bool() & view[:, :, None, None, None]
        inputs = torch.cat((torch.where(valid, y, 0), valid.to(y.dtype),
                            torch.where(valid, batch['input_path_length_m'], 0) / (math.sqrt(3)*.6)), 2)
        # Padding geometry can contain arbitrary contents, including NaN.
        geometry = dict(batch)
        for key in ('station_origin_m', 'center_direction', 'right_direction', 'up_direction'):
            field = 'input_' + key
            geometry[field] = torch.where(view[:, :, None], batch[field], 0)
        b, v = inputs.shape[:2]
        features = self.encoder(inputs.flatten(0, 1)).reshape(b, v, 32, 64, 64)
        lifted, visibility = self.lifting(features, geometry)
        return self.decoder(masked_scaled_sum(lifted, visibility, view, self.v_ref))
