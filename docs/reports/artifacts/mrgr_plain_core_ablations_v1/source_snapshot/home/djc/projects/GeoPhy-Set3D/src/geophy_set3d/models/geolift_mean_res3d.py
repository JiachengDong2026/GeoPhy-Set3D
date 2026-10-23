"""GeoLift-Mean-Res3D: supervised-only variable-view reconstruction baseline."""
import math
import torch
from torch import nn
from .encoder2d import SharedEncoder2D
from .lifting import GeometryLifting
from .fusion import masked_mean
from .decoder3d import ResidualDecoder3D


class GeoLiftMeanRes3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder2D()
        self.lifting = GeometryLifting()
        self.decoder = ResidualDecoder3D()

    def forward(self, batch):
        y = batch['input_projection']
        inputs = torch.cat((y, batch['input_valid_ray'].to(y.dtype),
                            batch['input_path_length_m'] / (math.sqrt(3)*.6)), dim=2)
        # Remove padded contents before convolutions as well as at fusion.
        inputs = torch.where(batch['view_mask'][:, :, None, None, None], inputs, 0)
        b, v = inputs.shape[:2]
        features = self.encoder(inputs.flatten(0, 1)).reshape(b, v, 32, 64, 64)
        lifted, visibility = self.lifting(features, batch)
        return self.decoder(masked_mean(lifted, visibility))
