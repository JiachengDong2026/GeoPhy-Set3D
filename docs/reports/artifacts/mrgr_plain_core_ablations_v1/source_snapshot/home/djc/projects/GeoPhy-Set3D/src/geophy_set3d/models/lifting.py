"""Explicit XYZ voxel to scanner angular coordinates, endpoint aligned."""
import torch
from torch import nn
from torch.nn import functional as F


class GeometryLifting(nn.Module):
    def __init__(self, resolution=32, extent_m=.6):
        super().__init__()
        self.resolution = resolution
        axis = torch.linspace(0, extent_m, resolution)
        self.register_buffer('voxel_xyz', torch.stack(torch.meshgrid(axis, axis, axis,
                             indexing='ij'), -1).reshape(-1, 3), persistent=False)

    def angular_grid(self, batch):
        delta = self.voxel_xyz[None, None] - batch['input_station_origin_m'][:, :, None]
        f, r, u = [(delta * batch['input_' + key][:, :, None]).sum(-1)
                   for key in ('center_direction', 'right_direction', 'up_direction')]
        alpha = torch.rad2deg(torch.atan2(r, f))
        beta = torch.rad2deg(torch.atan2(u, torch.sqrt(f.square() + r.square())))
        grids, visible = [], (f > 0) & batch['view_mask'][:, :, None]
        for angles, key in ((alpha, 'horizontal_angle_deg'), (beta, 'vertical_angle_deg')):
            endpoints = batch[key]
            low, high = endpoints[:, :1, None], endpoints[:, -1:, None]
            if not bool(torch.all(high > low)):
                raise ValueError('Steering angles must have increasing endpoints')
            grids.append(2 * (angles - low) / (high - low) - 1)
            visible = visible & (angles >= low) & (angles <= high)
        return torch.stack(grids, -1), visible

    def forward(self, features, batch):
        b, v, c, _, _ = features.shape
        grid, visible = self.angular_grid(batch)
        # grid_sample expects [horizontal, vertical], not physical XYZ axes.
        sampled = F.grid_sample(features.flatten(0, 1), grid.reshape(b*v, -1, 1, 2),
                                mode='bilinear', padding_mode='zeros', align_corners=True)
        shape = (b, v, c, *([self.resolution]*3))
        return sampled.reshape(shape), visible.reshape(b, v, *([self.resolution]*3))
