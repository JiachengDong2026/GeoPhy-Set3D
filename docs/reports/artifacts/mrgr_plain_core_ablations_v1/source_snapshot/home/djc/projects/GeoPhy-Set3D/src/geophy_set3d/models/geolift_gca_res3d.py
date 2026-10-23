"""B1: exactly the B0 encoder, geometry lifting and decoder with learned fusion."""
import math
import torch
from .geolift_mean_res3d import GeoLiftMeanRes3D
from .attention_fusion import GeometryConditionedAttentionPooling


class GeoLiftGCARes3D(GeoLiftMeanRes3D):
    def __init__(self, distance_scale_m=6.5, azimuth_scale_deg=180.0):
        # Initialize all shared modules before fusion: same seed gives the same B0 weights.
        super().__init__()
        self.fusion = GeometryConditionedAttentionPooling()
        self.distance_scale_m = distance_scale_m
        self.azimuth_scale_deg = azimuth_scale_deg

    def geometry_descriptor(self, batch):
        grid, _ = self.lifting.angular_grid(batch)
        delta = self.lifting.voxel_xyz[None, None] - batch['input_station_origin_m'][:, :, None]
        distance = torch.linalg.vector_norm(delta, dim=-1) / self.distance_scale_m
        azimuth = batch['input_station_azimuth_deg'][:, :, None].expand_as(distance) / self.azimuth_scale_deg
        descriptor = torch.cat((grid, distance[..., None], azimuth[..., None]), -1)
        b, v = descriptor.shape[:2]
        return descriptor.transpose(2, 3).reshape(b, v, 4, 32, 32, 32)

    def forward(self, batch, return_attention=False):
        y = batch['input_projection']
        inputs = torch.cat((y, batch['input_valid_ray'].to(y.dtype),
                            batch['input_path_length_m'] / (math.sqrt(3)*.6)), dim=2)
        inputs = torch.where(batch['view_mask'][:, :, None, None, None], inputs, 0)
        b, v = inputs.shape[:2]
        features = self.encoder(inputs.flatten(0, 1)).reshape(b, v, 32, 64, 64)
        lifted, visibility = self.lifting(features, batch)
        fused, weights = self.fusion(lifted, self.geometry_descriptor(batch), visibility, batch['view_mask'])
        prediction = self.decoder(fused)
        if return_attention:
            return prediction, dict(attention_weights=weights, visibility=visibility)
        return prediction
