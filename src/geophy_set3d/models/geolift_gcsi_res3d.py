"""B2: geometry-conditioned set interaction followed by the unchanged B1 GCA."""
import math
import torch
from .geolift_gca_res3d import GeoLiftGCARes3D
from .set_interaction import GeometryConditionedSetInteraction


class GeoLiftGCSIRes3D(GeoLiftGCARes3D):
    def __init__(self, distance_scale_m=6.5, azimuth_scale_deg=180., **interaction_settings):
        # Same seed preserves the initial encoder/decoder AND B1 pooling parameters.
        super().__init__(distance_scale_m, azimuth_scale_deg)
        self.interaction = GeometryConditionedSetInteraction(**interaction_settings)

    def forward(self, batch, return_attention=False):
        y = batch['input_projection']
        inputs = torch.cat((y, batch['input_valid_ray'].to(y.dtype),
                            batch['input_path_length_m'] / (math.sqrt(3)*.6)), dim=2)
        inputs = torch.where(batch['view_mask'][:, :, None, None, None], inputs, 0)
        b, v = inputs.shape[:2]
        features = self.encoder(inputs.flatten(0, 1)).reshape(b, v, 32, 64, 64)
        lifted, visibility = self.lifting(features, batch)
        geometry = self.geometry_descriptor(batch)
        context = self.interaction(lifted, geometry, visibility, batch['view_mask'], return_attention)
        if return_attention:
            context, matrices = context
        fused, weights = self.fusion(context, geometry, visibility, batch['view_mask'])
        prediction = self.decoder(fused)
        if return_attention:
            return prediction, dict(attention_weights=weights, interaction_attention=matrices,
                                    visibility=visibility, context_features=context)
        return prediction
