"""OARI changes only B2's set interaction; encoder/lifting/GCA/decoder are shared."""
import math
import torch
from .geolift_gca_res3d import GeoLiftGCARes3D
from .ray_relational_interaction import (
    RayGeometryDescriptor, LocalObservabilityDescriptor, ObservabilityAwareRayInteraction)


class GeoLiftOARIRes3D(GeoLiftGCARes3D):
    def __init__(self, distance_scale_m=6.5, azimuth_scale_deg=180., **settings):
        # Initialize shared modules first, exactly as B2 does under the same seed.
        super().__init__(distance_scale_m, azimuth_scale_deg)
        self.ray_geometry = RayGeometryDescriptor()
        self.observability = LocalObservabilityDescriptor()
        self.interaction = ObservabilityAwareRayInteraction(**settings)

    def forward(self, batch, return_attention=False, relation_permutation=None):
        """relation_permutation is an evaluation-only relation mismatch probe."""
        if relation_permutation is not None and self.training:
            raise ValueError('Relation mismatch is allowed only during evaluation')
        y = batch['input_projection']
        inputs = torch.cat((y, batch['input_valid_ray'].to(y.dtype),
                            batch['input_path_length_m']/(math.sqrt(3)*.6)), 2)
        inputs = torch.where(batch['view_mask'][:, :, None, None, None], inputs, 0)
        b, v = inputs.shape[:2]
        features = self.encoder(inputs.flatten(0, 1)).reshape(b, v, 32, 64, 64)
        lifted, visibility = self.lifting(features, batch)
        geometry = self.geometry_descriptor(batch)
        valid = visibility & batch['view_mask'][:, :, None, None, None]
        mask = valid.flatten(2).transpose(1, 2).reshape(-1, v)
        directions, relations = self.ray_geometry(batch['input_station_origin_m'],
            self.lifting.voxel_xyz, batch['input_station_azimuth_deg'], valid)
        input_count = batch['view_mask'].sum(1)[:, None].expand(-1, 32**3).reshape(-1)
        observability = self.observability(directions, mask, input_count)
        if relation_permutation is not None:
            relations = relations[:, relation_permutation][:, :, relation_permutation]
        context = self.interaction(lifted, geometry, visibility, batch['view_mask'],
                                   relations, observability, return_attention)
        if return_attention:
            context, matrices, bias = context
        fused, weights = self.fusion(context, geometry, visibility, batch['view_mask'])
        prediction = self.decoder(fused)
        if return_attention:
            return prediction, dict(attention_weights=weights, interaction_attention=matrices,
                pairwise_bias=bias, ray_directions=directions, pairwise_geometry=relations,
                observability_descriptor=observability, visibility=visibility, context_features=context)
        return prediction
