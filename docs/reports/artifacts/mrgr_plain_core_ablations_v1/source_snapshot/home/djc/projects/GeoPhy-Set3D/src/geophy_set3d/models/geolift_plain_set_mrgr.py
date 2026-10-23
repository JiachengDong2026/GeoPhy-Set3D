"""Plain set interaction control; inherit the canonical MRGR feedback unchanged."""
import math
import torch
from .geolift_oari_res3d import GeoLiftOARIRes3D
from .geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D


class PlainSetStage0(GeoLiftOARIRes3D):
    def forward(self, batch, return_attention=False, relation_permutation=None):
        if relation_permutation is not None:
            raise ValueError('Plain interaction has no ray-relation permutation interface')
        y = batch['input_projection']
        inputs = torch.cat((y, batch['input_valid_ray'].to(y.dtype),
                            batch['input_path_length_m']/(math.sqrt(3)*.6)), 2)
        inputs = torch.where(batch['view_mask'][:, :, None, None, None], inputs, 0)
        b, v = inputs.shape[:2]
        features = self.encoder(inputs.flatten(0, 1)).reshape(b, v, 32, 64, 64)
        lifted, visibility = self.lifting(features, batch)
        geometry = self.geometry_descriptor(batch)
        context = self.interaction(lifted, geometry, visibility, batch['view_mask'],
                                   return_attention=return_attention)
        if return_attention:
            context, matrices = context
        fused, weights = self.fusion(context, geometry, visibility, batch['view_mask'])
        prediction = self.decoder(fused)
        if return_attention:
            return prediction, dict(attention_weights=weights, interaction_attention=matrices,
                visibility=visibility, context_features=context, interaction_kind='plain_set')
        return prediction


class GeoLiftPlainSetMRGR(GeoLiftOARIMRGRRes3D, PlainSetStage0):
    """C3 MRO routes the unchanged MRGR super().forward to PlainSetStage0."""
    pass
