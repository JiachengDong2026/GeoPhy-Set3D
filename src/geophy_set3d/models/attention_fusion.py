"""Shared voxel/view scoring with geometry conditioning and safe masked softmax."""
import torch
from torch import nn


class GeometryConditionedAttentionPooling(nn.Module):
    def __init__(self, feature_channels=32):
        super().__init__()
        self.geometry_embedding = nn.Sequential(
            nn.Conv3d(4, 16, 1), nn.SiLU(), nn.Conv3d(16, 16, 1), nn.SiLU())
        self.score = nn.Sequential(
            nn.Conv3d(feature_channels + 16, 32, 1), nn.SiLU(), nn.Conv3d(32, 1, 1))

    def forward(self, lifted, geometry, visibility, view_mask):
        valid = visibility & view_mask[:, :, None, None, None]
        mask = valid[:, :, None]
        # Inactive contents cannot affect scores, reductions or parameter gradients.
        features = torch.where(mask, lifted, 0)
        geometry = torch.where(mask, geometry, 0)
        b, v, c, x, y, z = features.shape
        embedded = self.geometry_embedding(geometry.flatten(0, 1))
        logits = self.score(torch.cat((features.flatten(0, 1), embedded), 1))
        logits = logits.reshape(b, v, x, y, z).masked_fill(~valid, -torch.inf)
        any_valid = valid.any(1, keepdim=True)
        # Softmax of all -inf is undefined; replace only empty voxel sets by zeros.
        safe_logits = torch.where(any_valid, logits, 0)
        weights = torch.softmax(safe_logits, dim=1).masked_fill(~valid, 0)
        fused = (weights[:, :, None] * features).sum(1)
        return fused, weights


def normalized_attention_entropy(weights, visibility):
    """Mean H/log(n_visible); singleton/empty voxels carry no entropy evidence."""
    count = visibility.sum(1)
    eligible = count > 1
    entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(1)
    normalized = entropy / count.clamp_min(2).to(weights.dtype).log()
    return normalized[eligible].mean() if eligible.any() else weights.new_zeros(())
