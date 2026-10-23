"""View-only ray relations and a geometry-derived local directional descriptor.

These descriptors describe directional diversity, not a Fisher matrix or a
complete tomography observability criterion. Geometry is fixed acquisition data;
its eigendecomposition does not participate in learned-parameter differentiation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .set_interaction import GeometryConditionedSetInteraction


class RayGeometryDescriptor(nn.Module):
    @torch.no_grad()
    def forward(self, station_origins, voxel_xyz, azimuth_deg, valid):
        """Return [B*XYZ,V,3] directions and [B*XYZ,V,V,3] relations."""
        b, v = valid.shape[:2]
        mask = valid.flatten(2).transpose(1, 2).reshape(-1, v)
        delta = voxel_xyz[None, None] - station_origins[:, :, None]
        delta = torch.where(valid.flatten(2)[..., None], delta, 0)
        directions = F.normalize(delta, dim=-1, eps=1e-8)
        directions = directions.transpose(1, 2).reshape(-1, v, 3)
        cosine = (directions @ directions.transpose(-1, -2)).clamp(-1, 1).abs()
        # Exact self-pairs, including roundoff in normalization.
        cosine = torch.where(torch.eye(v, dtype=torch.bool, device=valid.device), 1., cosine)
        sine = (1-cosine.square()).clamp_min(0).sqrt()
        azimuth = torch.where(valid.flatten(2), azimuth_deg[:, :, None], 0)
        azimuth = azimuth.transpose(1, 2).reshape(-1, v)
        # Frozen acquisition family spans [0,180] degrees, without a wrap seam.
        distance = (azimuth[:, :, None]-azimuth[:, None, :]).abs()/180.
        relations = torch.stack((cosine, sine, distance), -1)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        return directions, torch.where(pair_mask[..., None], relations, 0)


class LocalObservabilityDescriptor(nn.Module):
    @torch.no_grad()
    def forward(self, directions, valid, input_view_count, return_tensor=False):
        """Descending trace-normalized eigenvalues, lambda3/lambda1, fraction.

        input_view_count excludes padded views but includes locally invisible
        input views; this preserves padding invariance without hiding coverage.
        """
        directions = torch.where(valid[..., None], directions, 0)
        count = valid.sum(-1)
        tensor = directions.transpose(-1, -2) @ directions
        tensor = tensor / count.clamp_min(1)[:, None, None]
        # cuSOLVER's batched 3x3 solver rejects large batches on some CUDA
        # versions. Bound solver batches; the per-voxel mathematics is identical.
        eigenvalues = torch.cat([torch.linalg.eigvalsh(part)
                                 for part in tensor.split(4096)], 0).flip(-1).clamp_min(0)
        eigenvalues = eigenvalues / eigenvalues.sum(-1, keepdim=True).clamp_min(1e-8)
        isotropy = eigenvalues[:, 2]/eigenvalues[:, 0].clamp_min(1e-8)
        fraction = count/input_view_count.clamp_min(1)
        descriptor = torch.cat((eigenvalues, isotropy[:, None], fraction[:, None]), -1)
        descriptor = torch.where((count > 0)[:, None], descriptor, 0)
        return (descriptor, tensor) if return_tensor else descriptor


class ObservabilityAwareRayInteraction(GeometryConditionedSetInteraction):
    """One d32/h4/FFN64 block, with observable tokens and per-head ray bias."""
    def __init__(self, **settings):
        super().__init__(**settings)
        self.observability_embedding = nn.Sequential(nn.Linear(5, 8), nn.SiLU())
        self.token_projection = nn.Linear(56, 32)
        self.pairwise_bias = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))
        nn.init.zeros_(self.pairwise_bias[-1].weight)
        nn.init.zeros_(self.pairwise_bias[-1].bias)

    def forward(self, features, geometry, visibility, view_mask, relations,
                observability, return_attention=False):
        valid = visibility & view_mask[:, :, None, None, None]
        b, v, c, x, y, z = features.shape
        features = torch.where(valid[:, :, None], features, 0)
        geometry = torch.where(valid[:, :, None], geometry, 0)
        embedded = self.geometry_embedding(geometry.flatten(0, 1)).reshape(b, v, 16, x, y, z)
        tokens = torch.cat((features, embedded), 2).permute(0, 3, 4, 5, 1, 2).reshape(-1, v, 48)
        mask = valid.permute(0, 2, 3, 4, 1).reshape(-1, v)
        token_mask = mask[:, :, None]
        pair_mask = mask[:, :, None] & mask[:, None, :]
        observability = torch.where(mask.any(1)[:, None], observability, 0)
        obs_embedding = self.observability_embedding(observability)
        tokens = torch.cat((tokens, obs_embedding[:, None].expand(-1, v, -1)), -1)
        tokens = torch.where(token_mask, self.token_projection(tokens), 0)
        head_dim = c//self.num_heads
        qkv = self.qkv(self.norm1(tokens)).reshape(-1, v, 3, self.num_heads, head_dim)
        q, k, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        values = torch.where(mask[:, None, :, None], values, 0)
        relations = torch.where(pair_mask[..., None], relations, 0)
        pair_inputs = torch.cat((relations, observability[:, None, None].expand(-1, v, v, -1)), -1)
        bias = self.pairwise_bias(pair_inputs).permute(0, 3, 1, 2)
        bias = torch.where(pair_mask[:, None], bias, 0)
        logits = (q @ k.transpose(-1, -2))/math.sqrt(head_dim) + bias
        logits = logits.masked_fill(~mask[:, None, None, :], -torch.inf)
        logits = torch.where(mask.any(1)[:, None, None, None], logits, 0)
        weights = torch.softmax(logits, -1).masked_fill(~pair_mask[:, None], 0)
        attended = (weights @ values).transpose(1, 2).reshape(-1, v, c)
        tokens = torch.where(token_mask, tokens+self.attention_output(attended), 0)
        tokens = torch.where(token_mask, tokens+self.ffn(self.norm2(tokens)), 0)
        context = tokens.reshape(b, x, y, z, v, c).permute(0, 4, 5, 1, 2, 3).contiguous()
        if return_attention:
            matrices = weights.reshape(b, x, y, z, self.num_heads, v, v)
            matrices = matrices.permute(0, 4, 5, 6, 1, 2, 3).contiguous()
            return context, matrices, bias
        return context
