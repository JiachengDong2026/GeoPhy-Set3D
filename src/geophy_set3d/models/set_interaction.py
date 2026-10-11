"""One geometry-conditioned, permutation-equivariant attention block over views only."""
import math
import torch
from torch import nn


class GeometryConditionedSetInteraction(nn.Module):
    def __init__(self, d_model=32, num_heads=4, num_blocks=1, ffn_channels=64):
        super().__init__()
        if (d_model, num_heads, num_blocks, ffn_channels) != (32, 4, 1, 64):
            raise ValueError('B2 v1 fixes one 32-channel, four-head block with FFN64')
        self.d_model, self.num_heads = d_model, num_heads
        self.geometry_embedding = nn.Sequential(
            nn.Conv3d(4, 16, 1), nn.SiLU(), nn.Conv3d(16, 16, 1), nn.SiLU())
        self.token_projection = nn.Linear(48, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.attention_output = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_channels), nn.SiLU(),
                                 nn.Linear(ffn_channels, d_model))

    def forward(self, features, geometry, visibility, view_mask, return_attention=False):
        valid = visibility & view_mask[:, :, None, None, None]
        b, v, c, x, y, z = features.shape
        features = torch.where(valid[:, :, None], features, 0)
        geometry = torch.where(valid[:, :, None], geometry, 0)
        embedded = self.geometry_embedding(geometry.flatten(0, 1)).reshape(b, v, 16, x, y, z)
        # Every voxel is an independent batch element, with V exchangeable tokens.
        tokens = torch.cat((features, embedded), 2).permute(0, 3, 4, 5, 1, 2).reshape(-1, v, 48)
        mask = valid.permute(0, 2, 3, 4, 1).reshape(-1, v)
        token_mask = mask[:, :, None]
        tokens = torch.where(token_mask, self.token_projection(tokens), 0)
        head_dim = c // self.num_heads
        qkv = self.qkv(self.norm1(tokens)).reshape(-1, v, 3, self.num_heads, head_dim)
        q, k, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        values = torch.where(mask[:, None, :, None], values, 0)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        logits = logits.masked_fill(~mask[:, None, None, :], -torch.inf)
        safe_logits = torch.where(mask.any(1)[:, None, None, None], logits, 0)
        weights = torch.softmax(safe_logits, -1)
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        weights = weights.masked_fill(~pair_mask, 0)
        attended = (weights @ values).transpose(1, 2).reshape(-1, v, c)
        tokens = torch.where(token_mask, tokens + self.attention_output(attended), 0)
        tokens = torch.where(token_mask, tokens + self.ffn(self.norm2(tokens)), 0)
        context = tokens.reshape(b, x, y, z, v, c).permute(0, 4, 5, 1, 2, 3).contiguous()
        if return_attention:
            matrices = weights.reshape(b, x, y, z, self.num_heads, v, v)
            matrices = matrices.permute(0, 4, 5, 6, 1, 2, 3).contiguous()
            return context, matrices  # [B,H,V_query,V_key,X,Y,Z]
        return context
