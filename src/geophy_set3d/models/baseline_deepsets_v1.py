"""Independent DeepSets baseline for ray-integral observations.

Zaheer et al., Deep Sets (NeurIPS 2017), https://arxiv.org/abs/1703.06114.
This implementation is written for this repository; no third-party code is copied.
Only the fixed, measurement-only ray tokenization is shared with Ray-Senseiver.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .ray_senseiver_ss import fourier, ray_tokens


class RayDeepSetsV1(nn.Module):
    """Shared phi, masked mean, and an independent coordinate-conditioned rho.

    Queries passed to ``decode_queries`` are XYZ coordinates in metres. No
    trainable lifting, feedback, attention, or primary-model decoder is reused.
    Empty observation sets use a zero latent vector; they do not aggregate the
    token packer's sentinel. Output is CH4 mass fraction, with sigmoid bounds.
    """

    def __init__(self, hidden_dim=128, query_chunk_size=16384,
                 decoder_checkpoint=True, output_bias_logit=0.0):
        super().__init__()
        if hidden_dim < 1 or query_chunk_size < 1:
            raise ValueError('Hidden dimension and query chunk size must be positive')
        if not math.isfinite(output_bias_logit):
            raise ValueError('Output bias must be finite')
        self.phi = nn.Sequential(
            nn.Linear(203, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 96, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1))
        nn.init.constant_(self.decoder[-1].bias, output_bias_logit)
        axis = torch.linspace(0., .6, 64)
        coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
        self.register_buffer('query_coordinates_m', coords, persistent=False)
        self.query_chunk_size = int(query_chunk_size)
        self.decoder_checkpoint = bool(decoder_checkpoint)

    def encode(self, batch):
        tokens, excluded, counts = ray_tokens(batch)
        encoded = self.phi(tokens)
        encoded = encoded.masked_fill(excluded.unsqueeze(-1), 0.)
        pooled = encoded.sum(1) / counts.clamp_min(1).unsqueeze(-1)
        return torch.where(counts[:, None] > 0, pooled, torch.zeros_like(pooled))

    def _decode_chunk(self, latent, coordinates_m):
        coordinates = fourier(2 * coordinates_m / .6 - 1)
        features = torch.cat((latent[:, None, :].expand(-1, len(coordinates), -1),
                              coordinates[None].expand(len(latent), -1, -1)), -1)
        return self.decoder(features).sigmoid().float()

    def decode_queries(self, latent, queries, chunk_size=None, checkpoint_chunks=None):
        if queries.ndim != 2 or queries.shape[-1] != 3:
            raise ValueError('Queries must have shape [Q,3] in metres')
        size = self.query_chunk_size if chunk_size is None else int(chunk_size)
        if size < 1:
            raise ValueError('Query chunk size must be positive')
        use_checkpoint = self.decoder_checkpoint if checkpoint_chunks is None else checkpoint_chunks
        if not len(queries):
            return latent.new_empty((len(latent), 0, 1), dtype=torch.float32)
        outputs = []
        for coordinates in queries.split(size):
            if use_checkpoint and self.training and torch.is_grad_enabled():
                value = checkpoint(self._decode_chunk, latent, coordinates, use_reentrant=False)
            else:
                value = self._decode_chunk(latent, coordinates)
            outputs.append(value)
        return torch.cat(outputs, 1)

    def forward(self, batch):
        latent = self.encode(batch)
        output = self.decode_queries(latent, self.query_coordinates_m)
        return output.transpose(1, 2).reshape(len(latent), 1, 64, 64, 64)
