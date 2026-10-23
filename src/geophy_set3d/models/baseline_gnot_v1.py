"""Ray-GNOT-v1: independently implemented GNOT mathematics for ray inverse maps.

Reference: Hao et al., GNOT, ICML 2023, arXiv:2302.14376v3.
This is an adaptation, not a vendored original implementation: heterogeneous
normalized linear attention and coordinate-gated expert networks are retained.
See docs/reports/baseline_gnot_implementation_v1_zh.md for provenance and changes.
"""
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .ray_senseiver_ss import ray_tokens


def _mlp(in_features, hidden, out_features):
    return nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(),
                         nn.Linear(hidden, out_features))


class NormalizedOperatorAttention(nn.Module):
    """Feature-softmax linear operator attention with heterogeneous sources.

    A_s(q) = q (sum_j k_sj outer v_sj) / (q dot sum_j k_sj).
    The projected result is q + sum_s A_s(q), as in GNOT. Masks remove
    samples before feature-softmax: excluded NaNs never enter arithmetic.
    """

    def __init__(self, width, heads, sources=1):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by heads')
        self.width, self.heads, self.sources = width, heads, sources
        self.query = nn.Linear(width, width)
        self.keys = nn.ModuleList(nn.Linear(width, width) for _ in range(sources))
        self.values = nn.ModuleList(nn.Linear(width, width) for _ in range(sources))
        self.output = nn.Linear(width, width)

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads,
                         self.width // self.heads).transpose(1, 2)

    def statistics(self, source, branch=0, excluded=None):
        if excluded is not None:
            source = source.masked_fill(excluded.unsqueeze(-1), 0.)
        keys = self._heads(self.keys[branch](source)).softmax(-1)
        values = self._heads(self.values[branch](source))
        if excluded is not None:
            included = (~excluded)[:, None, :, None]
            keys = keys * included
            values = values * included
        return keys.sum(-2), keys.transpose(-2, -1) @ values

    def apply_statistics(self, queries, statistics):
        q = self._heads(self.query(queries)).softmax(-1)
        attended = q
        for key_sum, context in statistics:
            denominator = (q * key_sum.unsqueeze(-2)).sum(-1, keepdim=True)
            # Empty sets contribute exactly zero; q residual is retained.
            attended = attended + (q @ context) / denominator.clamp_min(1.e-12)
        joined = attended.transpose(1, 2).reshape(queries.shape)
        return self.output(joined)


class GeometryOperatorBlock(nn.Module):
    """Cross operator, gated experts, global self operator, gated experts."""

    def __init__(self, width, heads, experts, inner_multiplier):
        super().__init__()
        hidden = width * inner_multiplier
        self.cross = NormalizedOperatorAttention(width, heads, sources=2)
        self.self_operator = NormalizedOperatorAttention(width, heads)
        self.query_norm = nn.LayerNorm(width)
        self.branch_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(2))
        self.self_norm = nn.LayerNorm(width)
        self.expert_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(2))
        self.experts = nn.ModuleList(nn.ModuleList(_mlp(width, hidden, width)
                                                  for _ in range(experts))
                                     for _ in range(2))
        self.gate = nn.Sequential(nn.Linear(3, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, experts))

    def _experts(self, x, coordinates, stage):
        weights = self.gate(coordinates).softmax(-1)
        # Sum one expert at a time instead of allocating [B,Q,width,experts].
        mixed = sum(expert(x) * weights[..., i:i+1]
                    for i, expert in enumerate(self.experts[stage]))
        return x + self.expert_norms[stage](mixed)

    def before_self(self, x, coordinates, *flat_statistics):
        stats = list(zip(flat_statistics[0::2], flat_statistics[1::2]))
        x = x + self.cross.apply_statistics(self.query_norm(x), stats)
        return self._experts(x, coordinates, 0)

    def self_statistics(self, x):
        return self.self_operator.statistics(self.self_norm(x))

    def after_self(self, x, coordinates, key_sum, context):
        x = x + self.self_operator.apply_statistics(self.self_norm(x),
                                                    [(key_sum, context)])
        return self._experts(x, coordinates, 1)


class RayGNOTV1(nn.Module):
    """Ray-integral adaptation with exact global, chunked operator attention.

    Measurement branch: all 203 legal ray features; geometry branch: 202
    non-measurement ray features. Raw integrals retain the frozen training-only
    scale from ray_tokens; neither branch sees GT or held-out measurements.
    Query coordinates are XYZ endpoint nodes in [0,.6]^3 meters.
    """

    def __init__(self, width=64, layers=2, heads=4, experts=2,
                 inner_multiplier=4, query_chunk_size=8192,
                 checkpoint_chunks=True, output_bias_logit=0.):
        super().__init__()
        if width < 1 or layers < 1 or heads < 1 or experts < 1:
            raise ValueError('positive width/layers/heads/experts required')
        if query_chunk_size < 1 or inner_multiplier < 1:
            raise ValueError('positive chunk size and inner multiplier required')
        self.width = width
        self.query_chunk_size = query_chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.measurement_encoder = _mlp(203, width, width)
        self.geometry_encoder = _mlp(202, width, width)
        self.coordinate_encoder = _mlp(3, width, width)
        self.blocks = nn.ModuleList(GeometryOperatorBlock(width, heads, experts,
                                                         inner_multiplier)
                                    for _ in range(layers))
        self.head = _mlp(width, width, 1)
        # Neutral logit only: no measured/GT concentration prior or scale search.
        nn.init.constant_(self.head[-1].bias, output_bias_logit)
        axis = torch.linspace(0., .6, 64)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1)
        self.register_buffer('query_coordinates_m', grid.reshape(-1, 3),
                             persistent=False)

    def _run(self, fn, *args):
        if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def encode_tokens(self, tokens, excluded, counts=None):
        """Public token adapter for permutation/padding audits."""
        tokens = tokens.float()
        if counts is not None:
            # ray_tokens uses one synthetic zero token for an empty sample.
            excluded = excluded | (counts == 0).unsqueeze(-1)
        safe = tokens.masked_fill(excluded.unsqueeze(-1), 0.)
        if not torch.isfinite(safe).all():
            raise FloatingPointError('Nonfinite unmasked GNOT ray token')
        return (self.measurement_encoder(safe),
                self.geometry_encoder(safe[..., 1:])), excluded

    def decode_queries(self, branches, excluded, coordinates_m,
                       chunk_size=None):
        size = self.query_chunk_size if chunk_size is None else chunk_size
        if size < 1 or len(coordinates_m) == 0:
            raise ValueError('nonempty coordinates and positive chunk size required')
        batch_size = branches[0].shape[0]
        positions = (2 * coordinates_m.float()/.6 - 1).unsqueeze(0)
        positions = positions.expand(batch_size, -1, -1)
        coordinate_chunks = positions.split(size, 1)
        states = [self._run(self.coordinate_encoder, pos)
                  for pos in coordinate_chunks]
        for block in self.blocks:
            stats = []
            for i, branch in enumerate(branches):
                stats.extend(block.cross.statistics(block.branch_norms[i](branch),
                                                     i, excluded))
            intermediate = [self._run(block.before_self, state, pos, *stats)
                            for state, pos in zip(states, coordinate_chunks)]
            # The reduction covers ALL queries before ANY chunk is decoded.
            # Chunking changes summation roundoff only, never attention scope.
            global_key_sum = global_context = None
            for state in intermediate:
                key_sum, context = self._run(block.self_statistics, state)
                global_key_sum = key_sum if global_key_sum is None else global_key_sum + key_sum
                global_context = context if global_context is None else global_context + context
            states = [self._run(block.after_self, state, pos,
                                global_key_sum, global_context)
                      for state, pos in zip(intermediate, coordinate_chunks)]
        return torch.cat([self._run(self.head, state).sigmoid()
                          for state in states], 1).float()

    def forward_tokens(self, tokens, excluded, counts=None, coordinates_m=None,
                       chunk_size=None):
        device_type = tokens.device.type
        # The formal experiment is FP32; explicitly reject ambient autocast.
        context = torch.autocast(device_type=device_type, enabled=False) if device_type in ('cpu', 'cuda') else nullcontext()
        with context:
            branches, excluded = self.encode_tokens(tokens, excluded, counts)
            coords = self.query_coordinates_m if coordinates_m is None else coordinates_m
            return self.decode_queries(branches, excluded, coords, chunk_size)

    def forward(self, batch):
        tokens, excluded, counts = ray_tokens(batch)
        decoded = self.forward_tokens(tokens, excluded, counts)
        return decoded.transpose(1, 2).reshape(tokens.shape[0], 1, 64, 64, 64)
