"""Ray-Senseiver-SS: complete ray-integral inputs and full XYZ coordinate queries."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from geophy_set3d.data.measurement_only import assert_no_gt
from .senseiver_core import Encoder, Decoder

Y_SCALE = 32284.894580078173


def fourier(x, bands=16):
    """Official linear frequencies and axis order: sin XYZ then cos XYZ."""
    phase = math.pi * x[..., :, None] * torch.linspace(1., 32., bands, device=x.device)
    return torch.cat([phase.sin().flatten(-2), phase.cos().flatten(-2)], -1)


def ray_tokens(batch):
    """Pack every valid ray, including zero integral; a zero token for empty sets."""
    assert_no_gt(batch)
    mask = batch['input_valid_ray'][:, :, 0].bool() & batch['view_mask'][:, :, None, None]
    tokens = []
    for i in range(len(mask)):
        valid = mask[i]
        if not valid.any():
            tokens.append(batch['input_projection_raw'].new_zeros((1, 203)))
            continue
        direction = batch['input_ray_direction'][i][valid]
        origin = batch['input_station_origin_m'][i, :, None, None, :].expand_as(batch['input_ray_direction'][i])[valid]
        near = batch['input_t_near'][i, :, 0][valid, None]
        length = batch['input_path_length_m'][i, :, 0][valid, None]
        entry = origin + direction * near
        exit = entry + direction * length
        entry, exit = 2 * entry / .6 - 1, 2 * exit / .6 - 1
        value = batch['input_projection_raw'][i, :, 0][valid, None] / Y_SCALE
        token = torch.cat([value, entry, exit, direction, length / (math.sqrt(3) * .6), fourier(entry), fourier(exit)], -1)
        if not torch.isfinite(token).all():
            raise FloatingPointError('Nonfinite valid ray geometry/measurement')
        tokens.append(token)
    maximum = max(len(x) for x in tokens)
    padded = tokens[0].new_zeros(len(tokens), maximum, 203)
    excluded = torch.ones(len(tokens), maximum, dtype=torch.bool, device=padded.device)
    for i, x in enumerate(tokens):
        padded[i, :len(x)] = x
        excluded[i, :len(x)] = False
    return padded, excluded, mask.flatten(1).sum(1)


class RaySenseiverSS(nn.Module):
    def __init__(self, query_chunk_size=16384, decoder_checkpoint=True):
        super().__init__()
        self.encoder = Encoder(203, 64, 64, 64, num_layers=3,
            num_cross_attention_heads=4, num_self_attention_heads=4,
            num_self_attention_layers_per_block=3, dropout=0.)
        self.decoder = Decoder(96, 64, 64, 1, 1, num_cross_attention_heads=4, dropout=0.)
        axis = torch.linspace(0., .6, 64)
        coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
        self.register_buffer('query_coordinates_m', coords, persistent=False)
        self.register_buffer('query_encoding', fourier(2 * coords / .6 - 1), persistent=False)
        self.query_chunk_size = query_chunk_size
        self.decoder_checkpoint = decoder_checkpoint

    def encode(self, batch):
        tokens, mask, _ = ray_tokens(batch)
        return self.encoder(tokens, mask)

    def decode_queries(self, latent, queries, chunk_size=None, checkpoint_chunks=None):
        size = self.query_chunk_size if chunk_size is None else chunk_size
        use_checkpoint = self.decoder_checkpoint if checkpoint_chunks is None else checkpoint_chunks
        outputs = []
        for coords in queries.split(size):
            coords = coords.unsqueeze(0).expand(len(latent), -1, -1)
            if use_checkpoint and self.training and torch.is_grad_enabled():
                pred = checkpoint(self.decoder, latent, coords, use_reentrant=False)
            else:
                pred = self.decoder(latent, coords)
            outputs.append(pred.sigmoid())
        return torch.cat(outputs, 1)

    def forward(self, batch):
        latent = self.encode(batch)
        return self.decode_queries(latent, self.query_encoding).transpose(1, 2).reshape(len(latent), 1, 64, 64, 64)
