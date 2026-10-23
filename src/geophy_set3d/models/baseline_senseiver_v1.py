"""Independent comparison entry for the existing ray-integral Senseiver adapter.

The architecture and initialisation are exactly RaySenseiverSS, not a new repair.
The pinned BSD-3-Clause core and notice live in senseiver_core.py and
third_party/senseiver/. No GeoPhy GeometryLifting or trained weights are used.
"""
import torch

from .ray_senseiver_ss import RaySenseiverSS


class RaySenseiverV1(RaySenseiverSS):
    """All valid rays -> 64 learned latents -> complete 64³ XYZ queries.

    This wrapper only validates memory controls and makes the FP32 contract
    explicit. It does not apply the historically unsuccessful initialisation,
    loss changes, global latent residual, or supervised warm starts.
    """

    def __init__(self, query_chunk_size=16384, decoder_checkpoint=True):
        if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size < 1:
            raise ValueError('query_chunk_size must be a positive integer')
        if not isinstance(decoder_checkpoint, bool):
            raise ValueError('decoder_checkpoint must be boolean')
        super().__init__(query_chunk_size=query_chunk_size,
                         decoder_checkpoint=decoder_checkpoint)

    def forward(self, batch):
        parameter = next(self.parameters())
        if parameter.dtype != torch.float32:
            raise ValueError('RaySenseiverV1 requires FP32 model parameters')
        # The unified experiment uses FP32. Nested caller autocast must not
        # silently change the measured architecture's inference arithmetic.
        with torch.autocast(device_type=parameter.device.type, enabled=False):
            return super().forward(batch)
