"""32-cubed to 64-cubed residual XYZ decoder."""
from torch import nn
from .blocks import ResidualBlock


class ResidualDecoder3D(nn.Sequential):
    def __init__(self):
        super().__init__(*[ResidualBlock(32, 3) for _ in range(3)],
                         nn.Upsample(size=(64, 64, 64), mode='trilinear', align_corners=True),
                         nn.Conv3d(32, 24, 3, padding=1), nn.GroupNorm(8, 24), nn.SiLU(),
                         *[ResidualBlock(24, 3) for _ in range(2)],
                         nn.Conv3d(24, 1, 1), nn.Sigmoid())
