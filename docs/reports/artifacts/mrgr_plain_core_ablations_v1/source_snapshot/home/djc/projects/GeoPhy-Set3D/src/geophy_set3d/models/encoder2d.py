"""One shared, full angular resolution encoder for every station."""
from torch import nn
from .blocks import ResidualBlock


class SharedEncoder2D(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32),
                         nn.SiLU(), *[ResidualBlock(32, 2) for _ in range(3)])
