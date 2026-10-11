"""Small pre-activation-free residual blocks; spatial axes are never permuted."""
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, dimensions):
        super().__init__()
        conv = nn.Conv2d if dimensions == 2 else nn.Conv3d
        self.body = nn.Sequential(conv(channels, channels, 3, padding=1),
                                  nn.GroupNorm(8, channels), nn.SiLU(),
                                  conv(channels, channels, 3, padding=1),
                                  nn.GroupNorm(8, channels))
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.body(x))
