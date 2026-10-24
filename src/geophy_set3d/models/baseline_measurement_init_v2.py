"""Train-measurement-calibrated initialization of independent ray baselines.

Only the final linear weight/bias initialization changes. The architecture,
forward, sigmoid range and token information are inherited unchanged from v1.
No checkpoint/label/statistics file is loaded by these constructors.
"""
import math
from torch import nn
from .baseline_deepsets_v1 import RayDeepSetsV1
from .baseline_gnot_v1 import RayGNOTV1


def initialize_readout(linear, mass_fraction, weight_multiplier=.01):
    if not isinstance(linear, nn.Linear):
        raise TypeError('A scalar linear readout is required')
    if linear.out_features != 1 or not math.isfinite(mass_fraction) or not 0 < mass_fraction < 1:
        raise ValueError('Finite strictly interior measurement-derived mass fraction required')
    if weight_multiplier != .01:
        raise ValueError('This version registers exactly0.01 readout scaling, no scale sweep')
    linear.weight.data.mul_(weight_multiplier)
    nn.init.constant_(linear.bias, math.log(mass_fraction / (1 - mass_fraction)))


class RayDeepSetsMeasurementInitV2(RayDeepSetsV1):
    def __init__(self, mass_fraction, **kwargs):
        super().__init__(**kwargs)
        initialize_readout(self.decoder[-1], mass_fraction)


class RayGNOTMeasurementInitV2(RayGNOTV1):
    def __init__(self, mass_fraction, **kwargs):
        super().__init__(**kwargs)
        initialize_readout(self.head[-1], mass_fraction)
