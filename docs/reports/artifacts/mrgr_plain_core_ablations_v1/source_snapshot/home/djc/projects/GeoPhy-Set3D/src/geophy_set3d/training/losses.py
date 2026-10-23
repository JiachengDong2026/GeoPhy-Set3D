import torch
from torch.nn import functional as F


def supervised_loss(prediction, target, smooth_l1=1., relative_l1=.1, relative_epsilon=1e-6):
    smooth = F.smooth_l1_loss(prediction, target)
    relative = ((prediction-target).abs().flatten(1).sum(1) /
                target.abs().flatten(1).sum(1).clamp_min(relative_epsilon)).mean()
    total = smooth_l1*smooth + relative_l1*relative
    return total, {'smooth_l1': smooth, 'relative_l1': relative, 'total': total}
