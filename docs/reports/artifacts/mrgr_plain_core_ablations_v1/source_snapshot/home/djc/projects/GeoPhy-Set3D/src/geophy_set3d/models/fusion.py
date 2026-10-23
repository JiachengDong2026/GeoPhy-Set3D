"""Permutation invariant, visibility and padding aware mean."""
import torch


def masked_mean(lifted, visibility):
    mask = visibility[:, :, None]
    # where also prevents inactive NaNs from entering the reduction.
    numerator = torch.where(mask, lifted, 0).sum(1)
    return numerator / mask.sum(1).clamp_min(1)
