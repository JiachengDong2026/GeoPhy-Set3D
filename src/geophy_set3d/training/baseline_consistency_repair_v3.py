"""Baseline-only homogeneous relative consistency; original selfsup stays intact.

The fixed+1e-6 denominator makes L(alpha*A,alpha*B) decrease to zero with
alpha even when relative disagreement is unchanged. This version removes that
radial incentive. FP64 reductions protect cancellation; zero/zero is zero.
"""
import torch


def homogeneous_consistency(ca,cb):
    if ca.shape!=cb.shape or ca.ndim<2:
        raise ValueError('Aligned batch fields required')
    if not torch.isfinite(ca).all() or not torch.isfinite(cb).all() or ca.min()<0 or cb.min()<0:
        raise FloatingPointError('Consistency expects finite nonnegative mass fractions')
    a,b=ca.double(),cb.double()
    numerator=(a-b).abs().flatten(1).mean(1)
    denominator=.5*(a.abs().flatten(1).mean(1)+b.abs().flatten(1).mean(1))
    safe=denominator.clamp_min(torch.finfo(torch.double).tiny)
    value=torch.where(denominator>0,numerator/safe,torch.zeros_like(numerator))
    return value.mean().to(ca.dtype)
