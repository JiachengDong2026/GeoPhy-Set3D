"""Read-only frozen-model stages, masked reductions, and case-level diagnostics."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from .clustered_one_swap import ClusteredEvaluationReadGuard

METRICS = ['relative_l1', 'relative_l2', 'mae', 'rmse',
           'integrated_concentration_relative_error', 'peak_value_absolute_error',
           'observed_relative_l1', 'observed_relative_l2', 'heldout_relative_l1', 'heldout_relative_l2']
AMPLITUDES = ['context0_rms', 'delta_rms', 'delta_context_ratio',
              'voxel_correction_mae', 'voxel_correction_rms', 'signed_residual_scaled_rms']
GROUPS = ['G1', 'G2', 'G3', 'G4', 'G5']


class DiagnosticReadGuard(ClusteredEvaluationReadGuard):
    """Inference cannot read GT; posthoc cannot read checkpoints/measurements."""
    def __init__(self, dataset_root, checkpoints, output, phase):
        if phase not in ('inference', 'gt'):
            raise ValueError('Explicit inference or gt phase required')
        super().__init__(dataset_root, checkpoints, output)
        self.phase = phase

    def _audit(self, event, args):
        if self.active and event == 'open' and isinstance(args[0], (str, bytes, Path)):
            import os
            p = Path(os.fsdecode(args[0])).resolve()
            if self.phase == 'inference' and p in self.gt64:
                self.deny('nonval_or_unknown_npy_attempts', 'GT64 forbidden during inference')
            if self.phase == 'gt' and (p in self.projections or p.suffix in ('.pt', '.pth', '.ckpt')):
                self.deny('nonval_or_unknown_npy_attempts', 'Posthoc reads saved outputs and Val GT64 only')
        super()._audit(event, args)

    def summary(self):
        return dict(super().summary(), phase=self.phase,
                    policy='Inference: Val Pool A + original checkpoint, no GT. GT: saved outputs + Val GT64, no checkpoint/measurement. Neither: GT128/Pool B/test/writes.')


@contextmanager
def capture_context(model):
    """Capture actual tensors without recomputing or modifying the forward."""
    result = {}
    def fusion_hook(module, args):
        if 'context0' not in result:
            result['context0'] = args[0]
    def interaction_hook(module, args):
        result['local_observability'] = args[5]
    handles = [model.fusion.register_forward_pre_hook(fusion_hook),
               model.interaction.register_forward_pre_hook(interaction_hook)]
    try:
        yield result
    finally:
        for handle in handles:
            handle.remove()


def masked_moments(value, mask):
    """FP64 reduction across selected scalar elements; invalid values excluded."""
    expanded = torch.broadcast_to(mask.bool(), value.shape)
    selected = value[expanded].double()
    if not selected.numel():
        return dict(count=0, mean=None, mae=None, rms=None)
    if not torch.isfinite(selected).all():
        raise FloatingPointError('Nonfinite valid diagnostic region')
    return dict(count=selected.numel(), mean=float(selected.mean()),
                mae=float(selected.abs().mean()), rms=float(selected.square().mean().sqrt()))


def safe_ratio(numerator, denominator, epsilon):
    if numerator is None or denominator is None or denominator <= epsilon:
        return None
    return float(numerator / denominator)


def stage_amplitudes(stages, context0, batch, epsilon):
    visible = stages['visibility_base'] & batch['view_mask'][:, :, None, None, None]
    c = masked_moments(context0, visible[:, :, None])
    d = masked_moments(stages['residual_context'], visible[:, :, None])
    valid = batch['input_valid_ray'].bool() & batch['view_mask'][:, :, None, None, None]
    r = masked_moments(stages['measurement_residual'], valid)
    rs = masked_moments(stages['residual_scaled'], valid)
    correction = (stages['C1'] - stages['C0']).double()
    ratio = safe_ratio(d['rms'], c['rms'], epsilon)
    # Independent exact context contract; no estimated context0=context1-delta.
    assert torch.equal(stages['context1'], torch.where(visible[:, :, None], context0 + stages['residual_context'], 0))
    return dict(context0_rms=c['rms'], delta_rms=d['rms'], delta_context_ratio=ratio,
                ratio_undefined=int(ratio is None), ratio_epsilon=epsilon,
                valid_context_scalar_count=c['count'], valid_view_voxel_count=int(visible.sum()),
                signed_residual_mean_ppm_m=r['mean'], signed_residual_mae_ppm_m=r['mae'],
                signed_residual_rms_ppm_m=r['rms'], signed_residual_scaled_rms=rs['rms'],
                valid_residual_ray_count=r['count'], voxel_correction_mae=float(correction.abs().mean()),
                voxel_correction_rms=float(correction.square().mean().sqrt()), context_contract_exact=True)


def aggregate_cases(rows):
    frame = pd.DataFrame(rows)
    identity = ['case_id', 'case_index', 'group', 'draw', 'geometry_id']
    if frame.duplicated(['case_id', 'geometry_id']).any():
        raise ValueError('Repeated case/layout')
    for _, part in frame.groupby(['case_id', 'group']):
        if len(part) != 5 or sorted(part['draw'].tolist()) != list(range(5)):
            raise ValueError('Five unique draws required before case-level statistics')
    cols = [c for c in frame.select_dtypes(include='number').columns if c not in identity]
    return frame.groupby(['case_id', 'case_index', 'group'], sort=True)[cols].mean().reset_index()


def holm(values):
    p = np.asarray(values, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return adjusted


def paired_summary(a, b, indices, tie_epsilon):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite paired case means required')
    delta = b - a
    ci = np.quantile(delta[indices].mean(1), [.025, .975]).tolist()
    p = 1.0 if np.all(delta == 0) else float(wilcoxon(delta).pvalue)
    return dict(before_mean=float(a.mean()), after_mean=float(b.mean()),
                delta_mean=float(delta.mean()), delta_median=float(np.median(delta)),
                ci_low=ci[0], ci_high=ci[1], improved=int((delta < -tie_epsilon).sum()),
                worsened=int((delta > tie_epsilon).sum()), tied=int((np.abs(delta) <= tie_epsilon).sum()),
                exact_improved=int((delta < 0).sum()), exact_worsened=int((delta > 0).sum()),
                exact_tied=int((delta == 0).sum()), wilcoxon_p=p, case_count=len(delta))
