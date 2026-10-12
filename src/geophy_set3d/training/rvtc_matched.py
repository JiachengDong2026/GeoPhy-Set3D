"""Independent matched contrast; never modify the original RVTC objective/constants."""
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from geophy_set3d.training import selfsup_v2 as v2
from geophy_set3d.training.utils import rng_state, restore_rng
from geophy_set3d.evaluation.rvtc_failure_diagnostics import weighted_losses, vector_geometry


def contrast_loss(parts, weight):
    if weight not in (0., .1):
        raise ValueError('Only the registered 0 / 0.1 contrast is permitted')
    weighted = weighted_losses(parts)
    measure = weighted['student_measurement'] + weighted['teacher_measurement']
    # Control has no computational dependence on consistency in its training loss.
    return measure if weight == 0 else measure + weighted['consistency']


def state_hash(state):
    """Canonical tensor-content hash, independent of torch serialization headers."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def validate_pair(configs):
    if len(configs) != 2 or [c['arm'] for c in configs] != ['Control', 'Treatment']:
        raise ValueError('Exactly Control and Treatment, in registered order')
    a, b = [{k: v for k, v in c.items() if k not in ('arm', 'consistency_weight')} for c in configs]
    if a != b or [c['consistency_weight'] for c in configs] != [0., .1]:
        raise ValueError('Arms differ beyond their registered consistency weight')
    expected = dict(seed=20261011, epochs=80, batch_size=1, gradient_accumulation=2,
        optimizer_steps=320, precision='fp32', optimizer=dict(name='adamw', lr=2e-4, weight_decay=1e-4),
        gradient_clip_norm=1., scheduler=None, shuffle=False, station_epoch=0,
        projector=dict(samples=96, all_rays=True, view_chunk=2),
        gradient_epochs=list(range(10, 81, 10)), gradient_cases='all gate8 plus first4 Val-A in frozen order',
        bootstrap=dict(samples=10000, seed=2026101202, unit='paired case', confidence=.95))
    if any(a.get(k) != v for k, v in expected.items()):
        raise ValueError('Matched finite protocol changed')


class ContrastReadGuard(v2.V2ReadGuard):
    """Allow only this experiment's exact lineage files, forbid all external weights."""
    def __init__(self, dataset_root, checkpoint_root, permitted, audit_log=None):
        super().__init__(dataset_root, checkpoint_root)
        self.permitted = {Path(p).resolve() for p in permitted}
        self.checkpoint_reads = 0
        self.checkpoint_writes = 0
        self.audit_log = Path(audit_log) if audit_log else None

    def __exit__(self, *args):
        super().__exit__(*args)
        if self.audit_log:
            with self.audit_log.open('a') as f:
                f.write(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    failed=args[0] is not None, **self.summary()))+'\n')

    def _audit(self, event, args):
        if self.active and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix in ('.pt', '.pth', '.ckpt') or (path.suffix == '.tmp' and path.is_relative_to(self.checkpoint_root.resolve())):
                base = path.with_suffix('.pt') if path.suffix == '.tmp' else path
                if base not in self.permitted:
                    with self.supervised_checkpoint_attempts.get_lock():
                        self.supervised_checkpoint_attempts.value += 1
                    raise PermissionError('Only registered own-experiment checkpoint lineage is permitted')
                mode, flags = args[1:3]
                writing = any(c in mode for c in 'wa+') if isinstance(mode, str) else bool(flags & (os.O_WRONLY | os.O_RDWR))
                if writing:
                    self.checkpoint_writes += 1
                else:
                    self.checkpoint_reads += 1
        super()._audit(event, args)

    def summary(self):
        return dict(super().summary(), own_checkpoint_reads=self.checkpoint_reads,
            own_checkpoint_writes=self.checkpoint_writes, loader_workers=0,
            external_weights_loaded=False, final_test_used=False)


@contextlib.contextmanager
def diagnostic_context(model):
    """Read-only diagnostics preserve tensors, mode, RNG, and parameter gradients."""
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError('Diagnostics require a completed update boundary')
    generator = torch.Generator()
    before_rng = rng_state(generator)
    before_state = state_hash(model.state_dict())
    mode = model.training
    try:
        model.eval()
        yield
        after_rng = rng_state(generator)
        if before_rng['python'] != after_rng['python'] or not np.array_equal(before_rng['numpy'][1], after_rng['numpy'][1]) or before_rng['numpy'][2:] != after_rng['numpy'][2:]:
            raise AssertionError('Diagnostic changed Python/NumPy random state')
        if not torch.equal(before_rng['torch'], after_rng['torch']) or any(not torch.equal(a,b) for a,b in zip(before_rng['cuda'],after_rng['cuda'])):
            raise AssertionError('Diagnostic changed torch random state')
        if before_state != state_hash(model.state_dict()) or any(p.grad is not None for p in model.parameters()):
            raise AssertionError('Diagnostic mutated model tensors or gradients')
    finally:
        restore_rng(before_rng, generator)
        model.train(mode)


def paired_effect(control, treatment, samples=10000, seed=2026101202):
    control, treatment = np.asarray(control), np.asarray(treatment)
    if control.shape != treatment.shape or control.ndim != 1 or len(control) == 0:
        raise ValueError('Paired case arrays must have identical nonempty shape')
    delta = treatment - control
    if not np.isfinite(delta).all():
        raise FloatingPointError('Nonfinite paired effect')
    rng = np.random.default_rng(seed)
    interval = np.quantile(delta[rng.integers(0, len(delta), (samples, len(delta)))].mean(1), [.025,.975])
    decision = 'extra_measurement_benefit' if delta.mean() < 0 and interval[1] < 0 else 'measurement_harm' if delta.mean() > 0 and interval[0] > 0 else 'inconclusive'
    return dict(control_mean=float(control.mean()), treatment_mean=float(treatment.mean()),
        mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
        treatment_wins=int((delta<0).sum()), treatment_losses=int((delta>0).sum()), ties=int((delta==0).sum()),
        paired_case_95ci=interval.tolist(), decision=decision, case_count=len(delta),
        bootstrap_samples=samples, bootstrap_seed=seed, unit='paired case')


def applied_gradient_geometry(vectors, arm):
    hypothetical = vector_geometry(vectors)
    applied = vector_geometry(dict(vectors, consistency=vectors['consistency'] if arm == 'Treatment' else torch.zeros_like(vectors['consistency'])))
    return dict(applied=applied, hypothetical_original_weight_consistency=hypothetical)
