"""Independent train-GT64 diagnostic. Historical no-GT guards remain untouched."""
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.models.ray_senseiver_ss import RaySenseiverSS
from geophy_set3d.training.ray_senseiver import tensor_identity

NAME = 'ray_senseiver_supervised_probe_v1'
STATIONS = [26, 27, 45, 42]


class SupervisedProbeGuard:
    """Exact train projection/GT64 whitelist; no foreign checkpoint or array reads.

    Independent audit hook, deliberately does not override historical GTReadGuard.
    Construction reads only the source manifest. GT128 is never opened, even to hash.
    """
    def __init__(self, root, own_root, indices):
        self.root = Path(root).resolve()
        self.own_root = Path(own_root).resolve()
        rows = [json.loads(s) for s in (self.root/'source_index.jsonl').read_text().splitlines() if s.strip()]
        train = [r for r in rows if r['split'] == 'train']
        if not indices or any(i not in range(8) for i in indices):
            raise ValueError('Only registered first eight train indices')
        self.allow = {}
        for i in indices:
            r = train[i]
            for k, label in [('train_projection_path', 'projection'), ('gt_volume_64_path', 'train_GT64')]:
                self.allow[str((self.root/r[k]).resolve())] = label
        self.geometry = str((self.root/'geometry/train_geometry.npz').resolve())
        self.opens = Counter(); self.denied = []; self.active = False

    def _audit(self, event, args):
        if not self.active or event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).resolve())
        mode, flags = args[1:3]
        reading = ('r' in mode or '+' in mode) if isinstance(mode, str) else not bool(flags & os.O_WRONLY)
        p = Path(path)
        if not reading:
            if p.is_relative_to(self.root):
                self.denied.append(path); raise PermissionError('Frozen dataset is read-only')
            return
        category = None
        if p.suffix == '.npy':
            category = self.allow.get(path)
            if category is None and not p.is_relative_to(self.own_root):
                self.denied.append(path); raise PermissionError('Only registered train Pool A / GT64 arrays')
        elif p.suffix == '.npz':
            if path != self.geometry and not p.is_relative_to(self.own_root):
                self.denied.append(path); raise PermissionError('Only Pool A geometry / own derived arrays')
            category = 'pool_A_geometry' if path == self.geometry else 'own_array'
        elif p.suffix in ('.pt', '.pth', '.ckpt'):
            if not p.is_relative_to(self.own_root):
                self.denied.append(path); raise PermissionError('No historical or foreign weights')
            category = 'own_checkpoint'
        if category:
            self.opens[(category, path)] += 1

    def __enter__(self):
        self.active = True; sys.addaudithook(self._audit); return self

    def __exit__(self, *args):
        self.active = False

    def summary(self):
        return dict(policy='Only registered train GT64 and Pool A; GT128/Val/test/Pool B/foreign weights forbidden',
                    denied_attempts=self.denied, reads=[dict(category=c, path=p, count=n) for (c,p),n in sorted(self.opens.items())])


class FixedStations:
    def sample(self, *args, **kwargs):
        return np.array(STATIONS, dtype=np.int64), np.empty(0, dtype=np.int64)


def dataset(root):
    ds = ProjectionPoolDataset(root, split='train', geometry_pool='train',
        min_input_views=4, max_input_views=4, hold_views=0, base_seed=20261011,
        normalization='linear_p99_5', load_gt128=False)
    ds.records = ds.records[:8]
    ds.sampler = FixedStations()
    return ds


def separate(sample):
    """Keep supervised labels entirely outside the unmodified no-GT model API."""
    gt = sample['gt_volume_64']
    inputs = {k:v for k,v in sample.items() if k != 'gt_volume_64'}
    return inputs, gt


def normalized_mse(prediction, target):
    return ((prediction-target).square().flatten(1).mean(1) /
            target.square().flatten(1).mean(1).clamp_min(1e-12)).mean()


def volume_metrics(prediction, target):
    p = prediction.detach().double().flatten(); g = target.detach().double().flatten()
    if not torch.isfinite(p).all() or not torch.isfinite(g).all():
        raise FloatingPointError('Nonfinite volume')
    d = p-g; region = g >= .1*g.max()
    def err(a, b):
        diff = a-b
        return dict(nmse=float(diff.square().mean()/b.square().mean().clamp_min(1e-12)),
                    relative_l1=float(diff.abs().sum()/b.abs().sum().clamp_min(1e-12)),
                    relative_l2=float(diff.square().sum().sqrt()/b.square().sum().sqrt().clamp_min(1e-12)))
    return dict(**err(p,g), mean=float(p.mean()), peak=float(p.max()), minimum=float(p.min()),
        spatial_std=float(p.std(unbiased=False)), gt_mean=float(g.mean()), gt_peak=float(g.max()),
        gt_spatial_std=float(g.std(unbiased=False)), total_ratio=float(p.sum()/g.sum().clamp_min(1e-12)),
        plume_voxels=int(region.sum()), plume=err(p[region],g[region]))


def init_scale(targets):
    mean = float(torch.stack(targets).double().mean())
    value = min(max(mean, 1e-6), 1-1e-6)
    return dict(gt_all_voxel_mean=mean, bias_input=value, clamp_applied=mean != value,
                bias_logit=math.log(value/(1-value)), output_weight_factor=.01)


def fresh(scale, chunk=16384, checkpoint=True):
    model = RaySenseiverSS(query_chunk_size=chunk, decoder_checkpoint=checkpoint)
    with torch.no_grad():
        model.decoder.postproc.weight.mul_(.01)
        model.decoder.postproc.bias.fill_(scale['bias_logit'])
    return model


def metrics_mean(rows):
    return {k:sum(r[k] for r in rows)/len(rows) for k in ['nmse','relative_l1','relative_l2']}


def criterion(stage, mean, limits, dependence, template_nmse):
    good = mean['nmse'] <= limits['nmse_threshold'] and mean['relative_l1'] <= limits['relative_l1_threshold']
    if stage == 'B':
        good = good and mean['nmse'] <= (1-limits['template_reduction'])*template_nmse
        good = good and dependence is not None and sum(r['relative_MA'] >= .001 for r in dependence) >= 2
    return bool(good)


def exit_event(stage, updates, losses, evaluation, limits, previous, elapsed, template_nmse=None):
    """Plateau uses adjacent disjoint 200-update training-loss windows.

    B joint passing checks use current (200-step) interventions, never stale ones.
    Other 100-step evaluations still update the plateau window.
    """
    current_dep = evaluation.get('interventions')
    joint = stage == 'A' or current_dep is not None
    passed = criterion(stage,evaluation['mean'],limits,current_dep,template_nmse)
    streak = previous.get('pass_streak',0)
    if joint:
        streak = streak+1 if passed else 0
    elif not (evaluation['mean']['nmse']<=limits['nmse_threshold'] and evaluation['mean']['relative_l1']<=limits['relative_l1_threshold']):
        streak = 0
    recent=older=improvement=None
    if len(losses) >= 400:
        recent=sum(losses[-200:])/200;older=sum(losses[-400:-200])/200
        improvement=(older-recent)/max(older,1e-12)
    plateau = updates >= limits['minimum_plateau_updates'] and improvement is not None and improvement < .02 and not passed
    failures = previous.get('plateau_streak',0)+1 if plateau else 0
    reason = ('PASSED' if streak>=2 and joint else 'PLATEAU_FAILED_TO_LEARN' if failures>=2 else
              'BUDGET_EXHAUSTED_UNRESOLVED' if updates>=limits['max_updates'] or elapsed>=limits['max_seconds'] else None)
    return dict(updates=updates, pass_condition=passed, pass_streak=streak,
                plateau_condition=plateau, plateau_streak=failures, recent200=recent,
                previous200=older, relative_improvement=improvement, reason=reason)


def verify_resume(saved, stage, config, registration_hash):
    if saved['stage']!=stage or saved['config']!=config or saved['registration_sha256']!=registration_hash:
        raise ValueError('Wrong experiment/stage/config')
    if saved['exit_state'].get('reason'):
        raise ValueError('Scientifically stopped run cannot resume')
    if saved['updates']!=len(saved['history']) or not all(k in saved for k in ['optimizer','rng','elapsed_seconds','initial_hash']):
        raise ValueError('Incomplete resume state')


def input_identity(batch):
    return tensor_identity(*[batch[k] for k in sorted(batch) if isinstance(batch[k],torch.Tensor)])
