"""Isolated, frozen paper-MRGR lifting diagnostic utilities."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts/paper_final_v1'))
from common import *
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from module_models_v2 import model_for
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
from probe_mrgr_multiscale_v1 import state_hash

EXPECTED = '650b4f2c0f0ab345b5fd99a7d629e0db388cede0937acb7e3191462cd57898d7'
CONDITIONS = ('Val-A-V4', 'spread-V4', 'V3', 'V6', 'clustered-V4')
MODES = ('flip_h', 'flip_v', 'flip_hv')
BASE = Path('/scratch/djc/GeoPhy-Set3D') / 'mrgr_lifting_mapping_v1'
ARTIFACT = PROJECT / 'docs/reports/artifacts/mrgr_lifting_mapping_v1'
REPORT = PROJECT / 'docs/reports/mrgr_lifting_mapping_v1_zh.md'
EPS = 1e-6
SEED = 20261022
METRICS = ('relative_l1', 'relative_l2', 'dice50', 'absolute_peak_ratio_error', 'centroid_distance_mm')


def method():
    return verified(BASE / 'method.json')


def identity():
    paths = {**sources()}
    for folder in (HERE, PROJECT / 'scripts/uniform_final_v1'):
        paths.update({str(p): sha(p) for p in folder.glob('*.py')})
    return paths


def tensor_hash(t):
    a = t.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def batch_hash(batch):
    parts = {k: tensor_hash(v) if isinstance(v, torch.Tensor) else v for k, v in sorted(batch.items())}
    h = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return h, parts


def frozen_state(model):
    assert not model.training and all(not m.training for m in model.modules())
    assert not torch.is_grad_enabled()
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return state_hash(model.state_dict())


def load_model(device):
    """Use the exact factory and strict loading path used by official evaluation.py."""
    m = method()
    assert sha(m['checkpoint']) == EXPECTED == m['checkpoint_sha256']
    ck = torch.load(m['checkpoint'], map_location='cpu', weights_only=False)
    assert ck['epoch'] == m['best_epoch'] == 278
    assert ck['config'] == m['config'] and not ck['volumetric_gt_used']
    assert ck['protocol_sha256'] == sha(Path(m['run']) / 'protocol.json')
    model = model_for(m['config'], m['arm'], device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    assert sum(p.numel() for p in model.parameters()) == m['parameters'] == 344678
    assert model.fusion.__class__.__name__ == 'GeometryConditionedAttentionPooling'
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class Guard(AccessGuard):
    def __init__(self, *, scoring=False):
        if scoring:
            seal = verified(BASE / 'all_predictions_sealed.json')
            assert seal['formal_forwards'] == 525 and seal['sealed_before_GT64']
            verify_hashes(seal['manifest_hashes'])
        super().__init__(splits=['test'], gt_splits=['test'] if scoring else [],
                         checkpoints=[method()['checkpoint']], output=BASE)
        for r in verified(BASE / 'baseline_manifest.json'):
            self.allowed[str(Path(r['path']).resolve())] = 'historical_sealed_prediction'


def reference(row):
    assert sha(row['path']) == row['sha256']
    with np.load(row['path'], allow_pickle=False) as z:
        out = {k: z[k] for k in ('C0', 'C1')}
    valid_fields(out)
    return out


def valid_fields(fields):
    assert set(fields) == {'C0', 'C1'}
    assert all(a.shape == (64, 64, 64) and a.dtype == np.float32 and
               np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1 for a in fields.values())


def differences(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(mean_abs=float(d.mean()), max_abs=float(d.max()),
                relative_change=float(d.mean() / (np.abs(b.astype(np.float64)).mean() + EPS)))


def prepare(model, batch):
    """Geometry-only reference; never run a baseline network prediction."""
    grid, visibility = model.lifting.angular_grid(batch)
    valid = visibility.reshape(1, batch['view_mask'].shape[1], 32, 32, 32)
    valid = valid & batch['view_mask'][:, :, None, None, None]
    directions, relations = model.ray_geometry(batch['input_station_origin_m'], model.lifting.voxel_xyz,
                                               batch['input_station_azimuth_deg'], valid)
    mask = valid.flatten(2).transpose(1, 2).reshape(-1, valid.shape[1])
    count = batch['view_mask'].sum(1)[:, None].expand(-1, 32**3).reshape(-1)
    obs = model.observability(directions, mask, count)
    return dict(grid=grid, visibility=visibility, geometry=model.geometry_descriptor(batch),
                relations=relations, observability=obs)


class MappingOnly:
    """Temporarily replace only this instance's forward, never angular_grid.

    The feedback lifting calls the original bound forward verbatim. All hooks
    are read-only checks. Geometry descriptors/ray relations are compared with
    geometry-only references computed before the network forward.
    """
    def __init__(self, model, mode, expected, *, engineering=False):
        assert mode in ('identity', *MODES)
        self.model, self.mode, self.expected = model, mode, expected
        self.engineering = engineering
        self.lifting_calls = self.interaction_calls = self.fusion_calls = 0
        self.records = []

    def lift(self, features, batch):
        self.lifting_calls += 1
        grid, visibility = self.model.lifting.angular_grid(batch)
        assert torch.equal(grid, self.expected['grid'])
        assert torch.equal(visibility, self.expected['visibility'])
        original_hash = tensor_hash(grid)
        used = grid.clone()
        if self.lifting_calls == 1:
            if self.mode in ('flip_h', 'flip_hv'):
                used[..., 0] = -used[..., 0]
            if self.mode in ('flip_v', 'flip_hv'):
                used[..., 1] = -used[..., 1]
            b, v, c, _, _ = features.shape
            sampled = F.grid_sample(features.flatten(0, 1), used.reshape(b*v, -1, 1, 2),
                                    mode='bilinear', padding_mode='zeros', align_corners=True)
            output = sampled.reshape(b, v, c, *([self.model.lifting.resolution] * 3))
            result = output, visibility.reshape(b, v, *([self.model.lifting.resolution] * 3))
        else:
            assert self.lifting_calls == 2
            result = self.original_forward(features, batch)
            assert torch.equal(result[1].flatten(2), visibility)
            if self.engineering:
                b, v, c, _, _ = features.shape
                independent = F.grid_sample(features.flatten(0, 1), grid.reshape(b*v, -1, 1, 2),
                                            mode='bilinear', padding_mode='zeros', align_corners=True)
                assert torch.equal(result[0], independent.reshape_as(result[0]))
        assert tensor_hash(grid) == original_hash
        assert result[0].dtype == features.dtype
        assert torch.equal(result[1].flatten(2), self.expected['visibility'])
        diff = (used - grid).abs()
        change = dict(mean_abs=float(diff.mean()), max_abs=float(diff.max()),
                      changed_coordinates=int((diff > 0).sum()), per_coordinate_mean_abs=diff.mean((0, 1, 2)).tolist())
        if self.lifting_calls == 1 and self.mode != 'identity':
            assert change['changed_coordinates'] > 0
        else:
            assert change['max_abs'] == 0
        self.records.append(dict(call=self.lifting_calls, feature_shape=list(features.shape),
            feature_sha256=tensor_hash(features), original_grid_sha256=original_hash,
            sampling_grid_sha256=tensor_hash(used), visibility_sha256=tensor_hash(visibility),
            returned_visibility_sha256=tensor_hash(result[1].flatten(2)), visibility_equal=True,
            original_grid_unmodified=True, original_forward_used=self.lifting_calls == 2,
            output_shape=list(result[0].shape), dtype=str(result[0].dtype), grid_change=change))
        return result

    def interaction(self, module, inputs):
        _, geometry, visibility, view_mask, relations, obs, *_ = inputs
        self.interaction_calls += 1
        assert torch.equal(geometry, self.expected['geometry'])
        assert torch.equal(visibility.flatten(2), self.expected['visibility'])
        assert torch.equal(relations, self.expected['relations'])
        assert torch.equal(obs, self.expected['observability'])
        self.geometry_hash = tensor_hash(geometry)
        self.relations_hash = tensor_hash(relations)
        self.obs_hash = tensor_hash(obs)

    def fusion(self, module, inputs):
        self.fusion_calls += 1
        _, geometry, visibility, _ = inputs
        assert torch.equal(geometry, self.expected['geometry'])
        assert torch.equal(visibility.flatten(2), self.expected['visibility'])

    def __enter__(self):
        self.had_forward = 'forward' in self.model.lifting.__dict__
        self.prior_attribute = self.model.lifting.__dict__.get('forward')
        self.original_forward = self.model.lifting.forward
        self.original_angular = self.model.lifting.angular_grid.__func__
        self.model.lifting.forward = self.lift
        self.hooks = [self.model.interaction.register_forward_pre_hook(self.interaction),
                      self.model.fusion.register_forward_pre_hook(self.fusion)]
        return self

    def __exit__(self, *args):
        if self.had_forward:
            self.model.lifting.forward = self.prior_attribute
        else:
            del self.model.lifting.forward
        for h in self.hooks:
            h.remove()
        assert self.model.lifting.angular_grid.__func__ is self.original_angular
        if args[0] is None:
            assert (self.lifting_calls, self.interaction_calls, self.fusion_calls) == (2, 1, 2)

    def audit(self):
        return dict(lifting_calls=self.lifting_calls, interaction_calls=self.interaction_calls,
                    fusion_calls=self.fusion_calls, lifting=self.records,
                    geometry_descriptor_sha256=self.geometry_hash, ray_relations_sha256=self.relations_hash,
                    observability_sha256=self.obs_hash, angular_grid_unchanged=True,
                    descriptors_equal=True, relations_equal=True, observability_equal=True,
                    feedback_original_forward=True, wrapper_restored='forward' not in self.model.lifting.__dict__)


def run_forward(model, batch, expected, mode, *, engineering=False):
    before, input_parts = batch_hash(batch)
    state = frozen_state(model)
    with MappingOnly(model, mode, expected, engineering=engineering) as probe:
        stages = model(batch, return_stages=True)
    fields = {k: stages[k][0, 0].cpu().numpy().copy() for k in ('C0', 'C1')}
    valid_fields(fields)
    assert frozen_state(model) == state
    assert batch_hash(batch)[0] == before
    return fields, dict(input_sha256=before, input_field_hashes=input_parts,
                       state_before=state, state_after=state, eval_mode=True, no_gradients=True,
                       checkpoint_sha256=EXPECTED, return_stages=True, model_forward_calls=1,
                       mapping=probe.audit())
