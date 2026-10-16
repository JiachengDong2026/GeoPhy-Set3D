"""Plan B: measurement-set-conditioned continuous concentration field.

Feature grids use endpoint-aligned physical XYZ axes. Query coordinates are
physical metres; x_norm=2*x/.6-1, with all four Fourier bands always enabled.
Each context is tied to its encoded input case and current model parameters;
callers must not retain it across optimizer updates or substitute another case.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from geophy_set3d.models.blocks import ResidualBlock
from geophy_set3d.models.encoder2d import SharedEncoder2D
from geophy_set3d.models.geolift_deepsets_sum import masked_scaled_sum
from geophy_set3d.models.lifting import GeometryLifting
from geophy_set3d.models.set_unrolled3d import _clean_inputs
from .physics import ContinuousRayIntegrator, Y_SCALE, mass_to_mole


def sample_trilinear(grid, xyz, extent_m=.6):
    """Sample XYZ-ordered [B,C,Nx,Ny,Nz] at [B,P,3], returning [B,P,C]."""
    if grid.ndim != 5 or xyz.ndim != 3 or xyz.shape[0] != grid.shape[0] or xyz.shape[-1] != 3:
        raise ValueError('Expected grid[B,C,Nx,Ny,Nz] and xyz[B,P,3]')
    normalized = xyz * (2. / extent_m) - 1.
    # Torch indexes W,H,D. Our storage is D=X,H=Y,W=Z.
    torch_grid = normalized[..., [2, 1, 0]].reshape(len(grid), -1, 1, 1, 3)
    values = F.grid_sample(grid, torch_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return values[:, :, :, 0, 0].transpose(1, 2)


@dataclass(frozen=True)
class FieldContext:
    features: torch.Tensor
    coverage: torch.Tensor

    @property
    def batch_size(self):
        return self.features.shape[0]


class SetConditionalField(nn.Module):
    """Reference section-8 model, with optional explicit small-grid debugging.

    Production fixes lift_resolution=32, width32, four width64 hidden layers,
    Q96, all four positional bands. No initialization from supervised weights.
    Checkpointing covers the entire feature-query/MLP/physics-reduction block,
    with non-reentrant autograd and attached shared encoder features.
    """
    def __init__(self, query_chunk=32768, ray_chunk=256, samples=96,
                 checkpoint_query=True, lift_resolution=32, y_scale=Y_SCALE,
                 cache_budget_bytes=512 * 1024 ** 2):
        super().__init__()
        if isinstance(query_chunk, bool) or not isinstance(query_chunk, int) or query_chunk < 1:
            raise ValueError('query_chunk must be a positive integer')
        if not isinstance(lift_resolution, int) or lift_resolution < 2:
            raise ValueError('lift_resolution must be an integer >= 2')
        if ray_chunk * samples > query_chunk:
            raise ValueError('ray_chunk*samples must not exceed query_chunk')
        self.query_chunk, self.checkpoint_query = query_chunk, bool(checkpoint_query)
        self.lift_resolution, self.y_scale = lift_resolution, float(y_scale)
        self.encoder = SharedEncoder2D()
        self.lifting = GeometryLifting(lift_resolution)
        self.refiner = nn.Sequential(*[ResidualBlock(32, 3) for _ in range(3)])
        self.decoder = nn.Sequential(nn.Linear(61, 64), nn.SiLU(),
            *[layer for _ in range(3) for layer in (nn.Linear(64, 64), nn.SiLU())],
            nn.Linear(64, 1), nn.Sigmoid())
        self.physics = ContinuousRayIntegrator(samples_per_ray=samples, ray_chunk=ray_chunk,
            projection_scale_ppm_m=y_scale, cache_budget_bytes=cache_budget_bytes)

    def encode(self, batch):
        batch = _clean_inputs(batch)
        geometry = self.physics.prepare(batch)
        y, valid = batch['input_projection'], batch['input_valid_ray']
        if y.dtype not in (torch.float32, torch.float64):
            raise ValueError('Reference B encoder requires FP32 or FP64 inputs')
        if not bool(torch.isfinite(y[valid]).all()):
            raise ValueError('Valid input measurements must be finite')
        view = batch['view_mask'].bool()
        inputs = torch.cat((torch.where(valid, y, 0), valid.to(y.dtype),
                            torch.where(valid, batch['input_path_length_m'], 0) / (math.sqrt(3) * .6)), 2)
        b, v = inputs.shape[:2]
        station_features = self.encoder(inputs.flatten(0, 1))
        station_features = station_features.reshape(b, v, *station_features.shape[1:])
        lifted, visible = self.lifting(station_features, batch)
        features = self.refiner(masked_scaled_sum(lifted, visible, view, 4))
        count = view.sum(1).clamp_min(1).to(features.dtype)
        visible_fraction = visible.sum(1).to(features.dtype) / count[:, None, None, None]
        transposed_one = self.physics.coverage(geometry, self.lift_resolution,
                                             device=features.device, dtype=features.dtype)
        log_coverage = torch.log1p(transposed_one)
        coverage_max = log_coverage.flatten(1).amax(1).reshape(b, 1, 1, 1, 1)
        normalized_coverage = log_coverage / coverage_max.clamp_min(torch.finfo(features.dtype).tiny)
        coverage = torch.cat((visible_fraction[:, None], normalized_coverage), 1)
        return FieldContext(features, coverage)

    def _query_direct(self, features, coverage, xyz):
        sampled = sample_trilinear(features, xyz)
        covered = sample_trilinear(coverage, xyz)
        normalized = xyz * (2. / .6) - 1.
        positional = []
        for band in range(4):
            phase = (2 ** band * math.pi) * normalized
            positional.extend((phase.sin(), phase.cos()))
        vector = torch.cat((sampled, normalized, *positional, covered), -1)
        return self.decoder(vector).squeeze(-1)

    def _query_block(self, features, coverage, xyz):
        if self.checkpoint_query and torch.is_grad_enabled():
            return checkpoint(self._query_direct, features, coverage, xyz, use_reentrant=False)
        return self._query_direct(features, coverage, xyz)

    def query(self, context, xyz, case_index=None):
        """Query physical XYZ; [P,3] broadcasts to all cases unless case_index.

        Batched coordinates [B,...,3] return [B,...]. Explicit case_index with
        [P,3] returns [P]; otherwise shared [P,3] returns [B,P]. Values outside the physical
        cube or nonfinite values are rejected, rather than silently extrapolated.
        """
        features, coverage = context.features, context.coverage
        xyz = torch.as_tensor(xyz, device=features.device, dtype=features.dtype)
        if xyz.shape[-1:] != (3,) or xyz.ndim < 2:
            raise ValueError('Query coordinates must end in physical XYZ')
        if not bool(torch.isfinite(xyz).all()) or bool(((xyz < -1e-6) | (xyz > .600001)).any()):
            raise ValueError('Query coordinates must be finite and within [0,.6]^3')
        shared = xyz.ndim == 2
        if case_index is not None:
            if not isinstance(case_index, int) or not 0 <= case_index < context.batch_size:
                raise ValueError('case_index outside context batch')
            features, coverage = features[case_index:case_index+1], coverage[case_index:case_index+1]
            if not shared:
                raise ValueError('case_index requires unbatched [P,3] coordinates')
        if shared:
            output_shape = (len(features), xyz.shape[0])
            xyz = xyz[None].expand(len(features), -1, -1)
        else:
            if xyz.shape[0] != len(features):
                raise ValueError('Batched query coordinates must match context batch size')
            output_shape = xyz.shape[:-1]
            xyz = xyz.reshape(len(features), -1, 3)
        if not xyz.shape[1]:
            result = features.new_empty(output_shape)
        else:
            result = torch.cat([self._query_block(features, coverage, xyz[:, start:start+self.query_chunk])
                for start in range(0, xyz.shape[1], self.query_chunk)], dim=1).reshape(output_shape)
        return result[0] if shared and case_index is not None else result

    def _integral_direct(self, features, coverage, xyz, length):
        values = self._query_direct(features, coverage, xyz.reshape(1, -1, 3))
        values = values.reshape(xyz.shape[:2])
        return mass_to_mole(values).sum(-1) * length * (1e6 / self.physics.samples_per_ray)

    def integrate(self, context, target_geometry, normalized=False):
        """Direct Q96 midpoint queries, never projecting a sampled voxel volume."""
        features, coverage = context.features, context.coverage
        geometry = self.physics.prepare(target_geometry)
        if geometry.projection_shape[0] != context.batch_size:
            raise ValueError('Target geometry and feature context batch sizes differ')
        if features.dtype not in (torch.float32, torch.float64):
            raise ValueError('Reference B integration requires FP32 or FP64 features')
        def block(b, xyz, length):
            args = (features[b:b+1], coverage[b:b+1], xyz, length)
            if self.checkpoint_query and torch.is_grad_enabled():
                return checkpoint(self._integral_direct, *args, use_reentrant=False)
            return self._integral_direct(*args)
        return self.physics.integrate(None, geometry, normalized=normalized, block_fn=block,
                                      device=features.device, dtype=features.dtype)

    def sample_grid(self, context, resolution=64):
        """Sample the same continuous function on an endpoint XYZ grid."""
        if not isinstance(resolution, int) or resolution < 2:
            raise ValueError('resolution must be an integer >= 2')
        axis = torch.linspace(0, .6, resolution, device=context.features.device, dtype=context.features.dtype)
        xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
        values = self.query(context, xyz)
        return values.reshape(context.batch_size, 1, resolution, resolution, resolution)

    def forward(self, batch, resolution=64):
        return self.sample_grid(self.encode(batch), resolution)
