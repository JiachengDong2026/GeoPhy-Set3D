"""Detached ray geometry and direct continuous midpoint integration for plan B.

No concentration volume or voxel projector appears in the projection path.
The only voxel scatter is the measurement-independent T^T 1 coverage feature.
All geometry is fixed: derivatives are with respect to the queried field.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math

import torch


Y_SCALE = 32284.894580078173


def mass_to_mole(concentration):
    """Methane mass fraction -> mole fraction, without a gradient-changing clamp."""
    methane, air = 1. / 16.043, 1. / 28.965
    return methane * concentration / (air + (methane - air) * concentration)


@dataclass(frozen=True)
class ContinuousStation:
    positions: torch.Tensor
    origin: torch.Tensor
    directions: torch.Tensor
    near: torch.Tensor
    length: torch.Tensor

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in
                   (self.positions, self.origin, self.directions, self.near, self.length))


@dataclass(frozen=True)
class ContinuousGeometry:
    projection_shape: tuple
    stations: tuple
    valid_ray: torch.Tensor
    valid_counts: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    signature: tuple


class ContinuousRayIntegrator:
    """Q-midpoint ray integration with bounded, detached CPU geometry LRU.

    ``query_fn(case_index, xyz[P,3])`` returns C[P]. ``integrate`` returns raw
    ppm m by default, or measurement-scale-normalized values on request.
    ``block_fn``, when supplied, executes the *complete* query/chi/reduction
    block, so callers can checkpoint it without retaining query activations.
    """
    def __init__(self, samples_per_ray=96, ray_chunk=256,
                 projection_scale_ppm_m=Y_SCALE, extent_m=.6,
                 cache_budget_bytes=512 * 1024 ** 2):
        for name, value in (("samples_per_ray", samples_per_ray), ("ray_chunk", ray_chunk)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(extent_m) or extent_m <= 0:
            raise ValueError('extent_m must be finite and positive')
        if not math.isfinite(projection_scale_ppm_m) or projection_scale_ppm_m <= 0:
            raise ValueError('projection_scale_ppm_m must be finite and positive')
        if cache_budget_bytes < 0:
            raise ValueError('cache_budget_bytes must be nonnegative')
        self.samples_per_ray, self.ray_chunk = samples_per_ray, ray_chunk
        self.projection_scale_ppm_m = float(projection_scale_ppm_m)
        self.extent_m = float(extent_m)
        self.cache_budget_bytes = int(cache_budget_bytes)
        self._cache, self._cache_bytes = OrderedDict(), 0

    @property
    def signature(self):
        return (self.samples_per_ray, self.extent_m, self.projection_scale_ppm_m)

    @property
    def cache_bytes(self):
        return self._cache_bytes

    def clear_cache(self):
        self._cache.clear()
        self._cache_bytes = 0

    @torch.no_grad()
    def prepare(self, batch):
        if isinstance(batch, ContinuousGeometry):
            if batch.signature != self.signature:
                raise ValueError('Geometry belongs to a different integration protocol')
            return batch
        origin = torch.as_tensor(batch['input_station_origin_m']).detach()
        if origin.ndim != 3 or origin.shape[-1] != 3:
            raise ValueError('station origins must be [B,V,3]')
        dtype = torch.float64 if origin.dtype == torch.float64 else torch.float32
        device = origin.device
        cpu = lambda x: torch.as_tensor(x).detach().to(device='cpu')
        origins = cpu(origin).to(dtype)
        mask = cpu(batch['view_mask']).bool()
        valid = cpu(batch['input_valid_ray']).bool()
        if valid.ndim != 5 or valid.shape[:3] != (*origin.shape[:2], 1):
            raise ValueError('valid rays must be [B,V,1,H,W]')
        if mask.shape != origin.shape[:2]:
            raise ValueError('view_mask must match [B,V]')
        valid = valid & mask[:, :, None, None, None]
        direction = cpu(batch['input_ray_direction']).to(dtype)
        near = cpu(batch['input_t_near']).to(dtype)
        length = cpu(batch['input_path_length_m']).to(dtype)
        bsz, views, _, height, width = valid.shape
        if direction.shape != (bsz, views, height, width, 3):
            raise ValueError('ray directions must be [B,V,H,W,3]')
        if near.shape != valid.shape or length.shape != valid.shape:
            raise ValueError('near and path lengths must match valid-ray shape')
        stations = []
        for b in range(bsz):
            for v in range(views):
                positions = valid[b, v].reshape(-1).nonzero().flatten()
                if not positions.numel():
                    continue
                o = origins[b, v]
                d = direction[b, v].reshape(-1, 3)[positions]
                n = near[b, v].reshape(-1)[positions]
                ell = length[b, v].reshape(-1)[positions]
                if (not all(torch.isfinite(x).all() for x in (o, d, n, ell))
                        or torch.any(n < 0) or torch.any(ell <= 0)):
                    raise ValueError('Valid rays require finite geometry, near >= 0, length > 0')
                if not torch.allclose(d.norm(dim=-1), torch.ones(len(d), dtype=dtype),
                                      rtol=2e-5, atol=2e-6):
                    raise ValueError('Valid ray directions must be unit vectors')
                # Intersect the supplied forward segment with the physical cube.
                # Zero components must not produce 0/0 at an endpoint face.
                parallel = d.abs() < torch.finfo(dtype).eps
                safe_d = torch.where(parallel, torch.ones_like(d), d)
                t0, t1 = -o / safe_d, (self.extent_m - o) / safe_d
                lower = torch.where(parallel, -torch.inf, torch.minimum(t0, t1))
                upper = torch.where(parallel, torch.inf, torch.maximum(t0, t1))
                outside = (parallel & ((o < 0) | (o > self.extent_m))).any(-1)
                clipped_near = torch.maximum(n, lower.max(-1).values)
                clipped_far = torch.minimum(n + ell, upper.min(-1).values)
                keep = (~outside) & (clipped_far > clipped_near)
                valid[b, v].reshape(-1)[positions[~keep]] = False
                positions, d = positions[keep], d[keep]
                n, ell = clipped_near[keep], (clipped_far - clipped_near)[keep]
                if not positions.numel():
                    continue
                digest = hashlib.sha256(repr((self.signature, dtype)).encode())
                for x in (positions, o, d, n, ell):
                    digest.update(x.contiguous().numpy().tobytes())
                key = digest.digest()
                station = self._cache.get(key)
                if station is None:
                    station = ContinuousStation(positions, o, d, n, ell)
                    if station.nbytes <= self.cache_budget_bytes:
                        while self._cache and self._cache_bytes + station.nbytes > self.cache_budget_bytes:
                            _, evicted = self._cache.popitem(last=False)
                            self._cache_bytes -= evicted.nbytes
                        self._cache[key] = station
                        self._cache_bytes += station.nbytes
                else:
                    self._cache.move_to_end(key)
                stations.append((b, v, station))
        counts = valid.flatten(1).sum(1)
        if torch.any(counts == 0):
            raise ValueError('Every case needs at least one valid in-domain ray')
        return ContinuousGeometry(tuple(valid.shape), tuple(stations), valid.to(device),
                                  counts.to(device), device, dtype, self.signature)

    def chunks(self, geometry, *, device=None, dtype=None):
        """Yield (case, flattened projection positions, midpoint XYZ, lengths)."""
        geometry = self.prepare(geometry)
        device = geometry.device if device is None else device
        dtype = geometry.dtype if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('Continuous physics requires FP32 or FP64')
        fractions = (torch.arange(self.samples_per_ray, device=device, dtype=dtype) + .5) / self.samples_per_ray
        views, pixels = geometry.projection_shape[1], math.prod(geometry.projection_shape[-2:])
        for b, v, station in geometry.stations:
            origin = station.origin.to(device=device, dtype=dtype)
            for start in range(0, len(station.positions), self.ray_chunk):
                stop = start + self.ray_chunk
                direction = station.directions[start:stop].to(device=device, dtype=dtype)
                near = station.near[start:stop].to(device=device, dtype=dtype)
                length = station.length[start:stop].to(device=device, dtype=dtype)
                xyz = origin + direction[:, None] * (near[:, None] + length[:, None] * fractions)[..., None]
                # Only roundoff can leave the clipped physical segment.
                xyz = xyz.clamp(0., self.extent_m)
                positions = station.positions[start:stop].to(device) + (b * views + v) * pixels
                yield b, positions, xyz, length

    def integrate(self, query_fn, batch_or_geometry, normalized=False, *,
                  device=None, dtype=None, block_fn=None):
        geometry = self.prepare(batch_or_geometry)
        device = geometry.device if device is None else device
        dtype = geometry.dtype if dtype is None else dtype
        positions, values = [], []
        # Physics stays FP32/FP64 even if called under an autocast context.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            for b, pos, xyz, length in self.chunks(geometry, device=device, dtype=dtype):
                if block_fn is None:
                    concentration = query_fn(b, xyz.reshape(-1, 3)).reshape(xyz.shape[:2])
                    if concentration.dtype not in (torch.float32, torch.float64):
                        raise ValueError('Query must return FP32 or FP64 concentrations')
                    value = mass_to_mole(concentration).sum(-1) * length * (1e6 / self.samples_per_ray)
                else:
                    value = block_fn(b, xyz, length)
                positions.append(pos)
                values.append(value)
            result = torch.zeros(math.prod(geometry.projection_shape), device=device, dtype=dtype)
            result = result.index_copy(0, torch.cat(positions), torch.cat(values)).reshape(geometry.projection_shape)
            return result / self.projection_scale_ppm_m if normalized else result

    @torch.no_grad()
    def coverage(self, batch_or_geometry, resolution=32, *, device=None, dtype=None):
        """Normalized T^T1, where T uses Q midpoints and normalized ppm m units."""
        if not isinstance(resolution, int) or resolution < 2:
            raise ValueError('coverage resolution must be >= 2')
        geometry = self.prepare(batch_or_geometry)
        device = geometry.device if device is None else device
        dtype = geometry.dtype if dtype is None else dtype
        shape = (geometry.projection_shape[0], 1, resolution, resolution, resolution)
        result = torch.zeros(math.prod(shape), device=device, dtype=dtype)
        cells = resolution ** 3
        for b, _, xyz, length in self.chunks(geometry, device=device, dtype=dtype):
            coord = xyz / self.extent_m * (resolution - 1)
            base = coord.floor().long().clamp(0, resolution - 2)
            frac = coord - base
            factor = length[:, None] * (1e6 / self.projection_scale_ppm_m / self.samples_per_ray)
            for x in (0, 1):
                for y in (0, 1):
                    for z in (0, 1):
                        ids = ((base[..., 0] + x) * resolution + base[..., 1] + y) * resolution + base[..., 2] + z
                        weights = ((frac[..., 0] if x else 1-frac[..., 0]) *
                                   (frac[..., 1] if y else 1-frac[..., 1]) *
                                   (frac[..., 2] if z else 1-frac[..., 2]))
                        result.scatter_add_(0, (ids + b*cells).flatten(), (weights * factor).flatten())
        return result.reshape(shape)
