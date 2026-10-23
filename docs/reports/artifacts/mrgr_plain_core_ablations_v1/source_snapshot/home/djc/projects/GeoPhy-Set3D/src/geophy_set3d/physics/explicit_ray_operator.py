"""Fixed-geometry midpoint rays with differentiable analytic Jacobian actions.

Volumes are endpoint samples in XYZ order. Interpolation precedes mass-to-mole
conversion. The linear coverage operator T includes the normalized ppm m scale;
its transpose is the Euclidean discrete transpose (no voxel-volume factor).
Geometry is detached and cached with bounded CPU and CUDA LRUs. This operator
differentiates concentration, never station coordinates.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn


@dataclass(frozen=True)
class _StationStencil:
    positions: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    factors: torch.Tensor
    cache_key: bytes = b''

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.positions, self.indices, self.weights, self.factors))


@dataclass(frozen=True)
class RayStencil:
    """Detached CPU interpolation data plus small original-device metadata."""
    grid_size: int
    projection_shape: tuple
    stations: tuple
    valid_ray: torch.Tensor
    valid_counts: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    operator_signature: tuple


class ExplicitRayOperator(nn.Module):
    """Nonlinear raw forward and normalized JVP/VJP, supporting higher gradients.

    Concentration must be FP32 or FP64 and lie in the model's physical [0,1]
    range. No extra clamp is inserted: it would change derivatives at bounds.
    CPU and CUDA LRUs are per operator, each default to 512 MiB, and are not
    checkpointed. Set device_cache_budget_bytes=0 for the streaming fallback.
    A prepared batch owns references to its CPU stencil until released.
    """
    def __init__(self, samples_per_ray=96, ray_chunk=256,
                 projection_scale_ppm_m=32284.894580078173,
                 domain_min_m=(0., 0., 0.), domain_max_m=(.6, .6, .6),
                 cache_budget_bytes=512 * 1024 ** 2,
                 device_cache_budget_bytes=512 * 1024 ** 2):
        super().__init__()
        if not isinstance(samples_per_ray, int) or samples_per_ray < 1:
            raise ValueError('samples_per_ray must be a positive integer')
        if not isinstance(ray_chunk, int) or ray_chunk < 1:
            raise ValueError('ray_chunk must be a positive integer')
        if not math.isfinite(projection_scale_ppm_m) or projection_scale_ppm_m <= 0:
            raise ValueError('projection_scale_ppm_m must be finite and positive')
        self.samples_per_ray = samples_per_ray
        self.ray_chunk = ray_chunk
        self.projection_scale_ppm_m = float(projection_scale_ppm_m)
        self.domain_min_m = tuple(map(float, domain_min_m))
        self.domain_max_m = tuple(map(float, domain_max_m))
        if len(self.domain_min_m) != 3 or len(self.domain_max_m) != 3 or any(
                not math.isfinite(a) or not math.isfinite(b) or b <= a
                for a, b in zip(self.domain_min_m, self.domain_max_m)):
            raise ValueError('domain bounds must be finite increasing XYZ triples')
        self.cache_budget_bytes = int(cache_budget_bytes)
        if self.cache_budget_bytes < 0:
            raise ValueError('cache_budget_bytes must be nonnegative')
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self.device_cache_budget_bytes = int(device_cache_budget_bytes)
        if self.device_cache_budget_bytes < 0:
            raise ValueError('device_cache_budget_bytes must be nonnegative')
        self._device_cache = OrderedDict()
        self._device_cache_bytes = 0

    @property
    def _signature(self):
        return (self.samples_per_ray, self.projection_scale_ppm_m,
                self.domain_min_m, self.domain_max_m)

    @property
    def cache_bytes(self):
        return self._cache_bytes

    def clear_cache(self):
        self._cache.clear()
        self._cache_bytes = 0
        self._device_cache.clear()
        self._device_cache_bytes = 0

    @property
    def device_cache_bytes(self):
        return self._device_cache_bytes

    @torch.no_grad()
    def prepare(self, batch, grid_size):
        """Prepare only valid rays, ignoring NaNs in invalid/padded geometry.

        Canonical measurement-only batches use input_* fields and view_mask.
        All cases must contain at least one valid forward ray. Geometries are
        hashed by values, not station IDs, so changed geometry cannot hit stale
        entries even if IDs are reused across pools.
        """
        if not isinstance(grid_size, int) or grid_size < 2:
            raise ValueError('grid_size must be an integer >= 2')
        origin = torch.as_tensor(batch['input_station_origin_m']).detach()
        if origin.ndim != 3 or origin.shape[-1] != 3:
            raise ValueError('station origins must be [B,V,3]')
        device = origin.device
        dtype = torch.float64 if origin.dtype == torch.float64 else torch.float32
        cpu = lambda x: torch.as_tensor(x).detach().to(device='cpu')
        origins = cpu(origin).to(dtype)
        valid = cpu(batch['input_valid_ray']).bool()
        if valid.ndim != 5 or valid.shape[:3] != (*origin.shape[:2], 1):
            raise ValueError('valid rays must be [B,V,1,H,W]')
        view_mask = cpu(batch['view_mask']).bool()
        if view_mask.shape != origin.shape[:2]:
            raise ValueError('view_mask must match [B,V]')
        valid = valid & view_mask[:, :, None, None, None]
        counts = valid.flatten(1).sum(1)
        if torch.any(counts == 0):
            raise ValueError('Every case needs at least one valid ray')
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
                if not len(positions):
                    continue
                o = origins[b, v]
                d = direction[b, v].reshape(-1, 3)[positions]
                n = near[b, v].reshape(-1)[positions]
                ell = length[b, v].reshape(-1)[positions]
                if (not all(torch.isfinite(t).all() for t in (o, d, n, ell))
                        or torch.any(n < 0) or torch.any(ell <= 0)):
                    raise ValueError('Valid rays need finite origins/directions, near >= 0 and length > 0')
                if not torch.allclose(d.norm(dim=-1), torch.ones(len(d), dtype=dtype),
                                      rtol=2e-5, atol=2e-6):
                    raise ValueError('Valid ray directions must be unit vectors')
                digest = hashlib.sha256(repr((grid_size, self._signature, dtype)).encode())
                for t in (positions, o, d, n, ell):
                    digest.update(t.contiguous().numpy().tobytes())
                key = digest.digest()
                station = self._cache.get(key)
                if station is None:
                    station = self._make_station(positions, o, d, n, ell, grid_size, key)
                    if station.nbytes <= self.cache_budget_bytes:
                        while self._cache and self._cache_bytes + station.nbytes > self.cache_budget_bytes:
                            _, evicted = self._cache.popitem(last=False)
                            self._cache_bytes -= evicted.nbytes
                        self._cache[key] = station
                        self._cache_bytes += station.nbytes
                else:
                    self._cache.move_to_end(key)
                stations.append((b, v, station))
        return RayStencil(grid_size, tuple(valid.shape), tuple(stations), valid.to(device),
                          counts.to(device), device, dtype, self._signature)

    def _make_station(self, positions, origin, directions, near, length, n, key=b''):
        dtype = origin.dtype
        lo = torch.tensor(self.domain_min_m, dtype=dtype)
        extent = torch.tensor(self.domain_max_m, dtype=dtype) - lo
        fractions = (torch.arange(self.samples_per_ray, dtype=dtype) + .5) / self.samples_per_ray
        indices, weights = [], []
        # Chunk construction as well as execution to bound CPU temporaries.
        for start in range(0, len(positions), self.ray_chunk):
            stop = start + self.ray_chunk
            t = near[start:stop, None] + length[start:stop, None] * fractions
            xyz = origin + directions[start:stop, None] * t[..., None]
            coordinate = ((xyz - lo) / extent * (n - 1)).clamp(0, n - 1)
            base = coordinate.floor().long().clamp(0, n - 2)
            frac = coordinate - base
            ix, wx = [], []
            for x in (0, 1):
                for y in (0, 1):
                    for z in (0, 1):
                        ix.append(((base[..., 0] + x) * n + base[..., 1] + y) * n + base[..., 2] + z)
                        wx.append((frac[..., 0] if x else 1-frac[..., 0]) *
                                  (frac[..., 1] if y else 1-frac[..., 1]) *
                                  (frac[..., 2] if z else 1-frac[..., 2]))
            indices.append(torch.stack(ix, -1))
            weights.append(torch.stack(wx, -1))
        factors = (1e6 / self.projection_scale_ppm_m) * length / self.samples_per_ray
        return _StationStencil(positions, torch.cat(indices), torch.cat(weights), factors, key)

    def _resolve(self, c, batch_or_stencil):
        if c.ndim != 5 or c.shape[1] != 1 or len(set(c.shape[-3:])) != 1:
            raise ValueError('Concentration must be [B,1,N,N,N] in XYZ order')
        if c.dtype not in (torch.float32, torch.float64):
            raise ValueError('Physics integration requires FP32 or FP64 concentration')
        stencil = batch_or_stencil if isinstance(batch_or_stencil, RayStencil) else self.prepare(batch_or_stencil, c.shape[-1])
        if stencil.grid_size != c.shape[-1] or stencil.projection_shape[0] != len(c):
            raise ValueError('Stencil and concentration shapes do not match')
        if stencil.operator_signature != self._signature:
            raise ValueError('Stencil belongs to a different operator configuration')
        return stencil

    def _resident(self, station, device, dtype):
        """Transfer each cached station once, rather than every physical step.

        A zero budget retains the original streaming path. Keys include actual
        geometry, grid, integration rule, device and dtype; station IDs alone
        never identify cached geometry. Autograd may retain evicted tensors for
        its active graph, so this bounds cache ownership, not total GPU memory.
        """
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if device.type != 'cuda' or not station.cache_key:
            return station
        nbytes = (station.positions.numel() * station.positions.element_size()
                  + station.indices.numel() * station.indices.element_size()
                  + (station.weights.numel() + station.factors.numel())
                  * torch.empty((), dtype=dtype).element_size())
        if nbytes > self.device_cache_budget_bytes:
            return station
        key = (station.cache_key, device, dtype)
        resident = self._device_cache.get(key)
        if resident is not None:
            self._device_cache.move_to_end(key)
            return resident
        while self._device_cache and self._device_cache_bytes + nbytes > self.device_cache_budget_bytes:
            _, evicted = self._device_cache.popitem(last=False)
            self._device_cache_bytes -= evicted.nbytes
        resident = _StationStencil(station.positions.to(device, non_blocking=True),
            station.indices.to(device, non_blocking=True),
            station.weights.to(device=device, dtype=dtype, non_blocking=True),
            station.factors.to(device=device, dtype=dtype, non_blocking=True), station.cache_key)
        self._device_cache[key] = resident
        self._device_cache_bytes += resident.nbytes
        return resident

    def _chunks(self, stencil, device, dtype):
        pixels = math.prod(stencil.projection_shape[-2:])
        views = stencil.projection_shape[1]
        for b, v, station in stencil.stations:
            station = self._resident(station, device, dtype)
            for start in range(0, len(station.positions), self.ray_chunk):
                sl = slice(start, start + self.ray_chunk)
                yield (b, station.indices[sl].to(device), station.weights[sl].to(device=device, dtype=dtype),
                       station.factors[sl].to(device=device, dtype=dtype),
                       (station.positions[sl] + (b * views + v) * pixels).to(device))

    @staticmethod
    def _sample(c, indices, weights):
        return (c[indices] * weights).sum(-1)

    @staticmethod
    def _chi(c):
        a, b = 1/16.043, 1/28.965
        return a * c / (b + (a-b) * c)

    @staticmethod
    def _chi_prime(c):
        a, b = 1/16.043, 1/28.965
        return a*b / (b + (a-b) * c).square()

    def forward(self, c, batch_or_stencil):
        """Return raw ppm m, with exact zeros for invalid/padded pixels."""
        return self.project_normalized(c, batch_or_stencil) * self.projection_scale_ppm_m

    def project_normalized(self, c, batch_or_stencil):
        stencil = self._resolve(c, batch_or_stencil)
        flat = c.reshape(len(c), -1)
        positions, values = [], []
        for b, ix, w, factor, pos in self._chunks(stencil, c.device, c.dtype):
            positions.append(pos)
            values.append(self._chi(self._sample(flat[b], ix, w)).sum(-1) * factor)
        result = c.new_zeros(math.prod(stencil.projection_shape))
        return result.index_copy(0, torch.cat(positions), torch.cat(values)).reshape(stencil.projection_shape)

    def jvp(self, c, v, batch_or_stencil):
        """Apply the Jacobian of normalized projection to volume direction v."""
        stencil = self._resolve(c, batch_or_stencil)
        if v.shape != c.shape or v.device != c.device or v.dtype != c.dtype:
            raise ValueError('JVP direction must match concentration')
        flat, vector = c.reshape(len(c), -1), v.reshape(len(c), -1)
        positions, values = [], []
        for b, ix, w, factor, pos in self._chunks(stencil, c.device, c.dtype):
            derivative = self._chi_prime(self._sample(flat[b], ix, w))
            values.append((derivative * self._sample(vector[b], ix, w)).sum(-1) * factor)
            positions.append(pos)
        return c.new_zeros(math.prod(stencil.projection_shape)).index_copy(
            0, torch.cat(positions), torch.cat(values)).reshape(stencil.projection_shape)

    def vjp(self, c, u, batch_or_stencil):
        """Exact normalized Euclidean transpose, also usable under no_grad."""
        stencil = self._resolve(c, batch_or_stencil)
        if tuple(u.shape) != stencil.projection_shape or u.device != c.device or u.dtype != c.dtype:
            raise ValueError('VJP ray vector must match projection shape, device and dtype')
        flat, rays = c.reshape(len(c), -1), u.reshape(-1)
        result = c.new_zeros(c.numel())
        cells = stencil.grid_size ** 3
        for b, ix, w, factor, pos in self._chunks(stencil, c.device, c.dtype):
            derivative = self._chi_prime(self._sample(flat[b], ix, w))
            contribution = derivative * (factor * rays[pos])[:, None]
            result = result.scatter_add(0, (ix + b*cells).reshape(-1),
                                        (contribution[..., None] * w).reshape(-1))
        return result.reshape_as(c)

    def residual_gradient(self, c, target, batch_or_stencil):
        """Normalized prediction and exact J(C)^T(p(C)-target), in one pass.

        All samples of a ray belong to one chunk. Sharing their interpolation
        between p and chi' avoids a second gather/transfer without detaching
        either residual or concentration. The caller applies per-case 1/m.
        """
        stencil = self._resolve(c, batch_or_stencil)
        if tuple(target.shape) != stencil.projection_shape or target.device != c.device or target.dtype != c.dtype:
            raise ValueError('Target must match normalized projection shape, device and dtype')
        flat, rays = c.reshape(len(c), -1), target.reshape(-1)
        result = c.new_zeros(c.numel())
        positions, values = [], []
        cells = stencil.grid_size ** 3
        for b, ix, w, factor, pos in self._chunks(stencil, c.device, c.dtype):
            sampled = self._sample(flat[b], ix, w)
            prediction = self._chi(sampled).sum(-1) * factor
            contribution = self._chi_prime(sampled) * (factor * (prediction - rays[pos]))[:, None]
            result = result.scatter_add(0, (ix + b*cells).reshape(-1),
                                        (contribution[..., None] * w).reshape(-1))
            positions.append(pos)
            values.append(prediction)
        projection = c.new_zeros(math.prod(stencil.projection_shape)).index_copy(
            0, torch.cat(positions), torch.cat(values)).reshape(stencil.projection_shape)
        return projection, result.reshape_as(c)

    @torch.no_grad()
    def coverage(self, stencil, *, device=None, dtype=None):
        """Return T^T 1 and T^T(T 1) for normalized linear ray integration."""
        if stencil.operator_signature != self._signature:
            raise ValueError('Stencil belongs to a different operator configuration')
        device = stencil.device if device is None else device
        dtype = stencil.dtype if dtype is None else dtype
        shape = (stencil.projection_shape[0], 1, *(stencil.grid_size,)*3)
        cells = stencil.grid_size ** 3
        first = torch.zeros(math.prod(shape), device=device, dtype=dtype)
        second = torch.zeros_like(first)
        for b, ix, w, factor, _ in self._chunks(stencil, device, dtype):
            transposed_one = factor[:, None, None] * w
            ids = (ix + b*cells).reshape(-1)
            first.scatter_add_(0, ids, transposed_one.expand_as(w).reshape(-1))
            # Each interpolation row sums to one, so T1 = factor * Q.
            second.scatter_add_(0, ids, (transposed_one * (factor*self.samples_per_ray)[:, None, None]).reshape(-1))
        return first.reshape(shape), second.reshape(shape)

    @torch.no_grad()
    def geometry_preconditioner(self, stencil, *, device=None, dtype=None):
        """Unfloored D0=(M_air/M_CH4)^2/m T^T(T1), separately per case."""
        _, gram_row_sum = self.coverage(stencil, device=device, dtype=dtype)
        counts = stencil.valid_counts.to(device=gram_row_sum.device, dtype=gram_row_sum.dtype)
        return gram_row_sum * (28.965/16.043)**2 / counts[:, None, None, None, None]
