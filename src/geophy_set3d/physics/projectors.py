"""Separate CPU reference and autograd-capable PyTorch midpoint projectors."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import numpy as np
from geophy_set3d.geometry.tdlas import subset_geometry


def mass_to_mole(c, methane=16.043, air=28.965):
    return (c / methane) / (c / methane + (1 - c) / air)


class ReferenceProjector:
    def __init__(self, config, cache_dir):
        self.config = config
        source = Path(__file__).with_name('reference_projector.cpp')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.source_hash = digest
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        lib = cache / f'reference_{digest[:16]}.so'
        if not lib.exists():
            temporary = lib.with_suffix('.tmp.so')
            subprocess.run(['g++', '-O3', '-std=c++17', '-fopenmp', '-shared', '-fPIC',
                            str(source), '-o', str(temporary)], check=True)
            temporary.replace(lib)
        self.library = ctypes.CDLL(str(lib))
        self.function = self.library.project_rays
        ptr = ctypes.c_void_p
        self.function.argtypes = [ptr, ctypes.c_int, ptr, ptr, ptr, ptr, ptr,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_int, ptr]
        self.function.restype = None

    def __call__(self, volume, geometry, samples=None):
        volume = np.ascontiguousarray(volume, dtype=np.float32)
        if volume.shape != (128, 128, 128) or not np.isfinite(volume).all():
            raise ValueError('Reference requires finite 128³ volume')
        count = len(geometry['station_origin_m'])
        hw = geometry['valid_ray'].shape[1:]
        result = np.empty((count, *hw), dtype=np.float32)
        chunk = self.config['view_chunk']
        for start in range(0, count, chunk):
            g = subset_geometry(geometry, slice(start, min(start + chunk, count)))
            arrays = [np.ascontiguousarray(g[k], dtype=np.float64) for k in
                      ('station_origin_m', 'ray_direction', 't_near', 't_far')]
            valid = np.ascontiguousarray(g['valid_ray'], dtype=np.uint8)
            out = result[start:start + len(g['station_origin_m'])]
            self.function(volume.ctypes.data, 128, *(a.ctypes.data for a in arrays),
                          valid.ctypes.data, valid.size, int(np.prod(hw)),
                          samples or self.config['reference_samples'],
                          self.config['methane_molar_mass'], self.config['air_molar_mass'],
                          self.config['threads'], out.ctypes.data)
            if not np.isfinite(out).all() or np.any(out < 0) or np.count_nonzero(out[~g['valid_ray']]):
                raise ValueError(f'Projection QA failed in station chunk [{start}, {start + len(out)})')
        return result


def differentiable_project(volume_xyz, geometry, samples=96):
    """Midpoint integration with grid_sample; XYZ input, grid coordinates ZYX."""
    import torch
    import torch.nn.functional as F
    dtype, device = volume_xyz.dtype, volume_xyz.device
    def tensor(a):
        return torch.as_tensor(a, dtype=dtype, device=device)
    origin = tensor(geometry['station_origin_m'])[:, None, None, None, :]
    direction = tensor(geometry['ray_direction'])[:, :, :, None, :]
    near = tensor(geometry['t_near'])[:, :, :, None]
    length = tensor(geometry['path_length_m'])[:, :, :, None]
    fraction = (torch.arange(samples, dtype=dtype, device=device) + 0.5) / samples
    t = near + length * fraction
    points = origin + direction * t[..., None]
    # torch input [N,C,D=x,H=y,W=z], normalized sampler coordinates [z,y,x].
    grid = (points[..., [2, 1, 0]] / 0.6 * 2 - 1).reshape(1, -1, 1, 1, 3)
    c = F.grid_sample(volume_xyz[None, None], grid, mode='bilinear',
                      padding_mode='border', align_corners=True).reshape(t.shape)
    chi = mass_to_mole(c.clamp(0, 1))
    pic = chi.mean(-1) * length[..., 0] * 1e6
    return pic * tensor(geometry['valid_ray'])
