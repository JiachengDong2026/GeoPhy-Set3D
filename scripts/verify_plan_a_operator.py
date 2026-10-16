#!/usr/bin/env python3
"""Reproduce Plan A's synthetic, label-free operator numerical gates and timings."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from geophy_set3d.geometry.tdlas import make_geometry, ray_box_intersection
from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator
from geophy_set3d.physics.projectors import differentiable_project, mass_to_mole


def canonical(geometry, dtype, device):
    tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    return dict(input_station_origin_m=tensor(geometry['station_origin_m'])[None],
                input_ray_direction=tensor(geometry['ray_direction'])[None],
                input_t_near=tensor(geometry['t_near'])[None, :, None],
                input_path_length_m=tensor(geometry['path_length_m'])[None, :, None],
                input_valid_ray=torch.as_tensor(geometry['valid_ray'], device=device)[None, :, None],
                view_mask=torch.ones(1, len(geometry['station_origin_m']), device=device, dtype=torch.bool))


def small_geometry():
    origin = np.array([[-.2, .11, .07], [.12, -.2, .19]])
    direction = np.array([[[[1., .21, .39], [1., .38, .16], [1., -.1, .43]]],
                          [[[.18, 1., .31], [.41, 1., -.12], [.07, 1., .47]]]])
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    near, far, length, valid = ray_box_intersection(origin[:, None, None], direction)
    return dict(station_origin_m=origin, ray_direction=direction, t_near=near,
                t_far=far, path_length_m=length, valid_ray=valid)


def relative(a, b):
    """Relative L2 discrepancy; stable for scalar transpose comparisons too."""
    a, b = a.detach(), b.detach()
    denominator = max(float(a.norm()), float(b.norm()), 1e-15)
    return float((a-b).norm())/denominator


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def numerical_checks(device, dtype):
    torch.manual_seed(20261015)
    geometry = small_geometry()
    batch = canonical(geometry, dtype, device)
    op = ExplicitRayOperator(samples_per_ray=96, ray_chunk=2)
    n = 7
    stencil = op.prepare(batch, n)
    c = (.1 + .2*torch.rand(1, 1, n, n, n, dtype=dtype, device=device)).requires_grad_()
    v = torch.randn_like(c)
    # Pass exactly the batch's finite precision geometry to the old projector.
    old = dict(station_origin_m=batch['input_station_origin_m'][0],
               ray_direction=batch['input_ray_direction'][0],
               t_near=batch['input_t_near'][0, :, 0],
               path_length_m=batch['input_path_length_m'][0, :, 0],
               valid_ray=batch['input_valid_ray'][0, :, 0])
    actual = op(c, stencil)
    legacy = differentiable_project(c[0, 0], old, 96)[None, :, None]
    p = op.project_normalized(c, stencil)
    u = torch.randn_like(p)
    target = torch.randn_like(p)
    m = stencil.valid_counts[0]
    jv = op.jvp(c, v, stencil)
    jt = op.vjp(c, u, stencil)
    jt_auto, = torch.autograd.grad((p*u).sum(), c, create_graph=True)
    residual = p-target
    g = op.vjp(c, residual, stencil)/m
    g_auto, = torch.autograd.grad(.5*residual.square().sum()/m, c, create_graph=True)
    hv, = torch.autograd.grad((g*v).sum(), c, retain_graph=True)
    hv_auto, = torch.autograd.grad((g_auto*v).sum(), c)
    scan = {}
    with torch.no_grad():
        for h in (1e-4, 1e-5, 1e-6, 1e-7):
            fd = (op.project_normalized(c+h*v, stencil)-op.project_normalized(c-h*v, stencil))/(2*h)
            scan[str(h)] = relative(jv, fd)
        h = 1e-6 if dtype == torch.float64 else 1e-3
        fd = (op.project_normalized(c+h*v, stencil)-op.project_normalized(c-h*v, stencil))/(2*h)
        gradient = lambda x: op.vjp(x, op.project_normalized(x, stencil)-target, stencil)/m
        hessian_fd = (gradient(c+h*v)-gradient(c-h*v))/(2*h)
        no_grad_error = relative(g.detach(), gradient(c))
    constant_errors = []
    for value in (0., .001, .5, 1.):
        const = torch.full_like(c, value)
        expected = batch['input_path_length_m']*1e6*mass_to_mole(value)
        constant_errors.append(relative(op(const, stencil), expected))
    chi_input = torch.linspace(.001, .99, 7, dtype=dtype, device=device).requires_grad_()
    chi_first, = torch.autograd.grad(op._chi(chi_input).sum(), chi_input, create_graph=True)
    chi_second, = torch.autograd.grad(chi_first.sum(), chi_input)
    a, b = 1/16.043, 1/28.965
    second_formula = -2*a*b*(a-b)/(b+(a-b)*chi_input).pow(3)
    errors = dict(forward_vs_legacy_relative_l2=relative(actual.detach(), legacy.detach()),
                  constant_max_relative_l2=max(constant_errors),
                  chi_prime_relative_l2=relative(op._chi_prime(chi_input), chi_first),
                  chi_second_relative_l2=relative(second_formula, chi_second),
                  jvp_finite_difference_relative_l2=relative(jv.detach(), fd),
                  transpose_identity_relative=relative((jv*u).sum().detach(), (v*jt).sum().detach()),
                  vjp_vs_autograd_relative_l2=relative(jt.detach(), jt_auto.detach()),
                  residual_gradient_vs_autograd_relative_l2=relative(g.detach(), g_auto.detach()),
                  hessian_vs_autograd_relative_l2=relative(hv.detach(), hv_auto.detach()),
                  hessian_finite_difference_relative_l2=relative(hv.detach(), hessian_fd),
                  no_grad_residual_gradient_relative_l2=no_grad_error)
    limits = dict(forward_vs_legacy_relative_l2=1e-10, constant_max_relative_l2=1e-10,
                  chi_prime_relative_l2=1e-10, chi_second_relative_l2=1e-10,
                  jvp_finite_difference_relative_l2=1e-5, transpose_identity_relative=1e-10,
                  vjp_vs_autograd_relative_l2=1e-10, residual_gradient_vs_autograd_relative_l2=1e-10,
                  hessian_vs_autograd_relative_l2=1e-10, hessian_finite_difference_relative_l2=1e-5,
                  no_grad_residual_gradient_relative_l2=1e-10)
    if dtype == torch.float32:
        limits = {key: (2e-3 if 'finite_difference' in key else 1e-5) for key in errors}
    checks = {key: dict(error=val, tolerance=limits[key], passed=val <= limits[key]) for key, val in errors.items()}
    return dict(device=str(device), dtype=str(dtype), grid_size=n, views=2,
                samples_per_ray=96, valid_rays=int(m), finite_difference_h=h,
                jvp_h_scan=scan, checks=checks, passed=all(x['passed'] for x in checks.values()))


def profile(device):
    """One synthetic 64^3 case, four 64x64 stations; no optimization or GT."""
    config = dict(azimuth_deg=[0, 180], radius_m=[5.5, 6.5], height_m=[.25, .35],
                  target_m=[.3, .3, .3], projection_hw=[64, 64],
                  horizontal_fan_deg=10, vertical_scan_deg=8)
    geometry = make_geometry(config, 4, seed=20261015)
    batch = canonical(geometry, torch.float32, device)
    op = ExplicitRayOperator()
    torch.manual_seed(20261015)
    c = .04*torch.rand((1, 1, 64, 64, 64), dtype=torch.float32, device=device)
    synchronize(device)
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    durations = {}
    def timed(name, action):
        synchronize(device)
        start = time.perf_counter()
        result = action()
        synchronize(device)
        durations[name] = time.perf_counter()-start
        return result
    with torch.no_grad():
        stencil = timed('prepare_cold_seconds', lambda: op.prepare(batch, 64))
        timed('prepare_cached_seconds', lambda: op.prepare(batch, 64))
        prediction = timed('forward_seconds', lambda: op(c, stencil))
        timed('jvp_seconds', lambda: op.jvp(c, c, stencil))
        timed('vjp_seconds', lambda: op.vjp(c, torch.ones_like(prediction), stencil))
        timed('coverage_seconds', lambda: op.coverage(stencil))
        operator_peak = (torch.cuda.max_memory_allocated(device)
                         if torch.device(device).type == 'cuda' else None)
        old = dict(station_origin_m=batch['input_station_origin_m'][0],
                   ray_direction=batch['input_ray_direction'][0],
                   t_near=batch['input_t_near'][0, :, 0],
                   path_length_m=batch['input_path_length_m'][0, :, 0],
                   valid_ray=batch['input_valid_ray'][0, :, 0])
        reference = differentiable_project(c[0, 0], old, 96)[None, :, None]
        forward_error = relative(prediction, reference)
    return dict(device=str(device), dtype=str(c.dtype), grid_size=64, views=4,
                image_hw=[64, 64], samples_per_ray=96, ray_chunk=256,
                valid_rays=int(stencil.valid_counts[0]), cached_cpu_bytes=op.cache_bytes,
                cpu_cache_budget_bytes=op.cache_budget_bytes,
                peak_cuda_allocated_bytes=operator_peak,
                forward_vs_legacy_relative_l2=forward_error,
                forward_vs_legacy_max_abs_ppm_m=float((prediction-reference).abs().max()),
                forward_vs_legacy_tolerance=1e-5,
                passed=forward_error <= 1e-5,
                context='torch.no_grad; wall-clock single calls, includes chunk CPU/device transfers',
                timings=durations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/reports/artifacts/plan_a_v1/operator_numerics.json')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--skip-profile', action='store_true')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    torch.set_num_threads(2)
    devices = [('cpu', torch.float64)]
    if args.device != 'cpu' and torch.cuda.is_available():
        devices.append(('cuda', torch.float32))
    report = dict(protocol='plan_a_operator_v1', seed=20261015,
                  provenance='Synthetic analytic fields and geometry only; no dataset, CFD or GT files opened.',
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   cuda_runtime=torch.version.cuda, gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)),
                  numerical=[numerical_checks(device, dtype) for device, dtype in devices])
    if not args.skip_profile:
        report['performance'] = profile(devices[-1][0])
    report['passed'] = all(row['passed'] for row in report['numerical'])
    if 'performance' in report:
        report['passed'] = report['passed'] and report['performance']['passed']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
