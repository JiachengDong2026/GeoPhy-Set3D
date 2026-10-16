"""Describe late-window concentration changes from an explicit external manifest.

No universal steady-state pass threshold is imposed. Example JSON schema:
{"axes":["x","y","z"], "bounds_m":[[0,.6],[0,.6],[0,.6]],
 "frames":[{"case_id":"case1", "time_s":1.5,
            "concentration_path":"C_1p5.npy", "density_path":"rho_1p5.npy"}, ...]}
Density is optional, but must be supplied at all selected times or none per case.
Paths are resolved relative to the manifest. Grids must be aligned endpoint nodes;
the caller must verify their physical provenance, not just their array shapes.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(times, concentrations, bounds, densities=None):
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=np.float64)
    assert len(t) >= 3 and np.isfinite(t).all() and np.all(np.diff(t) > 0)
    assert c.ndim == 4 and c.shape[0] == len(t) and min(c.shape[1:]) >= 2
    assert np.isfinite(c).all() and c.min() >= 0 and c.max() <= 1
    bounds = np.asarray(bounds, dtype=float)
    assert bounds.shape == (3, 2) and np.isfinite(bounds).all() and np.all(bounds[:, 1] > bounds[:, 0])
    weights, coords = [], []
    for n, (lo, hi) in zip(c.shape[1:], bounds):
        w = np.full(n, (hi - lo) / (n - 1))
        w[[0, -1]] *= .5
        weights.append(w)
        coords.append(np.linspace(lo, hi, n))
    volume_w = np.einsum('i,j,k->ijk', *weights)
    def integral(a): return float(np.sum(a * volume_w))
    q = np.array([integral(a) for a in c])
    assert q.min() > 0
    dt = np.diff(t)
    tw = np.zeros(len(t))
    tw[:-1] += dt / 2
    tw[1:] += dt / 2
    tw /= t[-1] - t[0]
    mean = np.einsum('t,tijk->ijk', tw, c)
    qref = q[-1]
    centers = np.array([[integral(a * coord.reshape(tuple(len(coord) if i == axis else 1 for i in range(3)))) / total
                         for axis, coord in enumerate(coords)] for a, total in zip(c, q)])
    changes = [integral(abs(c[i+1] - c[i])) / qref for i in range(len(t)-1)]
    pairs = [(integral(abs(c[j]-c[i])) / qref, i, j) for i in range(len(t)) for j in range(i+1, len(t))]
    worst, i, j = max(pairs)
    result = dict(
        frame_count=len(t), time_window_s=[float(t[0]), float(t[-1])], times_s=t.tolist(),
        first_last_field_relative_l1=integral(abs(c[-1] - c[0])) / qref,
        terminal_vs_time_mean_relative_l1=integral(abs(c[-1] - mean)) / integral(mean),
        worst_pair_field_relative_l1=worst, worst_pair_times_s=[float(t[i]), float(t[j])],
        adjacent_field_relative_l1=changes,
        adjacent_rate_over_terminal_C_integral_per_s=(np.array(changes)/dt).tolist(),
        integral_C_m3=q.tolist(), integral_C_relative_range=float(np.ptp(q)/qref),
        centroid_xyz_m=centers.tolist(),
        max_centroid_distance_from_terminal_mm=float(np.linalg.norm(centers-centers[-1], axis=1).max()*1000),
        peak_C=c.max(axis=(1,2,3)).tolist(),
        time_weighting='trapezoidal over actual timestamps',
        conclusion='Continuous temporal diagnostics only; tolerances and physical time scales must be assessed',
    )
    if densities is not None:
        rho = np.asarray(densities, dtype=np.float64)
        assert rho.shape == c.shape and np.isfinite(rho).all() and rho.min() > 0
        mass = np.array([integral(a*b) for a, b in zip(c, rho)])
        result['methane_mass_kg'] = mass.tolist()
        result['methane_mass_relative_range'] = float(np.ptp(mass)/mass[-1])
    else:
        result['methane_mass_kg'] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, default=1.5)
    parser.add_argument('--end', type=float, default=2.0)
    args = parser.parse_args()
    assert args.start < args.end and not args.output.exists()
    manifest = args.manifest.resolve()
    spec = json.loads(manifest.read_text())
    assert spec['axes'] == ['x', 'y', 'z']
    grouped, reads = defaultdict(list), []
    for r in spec['frames']:
        assert np.isfinite(float(r['time_s']))
        if args.start <= float(r['time_s']) <= args.end: grouped[r['case_id']].append(r)
    assert grouped, 'No frames in requested time window'
    results = {}
    for case, records in sorted(grouped.items()):
        records.sort(key=lambda r: float(r['time_s']))
        assert abs(float(records[-1]['time_s']) - args.end) < 1e-8, (case, 'Missing requested terminal time')
        assert abs(float(records[0]['time_s']) - args.start) < 1e-8, (case, 'Missing requested window start')
        cs, densities = [], []
        density_supplied = ['density_path' in r for r in records]
        assert all(density_supplied) or not any(density_supplied)
        for r in records:
            for key, values in [('concentration_path', cs), ('density_path', densities)]:
                if key not in r: continue
                path = (manifest.parent / r[key]).resolve()
                values.append(np.load(path, allow_pickle=False))
                reads.append(dict(case_id=case, time_s=r['time_s'], quantity=key, path=str(path), sha256=digest(path)))
        results[case] = analyze([r['time_s'] for r in records], cs, spec['bounds_m'], densities or None)
    output = dict(checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  manifest=str(manifest), manifest_sha256=digest(manifest), array_reads=reads,
                  scope='Only explicitly supplied histories; coordinate alignment and CFD provenance require independent verification',
                  cases=results, dataset_wide_steadiness_certified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f'Analyzed {len(results)} time histories: {args.output}')


if __name__ == '__main__': main()
