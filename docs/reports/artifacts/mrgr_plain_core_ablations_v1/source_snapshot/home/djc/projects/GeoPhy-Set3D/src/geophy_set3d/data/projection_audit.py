"""Train/Pool-A-only resolution mismatch audit against frozen physical measurements."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from geophy_set3d.data.build_utils import digest, timestamp, write_json, write_jsonl, atomic_text
from geophy_set3d.data.projection_pool_dataset import load_source_index, artifact_path
from geophy_set3d.geometry.tdlas import subset_geometry
from geophy_set3d.physics.projectors import differentiable_project


class ErrorTotals:
    """Accumulate float64 sums, so global results are ray-weighted, not mean ratios."""
    def __init__(self):
        self.count = 0
        self.absolute = self.squared = self.signed = 0.0
        self.reference_absolute = self.reference_squared = self.maximum = 0.0

    def update(self, prediction, reference, valid):
        p = np.asarray(prediction, dtype=np.float64)[valid]
        r = np.asarray(reference, dtype=np.float64)[valid]
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            raise ValueError('Nonfinite measurement in audit')
        delta = p - r
        self.count += len(delta)
        self.absolute += float(np.abs(delta).sum())
        self.squared += float(np.square(delta).sum())
        self.signed += float(delta.sum())
        self.reference_absolute += float(np.abs(r).sum())
        self.reference_squared += float(np.square(r).sum())
        self.maximum = max(self.maximum, float(np.abs(delta).max(initial=0)))

    def merge(self, other):
        for key in ('count', 'absolute', 'squared', 'signed', 'reference_absolute', 'reference_squared'):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        self.maximum = max(self.maximum, other.maximum)

    def metrics(self):
        if not self.count:
            raise ValueError('No valid rays in audit')
        return dict(valid_ray_count=self.count,
                    relative_l1=self.absolute / max(self.reference_absolute, 1e-20),
                    relative_l2=float(np.sqrt(self.squared) / max(np.sqrt(self.reference_squared), 1e-20)),
                    mae_ppm_m=self.absolute / self.count,
                    rmse_ppm_m=float(np.sqrt(self.squared / self.count)),
                    maximum_absolute_error_ppm_m=self.maximum,
                    mean_signed_error_ppm_m=self.signed / self.count)


def extra_error(operator, total):
    return {key + '_increase': total[key] - operator[key] for key in ('relative_l1', 'relative_l2')}


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {**{f'p{p}': float(np.percentile(values, p)) for p in (50, 90, 95, 99)},
            'max': float(values.max()), 'mean': float(values.mean())}


def figures(output, station_rows, case_rows, example):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    for metric in ('relative_l1', 'relative_l2'):
        fig, ax = plt.subplots(layout='constrained')
        for comparison in ('E_operator', 'E_total64'):
            values = [100 * row[metric] for row in station_rows if row['comparison'] == comparison]
            ax.hist(values, bins=50, alpha=.55, label=comparison)
        ax.set(xlabel=f'Station {metric} (%) on valid rays', ylabel='Case-station count')
        ax.legend()
        fig.savefig(output / f'{metric}_histogram.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, metric in zip(axes, ('relative_l1', 'relative_l2')):
        a = [100 * row['E_operator'][metric] for row in case_rows]
        b = [100 * row['E_total64'][metric] for row in case_rows]
        ax.scatter(a, b, s=14)
        ax.set(xlabel=f'C128 {metric} (%)', ylabel=f'C64 {metric} (%)', title='Per-case valid rays')
    fig.savefig(output / 'c128_vs_c64_scatter.png', dpi=150)
    plt.close(fig)
    # First training case, station with largest reference signal: no val/test selection.
    ref, p128, p64, valid, case_id, station = example
    fig, axes = plt.subplots(1, 5, figsize=(19, 4), layout='constrained')
    arrays = [ref, p128, p64, p128 - ref, p64 - ref]
    titles = ['Saved P_ref(C128,256)', 'P_diff(C128,96)', 'P_diff(C64,96)', 'C128 residual', 'C64 residual']
    vmax = max(float(np.max(a[valid])) for a in arrays[:3])
    residual_max = max(float(np.abs(a[valid]).max()) for a in arrays[3:])
    for i, (ax, array, title) in enumerate(zip(axes, arrays, titles)):
        limit = vmax if i < 3 else residual_max
        image = ax.imshow(np.ma.array(array, mask=~valid), origin='lower',
                          cmap='viridis' if i < 3 else 'coolwarm',
                          vmin=0 if i < 3 else -limit, vmax=limit)
        ax.set(title=title, xlabel='Horizontal steering index', ylabel='Vertical steering index')
        fig.colorbar(image, ax=ax, label='ppm m', shrink=.65)
    fig.suptitle(f'{case_id}; Pool A station {station}; invalid rays masked')
    fig.savefig(output / 'representative_projection_difference.png', dpi=150)
    plt.close(fig)


def run_audit(dataset_root, output_root, *, samples=96, split='train', view_chunk=4,
              device='cpu', dtype='float64', threads=8, max_cases=None):
    if split != 'train':
        raise ValueError('This audit is strictly train split / Pool A only')
    if samples < 1 or view_chunk < 1 or threads < 1 or (max_cases is not None and max_cases < 1):
        raise ValueError('Sampling, chunk, threads and case limits must be positive')
    root, output = Path(dataset_root).resolve(), Path(output_root).resolve()
    build_path = root / 'dataset_build_manifest.json'
    build = json.loads(build_path.read_text())
    source = Path(build['complete_effective_config']['dataset']['source_root']).resolve()
    if output.is_relative_to(root) or output.is_relative_to(source) or root.is_relative_to(output):
        raise ValueError('Audit output must be outside frozen/source datasets and their ancestors')
    if build['status'] != 'frozen' or build['reference_samples'] != 256:
        raise ValueError('Expected frozen P_ref(C128,256) dataset')
    if dtype not in ('float32', 'float64'):
        raise ValueError('dtype must be float32 or float64')
    torch.set_num_threads(threads)
    rows = [row for row in load_source_index(root) if row['split'] == 'train']
    if max_cases is not None:
        rows = rows[:max_cases]
    geometry_path = root / 'geometry/train_geometry.npz'
    with np.load(geometry_path, allow_pickle=False) as arrays:
        geometry = dict(arrays)
    station_count = len(geometry['station_origin_m'])
    provenance = {'source_index.jsonl': digest(root / 'source_index.jsonl'),
                  'dataset_build_manifest.json': digest(build_path),
                  'geometry/train_geometry.npz': digest(geometry_path)}
    if (provenance['source_index.jsonl'] != build['source_index_sha256']
            or provenance['geometry/train_geometry.npz'] != build['geometry_hashes']['train_geometry.npz']):
        raise ValueError('Index or Pool A geometry differs from frozen manifest')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(started_at=timestamp(), dataset_root=str(root), output_root=str(output),
                  split='train', geometry_pool='A', samples=samples, reference_samples=256,
                  device=device, dtype=dtype, view_chunk=view_chunk, threads=threads,
                  case_limit=max_cases, case_ids=[r['case_id'] for r in rows],
                  metric_mask='valid_ray only', bias_convention='prediction minus saved reference',
                  global_aggregation='pooled ray sums; not the mean of case/station ratios',
                  denominator_floor=1e-20, input_hashes=provenance,
                  projector_sha256=digest(Path(__file__).parents[1] / 'physics/projectors.py'))
    atomic_text(output / 'run_config.yaml', yaml.safe_dump(config, sort_keys=False))
    totals = {name: ErrorTotals() for name in ('E_operator', 'E_total64', 'E_resolution96')}
    cases, stations, example = [], [], None
    start = time.monotonic()
    for index, row in enumerate(rows):
        reference_path = artifact_path(root, row['train_projection_path'])
        gt64_path = artifact_path(root, row['gt_volume_64_path'])
        meta = json.loads((reference_path.parent / 'sample_meta.json').read_text())
        if meta['split'] != 'train' or meta['reference_samples'] != 256:
            raise ValueError('Case metadata must confirm train / K_ref=256')
        for p in (reference_path, gt64_path):
            if digest(p) != meta['hashes'][p.name]:
                raise ValueError(f'Frozen array hash mismatch: {p}')
        if digest(row['source_target_path']) != row['source_gt_sha256']:
            raise ValueError('Source GT hash mismatch')
        reference = np.load(reference_path, mmap_mode='r', allow_pickle=False)
        c64 = np.load(gt64_path, mmap_mode='r', allow_pickle=False)
        c128 = np.load(row['source_target_path'], mmap_mode='r', allow_pickle=False)
        if c64.shape != (64,)*3 or c128.shape != (128,)*3 or reference.shape != geometry['valid_ray'].shape:
            raise ValueError('Incorrect frozen volume/projection shape')
        if not all(np.isfinite(v).all() for v in (c64, c128, reference)):
            raise ValueError('Nonfinite audit input')
        tensors = [torch.tensor(np.array(v, copy=True), dtype=getattr(torch, dtype), device=device)
                   for v in (c128, c64)]
        predictions = [np.empty(reference.shape, dtype=np.float64) for _ in range(2)]
        with torch.no_grad():
            for offset in range(0, station_count, view_chunk):
                sl = slice(offset, min(offset + view_chunk, station_count))
                chunk = subset_geometry(geometry, sl)
                for volume, prediction in zip(tensors, predictions):
                    prediction[sl] = differentiable_project(volume, chunk, samples).cpu().numpy()
        case_totals = {name: ErrorTotals() for name in totals}
        # E_resolution96 isolates the signed C64-C128 change at the SAME midpoint K.
        # Its denominator is the saved reference; subtracting norms alone cannot isolate it.
        comparisons = {'E_operator': (predictions[0], reference),
                       'E_total64': (predictions[1], reference),
                       'E_resolution96': (predictions[1] - predictions[0] + reference, reference)}
        for comparison, (pred, ref) in comparisons.items():
            for station in range(station_count):
                accumulator = ErrorTotals()
                accumulator.update(pred[station], ref[station], geometry['valid_ray'][station])
                case_totals[comparison].merge(accumulator)
                stations.append(dict(case_id=row['case_id'], station_index=station,
                                     station_azimuth_deg=float(geometry['station_azimuth_deg'][station]),
                                     comparison=comparison, **accumulator.metrics()))
            totals[comparison].merge(case_totals[comparison])
        metrics = {name: value.metrics() for name, value in case_totals.items()}
        cases.append(dict(case_id=row['case_id'], sample_id=row['sample_id'],
                          condition_meta=row['condition_meta'], **metrics,
                          extra_error_from_64=extra_error(metrics['E_operator'], metrics['E_total64'])))
        if example is None:
            station = int(np.asarray(reference).sum(axis=(1, 2)).argmax())
            example = (reference[station].copy(), predictions[0][station], predictions[1][station],
                       geometry['valid_ray'][station], row['case_id'], station)
        print(f'{index+1}/{len(rows)} {row["case_id"]} '
              f'C128 L1={metrics["E_operator"]["relative_l1"]:.6%} '
              f'C64 L1={metrics["E_total64"]["relative_l1"]:.6%} '
              f'elapsed={time.monotonic()-start:.1f}s', flush=True)
    global_metrics = {name: value.metrics() for name, value in totals.items()}
    needs_review = bool(global_metrics['E_total64']['relative_l1'] > .02
                        or global_metrics['E_total64']['relative_l2'] > .03)
    flagged = [row['case_id'] for row in cases
               if row['E_total64']['relative_l1'] > .02 or row['E_total64']['relative_l2'] > .03]
    summary = dict(completed_at=timestamp(), split='train', geometry_pool='A',
                   case_count=len(rows), station_count=station_count, samples=samples,
                   reference_samples=256, dtype=dtype, device=device,
                   metric_domain='valid rays only', global_metrics=global_metrics,
                   extra_error_from_64=extra_error(global_metrics['E_operator'], global_metrics['E_total64']),
                   E_resolution96_definition='P_diff(C64,K)-P_diff(C128,K), normalized by saved P_ref',
                   station_distribution={name: {metric: percentiles([s[metric] for s in stations if s['comparison'] == name])
                       for metric in ('relative_l1', 'relative_l2')} for name in totals},
                   case_distribution={name: {metric: percentiles([c[name][metric] for c in cases])
                       for metric in ('relative_l1', 'relative_l2')} for name in totals},
                   review=dict(status='needs review' if needs_review or flagged else 'within suggested review levels',
                       global_needs_review=needs_review, flagged_case_count=len(flagged), flagged_case_ids=flagged,
                       suggested_relative_l1_level=.02, suggested_relative_l2_level=.03,
                       policy='Warnings only; no dataset validity decision or automatic parameter changes'),
                   elapsed_seconds=time.monotonic()-start, input_hashes=provenance)
    write_jsonl(output / 'per_case_metrics.jsonl', cases)
    with (output / 'per_station_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stations[0]))
        writer.writeheader()
        writer.writerows(stations)
    write_json(output / 'audit_summary.json', summary)
    figures(output / 'figures', stations, cases, example)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
    parser.add_argument('--output-root', default='/data/djc/GeoPhy-Set3D/outputs/audits/c64_projection_mismatch_v1')
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--split', choices=['train'], default='train')
    parser.add_argument('--view-chunk', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-cases', type=int)
    args = parser.parse_args()
    result = run_audit(**vars(args))
    print(json.dumps(result['global_metrics'], indent=2))
