#!/usr/bin/env python3
"""Plot frozen SelfSup v1 validation reconstructions; no training or selection."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import binary_erosion

from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.evaluation.protocols import evaluation_dataset, load_model
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.utils import load_config, move_batch

REFERENCE = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v1')
OUTPUT = Path('/data/djc/GeoPhy-Set3D/outputs/visualizations/selfsup_mrgr_v1')


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10, 'figure.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False})


def draw_overview(cases, output):
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12), layout='constrained')
    for row, case in enumerate(cases):
        gt, pred = case['gt'], case['prediction']
        maximum = max(float(gt.max()), float(pred.max()))
        error = np.abs(pred - gt)
        images = (gt.max(axis=2).T, pred.max(axis=2).T, error.max(axis=2).T)
        row_images = []
        for col, data in enumerate(images):
            ax = axes[row, col]
            im = ax.imshow(data, origin='lower', extent=(0, .6, 0, .6), vmin=0,
                           vmax=float(error.max()) if col == 2 else maximum,
                           cmap='magma' if col == 2 else 'viridis', interpolation='nearest')
            row_images.append(im)
            ax.set(xlabel='x (m)', ylabel='y (m)', aspect='equal')
            if row == 0:
                ax.set_title(('GT64 (evaluation only)', 'SelfSup v1 reconstruction', 'Max absolute voxel error')[col], pad=14)
            if col == 0:
                ax.text(.02, .98, f"{case['label']}\nvoxel L1 = {case['metrics']['relative_l1']:.4f}",
                        transform=ax.transAxes, fontsize=9, fontweight='bold', color='white', va='top')
            if col == 1:
                ax.text(.02, .98, f"4 views: {case['station_ids']}", transform=ax.transAxes,
                        va='top', fontsize=8, color='white', bbox=dict(facecolor='black', alpha=.45, edgecolor='none'))
        fig.colorbar(row_images[0], ax=axes[row, :2], shrink=.78, pad=.015, label='CH4 mass fraction')
        fig.colorbar(row_images[2], ax=axes[row, 2], shrink=.78, pad=.015, label='Absolute error')
    fig.suptitle('Successful SelfSup-MRGR v1: frozen Val-A / 4-view reconstructions\n'
                 'XY maximum-intensity projections; GT / prediction share a scale; errors use separate colorbars', fontsize=15)
    fig.savefig(output / 'selfsup_v1_reconstruction_overview.png', dpi=180)
    plt.close(fig)


def draw_slices(case, output):
    gt, pred = case['gt'], case['prediction']
    weight = gt.astype(np.float64)
    centroid = [int(np.clip(round((weight.sum(axis=tuple(i for i in range(3) if i != a)) * np.arange(64)).sum()
                      / max(weight.sum(), 1e-12)), 0, 63)) for a in range(3)]
    maximum = max(float(gt.max()), float(pred.max()))
    fig, axes = plt.subplots(3, 3, figsize=(12, 11.5), layout='constrained')
    concentration_image = error_image = None
    for row, (plane, axis, labels) in enumerate((('XY', 2, ('x', 'y')), ('XZ', 1, ('x', 'z')), ('YZ', 0, ('y', 'z')))):
        for col, volume in enumerate((gt, pred, np.abs(pred - gt))):
            ax = axes[row, col]
            im = ax.imshow(np.take(volume, centroid[axis], axis=axis).T, origin='lower', extent=(0, .6, 0, .6),
                           vmin=0, vmax=float(np.abs(pred-gt).max()) if col == 2 else maximum,
                           cmap='magma' if col == 2 else 'viridis', interpolation='nearest')
            if col == 2: error_image = im
            else: concentration_image = im
            ax.set(xlabel=labels[0]+' (m)', ylabel=labels[1]+' (m)', aspect='equal')
            if row == 0:
                ax.set_title(('GT64', 'SelfSup v1', 'Absolute voxel error')[col], pad=14)
            if col == 0:
                position = (centroid[axis]+.5)*.6/64
                ax.text(0, 1.04, f'{plane}: {"xyz"[axis]} = {position:.3f} m', transform=ax.transAxes, fontweight='bold')
    fig.colorbar(concentration_image, ax=axes[:, :2], shrink=.65, pad=.015, label='CH4 mass fraction (shared)')
    fig.colorbar(error_image, ax=axes[:, 2], shrink=.65, pad=.015, label='Absolute error')
    fig.suptitle(f"Typical Val-A case: SelfSup v1 from 4 measurements\nVoxel relative L1 = {case['metrics']['relative_l1']:.4f}, "
                 f"L2 = {case['metrics']['relative_l2']:.4f}; slices through GT mass centroid", fontsize=14)
    fig.savefig(output / 'selfsup_v1_typical_orthogonal_slices.png', dpi=180)
    plt.close(fig)
    return centroid


def draw_3d(cases, output):
    fig = plt.figure(figsize=(12, 13), layout='constrained')
    for row, case in enumerate(cases):
        threshold = .05 * float(case['gt'].max())
        maximum = max(float(case['gt'].max()), float(case['prediction'].max()))
        norm = Normalize(0, maximum)
        for col, key in enumerate(('gt', 'prediction')):
            ax = fig.add_subplot(3, 2, row*2+col+1, projection='3d')
            volume = case[key]
            mask = volume >= threshold
            # Plot the boundary of the thresholded *full 64^3* voxel mask.
            # This changes visualization only, without resampling the prediction.
            surface = mask & ~binary_erosion(mask)
            indices = np.argwhere(surface)
            points = (indices+.5)*.6/64
            values = volume[surface]
            im = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values,
                            norm=norm, cmap='viridis', s=5, alpha=.75, linewidths=0, depthshade=False)
            ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)',
                   xlim=(0,.6), ylim=(0,.6), zlim=(0,.6),
                   title=f"{case['label']} — {'GT64' if col == 0 else 'SelfSup v1'}")
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=22, azim=-58)
            ax.set_xticks((0,.3,.6));ax.set_yticks((0,.3,.6));ax.set_zticks((0,.3,.6))
            if col == 1:
                fig.colorbar(im, ax=ax, shrink=.5, pad=.08, label='CH4 mass fraction')
    fig.suptitle('SelfSup v1: 3D plume geometry from four views\n'
                 'Boundary point clouds at 5% of each GT peak; same threshold and colors for each pair', fontsize=14)
    fig.savefig(output / 'selfsup_v1_3d_plumes.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(2); style()
    freeze = json.loads((REFERENCE / 'frozen_checkpoint.json').read_text())
    completed = json.loads((REFERENCE / 'summary.json').read_text())
    assert freeze['completed_epochs'] == 300 and completed['posthoc_gt_evaluation_only']
    assert digest(freeze['checkpoint']) == freeze['sha256']
    config = load_config(REPO / 'configs/train_selfsup_oari_mrgr_v1.yaml')
    model, checkpoint = load_model(freeze['checkpoint'], args.device)
    assert checkpoint['volumetric_gt_used'] is False
    dataset = evaluation_dataset(config, pool='train', views=4, selection='reference')
    rows = [json.loads(line) for line in (REFERENCE / 'selfsup_oari_mrgr_v1/Val-A-V4/C1/per_case_metrics.jsonl').read_text().splitlines()]
    assert len(rows) == 36
    ordered = sorted(rows, key=lambda row: row['relative_l1'])
    selected = [(label, ordered[round((len(rows)-1)*q)]) for label, q in
                (('Lower-error quartile', .25), ('Median-error case', .5), ('Upper-error quartile', .75))]
    cases = []
    with torch.no_grad():
        for label, reference in selected:
            index = next(i for i, row in enumerate(dataset.records) if row['case_id'] == reference['case_id'])
            sample = dataset[index]
            assert sample['input_station_indices'].tolist() == reference['selected_station_ids']
            batch = move_batch(projection_pool_collate([sample]), args.device)
            prediction = model(batch)
            metrics = VolumeMetrics(); metrics.update(prediction, batch['gt_volume_64'], batch['case_id'])
            actual = metrics.rows[0]
            assert abs(actual['relative_l1']-reference['relative_l1']) <= 1e-7 + reference['relative_l1']*1e-5
            cases.append(dict(label=label, case_id=sample['case_id'], station_ids=reference['selected_station_ids'],
                              metrics=actual, gt=batch['gt_volume_64'][0,0].cpu().numpy(),
                              prediction=prediction[0,0].cpu().numpy()))
            print(f"{label}: {sample['case_id']} L1={actual['relative_l1']:.8f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    draw_overview(cases, args.output)
    centroid = draw_slices(cases[1], args.output)
    draw_3d(cases, args.output)
    manifest = dict(model='SelfSup-MRGR v1 (C1)', checkpoint=freeze,
        visualization_only=True, no_training_or_selection=True, actual_test_split_unused=True,
        concentration_color_scale='GT / prediction shared per case; absolute errors have separate labeled scales',
        protocol='Val-A fixed V4; exact frozen station IDs', cohort_size=36,
        case_selection='Ranks nearest 25%, 50%, 75% of frozen per-case voxel relative L1',
        slice_selection='Typical case GT mass centroid (evaluation visualization only)',
        typical_slice_indices_xyz=centroid, three_d_threshold='5% of GT peak, shared between GT and reconstruction',
        cases=[{k:v for k,v in case.items() if k not in ('gt','prediction')} for case in cases])
    (args.output / 'visualization_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Figures:', args.output, flush=True)


if __name__ == '__main__':
    main()
