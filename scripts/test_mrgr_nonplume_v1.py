#!/usr/bin/env python3
"""Small frozen-MRGR shape OOD experiment with an analytic Gaussian torus."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import probe_mrgr_multiscale_v1 as probe
import train_mrgr_multiscale_formal_v1 as formal
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset, assert_no_gt
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.evaluation.research import volume_metrics
from geophy_set3d.geometry.tdlas import subset_geometry
from geophy_set3d.physics.projectors import ReferenceProjector, differentiable_project
from geophy_set3d.training.utils import move_batch

RUN = formal.DEFAULT_OUT
DEFAULT_OUT = ROOT / 'outputs/research/mrgr_nonplume_torus_v1'
PARAMS = dict(center_xyz_m=[.3, .3, .3], major_radius_m=.12,
              tube_sigma_m=.025, peak_mass_fraction=.08, axis='y')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def target(n):
    x, y, z = np.meshgrid(*[np.linspace(0, .6, n)] * 3, indexing='ij')
    radial = np.sqrt((x-.3)**2 + (z-.3)**2)
    return (.08*np.exp(-((radial-.12)**2 + (y-.3)**2)/(2*.025**2))).astype(np.float32)


def project(volume, geometry, device):
    tensor = torch.as_tensor(volume, device=device)
    with torch.no_grad():
        return np.concatenate([differentiable_project(tensor, subset_geometry(geometry, slice(i, i+2)),
                                samples=96).cpu().numpy()
                               for i in range(0, len(geometry['station_origin_m']), 2)])


def shape_metrics(prediction, gt):
    result = volume_metrics(prediction, gt)
    result['peak_ratio'] = float(prediction.max()/gt.max())
    for fraction in (.1, .5):
        pm, gm = prediction >= fraction*gt.max(), gt >= fraction*gt.max()
        result[f'dice_gt_peak_{fraction}'] = float(2*(pm & gm).sum()/(pm.sum()+gm.sum()))
    x, y, z = np.meshgrid(*[np.linspace(0, .6, 64)]*3, indexing='ij')
    hole = ((x-.3)**2+(z-.3)**2 < .05**2) & (np.abs(y-.3) < .025)
    result['hole_mean_over_gt_peak'] = float(prediction[hole].mean()/gt.max())
    result['hole_max_over_gt_peak'] = float(prediction[hole].max()/gt.max())
    result['hole_fraction_above_10pct_gt_peak'] = float((prediction[hole] >= .1*gt.max()).mean())
    # Face connectivity at the SAME GT-derived threshold; descriptive, not a full topology invariant.
    from scipy.ndimage import label
    for fraction in (.1, .5):
        labels, count = label(prediction >= fraction*gt.max())
        sizes = np.bincount(labels.ravel())[1:]
        result[f'components_ge_8voxels_at_{fraction}'] = int((sizes >= 8).sum())
    return result


def render(out, gt, fields, geometry, observed, projected, ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    names, volumes = ['GT', 'C0 before feedback', 'C1 final'], [gt, fields['C0'], fields['C1']]
    norm = Normalize(0, max(v.max() for v in volumes))
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for col, (name, volume) in enumerate(zip(names, volumes)):
        planes = [volume[:, 31, :].T, volume[:, :, 31].T, volume[31, :, :]]
        for row, (plane, xlabel, ylabel) in enumerate(zip(planes, ['x (m)', 'x (m)', 'z (m)'],
                                                       ['z (m)', 'y height (m)', 'y height (m)'])):
            im = axes[row, col].imshow(plane, origin='lower', extent=(0, .6, 0, .6),
                                      cmap='magma', norm=norm, interpolation='nearest')
            axes[row, col].set(title=f'{name}: {"xz / y=0.295" if row == 0 else "xy / z=0.295" if row == 1 else "yz / x=0.295"}',
                               xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(im, ax=axes, shrink=.7, label='CH4 mass fraction; common linear scale')
    fig.suptitle('Non-plume Gaussian torus: frozen multiscale MRGR, ordinary V4')
    fig.savefig(out / 'reconstruction_slices.png', dpi=160)
    plt.close(fig)

    # Voxel surfaces use a common threshold and explicit XYZ -> x,z,y plotting axes.
    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    edges = np.linspace(-.6/126, .6+.6/126, 65)
    xx, zz, yy = np.meshgrid(edges, edges, edges, indexing='ij')
    for i, (name, volume) in enumerate(zip(names, volumes)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        occupied = (volume >= .5*gt.max()).transpose(0, 2, 1)
        ax.voxels(xx, zz, yy, occupied, facecolors='steelblue', edgecolor=None, alpha=.65)
        ax.set(xlabel='x (m)', ylabel='z (m)', zlabel='y height (m)', title=name,
               xlim=(0, .6), ylim=(0, .6), zlim=(0, .6))
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=25, azim=40)
    fig.suptitle('3D occupied voxels: same threshold = 50% GT peak')
    fig.savefig(out / 'reconstruction_3d.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(12, 8), constrained_layout=True)
    scale = max(observed[ids].max(), projected[ids].max())
    for col, sid in enumerate(ids):
        az = geometry['station_azimuth_deg'][sid]
        for row, a in enumerate((observed[sid], projected[sid], np.abs(projected[sid]-observed[sid]))):
            im = axes[row, col].imshow(a, origin='lower', vmin=0, vmax=scale, cmap='magma')
            axes[row, col].set_title(f'{["Observed", "C1 reprojection", "Absolute error"][row]}: #{sid}, {az:.1f} deg')
    fig.colorbar(im, ax=axes, shrink=.8, label='ppm m; common linear scale')
    fig.savefig(out / 'input_projections.png', dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    protocol = formal.verify(RUN)
    checkpoint = RUN / 'checkpoints/best.pt'
    checkpoint_sha = probe.digest(checkpoint)
    assert checkpoint_sha == '650b4f2c0f0ab345b5fd99a7d629e0db388cede0937acb7e3191462cd57898d7'
    config = protocol['config']
    ds = MeasurementOnlyPoolDataset(config['dataset']['root'], split='val', geometry_pool='train',
        min_input_views=4, max_input_views=4, hold_views=0,
        normalization=config['normalization']['method'], statistics_file=config['normalization']['statistics_file'])
    # Choose an existing ordinary layout by index before observing this synthetic target/model output.
    layout_path = RUN / 'evaluation/frozen_model_test_v1/registration.json'
    reference = json.loads(layout_path.read_text())
    setting = next(s for s in reference['settings'] if s['condition'] == 'Val-A-V4')
    reference_case = reference['cases'][0]
    ids = np.array(setting['input_ids'][reference_case], dtype=np.int64)
    hold_ids = np.setdiff1d(np.arange(ds.station_count), ids)
    registration = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        target='analytic Gaussian torus, new shape OOD single case', target_parameters=PARAMS,
        formula='0.08 exp(-((sqrt((x-.3)^2+(z-.3)^2)-.12)^2+(y-.3)^2)/(2*.025^2))',
        axes='XYZ, y vertical; endpoint grids over [0,0.6]^3 m',
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha, best_epoch=278,
        script_sha256=probe.digest(Path(__file__)), training_protocol_sha256=probe.digest(RUN/'protocol.json'),
        layout_source=str(layout_path), layout_source_sha256=probe.digest(layout_path),
        layout_reference_case=reference_case, input_station_ids=ids.tolist(),
        extra_station_ids=hold_ids.tolist(),
        station_azimuth_deg=ds.geometry['station_azimuth_deg'][ids].tolist(),
        station_origin_m=ds.geometry['station_origin_m'][ids].tolist(),
        selection='first registered ordinary V4 case; no target/model-dependent layout selection',
        projection_reference='analytic GT128 -> independent CPU reference, 256 samples, trapezoidal integration',
        normalization_scale_ppm_m=ds.y_ref, training=False, test_time_optimization=False,
        noise=False, final_test_used=False, external_volume_data_read=False,
        scope='one synthetic target, one ordinary layout, one frozen seed; exploratory only')
    write(out / 'registration.json', registration)
    gt128, gt64 = target(128), target(64)
    np.savez_compressed(out / 'target.npz', GT64=gt64, GT128=gt128)
    projector_config = yaml.safe_load((ROOT / 'configs/dataset_generation_v1.yaml').read_text())['projector']
    projector_config['threads'] = 2
    reference_projector = ReferenceProjector(projector_config, out/'projector_cache')
    measurements = reference_projector(gt128, ds.geometry)
    np.save(out / 'measurements_ppm_m.npy', measurements)
    np.savez_compressed(out / 'geometry.npz', **ds.geometry)
    sample = ds._measurement_sample(dict(case_id='synthetic_gaussian_torus', sample_id='synthetic_gaussian_torus'),
                                   measurements, ids, np.empty(0, dtype=np.int64))
    batch = move_batch(projection_pool_collate([sample]), args.device)
    inputs = {k: v for k, v in batch.items() if not k.startswith('hold_')}
    assert_no_gt(inputs)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ckpt['epoch'] == 278 and ckpt['volumetric_gt_used'] is False
    model = probe.model_for(ckpt['config'], 'multiscale', args.device)
    model.load_state_dict(ckpt['model'], strict=True); model.eval()
    before = probe.state_hash(model.state_dict())
    with torch.inference_mode():
        if args.device.startswith('cuda'): torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        stages = model(inputs, return_stages=True)
        if args.device.startswith('cuda'): torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter()-start
        fields = {k: stages[k][0, 0].cpu().numpy() for k in ('C0', 'C1')}
    assert all(v.shape == (64, 64, 64) and np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1 for v in fields.values())
    assert before == probe.state_hash(model.state_dict())
    np.savez_compressed(out / 'predictions.npz', **fields)
    # Predictions are saved before any score is calculated. GT is synthetic and used above only for forward generation.
    write(out / 'prediction_complete.json', dict(predictions_sha256=probe.digest(out/'predictions.npz'),
        weights_unchanged=True, optimizer_steps=0, inference_gt_keys=[],
        inference_input_keys=sorted(inputs), predictions_saved_before_scoring=True))
    metrics = {}
    all_projections = {}
    for name, volume in [('GT64 representation baseline', gt64), *fields.items()]:
        projected = project(volume, ds.geometry, args.device)
        all_projections[name] = projected
        metrics[name] = dict(volume=shape_metrics(volume, gt64), projection={})
        for label, selected in [('input_V4', ids), ('extra_V60', hold_ids)]:
            metrics[name]['projection'][label] = relative_projection_metrics(
                torch.from_numpy(projected[selected]), torch.from_numpy(measurements[selected]),
                torch.from_numpy(ds.geometry['valid_ray'][selected]))
    np.savez_compressed(out / 'reprojections.npz', GT64=all_projections['GT64 representation baseline'],
                        C0=all_projections['C0'], C1=all_projections['C1'])
    summary = dict(registration=registration, metrics=metrics, parameters=sum(p.numel() for p in model.parameters()),
        first_forward_seconds=elapsed, timing_note='single cold forward; excludes forward generation, reprojection and I/O',
        reference_projector_source_sha256=reference_projector.source_hash,
        measurement_max_ppm_m=float(measurements.max()),
        input_measurement_max_over_training_scale=float(measurements[ids].max()/ds.y_ref),
        gt64_mean=float(gt64.mean()), gt64_peak=float(gt64.max()),
        hole_definition='sqrt((x-.3)^2+(z-.3)^2)<.05 m and abs(y-.3)<.025 m; normalized to GT64 peak',
        metric_note='relative L1=sum(abs(pred-GT))/sum(GT); relative L2=norm(pred-GT)/norm(GT); Dice uses common GT-peak thresholds; integral/centroid use trapezoidal weights',
        predictions_saved_before_scoring=True, weights_unchanged=True)
    render(out, gt64, fields, ds.geometry, measurements, all_projections['C1'], ids)
    assert probe.digest(checkpoint) == checkpoint_sha
    summary['artifacts_sha256'] = {p.name: probe.digest(p) for p in out.iterdir() if p.is_file()}
    write(out / 'summary.json', summary)
    c0, c1 = metrics['C0'], metrics['C1']
    v = c1['volume']
    rel = '../../' + str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
    report = ROOT / 'docs/reports/mrgr_nonplume_torus_test_zh.md'
    lines = ['# 多尺度 MRGR：非羽流圆环目标小测试', '',
        f'完成时间：{registration["created_at"]}。使用已完成300轮训练、验证投影选出的第278轮冻结权重；单个新合成目标、单个普通V4布局。', '',
        f'最终三维相对L1为 **{100*v["relative_l1"]:.2f}%**，相对L2为 **{100*v["relative_l2"]:.2f}%**，Dice10为 **{v["dice_gt_peak_0.1"]:.3f}**，Dice50为 **{v["dice_gt_peak_0.5"]:.3f}**。', '',
        '## 目标与测量', '',
        '目标为水平中空高斯圆环：中心(0.3,0.3,0.3)m，大半径0.12m，管截面标准差0.025m，解析峰值质量分数0.08；y为竖直方向。目标参数及布局在计算模型输出前保存，未根据结果调参。它是用于检验形状迁移的标量场，不是流体模拟的稳态解。', '',
        f'复用既有普通V4评价第一例的站位{ids.tolist()}，方位角为{np.round(ds.geometry["station_azimuth_deg"][ids], 2).tolist()}°。没有新增聚簇困难条件。其余60站全部作为额外投影评价，未输入模型；它们是相关的几何观测，不是60个独立目标。', '',
        '在128³端点网格生成解析场，通过原始独立CPU参考投影器（256采样点）生成ppm·m测量；模型输入沿用训练集固定归一化尺度。64³解析场只用于评分及表示误差基线。没有训练、测试期优化或读取已有数据集的三维标签/最终test。', '',
        f'输入最大测量/训练归一化尺度为{summary["input_measurement_max_over_training_scale"]:.3f}，GT64平均质量分数为{gt64.mean():.5f}（训练统计均值约0.00392）。', '',
        '## 结果', '',
        '| 阶段 | 三维L1 | 三维L2 | Dice10 | Dice50 | 峰值比 | 质心偏移mm | 输入投影L1 | 额外60站投影L1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in metrics:
        m, p = metrics[name]['volume'], metrics[name]['projection']
        lines.append(f'| {name} | {100*m["relative_l1"]:.2f}% | {100*m["relative_l2"]:.2f}% | {m["dice_gt_peak_0.1"]:.3f} | {m["dice_gt_peak_0.5"]:.3f} | {m["peak_ratio"]:.3f} | {m["centroid_error_mm"]:.2f} | {100*p["input_V4"]["relative_l1"]:.2f}% | {100*p["extra_V60"]["relative_l1"]:.2f}% |')
    lines += ['', 'L1和L2均以对应GT范数归一化；Dice10/50使用GT64峰值10%/50%的共同阈值。质心采用物理梯形积分。圆环具有对称峰，单个argmax位置差不作为主要形态结论。GT64表示基线的投影误差包含128→64表示及参考/重投影积分差异。', '',
        f'圆心空洞区域定义为距y轴小于50mm、距y=0.3m小于25mm。GT区域平均浓度/GT峰值为{metrics["GT64 representation baseline"]["volume"]["hole_mean_over_gt_peak"]:.4f}，C0为{c0["volume"]["hole_mean_over_gt_peak"]:.4f}，C1为{v["hole_mean_over_gt_peak"]:.4f}；C1在该空洞内超过GT峰值10%的体素比例为{100*v["hole_fraction_above_10pct_gt_peak"]:.2f}%。该指标反映局部填洞，不能单独证明完整三维拓扑正确。', '',
        f'![相同位置、相同色标的切片]({rel}/reconstruction_slices.png)', '',
        f'![共同50%阈值的三维体素表面]({rel}/reconstruction_3d.png)', '',
        f'![输入测量与最终重投影]({rel}/input_projections.png)', '',
        '本测试只支持对该圆环及该常规布局的探索性判断，不能代表所有非羽流目标。C0/C1为同一个联合训练模型的反馈前后输出，差异不等于独立模块消融；本次合成误差也不与真实羽流验证集均值直接作排名。', '',
        f'产物：[汇总]({rel}/summary.json)、[目标]({rel}/target.npz)、[冻结预测]({rel}/predictions.npz)、[可复现脚本](../../scripts/test_mrgr_nonplume_v1.py)。', '',
        '复现命令（输出目录必须不存在）：', '',
        '```bash', '/scratch/djc/envs/gas-torch/bin/python scripts/test_mrgr_nonplume_v1.py --device cuda:0 --out outputs/research/mrgr_nonplume_torus_v1_repeat', '```']
    report.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(report=str(report), metrics=metrics), indent=2), flush=True)


if __name__ == '__main__':
    main()
