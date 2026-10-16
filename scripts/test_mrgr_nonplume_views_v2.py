#!/usr/bin/env python3
"""Compare frozen MRGR on identical saved torus measurements with wider V4/V6 layouts."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import test_mrgr_nonplume_v1 as base
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset, assert_no_gt
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler, coverage_descriptor
from geophy_set3d.training.utils import move_batch

SOURCE = base.DEFAULT_OUT
DEFAULT_OUT = ROOT / 'outputs/research/mrgr_nonplume_views_v2'
LABELS = {'ordinary_V4': 'Original V4', 'spread_V4': 'Spread V4',
          'ordinary_V6': 'Ordinary V6', 'spread_V6': 'Spread V6'}


def read(path): return json.loads(path.read_text())
def write(path, value): base.write(path, value)
def sha(path): return base.probe.digest(path)


def render(out, gt, fields, geometry, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    names = ['GT', *[LABELS[k] for k in LABELS]]
    volumes = [gt, *[fields[k]['C1'] for k in LABELS]]
    norm = Normalize(0, max(v.max() for v in volumes))
    fig, axes = plt.subplots(3, 5, figsize=(16, 9), constrained_layout=True)
    for col, (name, v) in enumerate(zip(names, volumes)):
        for row, (plane, xlabel, ylabel, cut) in enumerate([
                (v[:, 31, :].T, 'x (m)', 'z (m)', 'y=0.295 m'),
                (v[:, :, 31].T, 'x (m)', 'y height (m)', 'z=0.295 m'),
                (v[31, :, :], 'z (m)', 'y height (m)', 'x=0.295 m')]):
            im = axes[row, col].imshow(plane, origin='lower', extent=(0, .6, 0, .6),
                                      cmap='magma', norm=norm, interpolation='nearest')
            axes[row, col].set(title=f'{name}: {cut}', xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(im, ax=axes, shrink=.7, label='Mass fraction; common linear scale')
    fig.suptitle('Same torus and frozen multiscale MRGR; C1 final output')
    fig.savefig(out/'comparison_slices.png', dpi=160); plt.close(fig)

    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    edges = np.linspace(-.6/126, .6+.6/126, 65)
    xx, zz, yy = np.meshgrid(edges, edges, edges, indexing='ij')
    for i, (name, v) in enumerate([('GT', gt), ('Original V4', fields['ordinary_V4']['C1']),
                                  ('Spread V6', fields['spread_V6']['C1'])]):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.voxels(xx, zz, yy, (v >= .5*gt.max()).transpose(0, 2, 1),
                  facecolors='steelblue', edgecolor=None, alpha=.65)
        ax.set(title=name, xlabel='x (m)', ylabel='z (m)', zlabel='y height (m)',
               xlim=(0, .6), ylim=(0, .6), zlim=(0, .6))
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=25, azim=40)
    fig.suptitle('3D occupied voxels: same threshold = 50% GT peak')
    fig.savefig(out/'comparison_3d.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    origins = geometry['station_origin_m']
    for ax, setting in zip(axes, settings):
        ids = setting['input_ids']
        ax.scatter(origins[:, 0], origins[:, 2], s=6, color='lightgray')
        ax.scatter(origins[ids, 0], origins[ids, 2], s=35)
        for sid in ids:
            ax.plot([.3, origins[sid, 0]], [.3, origins[sid, 2]], lw=.7, alpha=.5)
            ax.annotate(str(sid), origins[sid, [0, 2]], fontsize=8)
        ax.scatter([.3], [.3], marker='x', c='red')
        ax.set(title=LABELS[setting['name']], xlabel='x (m)', ylabel='z (m)', aspect='equal')
    fig.suptitle('Station positions in the original Pool A; top view')
    fig.savefig(out/'view_layouts.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    previous = read(SOURCE/'summary.json')
    previous_registration = previous['registration']
    for name, digest in previous['artifacts_sha256'].items():
        assert sha(SOURCE/name) == digest, name
    assert sha(ROOT/'scripts/test_mrgr_nonplume_v1.py') == previous_registration['script_sha256']
    protocol = base.formal.verify(base.RUN)
    checkpoint = Path(previous_registration['checkpoint'])
    assert sha(checkpoint) == previous_registration['checkpoint_sha256']
    config = protocol['config']
    ds = MeasurementOnlyPoolDataset(config['dataset']['root'], split='val', geometry_pool='train',
        min_input_views=6, max_input_views=6, hold_views=0,
        normalization=config['normalization']['method'], statistics_file=config['normalization']['statistics_file'])
    with np.load(SOURCE/'geometry.npz') as saved:
        assert set(saved.files) == set(ds.geometry)
        assert all(np.array_equal(saved[k], ds.geometry[k]) for k in saved.files)
    assert ds.y_ref == previous_registration['normalization_scale_ppm_m']
    measurements = np.load(SOURCE/'measurements_ppm_m.npy', allow_pickle=False)
    old_fields = {k: v.copy() for k, v in np.load(SOURCE/'predictions.npz').items()}
    reference = read(Path(previous_registration['layout_source']))
    ordinary6 = next(s for s in reference['settings'] if s['condition']=='V6' and s['draw']==0)
    case = previous_registration['layout_reference_case']
    az = ds.geometry['station_azimuth_deg']
    selected = {'ordinary_V4': previous_registration['input_station_ids'],
        'spread_V4': EvaluationViewSampler(az, 4, 0, 'train', protocol='spread').select(case).tolist(),
        'ordinary_V6': ordinary6['input_ids'][case],
        'spread_V6': EvaluationViewSampler(az, 6, 0, 'train', protocol='spread').select(case).tolist()}
    common_hold = sorted(set(range(ds.station_count))-set(sum(selected.values(), [])))
    assert set(selected['spread_V4']).issubset(selected['spread_V6'])
    settings = [dict(name=name, input_ids=ids, station_azimuth_deg=az[ids].tolist(),
        coverage=coverage_descriptor(az[ids])) for name, ids in selected.items()]
    registration = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source=str(SOURCE), source_summary_sha256=sha(SOURCE/'summary.json'),
        source_artifact_sha256=previous['artifacts_sha256'], checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), script_sha256=sha(Path(__file__)),
        imported_experiment_source_sha256=sha(ROOT/'scripts/test_mrgr_nonplume_v1.py'),
        target_parameters=previous_registration['target_parameters'], settings=settings,
        common_unobserved_ids=common_hold, geometry_pool='unchanged Pool A, 64 stations',
        selection='geometry only; existing farthest-point spread sampler; ordinary V6 draw0 replays same first reference case',
        primary='spread V6 C1 vs original V4; spread V4 is a nested count control',
        secondary='ordinary V6, C0/C1; no layout chosen by synthetic GT or model score',
        score_order='save all new C0/C1 predictions before loading saved synthetic GT64 for scoring',
        ground_truth_in_model_input=False, training=False, test_time_optimization=False,
        final_test_used=False, normalization_scale_ppm_m=ds.y_ref)
    write(out/'registration.json', registration)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ckpt['epoch']==278 and ckpt['volumetric_gt_used'] is False
    model = base.probe.model_for(ckpt['config'], 'multiscale', args.device)
    model.load_state_dict(ckpt['model'], strict=True); model.eval()
    initial_state = base.probe.state_hash(model.state_dict())
    fields = {'ordinary_V4': old_fields}; manifest = []
    for setting in settings[1:]:
        name = setting['name']; ids = np.array(setting['input_ids'], dtype=np.int64)
        sample = ds._measurement_sample(dict(case_id='synthetic_gaussian_torus', sample_id='synthetic_gaussian_torus'),
            measurements, ids, np.empty(0, dtype=np.int64))
        batch = move_batch(projection_pool_collate([sample]), args.device)
        inputs = {k: v for k, v in batch.items() if not k.startswith('hold_')}
        assert_no_gt(inputs)
        with torch.inference_mode():
            if args.device.startswith('cuda'): torch.cuda.synchronize(args.device)
            start = time.perf_counter(); stages = model(inputs, return_stages=True)
            if args.device.startswith('cuda'): torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter()-start
            fields[name] = {k: stages[k][0, 0].cpu().numpy() for k in ('C0', 'C1')}
        assert all(v.shape==(64, 64, 64) and np.isfinite(v).all() and v.min()>=0 and v.max()<=1
                   for v in fields[name].values())
        path = out/f'{name}_predictions.npz'; np.savez_compressed(path, **fields[name])
        manifest.append(dict(name=name, prediction=path.name, sha256=sha(path),
            input_ids=ids.tolist(), inference_input_keys=sorted(inputs), forward_seconds=elapsed))
        print(f'{name}: predictions saved', flush=True)
    assert initial_state == base.probe.state_hash(model.state_dict())
    write(out/'prediction_complete.json', dict(manifest=manifest, predictions_saved_before_scoring=True,
        weights_unchanged=True, optimizer_steps=0, registration_sha256=sha(out/'registration.json')))
    # First load of the previous experiment's synthetic label in this run.
    with np.load(SOURCE/'target.npz') as saved: gt = saved['GT64']
    metrics = {}; projections = {}
    with np.load(SOURCE/'reprojections.npz') as saved:
        old_projections = {k: saved[k] for k in saved.files}
    for setting in settings:
        name = setting['name']; ids = setting['input_ids']
        panels = dict(input=ids, extra=sorted(set(range(64))-set(ids)),
                      common_unobserved=common_hold, all64=list(range(64)))
        for stage, volume in fields[name].items():
            key = name+'/'+stage
            projected = old_projections[stage] if name=='ordinary_V4' else base.project(volume, ds.geometry, args.device)
            projections[key.replace('/', '_')] = projected
            metrics[key] = dict(volume=base.shape_metrics(volume, gt), projection={})
            for label, stations in panels.items():
                metrics[key]['projection'][label] = relative_projection_metrics(torch.from_numpy(projected[stations]),
                    torch.from_numpy(measurements[stations]), torch.from_numpy(ds.geometry['valid_ray'][stations]))
    np.savez_compressed(out/'reprojections.npz', **projections)
    summary = dict(registration=registration, metrics=metrics, parameters=sum(p.numel() for p in model.parameters()),
        comparison_note='Same GT, all 64 saved measurements, geometry, normalization and checkpoint. Common unobserved panel excludes inputs from ALL layouts.',
        no_layout_score_selection=True, predictions_saved_before_scoring=True,
        gt_peak=float(gt.max()), gt_hole_mean_over_gt_peak=base.shape_metrics(gt, gt)['hole_mean_over_gt_peak'])
    write(out/'summary.json', summary)
    render(out, gt, fields, ds.geometry, settings)
    assert sha(checkpoint) == registration['checkpoint_sha256']
    write(out/'artifact_hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    v4 = metrics['ordinary_V4/C1']['volume']; v6 = metrics['spread_V6/C1']['volume']
    improvement = 100*(v4['relative_l1']-v6['relative_l1'])/v4['relative_l1']
    rel = '../../'+str(out.relative_to(ROOT))
    lines = ['# 多尺度 MRGR：相同非羽流圆环的分散六视角测试', '',
        f'完成时间：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}。沿用上次圆环、全部测量、冻结第278轮权重和训练归一化，改变输入站位。', '',
        f'**分散六视角最终三维L1为{100*v6["relative_l1"]:.2f}%，原四视角为{100*v4["relative_l1"]:.2f}%，相对误差降低{improvement:.2f}%。** Dice50由{v4["dice_gt_peak_0.5"]:.3f}变为{v6["dice_gt_peak_0.5"]:.3f}。', '',
        '## 布局与比较边界', '',
        '原四视角仅覆盖约49°–124°。本次在原Pool A内，以既有spread规则选端点、再按最远点补充站位；该规则只用站位几何，不读取圆环GT或模型得分。分散V4是分散V6的子集，用于观察增加两个视角的效果；普通V6复用相同参考案例既有draw0。未搜索最佳模型得分布局，也没有训练或测试期优化。', '',
        '| 布局 | 站位ID（保留输入顺序） | 方位角（度） | 方位覆盖范围 |',
        '|---|---|---|---:|']
    for s in settings:
        lines.append(f'| {LABELS[s["name"]]} | {s["input_ids"]} | {np.round(s["station_azimuth_deg"],2).tolist()} | {s["coverage"]["azimuth_span_deg"]:.2f}° |')
    lines += ['', f'![视角布局]({rel}/view_layouts.png)', '',
        f'所有布局用相同的{len(common_hold)}个共同未输入站位评价投影，另保存全64站投影指标。各布局自身输入/剩余站位不同，其投影误差不构成严格相同面板的配对比较；共同未输入站位与全64站适合横向比较。', '',
        '## 三维与投影结果', '',
        '| 布局/阶段 | 三维L1 | 三维L2 | Dice10 | Dice50 | 质心偏移mm | 自身输入投影L1 | 共同未输入投影L1 | 全64站投影L1 | 空洞均值/GT峰 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in settings:
        for stage in ('C0','C1'):
            m = metrics[s['name']+'/'+stage]; v,p=m['volume'],m['projection']
            lines.append(f'| {LABELS[s["name"]]}/{stage} | {100*v["relative_l1"]:.2f}% | {100*v["relative_l2"]:.2f}% | {v["dice_gt_peak_0.1"]:.3f} | {v["dice_gt_peak_0.5"]:.3f} | {v["centroid_error_mm"]:.2f} | {100*p["input"]["relative_l1"]:.2f}% | {100*p["common_unobserved"]["relative_l1"]:.2f}% | {100*p["all64"]["relative_l1"]:.2f}% | {100*v["hole_mean_over_gt_peak"]:.2f}% |')
    lines += ['', '三维L1/L2沿用上次归一化定义；Dice10/50使用相同GT峰值10%/50%阈值。质心按梯形积分。空洞定义仍为距y轴小于50mm且距y=0.3m小于25mm，GT空洞均值仅为GT峰值的0.47%。C0/C1为同一模型反馈前后，不能解释成独立模块消融。', '',
        f'分散V6空洞中超过GT峰值10%的体素占比为{100*v6["hole_fraction_above_10pct_gt_peak"]:.2f}%，空洞平均浓度为GT峰值的{100*v6["hole_mean_over_gt_peak"]:.2f}%。需结合下图判断中空形态，不能仅用整体误差代表结构恢复。', '',
        f'![共同切面与色标]({rel}/comparison_slices.png)', '',
        f'![共同50%阈值三维图]({rel}/comparison_3d.png)', '',
        '本次仅有同一个合成圆环和单个冻结训练种子。原V4到分散V6同时改变了数量与分布，分散V4→分散V6提供了相同选站规则下的局部数量对照；不能将所有变化归因于视角数量，也不能把该布局称为全局最优或推广到所有非羽流目标。', '',
        f'产物：[汇总]({rel}/summary.json)、[预测保存记录]({rel}/prediction_complete.json)、[可复现脚本](../../scripts/test_mrgr_nonplume_views_v2.py)。', '',
        '```bash', '/scratch/djc/envs/gas-torch/bin/python scripts/test_mrgr_nonplume_views_v2.py --device cuda:0 --out outputs/research/mrgr_nonplume_views_v2_repeat', '```']
    report = ROOT/'docs/reports/mrgr_nonplume_views_v2_zh.md'
    report.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(report), improvement_relative_percent=improvement,
        final={name:metrics[name+'/C1'] for name in selected}), indent=2), flush=True)


if __name__ == '__main__': main()
