#!/usr/bin/env python3
"""Report a completed local-detail MRGR run, with frozen matched baselines."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import test_mrgr_local_detail_v1 as ev


def independent_check(registration, records, manifest):
    """Recompute scalar errors from 12 fields without calling volume_metrics."""
    lookup = {(x['condition'], x['draw'], x['case_id'], x['stage']): x for x in records}
    source = {x['case_id']: x for x in ev.rows(ev.DATA / 'source_index.jsonl') if x['split'] == 'val'}
    selected = registration['cases'][::12]
    checks = []
    for x in manifest:
        if x['condition'] not in ('Val-A-V4', 'clustered-V4') or x['draw'] != 0 or x['case_id'] not in selected:
            continue
        g = np.load(ev.DATA / source[x['case_id']]['gt_volume_64_path'], allow_pickle=False).squeeze().astype('float64')
        with np.load(ev.OUT / x['prediction'], allow_pickle=False) as fields:
            for stage in ('C0', 'C1'):
                p = fields[stage].astype('float64')
                error = p-g
                actual = dict(relative_l1=float(np.sum(abs(error))/max(np.sum(abs(g)), 1e-6)),
                              relative_l2=float(np.sqrt(np.sum(error*error)/max(np.sum(g*g), 1e-12))))
                reported = lookup[x['condition'], 0, x['case_id'], stage]['volume']
                for metric, value in actual.items():
                    assert np.isclose(value, reported[metric], rtol=1e-12, atol=1e-14)
                checks.append(dict(condition=x['condition'], case_id=x['case_id'], stage=stage, **actual))
    assert len(checks) == 12
    return checks


def shape_value(morphology, metric):
    # Ratios may exceed one; taking abs per case prevents over/under-estimation cancellation.
    if metric == 'absolute_gt_peak_location_ratio_error':
        return abs(morphology['gt_peak_location_ratio'] - 1.)
    if metric == 'absolute_peak_ratio_error':
        return abs(morphology['peak_ratio'] - 1.)
    return morphology[metric]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    ev.RUN = args.run_dir.resolve(); ev.OUT = ev.RUN / 'evaluation/frozen_model_test_v1'
    registration = ev.verify_registration()
    summary = ev.read(ev.OUT / 'summary.json')
    assert summary['development_validation_only'] and not summary['final_test_evaluated']
    records = ev.rows(ev.OUT / 'per_case_metrics.jsonl')
    manifest = ev.rows(ev.OUT / 'prediction_manifest.jsonl')
    boundary = ev.read(ev.OUT / 'predictions_complete.json')
    assert boundary['manifest_sha256'] == ev.sha(ev.OUT / 'prediction_manifest.jsonl')
    assert boundary['all_predictions_saved_before_gt']
    for x in manifest:
        assert ev.sha(ev.OUT / x['prediction']) == x['sha256']
    current = {(x['condition'], x['draw'], x['case_id'], x['stage']): x for x in records}
    ms_records = ev.rows(registration['baseline_multiscale_metrics'])
    multiscale = {(x['condition'], x['draw'], x['case_id'], x['stage']): x for x in ms_records}
    checks = independent_check(registration, records, manifest)
    panel_checks = []
    for p in summary['panels']:
        values = [np.mean([current[p['condition'], d, c, 'C1']['volume'][p['metric']] for d in p['draws']])
                  for c in registration['cases']]
        assert np.isclose(np.mean(values), p['mean'], rtol=1e-12, atol=1e-14)
        assert np.isclose(np.quantile(values, .9), p['p90'], rtol=1e-12, atol=1e-14)
        panel_checks.append(dict(condition=p['condition'], draws=p['draws'], metric=p['metric'], passed=True))
    ev.write(ev.OUT / 'independent_verification.json', dict(completed_at=ev.now(), fields=checks, panels=panel_checks,
             formulas='L1=sum(abs(p-g))/sum(abs(g)); L2=sqrt(sum((p-g)^2)/sum(g^2)); no pooled cases'))

    shape_pairs = []
    for condition in ('Val-A-V4', 'clustered-V4'):
        for metric, lower in [('centroid_distance_mm', True), ('absolute_gt_peak_location_ratio_error', True),
                              ('absolute_peak_ratio_error', True), ('dice_0.5', False),
                              ('high_concentration_deficit', True), ('background_predicted_fraction', True)]:
            a = [shape_value(current[condition, 0, c, 'C1']['morphology'], metric) for c in registration['cases']]
            b = [shape_value(multiscale[condition, 0, c, 'C1']['morphology'], metric) for c in registration['cases']]
            paired = ev.paired(a, b)
            paired['lower_is_better'] = lower
            paired['improvement_wins'] = int(np.sum(np.array(a) < b if lower else np.array(a) > b))
            shape_pairs.append(dict(condition=condition, metric=metric, **paired))
    ev.write(ev.OUT / 'morphology_paired.json', shape_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    for ax, metric in zip(axes, ('relative_l1', 'relative_l2')):
        panels = [p for p in summary['panels'] if p['metric'] == metric and
                  (len(p['draws']) == 5 or p['condition'] in ('Val-A-V4', 'spread-V4'))]
        x = np.arange(len(panels)); width = .16
        for i, model in enumerate(('History', 'Multiscale', 'Local-detail', 'A', 'B')):
            y = [p['mean'] if model == 'Local-detail' else p['comparisons'][model]['mean_reference'] for p in panels]
            ax.bar(x+(i-2)*width, np.array(y)*100, width, label=model)
        ax.set_xticks(x, [p['condition'] for p in panels], rotation=15)
        ax.set(ylabel=metric+' (%)', title='Matched C1 validation, average layouts within case')
        ax.legend(fontsize=7); ax.grid(axis='y', alpha=.2)
    fig.savefig(ev.OUT / 'comparison_l1_l2.png', dpi=170); plt.close(fig)

    # Fixed examples selected in the prior A report, never reselect for the candidate.
    selections = ev.read(ev.AB_RUNS['A'] / 'evaluation/frozen_model_test/slice_selection.json')
    source = {x['case_id']: x for x in ev.rows(ev.DATA / 'source_index.jsonl') if x['split'] == 'val'}
    hist_root = ev.ROOT / 'outputs/research/historical_reconstruction_morphology_v1'
    historical = {(x['condition'], x['case_id']): x for x in ev.rows(hist_root / 'prediction_manifest.jsonl')}
    ms_root = Path(registration['baseline_multiscale_metrics']).parent
    ms_manifest = {(x['condition'], x['draw'], x['case_id']): x for x in ev.rows(ms_root / 'prediction_manifest.jsonl')}
    predictions = {(x['condition'], x['draw'], x['case_id']): x for x in manifest}
    fig, axes = plt.subplots(len(selections)*2, 4, figsize=(14, 5*len(selections)), constrained_layout=True)
    for i, selection in enumerate(selections):
        case = selection['case_id']
        g = np.load(ev.DATA / source[case]['gt_volume_64_path'], allow_pickle=False).squeeze()
        iz = np.unravel_index(g.argmax(), g.shape)[2]
        for j, condition in enumerate(('Val-A-V4', 'clustered-V4')):
            fields = [('GT', g)]
            for label, entry, directory in [('History', historical[condition, case], hist_root),
                     ('Multiscale', ms_manifest[condition, 0, case], ms_root),
                     ('Local-detail', predictions[condition, 0, case], ev.OUT)]:
                assert ev.sha(directory / entry['prediction']) == entry['sha256']
                with np.load(directory / entry['prediction'], allow_pickle=False) as archive:
                    fields.append((label, archive['C1'].copy()))
            norm = SymLogNorm(linthresh=float(g.max())*1e-3, vmin=0, vmax=max(float(f.max()) for _, f in fields))
            row = 2*i+j
            for col, (label, field) in enumerate(fields):
                im = axes[row, col].imshow(field[:, :, iz].T, origin='lower', extent=(0, .6, 0, .6),
                                          norm=norm, cmap='magma', interpolation='nearest')
                axes[row, col].set(title=f'{label} | {condition}', xlabel='x (m)', ylabel='y (m)')
            fig.colorbar(im, ax=axes[row], shrink=.7, label='CH4 mass fraction')
    fig.suptitle('Previously selected cases, GT-peak z planes; shared scales within rows')
    fig.savefig(ev.OUT / 'matched_reconstructions.png', dpi=150); plt.close(fig)

    report = args.report or ev.ROOT / f'docs/reports/{ev.RUN.name}_completed_model_test_zh.md'
    report = report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    def link(path):
        return os.path.relpath(path, report.parent)
    train = ev.read(ev.RUN / 'summary.json')
    lines = ['# 多尺度上下文＋轻量局部细节分支：冻结三维评价', '',
        f"完成时间：{summary['completed_at']}（北京时间）。训练300轮、24,900次更新，参数{registration['parameters']:,}；按原验证H规则选中第{registration['best_epoch']}轮。", '',
        '本报告为单种子开发验证评价，最终35例test未读取。不能把重复使用的验证集结果称为盲测，也不能据本次结果认定跨种子稳定收益。', '',
        '## 1. 协议', '',
        '36个验证工况，原始历史站位顺序，共17组布局、612次重建。全部C0/C1共1,224个64³浓度场在独立GT评分前保存并哈希封存；模型输入不含额外H或GT。最佳轮数按投影验证指标选取，没有用三维GT选checkpoint。', '',
        '相对L1=Σ|pred−GT|/Σ|GT|，相对L2=√(Σ(pred−GT)²/ΣGT²)。逐案例计算；五布局先在案例内平均，再对36例等权平均。五布局均值不是180个独立样本，L2均值也不是平均NMSE再开方。', '',
        '## 2. 三维误差与基线', '']
    for metric in ('relative_l1', 'relative_l2'):
        lines += [f'### {metric}', '', '| 条件 | 布局数 | 历史 | 原多尺度 | 局部细节版 | A | B | 对原多尺度降低 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for p in summary['panels']:
            if p['metric'] != metric: continue
            c = p['comparisons']
            lines.append(f"| {p['condition']} | {len(p['draws'])} | {c['History']['mean_reference']:.4%} | {c['Multiscale']['mean_reference']:.4%} | {p['mean']:.4%} | {c['A']['mean_reference']:.4%} | {c['B']['mean_reference']:.4%} | {c['Multiscale']['relative_error_reduction']:.2%} |")
        lines += ['']
    lines += [f"![L1/L2对比]({link(ev.OUT / 'comparison_l1_l2.png')})", '', '## 3. 尾部与配对不确定性', '',
        'Δ为局部细节版减原多尺度，负值表示误差降低；36案例bootstrap 10,000次，seed20261016。区间只反映案例变化，不覆盖种子变化；次要比较未做多重校正。', '',
        '| 条件 | 布局数 | 指标 | P90 | 最大案例均值 | 胜/平/负 | Δ95%区间（百分点） |', '|---|---:|---|---:|---:|---|---|']
    for p in summary['panels']:
        c = p['comparisons']['Multiscale']; lo, hi = np.array(c['ci95'])*100
        lines.append(f"| {p['condition']} | {len(p['draws'])} | {p['metric']} | {p['p90']:.4%} | {p['max']:.4%} | {c['wins']}/{c['ties']}/{c['losses']} | [{lo:.4f}, {hi:.4f}] |")
    lines += ['', '## 4. 高浓度区域、真实峰位置与质心', '',
        '形态指标使用与旧报告一致的等权节点质心和GT峰值相对阈值。volume字段另有梯形积分质心；两种口径不可混用。GT阈值仅用于评价，部署不依赖GT。', '',
        '| 条件 | 阶段/模型 | L1 | GT峰位置比 | 全局峰比 | 质心mm | Dice50 | 高浓度缺失率 | 背景占比 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in ('Val-A-V4', 'clustered-V4'):
        for label, lookup, stage in [('Multiscale', multiscale, 'C1'), ('Local-C0', current, 'C0'), ('Local-C1', current, 'C1')]:
            rr = [lookup[condition, 0, c, stage]['morphology'] for c in registration['cases']]
            mean = lambda k: np.mean([x[k] for x in rr])
            lines.append(f"| {condition} | {label} | {mean('relative_l1'):.4%} | {mean('gt_peak_location_ratio'):.4f} | {mean('peak_ratio'):.4f} | {mean('centroid_distance_mm'):.3f} | {mean('dice_0.5'):.4f} | {mean('high_concentration_deficit'):.2%} | {mean('background_predicted_fraction'):.2%} |")
    lines += ['', 'C0/C1来自同一联合训练模型，仅是阶段轨迹，不是重新训练的反馈消融。原始峰比仅作描述，接近1最好，不能把更大视为改善。下表对真实峰位置比及全局峰比逐案例计算|比值−1|再平均，避免高估与低估相互抵消，两种绝对比误差均越低越好。', '',
              '| 条件 | 形态指标 | 局部细节版 | 原多尺度 | 改善例数 | Δ95%区间 |', '|---|---|---:|---:|---:|---|']
    for p in shape_pairs:
        lo, hi = p['ci95']
        lines.append(f"| {p['condition']} | {p['metric']} | {p['mean_current']:.5f} | {p['mean_reference']:.5f} | {p['improvement_wins']}/36 | [{lo:.5f}, {hi:.5f}] |")
    lines += ['', '## 5. 固定示例重建图', '',
        '继续使用此前按A普通误差选出的三例及GT峰值z切面，每行共享色标，避免按新候选表现挑选有利案例。切面结果不代表整个三维场的拓扑。', '',
        f"![相同案例与布局]({link(ev.OUT / 'matched_reconstructions.png')})", '',
        '## 6. 测量一致性与成本', '', '| 条件 | draw | 输入投影L1 | 额外H4投影L1 |', '|---|---:|---:|---:|']
    for p in summary['measurement']:
        lines.append(f"| {p['condition']} | {p['draw']} | {p['observed']:.4%} | {p['heldout']:.4%} |")
    lines += ['', f"完整训练耗时{train['elapsed_seconds']/3600:.3f}小时；612次重建及保存耗时{boundary['elapsed_seconds']:.1f}秒。同步单例重建中位数（排除首例）{summary['timing']['reconstruct_seconds_median_excluding_first']:.4f}秒，排除外部重投影及I/O，不能作为不同计时边界下的跨模型速度排名。", '',
        '输入/额外H4投影与正式训练固定验证H面板不同。低投影误差不保证低三维误差，局部峰、形态与质心结果必须一起检查。', '',
        '## 7. 复核与证据', '',
        '正式协议、源码、参考结果与预测哈希验证通过。独立重算3例×2布局×C0/C1共12场的L1/L2，以及全部16个误差面板均值/P90。预测阶段GT/禁止NPY/外来监督权重读取审计必须为零。最终test与GT128未读取。', '',
        f"checkpoint SHA256：`{registration['checkpoint_sha256']}`。", '',
        f"产物：[汇总]({link(ev.OUT / 'summary.json')})、[逐例指标]({link(ev.OUT / 'per_case_metrics.jsonl')})、[注册协议]({link(ev.OUT / 'registration.json')})、[封存边界]({link(ev.OUT / 'predictions_complete.json')})、[独立复核]({link(ev.OUT / 'independent_verification.json')})。"]
    report.write_text('\n'.join(lines)+'\n')
    print(report)


if __name__ == '__main__': main()
