#!/usr/bin/env python3
"""Render completed multiscale validation metrics and matched reconstruction images."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import test_mrgr_multiscale_frozen_v1 as ev

ROOT, OUT, RUN = ev.ROOT, ev.OUT, ev.RUN


def main():
    summary = ev.read(OUT / 'summary.json')
    r = ev.verify_registration()
    panels = summary['panels']
    shape = summary['morphology']
    morphology_old = ev.read(ROOT / 'outputs/research/historical_reconstruction_morphology_v1/summary.json')['summary']
    records = ev.rows(OUT / 'per_case_metrics.jsonl')
    manifest = ev.rows(OUT / 'prediction_manifest.jsonl')
    current = {(x['condition'], x['draw'], x['case_id'], x['stage']): x for x in records}
    old_shape = ev.rows(ROOT / 'outputs/research/historical_reconstruction_morphology_v1/per_case_morphology.jsonl')
    morphology_pairs = []
    for condition in ('Val-A-V4', 'clustered-V4'):
        hist = {x['case_id']: x for x in old_shape if x['condition'] == condition and x['model'] == 'History'}
        for metric, lower in [('centroid_distance_mm', True), ('dice_0.5', False), ('dice_0.1', False)]:
            a = np.array([current[condition, 0, c, 'C1']['morphology'][metric] for c in r['cases']])
            b = np.array([hist[c][metric] for c in r['cases']])
            p = ev.paired(a, b)
            p['improvement_wins'] = int((a < b).sum() if lower else (a > b).sum())
            p['lower_is_better'] = lower
            morphology_pairs.append(dict(condition=condition, metric=metric, **p))
    ev.write(OUT / 'morphology_paired.json', morphology_pairs)
    per_draw = []
    for setting in r['settings']:
        if setting['condition'] not in ('V3', 'V6', 'clustered-V4'):
            continue
        old = {x['case_id']: x for x in ev.rows(setting['references']['History'])}
        values = [current[setting['condition'], setting['draw'], c, 'C1']['volume']['relative_l1'] for c in r['cases']]
        base = [old[c]['relative_l1'] for c in r['cases']]
        per_draw.append(dict(condition=setting['condition'], draw=setting['draw'],
                             mean=float(np.mean(values)), historical_mean=float(np.mean(base)),
                             relative_error_reduction=float(1-np.mean(values)/np.mean(base))))
    ev.write(OUT / 'per_draw_summary.json', per_draw)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    for ax, wanted, title in [(axes[0], [p for p in panels if len(p['draws']) == 1], 'Matched single-layout 3D error'),
                              (axes[1], [p for p in panels if len(p['draws']) == 5], 'Five layouts averaged within each case')]:
        x = np.arange(len(wanted)); width = .19
        for i, model in enumerate(('History', 'Multiscale', 'A', 'B')):
            values = [p['mean'] if model == 'Multiscale' else p['comparisons'][model]['mean_reference'] for p in wanted]
            ax.bar(x + (i-1.5)*width, np.array(values)*100, width, label=model)
        ax.set_xticks(x, [p['condition'] for p in wanted], rotation=15)
        ax.set(ylabel='3D relative L1 (%)', title=title)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.savefig(OUT / 'comparison.png', dpi=170); plt.close(fig)

    chosen = ev.read(ev.AB_RUNS['A'] / 'evaluation/frozen_model_test/slice_selection.json')
    source = {x['case_id']: x for x in ev.rows(ev.DATA / 'source_index.jsonl') if x['split'] == 'val'}
    hist_root = ROOT / 'outputs/research/historical_reconstruction_morphology_v1'
    historical = {(x['condition'], x['case_id']): x for x in ev.rows(hist_root / 'prediction_manifest.jsonl')}
    predictions = {(x['condition'], x['draw'], x['case_id']): x for x in manifest}
    ab = {}
    for model, run in ev.AB_RUNS.items():
        for condition in ('Val-A-V4', 'clustered-V4'):
            d = Path(ev.read(run / f'evaluation/frozen_model_test/{condition}_prediction.json')['directory'])
            boundary = ev.read(d / 'prediction_boundary.json')
            assert ev.sha(d / 'predictions_manifest.jsonl') == boundary['prediction_manifest_sha256']
            assert boundary['all_predictions_saved_before_gt']
            for row in ev.rows(d / 'predictions_manifest.jsonl'):
                assert ev.sha(d / row['prediction']) == row['prediction_sha256']
                ab[model, condition, row['case_id']] = d / row['prediction']
    fig, axes = plt.subplots(6, 5, figsize=(16, 17), constrained_layout=True)
    source_fields = {}
    for i, selection in enumerate(chosen):
        case = selection['case_id']
        g = np.load(ev.DATA / source[case]['gt_volume_64_path'], allow_pickle=False).squeeze()
        iz = np.unravel_index(g.argmax(), g.shape)[2]
        for j, condition in enumerate(('Val-A-V4', 'clustered-V4')):
            h = historical[condition, case]
            assert ev.sha(hist_root / h['prediction']) == h['sha256']
            with np.load(hist_root / h['prediction'], allow_pickle=False) as f: history = f['C1'].copy()
            p = predictions[condition, 0, case]
            assert ev.sha(OUT / p['prediction']) == p['sha256']
            with np.load(OUT / p['prediction'], allow_pickle=False) as f: ms = f['C1'].copy()
            fields = [('GT', g), ('History', history), ('Multiscale', ms)]
            fields += [(model, np.load(ab[model, condition, case], allow_pickle=False).squeeze()) for model in ('A', 'B')]
            if i == len(chosen)-1:
                source_fields[condition] = [(name, field.copy()) for name, field in fields]
            norm = SymLogNorm(linthresh=float(g.max())*1e-3, vmin=0, vmax=max(float(v.max()) for _, v in fields))
            row = 2*i+j
            for col, (name, field) in enumerate(fields):
                ax = axes[row, col]
                im = ax.imshow(field[:, :, iz].T, origin='lower', extent=(0, .6, 0, .6), cmap='magma', norm=norm, interpolation='nearest')
                ax.set(title=f'{name} | {condition}', xlabel='x (m)', ylabel='y height (m)')
            bar = fig.colorbar(im, ax=axes[row], shrink=.65, label='CH4 mass fraction')
            ticks = [0, float(g.max())*.01, float(g.max())*.1, float(g.max())]
            bar.set_ticks(ticks, labels=[f'{v:.2g}' for v in ticks])
    fig.suptitle('Same three previously selected cases; fixed GT-peak z planes; shared scale within each row', fontsize=13)
    fig.savefig(OUT / 'matched_reconstructions.png', dpi=150); plt.close(fig)
    peak = np.unravel_index(g.argmax(), g.shape)
    ix, iy, iz = peak
    xlo, xhi = max(0, ix-3), min(64, ix+4)
    zlo, zhi = max(0, iz-3), min(64, iz+4)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), constrained_layout=True)
    for row, condition in enumerate(('Val-A-V4', 'clustered-V4')):
        for col, (name, field) in enumerate(source_fields[condition]):
            im = axes[row, col].imshow(field[xlo:xhi, iy, zlo:zhi].T, origin='lower',
                cmap='magma', vmin=0, vmax=float(g.max()), interpolation='nearest',
                extent=(xlo*.6/63, (xhi-1)*.6/63, zlo*.6/63, (zhi-1)*.6/63))
            axes[row, col].set(title=f'{name} | {condition}', xlabel='x (m)', ylabel='z (m)')
        fig.colorbar(im, ax=axes[row], shrink=.8, label='CH4 mass fraction (linear)')
    fig.suptitle(f'Previously selected A ordinary-V4 high-error case: same GT-peak patch at y={iy*.6/63:.4f} m')
    fig.savefig(OUT / 'source_peak_detail.png', dpi=170); plt.close(fig)

    ordinary = next(p for p in panels if p['condition'] == 'Val-A-V4')
    cluster5 = next(p for p in panels if p['condition'] == 'clustered-V4' and len(p['draws']) == 5)
    train = ev.read(RUN / 'summary.json')
    lines = ['# 多尺度 MRGR：300轮完成后的冻结三维重建测试', '',
        f"完成时间：{summary['completed_at']}（北京时间）。单种子20261011，按原验证投影规则选中第{r['best_epoch']}轮，参数344,678。", '',
        f"**普通固定V4：三维相对L1为{ordinary['mean']:.4%}，历史为{ordinary['comparisons']['History']['mean_reference']:.4%}，相对误差降低{ordinary['comparisons']['History']['relative_error_reduction']:.2%}；胜/平/负为{ordinary['comparisons']['History']['wins']}/{ordinary['comparisons']['History']['ties']}/{ordinary['comparisons']['History']['losses']}。**", '',
        f"**聚簇五布局：三维相对L1为{cluster5['mean']:.4%}，历史为{cluster5['comparisons']['History']['mean_reference']:.4%}，相对误差降低{cluster5['comparisons']['History']['relative_error_reduction']:.2%}。** 普通与困难布局必须同时评价；投影收益不能代替三维收益。", '',
        '**使用判断：** 多尺度版在本次单种子开发评价中形成了“普通布局小幅改善、聚簇五布局较明显改善”的候选，保留了较低的参数与训练成本；但普通固定V4仍未达到A的9.0736%，极窄draw0的质心也没有改善，不能宣布全面优于A或解决自由视角消歧。', '',
        '## 1. 协议与数据边界', '',
        '36个官方验证案例，Pool A，64³ XYZ端点网格；逐例重放历史输入站位及顺序。共17组布局：普通/分散V4各1组，V3/V6/聚簇各5组，共612次重建；反馈前C0和反馈后C1共1,224场全部保存、哈希封存后，独立评分进程才读取验证GT64。', '',
        '模型输入剔除H测量；额外H4仅用于重投影评价，与A/B后验评价逐例相同。历史原评价H口径不同，本文不作历史H的严格配对排名。checkpoint选择不变，未用GT选择轮数；本轮没有训练或测试期优化。验证GT已用于开发，不能称为盲测。最终test、GT128均未读取。', '',
        '## 2. 三维误差配对比较', '',
        '| 条件 | 布局数 | 历史 | 多尺度MRGR | A | B | 多尺度对历史误差降低 | 胜/平/负 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for p in panels:
        comp = p['comparisons']; h = comp['History']
        lines.append(f"| {p['condition']} | {len(p['draws'])} | {h['mean_reference']:.4%} | {p['mean']:.4%} | {comp['A']['mean_reference']:.4%} | {comp['B']['mean_reference']:.4%} | {h['relative_error_reduction']:.2%} | {h['wins']}/{h['ties']}/{h['losses']} |")
    lines += ['', '布局数1的V3/V6/聚簇为draw0；布局数5先在每例内平均，再对36例汇总。单布局与五布局均值不可混用，180条布局结果不等于180个独立案例。', '',
        'V3、V6和聚簇的每一个draw宏平均都优于历史；其中聚簇draw0相对只改善4.36%，其余四个draw改善约18.50%–20.71%。五布局结果支持困难收益并非单一有利布局，但最极端布局的提升较弱，不能用五布局平均掩盖这一点。', '',
        '![四模型三维误差](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/comparison.png)', '',
        '## 3. 尾部误差与配对不确定性', '',
        'Δ为多尺度减历史，负值表示改善。案例bootstrap为36例、10,000次、seed20261016；95%区间只反映案例变化，不覆盖训练种子变化。普通固定V4为主终点，其余指标和面板为次要/探索性，未做多重检验校正。', '',
        '| 条件 | 布局数 | 中位数 | P90 | 最大案例均值 | ΔL1的95%区间（百分点） |', '|---|---:|---:|---:|---:|---|']
    for p in panels:
        ci = np.array(p['comparisons']['History']['ci95'])*100
        lines.append(f"| {p['condition']} | {len(p['draws'])} | {p['median']:.4%} | {p['p90']:.4%} | {p['max']:.4%} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines += ['', '## 4. 峰值、质心和高浓度形态', '',
        '本表对应普通V4与聚簇draw0。质心使用等权端点节点，与既有形态报告一致；物理体积分与另一套质心见逐例volume字段（梯形权重）。峰值比可能正负抵消，峰幅精度需另看绝对比误差。真实峰位置比指pred[GT argmax]/max(GT)。Dice使用该例GT峰值10%/50%的共同阈值，部署不依赖GT阈值。', '',
        '| 布局 | 模型 | 三维L1 | 全局峰比 | 真实峰位置比 | 质心mm | Dice10 | Dice50 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for condition in ('Val-A-V4', 'clustered-V4'):
        for model in ('History', 'Multiscale-C0', 'Multiscale', 'A', 'B'):
            if model.startswith('Multiscale'):
                st = 'C0' if model.endswith('C0') else 'C1'
                m = next(x['mean'] for x in shape if x['condition'] == condition and x['stage'] == st)
            else:
                s = next(x for x in morphology_old if x['condition'] == condition and x['model'] == model)
                m = {k.removeprefix('mean_'): v for k, v in s.items() if k.startswith('mean_')}
            lines.append(f"| {condition} | {model} | {m['relative_l1']:.4%} | {m['peak_ratio']:.3f} | {m['gt_peak_location_ratio']:.3f} | {m['centroid_distance_mm']:.2f} | {m['dice_0.1']:.3f} | {m['dice_0.5']:.3f} |")
    lines += ['', '多尺度C0/C1来自同一联合训练模型，是轨迹诊断，不是重新训练的反馈消融。Dice50为0仅指高阈值区域不交叠，不等于完全没有重建羽流。', '',
        '| 布局 | 指标 | 多尺度均值 | 历史均值 | 改善例数 | 多尺度−历史95%区间 |', '|---|---|---:|---:|---:|---|']
    for p in morphology_pairs:
        ci = p['ci95']
        lines.append(f"| {p['condition']} | {p['metric']} | {p['mean_current']:.5f} | {p['mean_reference']:.5f} | {p['improvement_wins']}/36 | [{ci[0]:.5f}, {ci[1]:.5f}] |")
    lines += ['', '**普通布局：** 多尺度质心均值1.51 mm（历史1.61 mm）、Dice50为0.835（历史0.827），方向上较好，但两项配对区间均跨0，不能称为可靠的全面定位提升。全局峰绝对比误差从历史0.1935降至0.1803；GT高浓度区缺失率从9.79%降至9.46%，幅度均有限。', '',
        '**聚簇draw0：** Dice50从0.207升至0.304，GT峰位置浓度比从0.141升至0.210；高浓度区缺失率从72.10%降至61.47%，背景预测占比从10.17%降至8.46%。这些改善有价值，但GT峰位置仍平均欠估约79%。同时质心由15.00增至17.29 mm，只有13/36例更准；该均值差区间跨0，不能称为确定退化，也不能忽略局部定位风险。', '',
        '**反馈仍有取舍：** 普通C0→C1的36例L1均改善；聚簇draw0为30例改善、6例变差。聚簇平均质心从C0的16.64变为C1的17.29 mm，GT峰位置浓度比由0.231变为0.210，而全场L1及Dice50改善。这进一步说明总体指标、高浓度恢复和空间定位不一定同步；不能由联合训练阶段比较推断反馈模块的独立因果。', '',
        '## 5. 相同切面与色标的重建图', '',
        '固定复用此前按A普通V4误差选出的低/中/高三例，不按本模型表现重新挑选。每例普通/聚簇两行，GT峰值z平面，相同色标。图像只能支持对应切面的描述，不能证明三维拓扑或模块因果。', '',
        '![相同案例与布局重建图](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/matched_reconstructions.png)', '',
        '在固定低误差双源示例的聚簇切面中，多尺度仍出现侧向鼓出和气团间隙变形，不能因整体均值更好就解释为轮廓处处更准确。中位和高误差示例也保留近地局部偏移。以下近源放大中，普通布局多尺度能恢复峰附近结构，但该例幅值仍低于GT；聚簇下该真实峰邻域在历史、多尺度和B中均严重缺失。上述仅是固定示例观察，不能推断全域所有峰或单个模块的因果。', '',
        '下图固定为上述A普通误差最高案例，在GT峰位置取相同y平面的7×7 x-z邻域，统一GT峰值线性色标。它展示真实峰附近的幅度和位置问题，不能代表裁剪外没有伪峰。', '',
        '![同一近源峰的局部放大](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/source_peak_detail.png)', '',
        '## 6. 输入与额外H4投影', '',
        '| 条件 | draw | 输入投影L1 | 额外H4投影L1 |', '|---|---:|---:|---:|']
    for p in summary['measurement']:
        lines.append(f"| {p['condition']} | {p['draw']} | {p['observed']:.4%} | {p['heldout']:.4%} |")
    lines += ['', '本表是科学评价布局的额外H4，不是正式训练日志的固定A4/B4/H4。原最佳验证H为3.10937%，不得把两者当同一面板。B为直接连续积分，离散场投影与其差异包含表示因素。', '',
        '## 7. 成本、核验与解释边界', '',
        f"训练完成300轮/24,900更新，耗时{train['elapsed_seconds']/3600:.3f}小时；参数相比历史增加3.80%。本轮GPU3全部612次推理与保存耗时{summary['prediction_boundary']['elapsed_seconds']:.1f}秒。同步单例C0/C1重建中位数（排除首例）{summary['timing']['reconstruct_seconds_median_excluding_first']:.4f}秒，不含外部重投影和I/O，不用于与其他报告不同计时边界的推理排名。", '',
        '正式注册源码、数据身份与冻结权重校验通过；本轮推理GT/禁止NPY/外来权重读取尝试均为0。正式训练summary/status保留为当时完成快照，不回写旧文件；本次评价完成状态以evaluation目录summary为准。', '',
        '独立复核另取3例×2布局×C0/C1共12个场，用独立公式重算L1、等权质心和Dice50；同时重算全部8个面板的均值及P90，与报告一致。详见[独立核验记录](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/independent_verification.json)。', '',
        f"最佳权重SHA256：`{r['checkpoint_sha256']}`。", '',
        '单种子和重复使用的开发验证集限制结论外推。多尺度分支和额外容量同时变化，未经容量匹配对照不能断言收益专门来自多尺度；多个次要指标的区间不能当成经多重校正的确认性证据。未做噪声/标定误差、未用于开发的新几何、多种子或最终test评价。', '',
        '## 8. 下一步建议', '',
        '优先保留该模型作为鲁棒性候选，与A分别代表不同取舍。先对冻结误差做共同前向/弱观测方向诊断，尤其检查极窄布局的质心偏移与真实峰位置；不要仅因五布局平均改善就删除失败案例。后续若开展训练，应分别验证局部细节路径、反馈控制或同容量对照中的一个因素，再以新种子确认。原始时序和流场未补齐前，不增加未经验证的稳态输运约束。', '',
        '产物：[评价汇总](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/summary.json)、[逐例指标](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/per_case_metrics.jsonl)、[逐布局汇总](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/per_draw_summary.json)、[注册协议](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/registration.json)、[预测封存与读取审计](../../outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/predictions_complete.json)、[独立测试脚本](../../scripts/test_mrgr_multiscale_frozen_v1.py)、[制图报告脚本](../../scripts/report_mrgr_multiscale_frozen_v1.py)。']
    report = ROOT / 'docs/reports/mrgr_multiscale_completed_model_test_zh.md'
    report.write_text('\n'.join(lines) + '\n')
    print(report)
    print(json.dumps(morphology_pairs, indent=2))


if __name__ == '__main__': main()
