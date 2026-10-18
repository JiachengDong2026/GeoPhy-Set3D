#!/usr/bin/env python3
"""Report the completed, independently verified non-OCMSF screens (scalars only)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'outputs/research'
OUT = RESEARCH / 'mrgr_parallel_optimization_20261017'
REPORT = ROOT / 'docs/reports/mrgr_parallel_optimization_20261017_zh.md'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired(reference, candidate, seed=20261017):
    delta = np.asarray(candidate) - np.asarray(reference)
    rng = np.random.default_rng(seed)
    ci = np.quantile(delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(1), [.025, .975])
    return dict(mean_delta=float(delta.mean()), ci95=ci.tolist(),
                wins=int((delta < 0).sum()), losses=int((delta > 0).sum()))


def main():
    verified = read(OUT / 'independent_verification.json')
    names = ('checkpoint_average', 'projection_detail', 'geometry_curriculum')
    assert all(verified[n]['passed_verification'] for n in names)
    averages = read(RESEARCH / 'mrgr_checkpoint_average_v1/summary.json')
    detail = read(RESEARCH / 'mrgr_projection_detail_v1/gate.json')
    curriculum = read(RESEARCH / 'mrgr_geometry_curriculum_v1/comparison.json')
    late = read(RESEARCH / 'mrgr_projection_detail_v1/late_intervention_audit.json')['detail']
    records = []
    for name, label, baseline, candidate, ordinary, p90, cluster, seconds, passed in [
        ('checkpoint_average', '固定检查点平均', 'best278', 'average250_275_300',
         {a: averages['comparisons']['ordinary'][b + '_mean'] for a, b in [('best278', 'reference'), ('average250_275_300', 'candidate')]},
         {a: averages['comparisons']['ordinary'][b + '_p90'] for a, b in [('best278', 'reference'), ('average250_275_300', 'candidate')]},
         {a: averages['comparisons']['clustered'][b + '_mean'] for a, b in [('best278', 'reference'), ('average250_275_300', 'candidate')]},
         averages['total_elapsed_seconds'], averages['promoted']),
        ('projection_detail', '投影梯度细节损失', 'multiscale', 'detail', detail['ordinary_H'], detail['ordinary_P90'],
         detail['clustered_H'], None, detail['passed']),
        ('geometry_curriculum', '低强度困难几何课程', 'control', 'curriculum',
         {a: curriculum['ordinary'][a]['mean'] for a in ('control', 'curriculum')},
         {a: curriculum['ordinary'][a]['p90'] for a in ('control', 'curriculum')},
         curriculum['clustered_five_draw'], None, curriculum['passed'])]:
        records.append(dict(name=name, label=label, baseline=baseline, candidate=candidate, ordinary=ordinary,
                            p90=p90, cluster=cluster, seconds=seconds, passed=passed))

    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, route, arms, casefile, title in [
        (axes[0], 'mrgr_projection_detail_v1', ('multiscale', 'detail'), 'epoch20_per_case.jsonl', 'Projection-gradient loss'),
        (axes[1], 'mrgr_geometry_curriculum_v1', ('control', 'curriculum'), 'final_per_case.jsonl', 'Mild geometry curriculum')]:
        out = RESEARCH / route
        ids = read(out / 'protocol.json')['validation_case_ids']
        metrics = {arm: rows(out / arm / 'metrics.jsonl') for arm in arms}
        ordinary = {arm: [x['heldout_relative_l1'] for x in rows(out / arm / casefile)] for arm in arms}
        hard = rows(out / 'hard_panel_per_case.jsonl')
        hardarrays = {arm: [np.mean([r['relative_l1'] for r in hard if r['arm'] == arm and r['case_id'] == case])
                           for case in ids] for arm in arms}
        stats[route] = dict(ordinary=paired(ordinary[arms[0]], ordinary[arms[1]]),
                           cluster5=paired(hardarrays[arms[0]], hardarrays[arms[1]]),
                           two_arm_epoch_seconds=sum(x['seconds'] for arm in arms for x in metrics[arm]),
                           peak_allocated_gib={arm: max(x['peak_allocated_bytes'] for x in metrics[arm]) / 2**30 for arm in arms})
        for arm in arms:
            ax.plot([x['epoch'] for x in metrics[arm]],
                    [x['val']['measurement']['heldout']['relative_l1'] * 100 for x in metrics[arm]],
                    label=arm, linewidth=1.6)
        ax.set(title=title, xlabel='Epoch', ylabel='Held-out projection relative L1 (%)', yscale='log')
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(OUT / 'validation_curves.png', dpi=180)
    fig.savefig(OUT / 'validation_curves.svg')
    plt.close(fig)
    (OUT / 'paired_statistics.json').write_text(json.dumps(stats, indent=2) + '\n')

    lines = ['# 多尺度 MRGR：非 OCMSF 三路线并行验证报告', '',
             '日期：2026-10-17。三条独立路线已完成；本文中的误差均为**留出投影相对 L1**，不是三维相对 L1/L2。', '',
             '本报告的三条路线均为 OCMSF 之外的实验，旧多尺度空间 gate 也不能替代 OCMSF 对照。用户随后另行授权 OCMSF 并行验证，其设计、状态与结果另见 [OCMSF 设计](mrgr_ocmsf_design_zh.md)和[OCMSF 筛查](mrgr_ocmsf_screening_zh.md)。', '',
             '## 1. 结果与决策', '',
             '| 路线 | 普通均值：对照→候选 | 均值相对变化 | 普通 P90：对照→候选 | 聚簇五布局：对照→候选 | 晋级门槛 |',
             '|---|---:|---:|---:|---:|---|']
    for r in records:
        a, b = r['baseline'], r['candidate']
        def arrow(values): return f'{100*values[a]:.4f}% → {100*values[b]:.4f}%'
        lines.append(f"| {r['label']} | {arrow(r['ordinary'])} | {100*(r['ordinary'][b]/r['ordinary'][a]-1):+.2f}% | {arrow(r['p90'])} | {arrow(r['cluster'])} | {'通过' if r['passed'] else '未通过'} |")
    n_pass = sum(r['passed'] for r in records)
    lines += ['', f'本轮 {n_pass}/3 条路线通过预登记门槛。未启动新的 300 轮训练。', '',
              '**比较边界：**检查点平均使用已完成 300 轮运行的权重；另两条是各自从相同初始化开始的 20 轮配对试验。只能在每条路线内比较对照与候选，不能将表中不同训练阶段的绝对误差排序。', '',
              '晋级要求：普通均值至少相对降低 1%，普通 P90 和聚簇五布局均值均不得相对恶化超过 2%。训练路线另要求耗时增量不超过 20%、300 轮外推不超过 5 小时，并通过数值检查。所有判据在训练前登记，固定第 20 轮，未选择最好轮数。', '',
              '## 2. 三条路线分别检验什么', '',
              '### 2.1 固定检查点平均：检查后期权重波动', '',
              '预先固定 epoch250/275/300 三份参数等权平均，与原 best278 比较；非浮点参数/缓冲需一致。使用双精度累加再恢复原类型，没有搜索平均窗口，也没有增加推理模型数量。这是训练后的参数平均，不能称为训练中 EMA 或正式 SWA。', '',
              '普通误差略升而聚簇略降，未达到普通均值收益门槛；两面板的配对均值差区间均跨零。因此当前固定平均窗口不替换原 best278，也不能据此排除所有 EMA 设置。', '',
              '### 2.2 投影梯度损失：检查测量域细节监督', '',
              '网络保持 344,678 参数，仅增加权重 0.05 的 H 投影相邻差 SmoothL1。使用原始测量尺度和有效相邻射线掩码，复用已有投影，不额外执行重建。H 只用于测量监督，不作为重建输入。', '',
              f"初始 smoke 中，新增项对原检查层的加权梯度范数约为原损失的 0.0502%。末轮在预定两个训练案例上的只读审计中，加权新增损失/原目标为 {late['weighted_to_base_loss']*100:.4f}%，全参数梯度范数占比 {late['all_parameter_added_to_base_grad_norm']*100:.4f}%，原检查层占比 {late['context_output_added_to_base_grad_norm']*100:.4f}%；干预弱但非零。", '',
              '本次通过工程门槛，但普通均值配对区间跨零，且收益小于重复对照漂移，因此只保留为待重复候选。结论仅适用于当前定义与固定权重，不能推广为所有细节损失都有效或无效。', '',
              '### 2.3 低强度几何课程：检查训练布局分布', '',
              '结构与损失均不变。前 5 轮原样采样；第 6–20 轮逐步提高替换配额，最多每轮 16/166 案例。总共替换 126/3,320 个案例—轮次，约 3.80%。替换 A 站位方位跨度限定 20–40°，保留原 A/B/H 数量、互斥性并重抽 B/H。', '',
              '这一干预同时改变站位边际和 B/H 分布，不能将结果解释为纯角度因果效应。全程 A/B/H 站位边际总变差约为 0.00777/0.00680/0.01084，强度低于旧高比例增强。独立多进程检查确认 epoch 更新、掩码和计划一致。', '',
              '本次普通均值、P90和困难均值均恶化，未晋级。由于未干预的前缀已经存在明显轨迹漂移，只能否定当前一次试验的晋级资格，不能证明课程采样稳定导致退化。', '',
              '## 3. 20 轮曲线、配对不确定性与成本', '',
              '![两条训练路线的留出投影误差](../../outputs/research/mrgr_parallel_optimization_20261017/validation_curves.png)', '',
              '曲线采用 300 轮余弦学习率日程的前 20 轮，尚不能作为收敛曲线。两路线均使用完整 166 训练/36 验证案例、batch2、FP32，每臂 1,660 次更新；单种子。配对 bootstrap 以 36 个案例为重采样单元，聚簇先在案例内平均 5 个布局，不将 180 条记录当作独立案例。本报告统一使用20,000次重采样、seed20261017；专项报告因重采样次数或seed不同，区间端点可略有差异，点估计一致。区间为描述性分析，未做多重比较校正。', '',
              '| 路线 | 普通配对均值差 95% 区间（百分点） | 聚簇配对均值差 95% 区间（百分点） | 普通胜/负 | 两臂轮耗时合计 | 候选每轮 / 300 轮外推 |',
              '|---|---:|---:|---:|---:|---:|']
    for key, label, summary, candidate in [
        ('mrgr_projection_detail_v1', '投影梯度损失', detail, 'detail'),
        ('mrgr_geometry_curriculum_v1', '几何课程', curriculum, 'curriculum')]:
        s = stats[key]
        def interval(panel): return '[' + ', '.join(f'{x*100:+.4f}' for x in s[panel]['ci95']) + ']'
        seconds = summary['warm_epoch_seconds'][candidate]
        lines.append(f"| {label} | {interval('ordinary')} | {interval('cluster5')} | {s['ordinary']['wins']}/{s['ordinary']['losses']} | {s['two_arm_epoch_seconds']/60:.2f} 分钟 | {seconds:.2f}s / {seconds*300/3600:.2f}h |")
    repeat = verified['repeat_control_diagnostic']
    prefix = verified['geometry_curriculum']['preintervention_prefix']
    prefix_max = max(abs(x['ordinary_H_delta']) for x in prefix)
    lines += ['', f"**重复对照诊断：**两训练路线的原多尺度对照具有相同初始权重 hash、每轮打乱和站位记录，但末轮 H 分别为 {repeat['projection_route']*100:.4f}% / {repeat['curriculum_route']*100:.4f}%，差 {repeat['absolute_difference']*100:.4f} 个百分点（相对后者 {repeat['relative_difference']*100:.2f}%）。这是两个进程/设备/入口的实际执行漂移，不是已隔离的 GPU 因果效应，也不足以估计种子方差。案例 bootstrap 不包含训练重复的不确定性，不能据其或1%门槛宣称确定改进。", '',
              f"几何课程内部两臂在未实施干预的前5轮也已出现漂移，普通 H 最大相差 {prefix_max*100:.4f} 个百分点，尽管初态、学习率、打乱及站位证据一致。它进一步限制了本次单轨迹对小幅效应的归因能力；不据此修改原判据或事后筛选轮数。", '',
              f"检查点平均评估用时 {averages['total_elapsed_seconds']:.2f} 秒；两训练路线分别使用 GPU2/GPU3 并行。上表两臂轮耗时包含训练和普通验证，不含启动、最终困难面板和补充审计；300 轮外推取预热后均值，实际受 GPU 负载、频率和评估频次影响。", '',
              '## 4. 除 OCMSF 外仍值得验证的路线', '',
              '| 优先级 | 路线 | 对应瓶颈与最小验证 | 主要风险 |',
              '|---|---|---|---|',
              '| 高 | 共享权重两步残差反馈 | 固定 1 步/2 步对照；每步重新计算投影残差；记录每步输入/H误差及修正范数，先各20轮 | 输入拟合改善未必改善三维；额外前向需重新核验5小时目标 |',
              '| 高 | 局部方向覆盖条件化反馈 | 用射线方向二阶矩、各向异性和有效视角数调节修正，与同参数非条件模块对照 | 几何覆盖不是校准后的不确定性；需验证旋转一致性及真实调制量 |',
              '| 中 | 边缘保持的修正正则 | 小权重 Huber/Charbonnier 仅约束反馈修正 ΔC，先各20轮；冻结后查峰位、Dice、背景 | 仍可能抹掉真实尖峰；属于空间先验，不是输运物理 |',
              '| 采集可改时 | 相同4站预算下提高方向互补性 | 固定射线总预算，对比随机与预定几何选站策略；作为采集协议实验独立报告 | 当前网络未必充分利用分散视角；不能通过换容易布局声称结构改善 |', '',
              '不建议现在堆叠全部模块。已有局部分支/P90、弱 gate、部分 teacher 与本次平均窗口的负结果，只排除具体实现和预算，不能排除整类机制。128×128 主要增加同方向采样密度，不能替代互补角度；在缺少可靠时序、局部速度、密度、扩散和源边界资产时，也不宜直接添加稳态 PDE。', '',
              '候选若通过本轮测量门槛，先做少量短程重复，检查收益是否超过上述训练漂移。几何课程下一次可共享同一前5轮完整状态（模型、优化器、调度器和随机状态）后再分叉，减少干预前差异；这也不能自动消除后续 CUDA 反向的非确定性。不要因一个末轮点的微小差异直接增加300轮预算。', '',
              '随后仍需先封存候选权重/预测，再独立比较三维 L1/L2、峰幅与位置、质心、Dice 和背景泄漏。投影改善只提供筛查信号，不构成三维改进证据。最终论文模型需方法冻结后的多种子、未用于开发的几何和最终测试。', '',
              '## 5. 审计与复核入口', '',
              '启动前原正式运行 `verify` 通过：116 个注册源码、205 个数据身份。父验证器重新从三路线逐案例标量计算普通均值/P90、困难五布局案例均值、配对采样和晋级门槛；另检查3320条采样计划、126次替换及序列化修补 hash。GT 访问审计均未出现禁止读取；35个最终测试案例保持未使用。', '',
              '- [父级登记与原模型审计](mrgr_current_state_audit_zh.md)',
              '- [检查点平均专项报告](mrgr_checkpoint_average_v1_zh.md)',
              '- [投影梯度损失专项报告](mrgr_projection_detail_v1_zh.md)',
              '- [几何课程专项报告](mrgr_geometry_curriculum_v1_zh.md)',
              '- [独立复核 JSON](../../outputs/research/mrgr_parallel_optimization_20261017/independent_verification.json)',
              '- [统一配对统计](../../outputs/research/mrgr_parallel_optimization_20261017/paired_statistics.json)',
              '- [几何课程结果及注册计划](../../outputs/research/mrgr_geometry_curriculum_v1/comparison.json)',
              '- [报告生成脚本](../../scripts/report_mrgr_parallel_screens_v1.py)', '']
    REPORT.write_text('\n'.join(lines))
    print(REPORT)


if __name__ == '__main__':
    main()
