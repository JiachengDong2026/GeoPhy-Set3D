#!/usr/bin/env python3
"""Render completed, independently scored remaining-draw development results."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/research/ab_remaining_draws_v1'


def main():
    summary=json.loads((BASE/'summary.json').read_text())
    registration=json.loads((BASE/'registration.json').read_text())
    full=[x for x in summary['results'] if x['draw']=='all5']
    assert len(full)==3
    lines=['# A / B：补齐历史剩余四个 draw 后的几何稳健性评价','',
        '本实验补齐 V3、V6、clustered-V4 的历史 draw1..4，复用已冻结评价的 draw0，得到五个既定布局的综合结果。它属于同一批 36 个验证案例上的开发评价；没有新增训练，没有访问最终 test。','',
        '五个布局的结果一致支持：A 的三/六视角平均收益具有跨既定布局的一致性，极窄聚簇退化也具有一致性，并非 draw0 偶然；B 在跨五 draw 的逐案例平均上仍全面落后。新 draw 中 B 存在少数个案胜利，不能将此前 draw0 的逐例全败扩大为所有新布局逐例全败。','',
        '## 五 draw 综合结果','',
        '**所有综合统计先在每个案例内部平均五个 draw 的误差，再对 36 个案例作统计。180 个案例布局不视为 180 个独立案例。** 相对 L1 越低越好；“改善”指相对历史模型的误差降低率。','',
        '| 条件 | 历史 L1 | A L1 | A 改善 | A 胜/负 | B L1 | B 改善 | B 胜/负 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in full:
        v=x['relative_l1'];a=x['A_versus_history'];b=x['B_versus_history']
        lines.append(f"| {x['condition']} | {100*v['history']:.4f}% | {100*v['A']:.4f}% | {100*a['relative_error_reduction']:+.2f}% | {a['wins']}/{a['losses']} | {100*v['B']:.4f}% | {100*b['relative_error_reduction']:+.2f}% | {b['wins']}/{b['losses']} |")
    lines.extend(['','| 条件 | A−历史均值差（百分点）及 95% CI | B−历史均值差（百分点）及 95% CI |',
        '|---|---:|---:|'])
    for x in full:
        a=x['A_versus_history'];b=x['B_versus_history']
        lines.append(f"| {x['condition']} | {100*a['mean_delta']:+.4f} [{100*a['ci95'][0]:+.4f}, {100*a['ci95'][1]:+.4f}] | {100*b['mean_delta']:+.4f} [{100*b['ci95'][0]:+.4f}, {100*b['ci95'][1]:+.4f}] |")
    lines.extend(['','置信区间使用案例级配对 bootstrap，10,000 次、种子 20261016；只反映给定五个布局中的案例差异，不包括训练种子不确定性，也不估计从新布局分布抽样的不确定性。','',
        '## 各 draw 分解','',
        '| 条件 | draw | 历史 L1 | A L1 | A 对历史改善 | B L1 | B 对历史改善 |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for x in summary['results']:
        if x['draw']=='all5':continue
        v=x['relative_l1']
        lines.append(f"| {x['condition']} | {x['draw']} | {100*v['history']:.4f}% | {100*v['A']:.4f}% | {100*x['A_versus_history']['relative_error_reduction']:+.2f}% | {100*v['B']:.4f}% | {100*x['B_versus_history']['relative_error_reduction']:+.2f}% |")
    lines.extend(['','![各 draw 的三维误差](../../outputs/research/ab_remaining_draws_v1/draw_comparison.png)','',
        '## 范围、正确性与成本','',
        '- 两模型均使用原来已完成 300 轮训练的冻结检查点；A 最佳第 299 轮，B 最佳第 288 轮，单训练种子 20261011。模型结构、参数及执行精度未改变。',
        '- 全部十二份历史 draw1..4 参考文件存在，每份 36 个官方验证案例。新评价逐例精确重放历史输入站点及顺序；A/B 使用相同的输入和留出站位。39 个数据文件（索引、训练统计、Pool A 几何及 36 个验证投影）在注册和预测时校验哈希。',
        '- 每模型新增 432 个预测场，共 864 个 FP32 64³ XYZ 端点场；全部保存并哈希封存后，独立进程才读取验证 GT64。预测阶段 GT/禁止文件/外部检查点打开尝试均为 0。',
        '- draw0 直接复用已有冻结评价的逐案例指标，校验文件哈希及与历史 draw0 完全相同的站点顺序，没有重选模型、阈值、布局或案例。',
        '- 新指标同时保存输入/H4 投影误差。B 使用连续函数直接积分，A 使用离散 64³ 场投影；这些投影指标的表示口径不同。三维 L1 在相同 GT64 网格上计算，是本报告的主要比较指标。',
        '- 新 draw 的 H 站点由预注册确定性规则选择，A/B 完全一致；历史 H 面板未重放，所以不拿历史 H 作配对因果比较。',''])
    for m in ('A','B'):
        b=json.loads((BASE/m/'prediction_boundary.json').read_text())
        lines.append(f"- {m} 预测阶段耗时 **{b['elapsed_seconds']:.2f} 秒**，峰值 PyTorch 分配显存 **{b['peak_allocated_bytes']/1024**3:.2f} GiB**，含模型加载、重建、输入/H 投影、压缩保存和校验。")
    lines.extend([f"- 两模型预测合计 **{summary['prediction_elapsed_seconds']:.2f} 秒**，在注册的 5 分钟预算内；物理 GPU1 上顺序执行，未与新 B 短程训练同时运行。GT 评分和本报告绘图另在 CPU 执行。共享机器上的该时间不等于独占卡基准，不能外推训练耗时。",'',
        '## 解释边界','',
        '补齐布局可以检验 draw0 结论是否偶然，但仍未增加独立案例或训练种子。五个 cluster 布局均为极窄角度；新增 draw1..4 的最短覆盖弧分别为 1.9175°、1.9779°、2.1031°、2.3852°，draw0 为约 1.6907°，不代表全部现实困难几何。','',
        '这是冻结模型的横向稳健性评价，不是针对初始化、网络容量、物理反馈或连续表示某一因素的严格因果消融。指标变化需要与同期短程控制实验及冻结步长干预共同解释。','',
        '## 产物','',
        '- [registration.json](../../outputs/research/ab_remaining_draws_v1/registration.json)、[all_predictions_complete.json](../../outputs/research/ab_remaining_draws_v1/all_predictions_complete.json)',
        '- [summary.json](../../outputs/research/ab_remaining_draws_v1/summary.json)、[per_case_metrics.jsonl](../../outputs/research/ab_remaining_draws_v1/per_case_metrics.jsonl)',
        '- [A manifest](../../outputs/research/ab_remaining_draws_v1/A/manifest.jsonl)、[B manifest](../../outputs/research/ab_remaining_draws_v1/B/manifest.jsonl)',
        '- [独立预测与评分入口](../../scripts/probe_ab_remaining_draws_v1.py)、[报告入口](../../scripts/report_ab_remaining_draws_v1.py)'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,condition in zip(axes,['V3','V6','clustered-V4']):
        group=[x for x in summary['results'] if x['condition']==condition and x['draw']!='all5']
        for model,label in [('history','Historical MRGR'),('A','A'),('B','B')]:
            ax.plot([x['draw'] for x in group],[100*x['relative_l1'][model] for x in group],marker='o',label=label)
        ax.set(title=condition,xlabel='Historical draw',ylabel='3D relative L1 (%)',xticks=range(5));ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(BASE/'draw_comparison.png',dpi=170);plt.close(fig)
    p=ROOT/'docs/reports/ab_remaining_draws_v1_zh.md';p.write_text('\n'.join(lines)+'\n')
    print(p)


if __name__=='__main__':main()
