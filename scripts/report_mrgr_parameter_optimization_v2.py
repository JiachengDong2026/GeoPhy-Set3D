#!/usr/bin/env python3
"""Reporting-only plotting/annotations for the registered parameter screens."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mrgr_parameter_optimization_v2 as run

def main():
    run.report()
    report=run.ROOT/'docs/reports/mrgr_parameter_optimization_v2_zh.md'
    text=report.read_text().replace('(../../outputs/','(../../../outputs/')
    text+='\n第二种子中 `training.seed` 同时改变初始化与普通验证站位；每个种子内部严格配对。聚簇五布局跨种子固定。第二次结果是新初始化及新普通布局的匹配复核，不能解释为仅初始化重复，不将两次36例合并成72个独立病例。\n'
    comparisons=[run.read(p) for p in sorted(run.OUT.glob('seed*/comparison.json'))]
    lr_pass=all(r['contrasts']['lr4e4']['gate_passed'] for r in comparisons) and len(comparisons)==2
    text+=f"\n原协议结论：LR4e-4 {'两次匹配筛查均通过，可交主agent开展正式训练验证' if lr_pass else '尚未通过完整两次复核'}；降低 consistency 到.01因首种子聚簇均值恶化2.464%超过2%上限，未晋级，不做第二种子。LR当前收益可解释为早期优化加快，不能据20轮预测300轮三维终态收益。\n"
    text+='\n## 损失和梯度诊断\n\n`standard_total` 始终按原 `.05` 一致性系数重组，实际 `total` 才使用处理组系数；不同系数的实际 total 不可直接比较优劣。下表梯度是每种子固定前两训练案例上的全参数梯度，末期不进行更新。\n\n'
    text+='| 种子/组 | 第20轮一致性原.05梯度/原总梯度 | 实际系数梯度/原总梯度 | 一致性与原总梯度余弦 |\n|---|---:|---:|---:|\n'
    for out in sorted(run.OUT.glob('seed*')):
        if not (out/'comparison.json').exists():continue
        p=run.read(out/'protocol.json');fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for arm in p['arms']:
            rows=[json.loads(x) for x in (out/arm/'metrics.jsonl').read_text().splitlines()];epochs=[r['epoch'] for r in rows]
            axes[0,0].plot(epochs,[r['train']['loss']['standard_total'] for r in rows],label=arm)
            axes[0,1].plot(epochs,[r['val']['measurement']['heldout']['relative_l1']*100 for r in rows],label=arm)
            axes[1,0].plot(epochs,[r['train']['loss']['consistency'] for r in rows],label=arm)
            axes[1,1].plot(epochs,[r['train']['loss']['weighted_consistency_actual']/r['train']['loss']['total']*100 for r in rows],label=arm)
            q=run.read(out/arm/'late_gradient_probe.json');ratio=q['original_consistency_to_total_gradient_ratio']
            text+=f"| {p['seed']}/{arm} | {ratio*100:.2f}% | {ratio*q['actual_weight']/.05*100:.2f}% | {q['cosine_consistency_with_total']:.4f} |\n"
        titles=['Training original-weight total','Validation held-out projection relative L1 (%)','Training raw 3D consistency','Actual weighted consistency / actual total (%)']
        for ax,title in zip(axes.flat,titles):
            ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend()
        axes[0,0].set_yscale('log');axes[0,1].set_yscale('log');fig.suptitle(f"Matched parameter screens: seed {p['seed']}")
        for suffix in ('png','svg'):fig.savefig(out/f'comparison_curves.{suffix}',dpi=160)
        plt.close(fig)
    text+='\n初始梯度占比很小、末期变大反映测量项下降后目标构成发生变化；方向与总梯度同向不能证明和测量项无竞争，也不能据其大小直接判定该正则有害。固定两例探针并非全训练分布的平均结论。\n\n'
    for out in sorted(run.OUT.glob('seed*')):
        if (out/'comparison.json').exists():
            text+=f"![{out.name} 训练曲线](../../../outputs/research/mrgr_parameter_optimization_v2/{out.name}/comparison_curves.png)\n\n"
            audit=out/'independent_verification.json'
            text+=f"独立 CPU 审计：[{out.name}](../../../outputs/research/mrgr_parameter_optimization_v2/{out.name}/independent_verification.json)，状态 {'已通过' if audit.exists() and run.read(audit)['passed'] else '待执行'}。\n\n"
    text+='\n## 投影相对L2补充指标\n\n相对L2没有用于晋级筛选，以下仍是投影指标；不能将L1改善当作所有范数都同步改善。\n\n| 种子/组 | 普通H投影相对L2均值 |\n|---|---:|\n'
    for out in sorted(run.OUT.glob('seed*')):
        if not (out/'comparison.json').exists():continue
        p=run.read(out/'protocol.json')
        for arm in p['arms']:
            cases=[json.loads(x) for x in (out/arm/'epoch20_per_case.jsonl').read_text().splitlines()]
            text+=f"| {p['seed']}/{arm} | {np.mean([x['heldout_relative_l2'] for x in cases])*100:.4f}% |\n"
    text+='\n第20轮逐轮平均诊断中，降低一致性系数的组，其训练 raw consistency .18693 对照 .15308（+22.1%），验证 .17670 对照 .14159（+24.8%）；实际 total 从 .03247 降到 .02664，但按原权重重组 standard_total 升到 .03412。因此较低实际 total 包含目标权衡的变化，不能直接当作整体拟合更佳的证据。LR组的一致性与控制接近。\n'
    secondary=run.OUT/'common_layout_secondary/summary.json'
    if secondary.exists():
        secondary=run.read(secondary)
        text+='\n## 统一普通布局补充审计\n\n此项在第二种子LR结果产生前独立注册，四组固定epoch20权重哈希全部封存后，统一回放首种子普通A/B/H布局。只读测量、无GT、不影响原门槛或正式晋级。\n\n| 训练种子 | 控制 H相对L1 | LR4e-4 | 相对变化 | 配对均值差95%CI（比例） |\n|---|---:|---:|---:|---|\n'
        for seed,x in secondary['results'].items():
            c=x['contrast'];text+=f"| {seed} | {x['control']['mean']*100:.4f}% | {x['lr4e4']['mean']*100:.4f}% | {(c['mean_ratio']-1)*100:+.2f}% | {c['paired_ci95']} |\n"
        text+='\n仅作初始化/训练变化在相同普通布局下的诊断，不用于挽救失败候选，不把144条记录视为144个独立病例。证据：[注册](../../../outputs/research/mrgr_parameter_optimization_v2/common_layout_secondary/registration.json)、[完整输出](../../../outputs/research/mrgr_parameter_optimization_v2/common_layout_secondary/summary.json)。\n'
    report.write_text(text)
if __name__=='__main__':main()
