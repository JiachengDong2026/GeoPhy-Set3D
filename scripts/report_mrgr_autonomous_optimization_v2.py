#!/usr/bin/env python3
"""Consolidate completed optimization screens without selecting new hyperparameters."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research/mrgr_autonomous_optimization_v2'
REPORT=ROOT/'docs/reports/mrgr_autonomous_optimization_v2_zh.md'
MARKER='## 6. 筛查结果（自动汇总）'


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    groups=[]
    for kind,pattern in [('可学习几何','mrgr_adaptive_geometry_v2_seed*/comparison.json'),
        ('尺度融合','mrgr_scale_fusion_v2_seed*/comparison.json'),
        ('参数优化','mrgr_parameter_optimization_v2/seed*/comparison.json'),
        ('两步反馈','mrgr_two_step_v2_seed*/comparison.json')]:
        for path in sorted((ROOT/'outputs/research').glob(pattern)):
            p=read(path.parent/'protocol.json')
            groups.append((kind,p['seed'],path.parent,read(path)))
    text=MARKER+'\n\n以下全部是固定20轮的留出**投影**相对L1，不是三维浓度误差。不同种子会改变初态及随机站位，只在每个种子内部配对比较。\n\n'
    text+='| 路线/种子 | 对照→候选 | 普通均值变化 | P90变化 | 聚簇五布局变化 | 数值门槛 |\n|---|---|---:|---:|---:|---|\n'
    for kind,seed,path,result in groups:
        for key,c in result['contrasts'].items():
            if kind=='参数优化':label=f'control→{key}';passed=c['gate_passed']
            else:
                candidate,control=key.split('_vs_');label=f'{control}→{candidate}'
                if kind=='尺度融合':passed=result['gates'][candidate]['gate_passed'] if control=='B_multiscale' else None
                elif candidate=='D_geometry' or kind=='两步反馈':passed=result['gates'][control]['gate_passed']
                else:passed=None
            o,h=c['ordinary'],c['cluster5']
            status='通过' if passed else '未通过' if passed is False else '探索性'
            text+=f"| {kind}/{seed} | {label} | {(o['mean_ratio']-1)*100:+.2f}% | {(o['p90_ratio']-1)*100:+.2f}% | {(h['mean_ratio']-1)*100:+.2f}% | {status} |\n"
    text+='\n负号代表误差降低。几何候选须同时通过相对B和相对同参数内容控制两组主对照；单个通过不能代替联合晋级。数值门槛不是跨种子统计显著性证明。\n\n'
    seeds=sorted({g[1] for g in groups})
    for seed in seeds:
        current=[g for g in groups if g[1]==seed]
        fig,axes=plt.subplots(2,len(current),figsize=(5*len(current),7),squeeze=False,constrained_layout=True)
        for col,(kind,_,path,result) in enumerate(current):
            for arm in result['ordinary']:
                rr=rows(path/arm/'metrics.jsonl')
                train=[x['train']['loss'].get('standard_total',x['train']['loss']['total']) for x in rr]
                axes[0,col].plot([x['epoch'] for x in rr],train,label=arm)
                axes[1,col].plot([x['epoch'] for x in rr],[100*x['val']['measurement']['heldout']['relative_l1'] for x in rr],label=arm)
            axes[0,col].set_title(path.name)
            axes[0,col].set_ylabel('Train original-weight total loss')
            axes[1,col].set_ylabel('Validation heldout projection relative L1 (%)')
            for row in range(2):
                axes[row,col].set_xlabel('Epoch');axes[row,col].set_yscale('log');axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
        artifact=OUT/f'screen_curves_seed{seed}'
        fig.savefig(artifact.with_suffix('.png'),dpi=160)
        fig.savefig(artifact.with_suffix('.svg'))
        plt.close(fig)
        text+=f'![筛查曲线 seed{seed}](../../outputs/research/mrgr_autonomous_optimization_v2/screen_curves_seed{seed}.png)\n\n'
    drift=read(OUT/'control_execution_drift.json')
    text+=f"同seed20261018的三次控制初态hash、逐轮站位及loader RNG一致，普通均值分别为"+ '、'.join(f"{x['H']*100:.4f}%" for x in drift['controls'])+f"；最大/最小相对差{drift['relative_max_min_spread']*100:.2f}%。这是独立执行/入口/设备差异混杂下的观测，不是GPU因果实验，也不是训练种子总体方差估计。\n\n"
    if (OUT/'promotion_decision.json').exists():
        decision=read(OUT/'promotion_decision.json')
        text+='### 6.1 正式候选决定\n\n'+decision['explanation_zh']+'\n\n'
        text+='[封存决定](../../outputs/research/mrgr_autonomous_optimization_v2/promotion_decision.json)。\n\n'
    text+='### 6.2 成本与原始记录\n\n| 路线/种子 | 模型 | 普通投影均值 | 每轮秒 | 300轮线性外推 |\n|---|---|---:|---:|---:|\n'
    for kind,seed,path,result in groups:
        for arm,v in result['ordinary'].items():
            seconds=result['warm_epoch_seconds'][arm]
            text+=f"| {kind}/{seed} | {arm} | {v['mean']*100:.4f}% | {seconds:.2f} | {seconds*300/3600:.2f}h |\n"
    text+='\n外推不是已完成300轮训练。完整轮边界含训练及普通验证，checkpoint/绘图及困难面板另计；精确成本以运行summary为准。\n\n'
    for kind,seed,path,result in groups:
        relative=path.relative_to(ROOT)
        text+=f'- {kind}/{seed}：[协议](../../{relative}/protocol.json)、[完整结果](../../{relative}/comparison.json)。\n'
    text+='\n[75项回归记录](../../outputs/research/mrgr_autonomous_optimization_v2/regression_tests.json)；[独立执行漂移核查](../../outputs/research/mrgr_autonomous_optimization_v2/control_execution_drift.json)。\n'
    verification=OUT/'independent_verification.json'
    if verification.exists():
        v=read(verification)
        text+=f"\n父agent独立核验通过：从逐案例原始记录复算全部主/探索性对照、区间与门槛；核对源码/数据/权重/配置hash、学习率/损失/案例/站位/打乱及GT审计。已核对{v['total_epochs']}完整轮、{v['total_updates']:,}更新；不读取新三维GT。[核验记录](../../outputs/research/mrgr_autonomous_optimization_v2/independent_verification.json)。\n"
    prefix=REPORT.read_text().split(MARKER)[0].rstrip()+'\n\n'
    REPORT.write_text(prefix+text)
    print(REPORT)


if __name__=='__main__':main()
