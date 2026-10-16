#!/usr/bin/env python3
"""Render completed bounded mechanism experiments without loading models/GT."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/research/plan_a_mechanisms_v1'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    registration=json.loads((BASE/'registration.json').read_text())
    stages=json.loads((BASE/'stages/summary.json').read_text())['results']
    changes=json.loads((BASE/'stage_changes.json').read_text())
    execution=json.loads((BASE/'execution.json').read_text())
    results=[]
    for setting in registration['settings']:
        arm=setting['arm'];run=BASE/'short_runs'/arm
        metrics=read_rows(run/'metrics.jsonl') if (run/'metrics.jsonl').exists() else []
        summary=json.loads((run/'training_summary.json').read_text())
        initial=json.loads((run/'initial_validation.json').read_text())
        best=min(metrics,key=lambda r:r['selection_metric']) if metrics else None
        sampling=read_rows(run/'sampling_manifest.jsonl')
        completed_sampling=[r for r in sampling if r['epoch']<len(metrics) or r.get('split')=='val']
        audit=json.loads((run/'gt_read_audit.json').read_text())
        assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        result=dict(arm=arm,parameters=setting['parameters'],completed=summary.get('completed',False),
            completed_epochs=len(metrics),updates=metrics[-1]['global_step'] if metrics else 0,
            initial_h=initial['measurement']['heldout']['relative_l1'],
            best_h=best['selection_metric'] if best else None,best_epoch=best['epoch']+1 if best else None,
            last_h=metrics[-1]['selection_metric'] if metrics else None,
            last_train_h=metrics[-1]['train']['measurement']['heldout']['relative_l1'] if metrics else None,
            last_validation_loss=metrics[-1]['validation']['loss']['total'] if metrics else None,
            elapsed_seconds=next(r['elapsed_seconds'] for r in execution['arms'] if r['arm']==arm),
            sampling=completed_sampling,gt_read_audit=audit)
        results.append(result)
    assert all(r['completed'] for r in results), 'Incomplete bounded arm: report separately; do not claim complete matched comparison'
    assert all(r['completed_epochs']==20 and r['updates']==160 for r in results)
    assert all(r['sampling']==results[0]['sampling'] for r in results)
    for r in results:del r['sampling']
    (BASE/'summary.json').write_text(json.dumps(dict(stages=stages,short_runs=results,
        same_sampling_verified=True,scope='single seed, early-learning measurement screening only; no short-run GT evaluation',
        completed=True),ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,condition in zip(axes,['Val-A-V4','clustered-V4']):
        group=[r for r in stages if r['condition']==condition]
        for key,label in [('relative_l1','3D relative L1'),('observed_l1','Observed projection'),('heldout_l1','Heldout projection')]:
            ax.plot(np.arange(len(group)),[100*r[key] for r in group],marker='o',ms=4,label=label)
        ax.set(xticks=np.arange(len(group)),xticklabels=[r['state'] for r in group],yscale='log',ylabel='Relative L1 (%)',title=condition)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(BASE/'stage_trajectory.png',dpi=180);fig.savefig(BASE/'stage_trajectory.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for r in results:
        metrics=read_rows(BASE/'short_runs'/r['arm']/'metrics.jsonl')
        ax.plot([m['epoch']+1 for m in metrics],[100*m['selection_metric'] for m in metrics],label=r['arm'])
    ax.set(xlabel='Epoch (16 training cases)',ylabel='Validation heldout relative L1 (%)',yscale='log',title='Short screen: same initial common weights and sampling')
    ax.grid(alpha=.2);ax.legend();fig.savefig(BASE/'short_learning_curves.png',dpi=180);plt.close(fig)
    lines=['# 方案 A 三项机制实验：冻结轨迹与短程对照','',
        '第一个实验重放正式 300 轮模型；后两个实验为独立 fresh 初始化、单种子的小预算筛查，不能作为正式 300 轮三维精度消融。正式 A/B 源码与检查点未更改。','',
        '**预算与公平性。** 使用物理 GPU2。短程每组 16 train/8 val、20 轮、160 次更新，训练预算每组 300 秒，监督进程额外设置 360 秒超时保护。共用初始 encoder/decoder/prior/步长/步嵌入权重，A2 与学习反馈控制全部参数初值相同；训练打乱、每轮 A/B/H 站位清单逐行一致。模型选择、训练和短程评价只读测量，不读 GT。三组都 fresh 初始化，没有使用正式模型 warm start。','',
        '| 短程版本 | 参数量 | 完成轮次/更新 | 最佳验证投影 L1 | 第20轮投影 L1 | 实际耗时 |','|---|---:|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['arm']} | {r['parameters']:,} | {r['completed_epochs']}/{r['updates']} | {100*r['best_h']:.3f}% | {100*r['last_h']:.3f}% | {r['elapsed_seconds']:.1f} 秒 |")
    lines+=['','reference 为当前预算 A2；learned_feedback 使用已有 A-control 更新规则，以归一化残差提升代替精确物理步，参数量相同；oari_initializer 仅恢复历史 OARI 初始化器，先验仍为相同预算结构，多 12,933 参数。后者是结构恢复探针，不是等总参数的对照。OARI 交互保持 FP32，共有 encoder/decoder/prior 采用原预算 BF16 设置；相应耗时不能归为单一算子的开销。','',
        '**短程结果的含义。** 当前 A2 的最佳验证投影 L1 为 17.037%，同参数量学习反馈控制为 21.221%，前者相对低约 19.72%；这支持物理更新对本次早期学习有价值，不支持“新增物理机制完全没有作用”的判断。恢复 OARI 的最佳值为 17.628%，本轮尚未显示收益；但短程普通几何投影指标不能排除其长期训练或困难几何下的作用，因此不据此启动 OARI 长训练，也不宣布该机制无效。三组训练合计 285.5 秒，包含各自初始化/编译/验证/保存，不含前面的冻结模型诊断。','',
        '**冻结模型逐步诊断。** 36 个官方验证案例分别在历史固定 V4、极窄聚簇 V4/draw0 下推理，共 72 次，保存 C0、Z1/C1…Z4/C4 全部 648 个状态；先封存全部输出及哈希，再由独立进程读取对应验证 GT64。Z 表示物理步后，C 表示先验修正后。','',
        '| 状态 | 常规 V4 三维 L1 | 聚簇 V4 三维 L1 |','|---|---:|---:|']
    names=[r['state'] for r in stages if r['condition']=='Val-A-V4']
    for name in names:
        values=[next(r['relative_l1'] for r in stages if r['state']==name and r['condition']==c) for c in ['Val-A-V4','clustered-V4']]
        lines.append(f'| {name} | {100*values[0]:.3f}% | {100*values[1]:.3f}% |')
    lines+=['',
        '**主要发现。** 常规组第三、四次物理步分别使 33/36、36/36 案例的三维 L1 上升；聚簇组第三、四次物理步均使 36/36 案例上升，同时平均输入投影误差下降。先验再将大部分空间误差修正回来。因此“输入拟合下降方向”和“三维精度改善方向”在后续步骤上存在可观测的错位，值得优先研究步长/更新目标与几何条件的适配。完整物理步＋先验的平均误差仍随四步下降，不能据此直接删除物理步骤或认定所有展开步骤无效。','',
        '初始化 C0 的三维 L1 很高，常规约 232%、聚簇约 358%。C0 没有独立重建损失，是联合训练的中间状态，不是独立训练 DeepSets 的结果；这一数值提示前后模块可能形成较强的共同适应，不能据此比较两种初始化架构的独立能力。','',
        '跨 GPU 重放 C4 的平均逐体素绝对差约 '+f"{changes['final_field_replay_mean_absolute_difference']:.3g}"+'，最大逐体素差约 '+f"{changes['final_field_replay_max_absolute_difference']:.3g}"+'，平均三维指标与原评价接近；保留全部重放输出，不覆盖原评价。本轮 H4 另行固定选择，与此前报告站位不同，只用于同次轨迹比较。','',
        '**解释限制。** 短程只观察 160 次更新，而正式训练有 24,900 次，且 20 轮 cosine 日程不同；first16/first8 案例及普通训练采样不能代表全部工况或极窄几何。A-control 的残差输入压缩与物理预条件更新并不等价，失败可能含信号尺度因素。单个种子和最佳短程投影指标不足以决定最终三维排名。GT 诊断只用于开发判断，不能作为下一版的独立盲测。','',
        '![冻结模型轨迹](../../outputs/research/plan_a_mechanisms_v1/stage_trajectory.png)','',
        '![短程学习曲线](../../outputs/research/plan_a_mechanisms_v1/short_learning_curves.png)','',
        '**核验。** 三组均完整完成，无 GT/禁止文件读取；初始共有权重、A2/控制完整初值、逐轮采样清单、完成更新数与最终 loader RNG 一致，短程源码哈希与注册相符。已有的“控制参数一致且不调用 Jacobian”及“OARI 初始化器保持原结构”两项针对性测试通过（2 passed）。正式 A/B 的来源文件哈希保持不变。','',
        '原始注册、共有初值哈希、逐案例状态/指标、逐轮训练记录、读取审计及汇总位于 `outputs/research/plan_a_mechanisms_v1/`。入口为 `scripts/probe_plan_a_mechanisms.py`；报告入口为 `scripts/report_plan_a_mechanisms.py`。']
    target=ROOT/'docs/reports/plan_a_mechanism_experiments_zh.md'
    target.write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2));print(target)


if __name__=='__main__':main()
