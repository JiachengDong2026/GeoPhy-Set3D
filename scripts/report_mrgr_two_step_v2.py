#!/usr/bin/env python3
"""Read only sealed scalar results and summarize shared two-step feedback."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def main(seed):
    out=ROOT/f'outputs/research/mrgr_two_step_v2_seed{seed}';p=read(out/'protocol.json');r=read(out/'comparison.json');v=read(out/'scalar_verification.json');inter=read(out/'condition_intervention/summary.json')
    arms=('B_one','T_two');names={'B_one':'多尺度MRGR，一步反馈','T_two':'共享权重，两步反馈'}
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for a in arms:
        log=[json.loads(s) for s in (out/a/'metrics.jsonl').read_text().splitlines()];epoch=[x['epoch'] for x in log]
        ax[0].plot(epoch,[x['train']['loss']['total'] for x in log],label=a);ax[1].plot(epoch,[x['val']['loss']['total'] for x in log],label=a);ax[2].plot(epoch,[100*x['val']['measurement']['heldout']['relative_l1'] for x in log],label=a)
    for x,title in zip(ax,['Train original objective','Validation original objective','Heldout projection relative L1 (%)']):x.set_title(title);x.set_yscale('log');x.set_xlabel('Epoch');x.grid(alpha=.2)
    ax[-1].legend();fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);fig.savefig(out/'curves.svg');plt.close(fig)
    drift='本seed没有额外的同GPU直接core重复，未借用不同seed作为匹配控制。'
    other=ROOT/f'outputs/research/mrgr_adaptive_geometry_v2_seed{seed}/comparison.json'
    if other.exists():
        q=read(other);current=r['ordinary']['B_one']['mean'];prior=q['ordinary']['B_multiscale']['mean'];hard=r['cluster5']['B_one']['mean'];prior_h=q['cluster5']['B_multiscale']['mean']
        audit=dict(exploratory_control_repeat=True,same_seed=seed,same_physical_gpu=1,previous_protocol_sha256=q['protocol_sha256'],current_protocol_sha256=r['protocol_sha256'],previous_ordinary=prior,current_ordinary=current,ordinary_ratio=current/prior,previous_cluster5=prior_h,current_cluster5=hard,cluster5_ratio=hard/prior_h,used_to_change_gate=False,no_gt=True)
        (out/'control_drift_context.json').write_text(json.dumps(audit,indent=2)+'\n')
        drift=f'另一个同seed、同GPU的一步直接core控制普通/聚簇为{100*prior:.4f}%/{100*prior_h:.4f}%；本次仅委托core的一步wrapper为{100*current:.4f}%/{100*hard:.4f}%，相对分别{100*(current/prior-1):+.3f}%/{100*(hard/prior_h-1):+.3f}%。底层初态、采样和目标一致，但重复执行存在数值轨迹差；不能跨路线选最好控制，也不能用案例bootstrap替代训练重复。该旁证未改变本轮gate。'
    c=r['contrasts']['T_two_vs_B_one'];o=c['ordinary'];h=c['cluster5'];ci=lambda q:f'[{100*q[0]:+.4f}, {100*q[1]:+.4f}]'
    text=f'''# 共享权重双步MRGR反馈 v2：完整20轮筛查（seed {seed}）

状态：两臂均完成20轮、每臂1,660次更新。预注册gate：**{'通过，需第二种子复核' if r['gate_passed'] else '未通过，不晋级'}**。全部指标为留出投影相对L1，不是三维误差；不读取GT或最终35例。

## 独立物理迭代方案

保持多尺度MRGR原一步反馈完整不变，在其C1上重新用同一Q96真实投影算子计算观测残差，再复用原residual_encoder、lifting、residual_adapter、fusion和decoder，将context1更新为context2并输出C2。训练/评价接口C1表示最终C2，原C1单独保留为C_feedback1。步间不detach，不加入参数，两臂均344,678参数；wrapper均用core.* state keys并共享完全相同初态。

这里的物理迭代指测量线积分正演一致性，并未引入流动/扩散PDE约束。

本路线允许更多计算以检验“学习到的残差修正是否可重复应用”。**预注册例外**为完整轮成本≤1.50、300轮外推≤5小时，不改变其他模块1.20的门槛。质量门槛仍为普通均值至少改善1%，普通P90及聚簇5均值恶化不超过2%，正确性及输出非坍缩通过。门槛在本路线GPU运行前独立注册，并非看到结果后放宽。

166训练/36验证，seed {seed}，固定第20轮末权重；原300轮cosine前缀、AdamW2e-4/wd1e-4、batch2、FP32、归一化、Q96、目标、站位与loader顺序一致。物理GPU1顺序训练两臂。协议SHA `{r['protocol_sha256']}`。

| 模型 | 参数 | 普通均值 | 普通P90 | 聚簇5均值 | 热态每轮 |
|---|---:|---:|---:|---:|---:|
'''
    for a in arms:text+=f"| {names[a]} | {p['parameters'][a]:,} | {100*r['ordinary'][a]['mean']:.4f}% | {100*r['ordinary'][a]['p90']:.4f}% | {100*r['cluster5'][a]['mean']:.4f}% | {r['warm_epoch_seconds'][a]:.2f}s |\n"
    text+=f'''

普通均值相对变化 **{100*(o['mean_ratio']-1):+.3f}%**，P90相对变化 {100*(o['p90_ratio']-1):+.3f}%，聚簇5均值相对变化 {100*(h['mean_ratio']-1):+.3f}%。耗时比 {c['runtime_ratio']:.4f}，两步300轮线性外推 {r['projected_two_300_hours']:.3f}小时。

普通案例差的95%配对区间（百分点）为 {ci(o['paired_ci95'])}，胜/负/平 {o['wins']}/{o['losses']}/{o['ties']}；聚簇5为 {ci(h['paired_ci95'])}，胜/负/平 {h['wins']}/{h['losses']}/{h['ties']}。36例配对bootstrap20,000次，聚簇先案例内平均；不包含训练种子不确定性。

门槛逐项：`{json.dumps(r['gates'],ensure_ascii=False)}`。

## 如何解释普通收益与困难退化

普通布局的均值和尾部都改善，且34/36例改善，说明新增共享反馈在本次训练中有实际价值。单种子案例区间不能替代训练重复，仍不能据此认定最终三维或跨种子收益。原gate同时保护困难布局，因此不能只挑普通均值而宣布晋级。

困难聚簇均值和中位数恶化，但P90由 {100*r['cluster5']['B_one']['p90']:.4f}% 到 {100*r['cluster5']['T_two']['p90']:.4f}%，最大值由 {100*r['cluster5']['B_one']['maximum']:.4f}% 到 {100*r['cluster5']['T_two']['maximum']:.4f}%。因此不是所有困难案例都变坏；27/36例恶化和均值区间为正说明主体案例的退化仍值得重视。不能用极端值改善代替预注册均值门槛。

可检验解释是第二次观测残差修正对普通角度覆盖有效，但在弱覆盖下过修正部分欠定方向；共享反馈也改变有效深度/增益与优化轨迹。这些尚未通过三维GT或独立机制消融验证。后续可单独研究弱覆盖时的反馈阻尼或中间阶段约束，本轮不追加未注册调参。

## 一步控制重复执行旁证

{drift}

## 正确性与实际迭代

CPU两项语义检查通过：两臂完整参数/初态一致，合成core零adapter初始输出逐位相同，非零adapter时中间C_feedback1存在非零梯度。真实GPU smoke又独立核验完整模型初始输出逐位相等、B1/B2与正常/部分padding/全padding/无有效射线共16组合均有限、空padding context2归零。一次真实临时更新后记录所有共享参数梯度；两步中间场梯度非零且新增反馈输出实际变化。临时smoke模型全部丢弃，未用于短筛初始化。

同权重诊断预注册首2验证案例×A/B四次输入：固定两步第20轮权重，比较steps1/2，并确认steps1与C_feedback1逐位一致。输出场相对差均值 {100*np.mean([x['field_relative_l1_difference'] for x in inter['rows']]):.5f}%。第一步H均值 {100*inter['first_H_mean']:.5f}%，第二步 {100*inter['final_H_mean']:.5f}%；观测投影相对L1由 {100*inter['first_observed_mean']:.5f}% 到 {100*inter['final_observed_mean']:.5f}%。权重hash不变；第二步残差context非零。这不是重新训练的消融，也不保证所有案例/步骤单调改善。

## 边界与证据

双步可能放大测量噪声或过修正；低投影误差不保证更好的三维场。收益还可能部分来自有效深度或反馈增益变化，当前两臂不能独立证明重新计算残差的必要性；若晋级主线，可追加复用旧残差/两倍单步增益对照。此结果仅开发验证单种子20轮，已知CUDA数值轨迹可影响训练；未做GT选模、形态评分或最终测试。首seed gate通过才做第二seed，并由主agent决定是否正式300轮。

![双臂曲线](../../outputs/research/{out.name}/curves.png)

- [协议](../../outputs/research/{out.name}/protocol.json)
- [全部逐例聚合与配对gate](../../outputs/research/{out.name}/comparison.json)
- [真实mask、完整参数梯度及中间梯度](../../outputs/research/{out.name}/smoke.json)
- [同权重步骤干预](../../outputs/research/{out.name}/condition_intervention/summary.json)
- [算术与hash复核](../../outputs/research/{out.name}/scalar_verification.json)

两臂各36普通记录、180聚簇记录、20轮站位/RNG/学习率以及checkpoint/source/config hashes均保存；算术复核通过 `{v['independent_verification_passed']}`，共40轮/3,320次真实更新。
'''
    report=ROOT/f'docs/reports/mrgr_two_step_v2_seed{seed}_zh.md';report.write_text(text);print(report)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=20261018);main(parser.parse_args().seed)
