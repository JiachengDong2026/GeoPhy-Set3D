"""Render final fixed-budget probe results, without choosing checkpoints."""
from probe_context import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    registration=read(ART / 'registration.json')
    independent=read(ART / 'independent_review.json')
    integrity=read(ART / 'execution_integrity.json')
    results={(index,arm):read(ART / f'case{index}_{arm}.json') for index in registration['indices'] for arm in registration['arms']}
    final={(r['index'],r['arm']):r for r in independent['final_results']}
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    colors={'voxel_sigmoid':'tab:blue','voxel_scaled':'tab:green','unet':'tab:orange'}
    for j,index in enumerate(registration['indices']):
        for arm in registration['arms']:
            metrics=results[index,arm]['metrics']
            for i,key in enumerate(['heldout','E']):
                values=[m['extra_E_H'] if key=='E' else m['measurement'][key] for m in metrics]
                axes[i,j].plot([m['step'] for m in metrics],values,'o-',label=arm,color=colors[arm],markersize=3)
                axes[i,j].axhline(1,color='grey',lw=.5)
                axes[i,j].set(xlabel='Actual optimizer updates',ylabel='Relative L1',title=f'Train case {index}: '+('fitted H rays' if i==0 else 'unseen E rays'))
        axes[0,j].legend(fontsize=8)
    fig.savefig(ART / 'probe_curves.png',dpi=180)
    plt.close(fig)
    lines=['# 固定案例体素优化与U-Net拟合诊断 v1','',
        '日期：2026-10-26（Asia/Shanghai）。本轮为机制诊断，不是正式基线或最终三维精度比较。','',
        f"已完成两个固定train案例×三个预登记臂×200次真实更新，共{integrity['actual_updates']}次；全部CPU双线程串行。未改原objective、物理投影器或冻结模型，未读取GT64/GT128、Pool B或最终test。",'',
        '## 固定设计与可比性','',
        '案例索引0、82，原seed20261011和epoch0 A/B/H。原H参与loss，不能作为独立泛化证据。E是A/B/H之外最小的四个站位ID，只做输出后的评价，未进入目标或网络，也不用于调整参数或选checkpoint。','',
        '三个臂从fresh U-Net同seed、冻结c0/alpha构造开始。两个直接体素臂初态为fresh A/B浓度均值；网络保留原A/B各自输出。直接体素场是每案例一个共享场，绕过网络和条件映射，A/B一致性严格为零。','',
        '| 臂 | 参数化 | AdamW lr | weight decay |','|---|---|---:|---:|',
        '| voxel_sigmoid | 共享262144体素logits→sigmoid | 2e-4 | 1e-4 |',
        '| voxel_scaled | 共享c0×u，每步投影浓度到[0,1] | 0.05 | 1e-4 |',
        '| unet | 原始五层U-Net A/B条件输出 | 2e-4 | 1e-4 |','',
        '均为FP32、完整64³、全部有效K96射线，clip1；cosine T_max300每16更新推进一次。单案例batch1是新诊断协议，不冒充原batch2/166例正式训练。尺度化臂故意使用不同变量尺度与学习率，因此它回答“存在可拟合路径吗”，不能用来作严格等优化器性能排名。','',
        '预登记操作判据：最终observed/cross/H三组relative L1均≤0.9且总loss下降，记为非平凡测量拟合；E≤0.99单独报告。这些判据不是正式GO资格。完整评价节点0/10/50/100/200，最终只判断200，不挑中途最佳。','',
        '## 末态与独立复核','',
        '| 案例 | 臂 | 初始loss→末态loss | 初始H→末态H | 末态E | 测量拟合 | E≤.99 |','|---|---|---|---|---:|---|---|']
    summary=[]
    for index in registration['indices']:
        for arm in registration['arms']:
            result=results[index,arm];first,last=result['metrics'][0],result['metrics'][-1];review=final[index,arm]
            lines.append(f"| {index} | {arm} | {first['loss']:.7g} → {last['loss']:.7g} | {first['measurement']['heldout']:.7g} → {last['measurement']['heldout']:.7g} | {last['extra_E_H']:.7g} | {'通过' if review['nontrivial_measurement_fit'] else '未通过'} | {'通过' if review['independent_E_below099'] else '未通过'} |")
            summary.append(dict(index=index,arm=arm,actual_updates=result['actual_updates'],initial_loss=first['loss'],final_loss=last['loss'],observed=last['measurement']['observed'],cross=last['measurement']['cross'],H=last['measurement']['heldout'],E=last['extra_E_H'],fit=review['nontrivial_measurement_fit'],E_below099=review['independent_E_below099']))
    with (ART/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
    lines+=['','![固定预算学习曲线](artifacts/voxel_probe_v1/probe_curves.png)','',
        f"独立NumPy从全部初末完整场重新计算三线性K96物理积分、测量误差及原objective：{independent['full_fields']}完整场、{independent['physical_integrals']}组积分，主实现与复核一致。两A/B相同的直接场仍逐分支复核；这不是24个独立案例或外部人员复核。最大投影差相对目标L1为{independent['max_projection_error_relative_to_target']:.5g}。",'',
        '## 对失败机制的含义','']
    for index in registration['indices']:
        scaled=final[index,'voxel_scaled'];sigmoid=final[index,'voxel_sigmoid'];network=final[index,'unet']
        lines.append(f"案例{index}：尺度化体素拟合{'通过' if scaled['nontrivial_measurement_fit'] else '未通过'}，直接sigmoid拟合{'通过' if sigmoid['nontrivial_measurement_fit'] else '未通过'}，U-Net拟合{'通过' if network['nontrivial_measurement_fit'] else '未通过'}。")
        if scaled['nontrivial_measurement_fit'] and not network['nontrivial_measurement_fit']:
            lines.append('此例说明原目标与这些射线存在可拟合路径，当前网络/优化条件没能达到；差异同时包含变量尺度、模型参数化、共享条件输出与一致性约束，不能唯一归因U-Net架构。')
        if scaled['nontrivial_measurement_fit'] and not sigmoid['nontrivial_measurement_fit']:
            lines.append('直接场也能因低浓度sigmoid参数化和固定小步长而停滞，支持优化尺度是重要因素；不是“只要去掉网络就一定立即收敛”。')
        if network['nontrivial_measurement_fit']:
            lines.append('原网络在此单案例可以降低测量误差，削弱“该网络/物理梯度完全不可学习”的解释；单案例可记忆一个场，仍不证明跨案例测量条件映射。')
        if scaled['nontrivial_measurement_fit'] and not scaled['independent_E_below099']:
            lines.append('拟合射线成功但E未通过，提示对已见射线的过拟合或稀疏约束非唯一性；不能把训练测量成功当作完整三维恢复。')
        if not any(final[index,a]['nontrivial_measurement_fit'] for a in registration['arms']):
            lines.append('全部臂在本200步预算内未达预定拟合标准，不能据此证明原目标无解；仍需区分步长/参数化和测量约束。保留曲线的实际改善程度，不强行归因。')
    lines+=['', '结合上一轮全166例梯度分解，relative-L1在弥散场上可提供明显幅度下压，而此前关闭relative/consistency均不足以解决Senseiver条件重建。当前对照应与这些证据共同解释，不能从单一成功或失败臂直接断言loss是唯一原因。','',
        '## 工程异常、真实更新与保护','',
        '首次voxel_sigmoid启动在初态checkpoint保存时，未向rng_state提供必须的generator参数；发生在0次优化更新。原run.py、失败日志与失败初态数组目录均保留；新run_v2.py只补充序列化generator，修订SHA在更新前登记。未修改目标、模型、初始化、学习率、预算或阈值，没有更新重放。','',
        '六个科学run各200次更新，逐步journal记录有限非零梯度与真实参数变化并fsync；最终checkpoint的每参数AdamW step=200、scheduler=12、RNG完整。全部CPU，无GPU抢占或外来进程终止。前序147源码、207输入、U-Net及共享诊断封存证据和冻结checkpoint身份再次核验。未合并main、未启动正式训练。','',
        '## 产物','',
        '- 协议：docs/voxel_network_probe_v1_protocol_zh.md；登记与小型证据：docs/reports/artifacts/voxel_probe_v1/。',
        '- 完整体素场、checkpoint、逐更新journal：/scratch/djc/GeoPhy-Set3D/voxel_probe_v1/。',
        '- 执行：scripts/voxel_probe_v1/run_v2.py；已完成目录拒绝覆盖，不应删除完成目录重跑。独立复核：review.py；报告：report.py。',
        '- 本报告只解释固定案例的测量拟合与优化，未使用真实三维GT，不能给出真实场精度结论。','']
    (PROJECT/'docs/reports/voxel_network_probe_v1_zh.md').write_text('\n'.join(lines))
    print('Bounded probe report and curve generated')


if __name__ == '__main__':
    main()
