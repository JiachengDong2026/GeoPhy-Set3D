"""Final synthesis, including the separately registered last path-dependence check."""
from probe_context import *
import report_all
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    report_all.main()
    assert read(ART/'refinement_independent_review.json')['passed']
    r=read(ART/'case0_original_refinement.json')
    d=read(ART/'case0_teacher_probe.json')
    fresh=read(ART/'case0_unet.json')
    network=read(ART/'case0_unet_lbfgs.json')
    f=r['final'];i=r['initial']
    passed=r['fitted_ABH'] and r['E_below099']
    path=PROJECT/'docs/reports/voxel_network_probe_v1_zh.md'
    text=path.read_text()
    lead=('**核心诊断：原物理目标有可拟合解，同一U-Net也能学习空间结构；本例的关键障碍是从弥散低浓度初态靠稀疏物理监督找到结构的优化过程。自生成可拟合场预热后，原目标能够继续降低误差。**' if passed else '**核心诊断：原物理目标有可拟合解，同一U-Net在稠密自生成目标下能学习空间结构；原目标继续优化的末态仍须结合下表判断，不能宣布修复。**')
    text=text.replace('日期：2026-10-26（Asia/Shanghai）。',lead+'\n\n日期：2026-10-26（Asia/Shanghai）。',1)
    text=text.replace('最后只登记一个稠密目标诊断','另行登记一个稠密目标诊断')
    text=text.replace('独立稠密目标Adam200更新；','独立稠密目标Adam200更新；最后原目标续训Adam100更新（Adam更新总计1500）；')
    text=text.replace('独立复核合计40个完整A/B场、160组K96积分','独立复核合计44个完整A/B场、176组K96积分').replace('并非40个独立案例','并非44个独立案例')
    section=['## 最后检查：空间结构预热后，原目标是否仍可继续优化','',
        '在稠密目标对照成功并完成独立复核后，单独预登记100步原目标续训：只加载本轮自己的200步自生成场拟合权重，重置AdamW，结构、输入、物理投影、原objective及裁剪均保持。此阶段不读稠密目标数组，E只评价，不能选择步数或checkpoint。没有加载主模型、GT监督模型或外来权重。末态固定为100步。','',
        '| train0条件 | 优化预算 | 原总loss | observed | cross | H（参与目标） | E（仅评价） |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,budget,m in [('fresh原目标AdamW','200更新',fresh['metrics'][-1]),('fresh原目标L-BFGS','20接受/42求值',network['final']),('自生成场稠密目标预热','200更新',d['final']),('预热后原目标续训','100更新',f)]:
        section.append(f"| {name} | {budget} | {m['loss']:.7g} | {m['measurement']['observed']:.7g} | {m['measurement']['cross']:.7g} | {m['measurement']['heldout']:.7g} | {m['extra_E_H']:.7g} |")
    section+=['',f"续训原loss：{i['loss']:.7g} → {f['loss']:.7g}；H：{i['measurement']['heldout']:.7g} → {f['measurement']['heldout']:.7g}；E：{i['extra_E_H']:.7g} → {f['extra_E_H']:.7g}。最终A/B/H拟合判据{'通过' if r['fitted_ABH'] else '未通过'}，E≤.99判据{'通过' if r['E_below099'] else '未通过'}。",'']
    if passed:
        section+=['原物理目标可以保持并进一步改善已学出的结构，排除了“本例原目标必然让所有网络输出塌缩”的强解释。结合fresh网络原目标失败、稠密自生成目标成功，证据支持初始空间定位/稀疏物理梯度的空间归因/优化路径为重要因素；尚不能把这些相互作用拆成唯一根因，也不能将一个U-Net案例外推至DeepONet/FNO等全部架构。','']
    else:
        section+=['末态没有同时满足两项判据；原目标续训仍存在问题。稠密目标的成功只证明同一网络可学习该自生成结构，不能据此宣布原训练协议得到修复。','']
    section+=['该两阶段过程累计300次AdamW更新，额外依赖案例专属体素求解及稠密目标；fresh对照为200次更新，不能作等预算性能排名。它回答可行路径和失败机制，不是新的正式基线结果。E是同案例未见射线，不是跨案例验证；全程没有GT，测量误差降低不能证明三维真值恢复正确。','',
        '![不同训练路径的诊断曲线](artifacts/voxel_probe_v1/diagnostic_path.png)','']
    text=text.replace('## 最终证据边界与后续处理','\n'.join(section)+'\n## 最终证据边界与后续处理')
    text=text.replace('下一步应围绕已定位的网络优化尺度/条件读出做一项明确假设的独立适配设计','下一步应在独立协议中验证“训练案例自生成可行场预热→原目标微调”能否跨案例推广，并统一预算与数据使用后再讨论正式适配')
    path.write_text(text)
    with (ART/'all_probe_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
    rows.append(dict(index=0,arm='original_refinement',actual_updates=100,initial_loss=i['loss'],final_loss=f['loss'],observed=f['measurement']['observed'],cross=f['measurement']['cross'],H=f['measurement']['heldout'],E=f['extra_E_H'],fit=r['fitted_ABH'],E_below099=r['E_below099'],kind='original_after_self_generated',accepted_iterations=100,function_evaluations=''))
    with (ART/'all_probe_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,key,title in zip(axes,['heldout','E'],['H rays (part of physical loss)','E rays (evaluation only)']):
        metric=lambda m:m['extra_E_H'] if key=='E' else m['measurement'][key]
        ax.plot([m['step'] for m in fresh['metrics']],[metric(m) for m in fresh['metrics']],'o-',label='Fresh, original loss')
        ax.plot([m['step'] for m in d['metrics']],[metric(m) for m in d['metrics']],'o-',label='Dense self-generated target')
        ax.plot([200+m['step'] for m in r['metrics']],[metric(m) for m in r['metrics']],'o-',label='Then original loss')
        ax.axvline(200,color='grey',ls=':',lw=1);ax.axhline(1,color='grey',lw=.5)
        ax.set(title=title,xlabel='Actual AdamW updates (different training paths)',ylabel='Relative L1')
        ax.legend(fontsize=7)
    fig.savefig(ART/'diagnostic_path.png',dpi=180);plt.close(fig)
    print('Final report includes last refinement; 44 fields / 176 independent integrations')


if __name__=='__main__':main()
