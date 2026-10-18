#!/usr/bin/env python3
"""Matched formal300 report, fixed historical examples, no candidate reselection."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import evaluate_mrgr_optimization_frozen_v2 as ev


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out).resolve()
    r=ev.verify(out);s=ev.read(out/'summary.json');verification=ev.read(out/'independent_verification.json')
    assert verification['passed']
    names=['frozen_multiscale']+[m['key'] for m in r['models']]
    labels={'frozen_multiscale':'Original MS'}
    for m in r['models']:
        lr=m['config']['optimizer']['lr'];weight=m['config']['loss']['consistency']
        labels[m['key']]=f"{m['candidate']} lr={lr:g} cons={weight:g}"
    def panel(key,condition='Val-A-V4',draws=(0,),metric='relative_l1'):
        return next(p for p in s['panels'] if p['model']==key and p['condition']==condition and p['draws']==list(draws) and p['metric']==metric)
    ordinary={n:panel(n) for n in names};hard={n:panel(n,'clustered-V4',range(5)) for n in names}
    current_baselines=[m['key'] for m in r['models'] if m['candidate'] in ('baseline','one_step_control')
        and m['config']['optimizer']['lr']==2e-4 and m['config']['loss']['consistency']==.05]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    conditions=[('Val-A-V4',[0]),('V3',list(range(5))),('V6',list(range(5))),('spread-V4',[0]),('clustered-V4',list(range(5)))]
    for ax,metric in zip(axes,('relative_l1','relative_l2')):
        width=.8/len(names);x=np.arange(len(conditions))
        for i,n in enumerate(names):
            y=[panel(n,c,d,metric)['mean']*100 for c,d in conditions]
            ax.bar(x+(i-(len(names)-1)/2)*width,y,width,label=labels[n])
        ax.set_xticks(x,[c for c,d in conditions],rotation=20);ax.set_ylabel(f'3D {metric} (%)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    curve_runs=[('frozen_multiscale',ev.previous.RUN)]+[(m['key'],Path(m['run'])) for m in r['models']]
    for key,run in curve_runs:
        rr=ev.rows(run/'metrics.jsonl');x=[a['epoch'] for a in rr]
        standard_loss=[sum(ev.base.selfsup.FROZEN_LOSS[k]*a['train']['loss'][k]
            for k in ('hold','cross','observed','consistency')) for a in rr]
        for ax,y in zip(axes,(standard_loss,
            [a['val']['measurement']['heldout']['relative_l1']*100 for a in rr],[a['learning_rate'] for a in rr])):
            ax.plot(x,y,label=labels[key],linewidth=1,linestyle='--' if key=='frozen_multiscale' else '-')
    for ax,title in zip(axes,('Original-weight training loss','Formal validation H relative L1 (%)','Learning rate')):
        ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set_yscale('log');axes[1].set_yscale('log')
    fig.savefig(out/'formal_curves.png',dpi=170);fig.savefig(out/'formal_curves.svg');plt.close(fig)
    chosen=ev.read(ev.previous.AB_RUNS['A']/'evaluation/frozen_model_test/slice_selection.json')
    source={x['case_id']:x for x in ev.rows(ev.DATA/'source_index.jsonl') if x['split']=='val'}
    paths={'frozen_multiscale':ev.REFERENCE,**{m['key']:out/m['key'] for m in r['models']}}
    manifests={n:{(x['condition'],x['draw'],x['case_id']):x for x in ev.rows(p/'prediction_manifest.jsonl')} for n,p in paths.items()}
    fields={};gt={}
    for selection in chosen:
        c=selection['case_id'];gt[c]=np.load(ev.DATA/source[c]['gt_volume_64_path'],allow_pickle=False).squeeze()
        for cond in ('Val-A-V4','clustered-V4'):
            for n in names:
                x=manifests[n][cond,0,c];path=paths[n]/x['prediction']
                assert ev.sha(path)==x['sha256']
                with np.load(path,allow_pickle=False) as f:fields[c,cond,n]=f['C1'].copy()
    columns=['GT']+names
    fig,axes=plt.subplots(6,len(columns),figsize=(3.2*len(columns),17),constrained_layout=True)
    for i,selection in enumerate(chosen):
        c=selection['case_id'];g=gt[c];iz=np.unravel_index(g.argmax(),g.shape)[2]
        vmax=max([float(g.max())]+[float(fields[c,cond,n].max()) for cond in ('Val-A-V4','clustered-V4') for n in names])
        norm=SymLogNorm(linthresh=float(g.max())*1e-3,vmin=0,vmax=vmax)
        for j,cond in enumerate(('Val-A-V4','clustered-V4')):
            row=2*i+j
            for col,n in enumerate(columns):
                field=g if n=='GT' else fields[c,cond,n]
                im=axes[row,col].imshow(field[:,:,iz].T,origin='lower',extent=(0,.6,0,.6),norm=norm,cmap='magma',interpolation='nearest')
                axes[row,col].set(title=f"{selection['selection']} / {'ordinary' if j==0 else 'cluster'}\n{labels.get(n,n)}",xlabel='x (m)',ylabel='y height (m)')
            fig.colorbar(im,ax=axes[row,:],shrink=.7)
    fig.suptitle('Fixed prior A low/median/high cases; GT-peak z planes; same scale across both layouts within each case')
    fig.savefig(out/'matched_reconstructions.png',dpi=145);plt.close(fig)
    c=chosen[-1]['case_id'];g=gt[c];ix,iy,iz=np.unravel_index(g.argmax(),g.shape)
    xl,xh=max(0,ix-3),min(64,ix+4);zl,zh=max(0,iz-3),min(64,iz+4)
    fig,axes=plt.subplots(2,len(columns),figsize=(3.1*len(columns),6),constrained_layout=True)
    for row,cond in enumerate(('Val-A-V4','clustered-V4')):
        for col,n in enumerate(columns):
            field=g if n=='GT' else fields[c,cond,n]
            im=axes[row,col].imshow(field[xl:xh,iy,zl:zh].T,origin='lower',vmin=0,vmax=float(g.max()),cmap='magma',interpolation='nearest',
                extent=(xl*.6/63,(xh-1)*.6/63,zl*.6/63,(zh-1)*.6/63))
            axes[row,col].set(title=f"{'ordinary' if row==0 else 'cluster'}\n{labels.get(n,n)}",xlabel='x (m)',ylabel='z (m)')
        fig.colorbar(im,ax=axes[row,:],shrink=.75)
    fig.suptitle(f'Fixed prior high-error case: identical GT-peak neighborhood at y={iy*.6/63:.4f}m; GT-maximum linear scale')
    fig.savefig(out/'source_peak_detail.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for col,selection in enumerate(chosen):
        c=selection['case_id'];g=gt[c]
        for row,cond in enumerate(('Val-A-V4','clustered-V4')):
            for n in columns:
                field=g if n=='GT' else fields[c,cond,n]
                axes[row,col].plot(np.arange(64)*.6/63,field.sum(axis=(0,2))/g.sum(),label=labels.get(n,n))
            axes[row,col].set(title=f"{selection['selection']} / {'ordinary' if row==0 else 'cluster'}",xlabel='y height (m)',ylabel='Nodal mass / total GT');axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=7)
    fig.savefig(out/'height_profiles.png',dpi=160);plt.close(fig)
    rel=out.relative_to(ev.ROOT);link=f'../../{rel}'
    lines=['# MRGR自主优化：300轮训练与冻结三维重建测试','',
        f"完成时间：{s['completed_at']}。本报告只解释已封存候选的开发验证结果，不据三维GT重新选轮或修改候选。",'',
        '## 1. 正式训练','', '| 模型 | 参数 | 初始LR | 一致性权重 | 最优轮 | 最低训练验证H | 总训练时间 |', '|---|---:|---:|---:|---:|---:|---:|']
    for m in r['models']:
        t=ev.read(Path(m['run'])/'summary.json');conf=m['config']
        lines.append(f"| {m['key']} | {m['parameters']:,} | {conf['optimizer']['lr']:g} | {conf['loss']['consistency']:g} | {m['best_epoch']} | {t['best_val_H']:.4%} | {t['elapsed_seconds']/3600:.3f}h |")
    lines+=['',f'![正式训练曲线]({link}/formal_curves.png)','',
        '全部全新初始化、seed20261011、300轮/24,900更新；AdamW/余弦T_max300。训练只读取测量，按最低普通验证H选轮。日志H为A4/B4/H4；冻结重建面板的额外H4口径不同。','',
        '## 2. 三维L1/L2与尾部','', '| 布局 | 布局数 | 模型 | 三维L1 | 三维L2 | L1 P90 | 对旧多尺度L1降低 |', '|---|---:|---|---:|---:|---:|---:|']
    for cond,draws in conditions:
        for n in names:
            a=panel(n,cond,draws);b=panel(n,cond,draws,'relative_l2')
            reduction=0 if n=='frozen_multiscale' else a['comparisons']['frozen_multiscale']['relative_error_reduction']
            lines.append(f"| {cond} | {len(draws)} | {labels[n]} | {a['mean']:.4%} | {b['mean']:.4%} | {a['p90']:.4%} | {reduction:+.2%} |")
    lines+=['',f'![三维误差对比]({link}/comparison.png)','',
        '五布局先逐案例平均，再对36案例汇总，不能视作180个独立样本。完整JSON还保留draw0、逐例最大误差和所有候选间配对对照。','',
        '| 候选 | 参考 | 普通ΔL1（百分点） | 95%区间（百分点） | 胜/平/负 | 普通P90变化 | 聚簇五布局变化 | ≥3%实用收益与安全门槛 |',
        '|---|---|---:|---|---|---:|---:|---|']
    decisions=[]
    for n in names[1:]:
        references=['frozen_multiscale']+[x for x in current_baselines if x!=n]
        for ref in references:
            a,b=ordinary[n],ordinary[ref];pair=a['comparisons'][ref];ci=np.array(pair['ci95'])*100
            p90=a['p90']/b['p90']-1;cluster=hard[n]['mean']/hard[ref]['mean']-1
            passed=pair['relative_error_reduction']>=.03 and p90<=.02 and cluster<=.02
            decisions.append(dict(model=n,reference=ref,practical_gate=passed,ordinary_reduction=pair['relative_error_reduction'],p90_change=p90,cluster_change=cluster))
            lines.append(f"| {labels[n]} | {labels[ref]} | {pair['mean_delta']*100:+.4f} | [{ci[0]:+.4f},{ci[1]:+.4f}] | {pair['wins']}/{pair['ties']}/{pair['losses']} | {p90:+.2%} | {cluster:+.2%} | {'通过' if passed else '未通过'} |")
    ev.write(out/'practical_assessment.json',decisions)
    lines+=['', '该门槛在本批GT评分前登记。实用门槛与统计区间分别解释，案例bootstrap为36例/20,000次/seed20261017，不覆盖训练种子方差，探索性比较未作多重校正。新执行基线用于观察执行漂移；不能只选择较差基线宣称收益。','',
        '## 3. 峰值、质心与形态','', '| 布局（draw0） | 模型 | 阶段 | 峰值比 | 峰幅绝对比误差 | GT峰处浓度比 | 质心mm | Dice50 | 背景占比 |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for x in s['morphology']:
        v=x['mean'];lines.append(f"| {x['condition']} | {labels[x['model']]} | {x['stage']} | {v['peak_ratio']:.3f} | {v['absolute_peak_ratio_error']:.3f} | {v['gt_peak_location_ratio']:.3f} | {v['centroid_distance_mm']:.3f} | {v['dice_0.5']:.3f} | {v['background_predicted_fraction']:.2%} |")
    lines+=['', '质心和背景比例采用与既有形态报告相同的等权端点节点；volume字段另存梯形物理体积分。Dice50共同阈值为该例GT峰值50%。峰值均值可能正负抵消，因此同时报告峰幅绝对误差。GT峰处浓度比是预测在GT峰坐标处的浓度除以GT峰值，并非最大峰位置距离；不能仅凭该比值声称峰位置已恢复。C0/C1阶段变化是联合训练轨迹诊断，不等于重训消融。','',
        '## 4. 固定案例重建图','', '严格沿用之前A普通布局低/中/高误差三例，不按本次结果挑例。每例使用GT峰值z平面，普通和聚簇共享色标。图像只说明对应切面，不能代替三维统计或证明拓扑。','',
        f'![同切面重建图]({link}/matched_reconstructions.png)','',f'![近源局部放大]({link}/source_peak_detail.png)','',f'![完整高度边缘分布]({link}/height_profiles.png)','',
        '局部放大固定为原高误差例、GT峰附近7×7邻域和同一y平面，色标上限为GT峰值；超出范围会饱和，需结合全域图和定量峰指标。','',
        '## 5. 观测与留出投影','', '| 模型 | 布局 | draw | 输入L1 | 额外H4 L1 | 输入L2 | 额外H4 L2 |', '|---|---|---:|---:|---:|---:|---:|']
    for n in names:
        for cond,d in [('Val-A-V4',0)]+[('clustered-V4',j) for j in range(5)]:
            rr=[x for x in manifests[n].values() if x['condition']==cond and x['draw']==d]
            values=[np.mean([x['measurement'][split][metric] for x in rr])
                for metric in ('relative_l1','relative_l2') for split in ('observed','heldout')]
            lines.append(f"| {labels[n]} | {cond} | {d} | "+' | '.join(f'{v:.4%}' for v in values)+' |')
    lines+=['', '投影误差与三维误差分别报告，不用投影收益代替三维收益。','',
        '## 6. 审计与限制','',
        '每模型17布局×36案例=612次重建，C0/C1合计1,224场。所有正式候选、权重和整批预测先封存，再在独立进程读取36例验证GT64；模型输入剔除H测量。没有GT128、最终35例test或评分后训练。开发验证集曾用于历史诊断，不能称盲测。','',
        f"独立核验通过：重算{verification['fields_independently_recomputed']:,}场L1/L2、{verification['panels_verified']}面板与配对区间、固定例形态指标，并核对300轮学习率/损失/选轮、源码/数据/权重/预测hash及GT读取审计。",'',
        f'[完整指标]({link}/summary.json)、[冻结协议]({link}/registration.json)、[批量封存]({link}/batch_predictions_sealed.json)、[独立核验]({link}/independent_verification.json)、[实用门槛判断]({link}/practical_assessment.json)。','']
    report=ev.ROOT/'docs/reports/mrgr_autonomous_optimization_v2_formal_test_zh.md'
    report.write_text('\n'.join(lines));print(report)


if __name__=='__main__':main()
