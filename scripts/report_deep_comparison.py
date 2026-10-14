#!/usr/bin/env python3
"""All-prespecified Chinese comparison report, figures, scalar/integrity delivery."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from geophy_set3d.training.deep_comparison import load_comparison_config,sha256,verify_registration
from geophy_set3d.training.utils import write_json
from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import DiagnosticReadGuard,GROUPS,METRICS
from diagnose_clustered_one_swap import snapshot


def table(rows,headers):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in r)+' |' for r in rows])


def plot(c,out,case,stats,training):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(figdir/(name+'.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,metric in zip(axs,['relative_l1','relative_l2']):
        for name in ['Ours','DeepSets']:
            means=[case[case.group==g][name+'_'+metric].mean() for g in GROUPS];ax.plot(GROUPS,means,'o-',label=name)
        ax.set(xlabel='Five fixed V4 geometry groups',ylabel='3D '+metric.replace('_',' '),title='36 cases; five layouts averaged within each case');ax.legend()
    save(fig,'group_error_curves')
    primary=stats[stats.metric=='relative_l1'].set_index('group').loc[GROUPS]
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    # Draw CI lines explicitly: percentile interval need not bracket the point estimate.
    for i,g in enumerate(GROUPS):
        r=primary.loc[g];ax.plot([r.ci_low,r.ci_high],[i,i],lw=3,color='tab:blue');ax.plot(r.mean_delta,i,'o',color='black')
    ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(5),yticklabels=GROUPS,xlabel='Ours - DeepSets 3D relative L1 (negative favors Ours)',title='Paired case bootstrap: 10,000 resamples; pointwise 95% CI');save(fig,'paired_delta_ci')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    history=[json.loads(l) for l in Path('/data/djc/GeoPhy-Set3D/outputs/logs/selfsup_oari_mrgr_v1/metrics.jsonl').read_text().splitlines()]
    for name,records,style in [('DeepSets',training,'-'),('Ours historical',history,'--')]:
        epochs=[r['epoch']+1 for r in records]
        for split in ['train','val']:
            axs[0,0].plot(epochs,[r[split]['measurement']['heldout']['relative_l1'] for r in records],style,label=name+' '+split)
            axs[0,1].plot(epochs,[r[split]['loss']['total'] for r in records],style,label=name+' '+split)
        axs[1,0].plot(epochs,[r['val']['measurement']['observed']['relative_l1'] for r in records],style,label=name+' observed')
        axs[1,0].plot(epochs,[r['val']['measurement']['cross']['relative_l1'] for r in records],style,label=name+' cross')
    for key in ['hold','cross','observed','consistency']:
        axs[1,1].plot([r['epoch']+1 for r in training],[r['val']['loss'][key] for r in training],label='DeepSets '+key)
    for ax,title in zip(axs.flat,['Held-out measurement relative L1','Unchanged self-supervised objective','Natural Val observed/cross relative L1','DeepSets natural Val loss components']):
        ax.set(xlabel='Epoch (1-based)',ylabel='Value',yscale='log',title=title);ax.legend(fontsize=8)
    save(fig,'training_measurement_curves')
    matrix=case.pivot(index='case_index',columns='group',values='delta_relative_l1').loc[range(36),GROUPS].to_numpy();limit=max(float(np.abs(matrix).max()),1e-12)
    fig,ax=plt.subplots(figsize=(7,10),layout='constrained');im=ax.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
    ax.set(xticks=range(5),xticklabels=GROUPS,yticks=range(36),yticklabels=range(36),ylabel='Original Val case index',title='Ours - DeepSets, case/group mean 3D L1');fig.colorbar(im,ax=ax);save(fig,'case_group_delta')
    source=[json.loads(l) for l in (Path(c['dataset']['root'])/'source_index.jsonl').read_text().splitlines()];source=[r for r in source if r['split']=='val']
    old=Path(c['comparison']['reference_predictions'])
    # Reuse only the nine prespecified archived Ours arrays, without any forward.
    for i in c['comparison']['representative_cases']:
        for g in c['comparison']['representative_groups']:
            token=f'case{i}_V4-{g}-d0';r=json.loads((old/'prediction_rows'/f'normal_{token}.json').read_text())
            assert sha256(r['prediction_file'])==r['prediction_sha256']
            target=out/'case_images'/f'ours_{token}.npz'
            if not target.exists():shutil.copy2(r['prediction_file'],target)
            assert sha256(target)==r['prediction_sha256']
    with DiagnosticReadGuard(c['dataset']['root'],[],out,'gt') as guard:
        for i in c['comparison']['representative_cases']:
            gt=np.load(Path(c['dataset']['root'])/source[i]['gt_volume_64_path'],allow_pickle=False)
            data=[]
            for g in c['comparison']['representative_groups']:
                token=f'case{i}_V4-{g}-d0';row=json.loads((out/'prediction_rows'/f'{token}.json').read_text());assert sha256(row['prediction_file'])==row['prediction_sha256']
                orow=json.loads((old/'prediction_rows'/f'normal_{token}.json').read_text());reused_ours=out/'case_images'/f'ours_{token}.npz';assert sha256(reused_ours)==orow['prediction_sha256']
                with np.load(row['prediction_file'],allow_pickle=False) as f:base=f['prediction']
                with np.load(reused_ours,allow_pickle=False) as f:ours=f['C1']
                data.append((g,ours,base))
            for view in ['slice','ymean']:
                def project(v):return v[:,c['comparison']['slice_y_index'],:].T if view=='slice' else v.mean(1).T
                maximum=max(float(project(gt).max()),*[float(project(v).max()) for _,a,b in data for v in [a,b]],1e-12)
                error_max=max(*[float(project(np.abs(v-gt)).max()) for _,a,b in data for v in [a,b]],1e-12)
                fig,axes=plt.subplots(3,5,figsize=(16,10),layout='constrained')
                for row,(g,a,b) in enumerate(data):
                    for col,v in enumerate([gt,a,b,np.abs(a-gt),np.abs(b-gt)]):
                        ax=axes[row,col];im=ax.imshow(project(v),origin='lower',extent=(0,.6,0,.6),vmin=0,vmax=maximum if col<3 else error_max,cmap='viridis' if col<3 else 'magma')
                        if col<3:con=im
                        else:err=im
                        ax.set(xlabel='x (m)',ylabel='z (m)',aspect='equal')
                        if row==0:ax.set_title(['GT64 posthoc','Ours C1','DeepSets','Ours absolute error','DeepSets absolute error'][col])
                        if col==0:ax.text(.02,.98,g,transform=ax.transAxes,va='top',color='white',bbox=dict(facecolor='black',alpha=.5))
                fig.colorbar(con,ax=axes[:,:3],shrink=.7,label='CH4 mass fraction, shared per case')
                fig.colorbar(err,ax=axes[:,3:],shrink=.7,label='Absolute mass fraction error, shared per case')
                title=f'y index32, height y={32*.6/63:.5f} m' if view=='slice' else 'mean along height y'
                fig.suptitle(f'Fixed original Val case {i}; draw0; {title}\nXYZ; y is height; groups/cases/display rules preregistered')
                save(fig,f'fixed_case{i}_{view}')
        write_json(out/'figure_gt_access.json',guard.summary())


def integrity(c,out):
    p=verify_registration(c);before=json.loads((out/'dataset_before.json').read_text());assert snapshot(c['dataset']['root'])==before
    history=json.loads((out/'history_before.json').read_text());assert all(snapshot(path)==value for path,value in history.items())
    for path,h in p['protected_source_sha256'].items():
        if path!='src/geophy_set3d/models/__init__.py':assert sha256(REPO/path)==h,path
    # A prior frozen metadata snapshot remains a read-free protection of GT128 contents.
    raw=json.loads((Path(c['comparison']['reference_predictions'])/'inventory_before.json').read_text())['gt128_metadata']
    for path,meta in raw.items():
        st=Path(path).stat();assert dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)==meta
    reference=p['reference_checkpoint'];assert sha256(reference['checkpoint'])==reference['sha256']
    rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];assert len(rows)==300 and [r['epoch'] for r in rows]==list(range(300))
    assert all(r['optimizer_steps']==83*(r['epoch']+1) and r['optimizer_steps_this_epoch']==83 for r in rows)
    events=[json.loads(l) for l in (out/'inference_events.jsonl').read_text().splitlines()];counts=collections.Counter(r['event'] for r in events);assert counts['forward_started']==counts['saved']==900
    audits={n:json.loads((out/n).read_text()) for n in ['training_access_audit.json','inference_access.json','gt_access.json','figure_gt_access.json']}
    assert not any(audits['training_access_audit.json'][k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
    assert all(audits[k]['forbidden_attempts']==0 for k in ['inference_access.json','gt_access.json','figure_gt_access.json'])
    value=dict(passed=True,dataset_metadata_unchanged=True,historical_metadata_unchanged=True,protected_source_hashes_unchanged_except_additive_registry=True,
               GT128_metadata_unchanged_contents_not_read=True,reference_checkpoint_unchanged=True,epochs=300,optimizer_steps=24900,
               historical_station_digests_match=all(r['historical_station_digest_equal'] for r in rows),forward_counts=dict(main_deepsets=900,selector_deepsets=36,timing_deepsets=360,timing_ours=360,formal_training_batch2_calls=49800,formal_validation_batch2_calls=10800),
               smoke_successful_model_forwards=7,smoke_optimizer_steps=1,smoke_failed_attempt_optimizer_steps=0,smoke_failed_attempt_model_forwards=2,selector_case_volumes=72,formal_smoke_initial_state_equal=True,
               regression='259 pretraining tests plus 3 comparison-statistics tests',boundaries=audits)
    write_json(out/'integrity.json',value);return value


def report(c,out,case,stats,training,tails,contrasts,integ):
    observer=json.loads((out/'training_process_observer.json').read_text());t=json.loads((out/'training_complete.json').read_text());freeze=json.loads((out/'frozen_best.json').read_text());timing=json.loads((out/'timing.json').read_text());selector=json.loads((out/'selector_replay.json').read_text())
    primary=stats[stats.metric=='relative_l1'].set_index('group').loc[GROUPS]
    bestgroups=[g for g in GROUPS if primary.loc[g].mean_delta<0];worse=[g for g in GROUPS if primary.loc[g].mean_delta>0]
    definite=[g for g in GROUPS if primary.loc[g].ci_high<0];uncertain=[g for g in GROUPS if primary.loc[g].ci_low<=0<=primary.loc[g].ci_high]
    ref='/data/djc/GeoPhy-Set3D/outputs/logs/selfsup_oari_mrgr_v1/training_summary.json';reftrain=json.loads(Path(ref).read_text())
    metricrows=[]
    for g in GROUPS:
        r=primary.loc[g];metricrows.append([g,f'{r.ours_mean:.6f}',f'{r.deepsets_mean:.6f}',f'{r.mean_delta:.6f}',f'{r.median_delta:.6f}',f'[{r.ci_low:.6f}, {r.ci_high:.6f}]',f'{int(r.ours_wins)}/{int(r.deepsets_wins)}/{int(r.ties)}',f'{100*r.ours_relative_error_reduction:.2f}%'])
    recent=[r['val']['measurement']['heldout']['relative_l1'] for r in training[-30:]];previous=[r['val']['measurement']['heldout']['relative_l1'] for r in training[-60:-30]]
    bestoutput=selector['validation']['output'];notcollapsed=(bestoutput['std']>1e-6 and bestoutput['mean']>1e-6 and bestoutput['fraction_near_one']<.01 and bestoutput['case_to_case_mean_absolute_difference']>1e-6)
    cost=[]
    for name in ['Ours','DeepSets']:
        ts=[r for r in timing['rows'] if r['model']==name];seconds=np.concatenate([r['seconds'] for r in ts]);idn=timing['identities'][name]
        cost.append([name,idn['parameter_count'],f"{idn['fp32_parameter_bytes']/2**20:.3f}",f"{timing['checkpoint_bytes'][name]/2**20:.3f}",f'{np.median(seconds)*1000:.3f}',f'{np.quantile(seconds,.9)*1000:.3f}',f"{max(r['peak_allocated_excluding_other_model_bytes'] for r in ts)/2**30:.3f}"])
    secondary=[]
    for m in ['relative_l2','observed_relative_l1','heldout_relative_l1','peak_value_absolute_error','integrated_concentration_relative_error']:
        for g in GROUPS:
            r=stats[(stats.group==g)&(stats.metric==m)].iloc[0];secondary.append([g,m,f'{r.ours_mean:.7g}',f'{r.deepsets_mean:.7g}',f'{r.mean_delta:.7g}',f'[{r.ci_low:.7g}, {r.ci_high:.7g}]',f'{int(r.ours_wins)}/{int(r.deepsets_wins)}/{int(r.ties)}'])
    tailrows=[]
    for m in ['relative_l1','relative_l2']:
        for g in GROUPS:
            for name in ['Ours','DeepSets']:
                r=tails[(tails.group==g)&(tails.metric==m)&(tails.model==name)].iloc[0];tailrows.append([g,m,name,f'{r.p90:.6f}',f'{r.p95:.6f}',f'{r.maximum:.6f}',int(r.worst_case_index)])
    cr=[]
    for r in contrasts:
        cr.append([r['comparison'],r['metric'],f"{r.get('ours_mean',r.get('ours')):.6f}",f"{r.get('deepsets_mean',r.get('deepsets')):.6f}",f"{r.get('mean_delta',r.get('delta')):.6f}",f"[{r['ci_low']:.6f}, {r['ci_high']:.6f}]"])
    last=training[-1];convergence=f"best在第{t['best_epoch']}轮；Val selector末30轮均值{np.mean(recent):.6f}，前30轮均值{np.mean(previous):.6f}，末轮{last['val']['measurement']['heldout']['relative_l1']:.6f}。"
    convergence+=('best位于末30轮且近期均值仍下降，有预算末期继续改善迹象；' if t['best_epoch']>270 and np.mean(recent)<np.mean(previous) else '曲线与末期变动仅用于判断预算内行为，不能证明全局收敛；')+'按300轮停止，不延长训练。'
    summary=f"Ours主L1均值更低的组：{', '.join(bestgroups) or '无'}；更高的组：{', '.join(worse) or '无'}。逐项95%CI完全低于0：{', '.join(definite) or '无'}；跨0：{', '.join(uncertain) or '无'}。"
    text=f'''# GeoLift-DeepSets-Sum 第一轮自监督同条件对照

**实际完成300轮、24,900次正式更新；原SelfSup-MRGR v1保持冻结。** {summary}

这是单seed、共享encoder/lifting/decoder的基础集合聚合对照，不代表最终强基线集合。允许基线胜出，未按本轮GT调整结构、布局、loss或训练预算。下一阶段优先任务：**按相同信息、标签、选模和冻结评价协议接入一个Senseiver任务适配基线，先独立预注册并验证接口**；当前DeepSets数值/协议检查通过，没有发现需要先修复的明确工程阻塞。本轮只建议，不自动实现或训练Senseiver。

协议：[训练前中文协议](../deepsets_selfsup_v1_protocol_zh.md)，登记commit `719d159`；训练实现commit `77c595a`。来源、配置、冻结清单和源码hash见[protocol.json](artifacts/deepsets_selfsup_v1/protocol.json)、[training_source_manifest.json](artifacts/deepsets_selfsup_v1/training_source_manifest.json)、[evaluation_source_manifest.json](artifacts/deepsets_selfsup_v1/evaluation_source_manifest.json)。结果与cost全部来自实测。

## 1. 实现、数值与完整训练

GeoLift-DeepSets-Sum（306,625参数）：原SharedEncoder2D接收每站完整64×64 measurement、valid-ray mask、path；原GeometryLifting保留真实站位与射线方向。在32³空间点，以真实view×visibility进行masked sum/4；固定V_ref=4，不除以实际V；原ResidualDecoder3D接sigmoid输出[B,1,64,64,64]质量分数。没有station编号embedding、OARI、GCA、MRGR或额外loss。invalid ray的测量/path置零；padding不会进入求和；全不可见聚合特征为0，decoder可非零。

Deep Sets原论文：Zaheer et al., https://arxiv.org/abs/1703.06114 。已读取结构定理及点云分类适配：共享元素映射+求和+后置映射提供不变形式；论文可数域定理与不可数固定集合大小的论证有适用条件。本任务元素是站位的二维场及几何，φ含空间lifting，ρ为空间卷积decoder，mask/固定尺度为任务适配。**不是原论文三维重建官方实现**，不将有限32通道表示称为普适唯一三维恢复保证。[来源与摘录](artifacts/deepsets_selfsup_v1/paper_source.json)。

独立临时模型A5/B5/H2、batch2真实forward/backward/AdamW一步通过，shape/range/有限梯度/置换/padding/全invalid及H排除均通过；置换最大差1.6391e−6，padding差0，峰值2.495GiB。首个检查在step前因诊断梯度字典键笔误终止；更正键后重启fresh smoke，没有科学设定更改，原日志保留。正式模型重新seed随机初始化，初态SHA=`56e43c54fe633c49c280075fd5bb5247b993a779e48111433fcec0636445eee2`，不读取smoke/监督/旧训练权重。

完成166 cases×300 epochs，每轮83次step，总24,900；FP32、batch2/accum1、seed20261011、AdamW lr2e−4/wd1e−4、clip1、cosine T_max300，无early stop和搜索。DeepSets训练+Val逐epoch墙钟合计{t['total_epoch_seconds']/3600:.3f}h，平均{t['mean_epoch_seconds']:.3f}s/epoch，完整训练进程墙钟{observer['wall_seconds']/3600:.3f}h（观察误差至多1秒），训练峰值{t['peak_gpu_allocated_bytes']/2**30:.3f}GiB；共享RTX4090，非独占benchmark。AdamW真实step计数及clip后norm已记录，没有NaN/Inf中断。

best第{t['best_epoch']}轮，natural Val平均held-out relative L1={t['best_val_heldout_relative_l1']:.8f}。best SHA256=`{freeze['sha256']}`，路径`{freeze['checkpoint']}`。完整预算结束后冻结，独立measurement-only selector重放={selector['replayed']:.8f}，容差abs1e−8+rel1e−5通过。

best Val输出min/max/mean/std={bestoutput['min']:.6g}/{bestoutput['max']:.6g}/{bestoutput['mean']:.6g}/{bestoutput['std']:.6g}，近1占比={bestoutput['fraction_near_one']:.6g}，case间MA差={bestoutput['case_to_case_mean_absolute_difference']:.6g}；按预定数值规则未塌缩={notcollapsed}。这些仅为数值行为证据，不把低measurement误差当作唯一正确场证明。

## 2. 同信息、标签、loss与选模身份

两方法均166 train/36 Val、Pool A、自然A3–5/B3–5/H2两两不交，Val固定A4/B4/H4；H不进network。归一化32284.894580078173 ppm·m，原P_diff/K96/all valid rays；完整调用原`training.selfsup.objective/run_epoch/selection_metric`，含内部SmoothL1 beta=.01、relative权重.1/epsilon1e−6与体积一致性epsilon1e−6，外权重1/.5/.1/.05，未复制或改写loss数值路径。

实际Val IDs与原v1一致；全部300轮case/station digest与原历史相同={t['all_historical_station_digests_equal']}。新run额外保存有序case/batch/V/station清单与digest，检查每轮166个case恰好一次、输入逐条符合stateless sampler。历史原v1缺少逐步参数/optimizer与完整有序batch trace，同seed不同结构也不等于相同初始权重；本比较为**同协议、同信息/标签/预算的历史冻结参考比较**，不能包装为严格随机化配对训练实验或逐步数值重放。

训练/selection GT64、GT128、Pool B/test与foreign checkpoint禁止尝试均0。GT仅在300轮及best/全部预测冻结后的独立saved-output evaluator读取36个Val GT64；GT128和最终test没有读取。第一GT边界与预测hash见[first_posthoc_gt_boundary.json](artifacts/deepsets_selfsup_v1/first_posthoc_gt_boundary.json)。[访问及完整性审计](artifacts/deepsets_selfsup_v1/integrity.json)。

统一接口由新model注册、独立严格config验证、模型身份、GT-free数据/objective/selector、来源/冻结、固定清单预测和配对统计构成；旧v1验证仍锁定MRGR，旧B0与所有数值源码保持原行为。后续基线可复用信息/评价合同，需独立模型/训练协议校验，不在本轮扩展白名单或伪造MRGR分支。

## 3. G1–G5主终点：同输入3D relative L1

原900条V4布局全部复用，input IDs及顺序、measurement/path/geometry、valid/view/lifting visibility masks和共同H4身份匹配。Ours采用最新normal C1已验证完整标量，原checkpoint SHA=`1573cd84dffeabeb58fbbfb1afec9c9dc865a0a674e21229711bb25b22e3c026`；此前900行十指标复现通过，本轮未重跑900个Ours结果。DeepSets900次独立冻结推理，先保存预测/identity hash再评价GT64。

先同case/group平均五layout，再36case配对bootstrap10,000次、seed2026101801；Δ=Ours−DeepSets，负值有利Ours。下表CI逐项95%，**未校正同时覆盖率**；本轮不计算p值、不作多重检验显著性断言。胜/负/平阈值|Δ|≤1e−8。900条、ray、voxel不作为独立样本。

{table(metricrows,['组','Ours','DeepSets','Δ均值','Δ中位数','95%CI','Ours胜/DeepSets胜/平','Ours相对降低'])}

![五档误差](artifacts/deepsets_selfsup_v1/figures/group_error_curves.png)

![配对差CI](artifacts/deepsets_selfsup_v1/figures/paired_delta_ci.png)

![case/group配对差](artifacts/deepsets_selfsup_v1/figures/case_group_delta.png)

{summary} 每组对预先固定的同输入进行比较，不因结果删除组或更换布局。详细[900配对行](artifacts/deepsets_selfsup_v1/paired_900.csv)、[180 case/group均值](artifacts/deepsets_selfsup_v1/case_group_180.csv)。

## 4. 普通精度与困难几何、退化幅度

{table(cr,['统计','指标','Ours','DeepSets','Δ','95%CI'])}

G5绝对误差见主表；G1–G4宏平均以每case四组等权均值计算。G5−G1是每case差值后平均；G5/G1是**组总体均值之比**，bootstrap每次重算两组均值，不是case比值均值。不能只凭退化比值小判断更好，要结合普通精度与G5绝对误差。固定旧25布局G5与历史clustered V4、增强确认G5不是同一评价身份；本报告不混用。

## 5. 辅助指标是否一致？

{table(secondary,['组','辅助指标','Ours','DeepSets','Δ','逐项95%CI','Ours胜/DeepSets胜/平'])}

辅助L2、input/H4、peak和integral的方向/CI完整保留，即使与主L1不一致也不筛掉。MAE/RMSE及峰值为无量纲CH4质量分数尺度；积分指标是冻结voxel sum相对proxy，不能称真实总质量。全部50项组×指标统计见[paired_statistics.csv](artifacts/deepsets_selfsup_v1/paired_statistics.csv)，未对辅助结果作经校正显著性宣称。测量拟合与三维重建分别解读，不将H4改善单独认定为正确三维恢复。

## 6. 参数、大小与实测推理成本

{table(cost,['模型','参数','FP32参数MiB','checkpoint MiB','前向median ms','前向P90 ms','峰值GiB'])}

同一{timing['gpu']}、{timing['device']}、batch1、FP32、TF32关闭；同9个固定case/layout，每model/输入warmup10、同步重复30，交替执行模型顺序。Ours前向包含内部P_diff；外部metric projector不计入forward。共各360次cost前向（含warmup），单独记录于[timing.json](artifacts/deepsets_selfsup_v1/timing.json)。峰值为CUDA allocated扣除另一个同进程驻留模型tensor字节；库/allocator共存状态可能影响测量，不当作独占硬件benchmark或系统总显存。checkpoint含optimizer/RNG，参数字节更直接表示架构存储成本。

DeepSets训练墙钟{t['total_epoch_seconds']/3600:.3f}h与原历史Ours约{reftrain['mean_epoch_seconds']*300/3600:.3f}h分别记录，GPU共享负载和历史时段不同，不能据此严格归因训练速度。参数变化之外，interaction/fusion/refinement一起改变，性能差不能单独归因OARI或MRGR。

## 7. 固定预算下收敛与最坏case

{convergence}

![训练measurement曲线](artifacts/deepsets_selfsup_v1/figures/training_measurement_curves.png)

{table(tailrows,['组','指标','模型','case均值P90','case均值P95','最大case均值','原Val索引'])}

tail基于36个case/group五layout均值；逐layout900行保留用于描述，不当独立推断。完整worst case ID与全部指标分位数见[tails.csv](artifacts/deepsets_selfsup_v1/tails.csv)。

## 8. 固定重建图与后续判断

预先固定原Val索引0/18/35、draw0、G1/G3/G5；y为高度，y32=0.30476m；x/z单位m。三方法显示浓度每case共享显示字段最大值色标，两个绝对误差另共享色标；同时保留固定水平切片与沿高度y的均值图，没有按GT误差挑例或切片。

![case0](artifacts/deepsets_selfsup_v1/figures/fixed_case0_slice.png)

![case18](artifacts/deepsets_selfsup_v1/figures/fixed_case18_slice.png)

![case35](artifacts/deepsets_selfsup_v1/figures/fixed_case35_slice.png)

本轮验证了一个具有同信息/同标签/同selector/同预算的基础集合参考；只一个seed、36例开发Val、固定站位，历史Ours没有同时fresh重训。无论优势/劣势方向，都不证明优于Senseiver、DeepONet或GNOT，不是独立训练模块消融。完整归档全部组和指标，保留原v1及增强NO-GO、C0/C1历史结论。

**唯一优先下一阶段：独立预注册并接入Senseiver同条件任务适配比较。** 当前实现/协议检查通过，停止于本次DeepSets报告，不用本轮Val GT修改Ours，不打开最终test，不自动启动新模型或超参搜索。

## 交付与访问边界

实际正式预算300epochs/24,900steps；主预测900、selector replay36次batch2前向/72个case预测、成本DeepSets360/Ours360；成功独立smoke七次batch2 forward/一步，失败检查两次batch2 forward/零步，不计正式预算；正式train/Val分别49,800/10,800次batch2 model调用。完整现有回归259项通过，新增独立配对统计三项通过。首次主评价因旧guard仅允许case_images子目录而阻止读取本run已保存的NPZ；更正输出子目录并保留失败审计。首个预测SHA核验后复用，仅补测量/身份记录，没有重复network forward；该行前向独立耗时未保存，记null，不估计，统一9输入成本benchmark完整。禁读原始GT/Pool B/test/foreign checkpoint尝试仍0。历史源码（除新增注册）、dataset、旧输出/checkpoint metadata/hash核验通过；GT128仅核对冻结metadata，内容未读。

配置 `configs/train_deepsets_selfsup_v1.yaml`；训练入口 `scripts/train_deepsets_selfsup.py`；统一接口 `src/geophy_set3d/training/deep_comparison.py`；独立评价 `scripts/evaluate_deep_comparison.py`；报告入口 `scripts/report_deep_comparison.py`。标量/图/协议/审计在本报告同目录artifacts；大预测与checkpoint仅外部目录`{out}`和`/data/djc/GeoPhy-Set3D/outputs/checkpoints/deepsets_selfsup_v1`，不提交Git。
'''
    path=REPO/'docs/reports/deepsets_selfsup_v1_zh.md';path.write_text(text)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/train_deepsets_selfsup_v1.yaml');a=ap.parse_args();c=load_comparison_config(a.config);out=Path(c['comparison']['output'])
    assert json.loads((out/'gt_complete.json').read_text())['rows']==900
    case=pd.read_csv(out/'case_group_180.csv');stats=pd.read_csv(out/'paired_statistics.csv');tails=pd.read_csv(out/'tails.csv');contrasts=json.loads((out/'contrasts.json').read_text());training=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()]
    plot(c,out,case,stats,training);integ=integrity(c,out);report(c,out,case,stats,training,tails,contrasts,integ)
    art=REPO/'docs/reports/artifacts/deepsets_selfsup_v1'
    for p in out.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.log','.sha256'] and p.name not in ['dataset_before.json','history_before.json']:
            shutil.copy2(p,art/p.name)
    for name in ['metrics.jsonl','actual_training_inputs.jsonl','inference_events.jsonl']:
        with gzip.open(art/(name+'.gz'),'wb') as f:f.write((out/name).read_bytes())
    shutil.copytree(out/'figures',art/'figures',dirs_exist_ok=True)
    write_json(art/'artifact_manifest.json',{str(p.relative_to(art)):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in art.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    print('Report, scalars, figures and integrity archive complete',flush=True)

if __name__=='__main__':main()
