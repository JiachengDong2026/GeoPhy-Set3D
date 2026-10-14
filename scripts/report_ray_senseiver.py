#!/usr/bin/env python3
"""Rebuild Chinese three-method report, fixed figures and preservation/access audit."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import shutil
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geophy_set3d.training.ray_senseiver import load_comparison_config,sha256,verify_registration
from geophy_set3d.training.utils import write_json
from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import DiagnosticReadGuard,GROUPS,METRICS
from diagnose_clustered_one_swap import snapshot,dataset,target_gt

MODELS=['Ours','DeepSets','Senseiver']


class FixedFigureReadGuard(DiagnosticReadGuard):
    """Exactly the nine preregistered historical arrays, read only after freeze."""
    def __init__(self,*args,historical,**kwargs):
        super().__init__(*args,**kwargs)
        self.historical={Path(p).resolve():h for p,h in historical.items()};self.history_reads=0
    def _audit(self,event,args):
        if self.active and event=='open' and isinstance(args[0],(str,bytes,Path)):
            p=Path(args[0]).resolve()
            if p in self.historical:
                mode=args[1]
                if not isinstance(mode,str) or any(c in mode for c in 'wa+'):
                    self.deny('dataset_write_attempts','Historical prediction arrays are read only')
                self.history_reads+=1;return
        super()._audit(event,args)
    def summary(self):return dict(super().summary(),fixed_historical_prediction_opens=self.history_reads,registered_historical_sha256={str(k):v for k,v in self.historical.items()})


def table(rows,headers):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def plots(c,out,case,stats,training,timing):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(figdir/(name+'.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,m in zip(axs,['relative_l1','relative_l2','heldout_relative_l1']):
        for n in MODELS:ax.plot(GROUPS,[case[case.group==g][n+'_'+m].mean() for g in GROUPS],'o-',label=n)
        ax.set(xlabel='Fixed V4 geometry group',ylabel=m.replace('_',' '));ax.legend()
    save(fig,'group_error_curves')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    primary=stats[(stats.family=='primary')&(stats.metric=='relative_l1')].set_index('group')
    for i,g in enumerate(GROUPS):
        r=primary.loc[g];ax.plot([r.ci_low,r.ci_high],[i,i],lw=3);ax.plot(r.mean_delta,i,'ko')
    ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(5),yticklabels=GROUPS,xlabel='Ours - Ray-Senseiver-SS 3D relative L1',title='36 paired cases; 10,000 bootstrap draws; pointwise 95% CI');save(fig,'paired_delta_ci')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    epochs=[r['epoch']+1 for r in training]
    for split in ['train','val']:
        axs[0,0].plot(epochs,[r[split]['measurement']['heldout']['relative_l1'] for r in training],label=split)
        axs[0,1].plot(epochs,[r[split]['loss']['total'] for r in training],label=split)
    for component in ['hold','cross','observed','consistency']:
        axs[1,0].plot(epochs,[r['val']['loss'][component] for r in training],label=component)
    for key in ['mean','std','case_to_case_mean_absolute_difference']:
        axs[1,1].plot(epochs,[r['val']['output'][key] for r in training],label=key)
    for ax,title in zip(axs.flat,['Held-out measurement relative L1','Original self-supervised loss','Val loss components','Val numerical output moments']):
        ax.set(xlabel='Actual epoch',title=title);ax.set_yscale('log');ax.legend()
    save(fig,'training_measurement_curves')
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    for n in MODELS:
        times=np.concatenate([r['seconds'] for r in timing['rows'] if r['model']==n]);error=case[case.group!='G5'][n+'_relative_l1'].mean()
        ax.scatter(np.median(times)*1000,error,s=100);ax.annotate(n,(np.median(times)*1000,error),xytext=(5,5),textcoords='offset points')
    ax.set(xlabel='Full 64^3 forward median (ms), same GPU / FP32',ylabel='G1-G4 macro 3D relative L1',title='Fixed-budget accuracy versus measured inference cost');save(fig,'accuracy_cost')
    records=json.loads((out/'station_subsets.json').read_text());selected=[r for r in records if r['case_index'] in [0,18,35] and r['group'] in ['G1','G3','G5'] and r['draw']==0]
    old=pd.read_csv(Path(c['comparison']['deepsets_artifact'])/'paired_900.csv').set_index(['case_id','group','draw'])
    historical={};paths={}
    for r in selected:
        token=f"case{r['case_index']}_{r['geometry_id']}";row=old.loc[(r['case_id'],r['group'],r['draw'])]
        op=Path(c['comparison']['reference_predictions'])/'case_images'/f"normal_case{r['case_index']}_{r['geometry_id']}.npz"
        dp=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/deepsets_selfsup_v1/case_images')/(token+'.npz')
        historical[str(op)]=row.ours_prediction_sha256;historical[str(dp)]=row.deepsets_prediction_sha256;paths[token]=(op,dp)
    # Hash checks themselves are guarded reads of exactly these registered files.
    with FixedFigureReadGuard(c['dataset']['root'],[],out,'gt',historical=historical) as guard:
        for p,h in historical.items():assert sha256(p)==h,p
        ds=dataset(dict(dataset=c['dataset']['root'],original_seed=c['training']['seed']))
        for i in [0,18,35]:
            gt=target_gt(ds,i)[0,0].numpy();fields=[]
            for g in ['G1','G3','G5']:
                r=next(r for r in selected if r['case_index']==i and r['group']==g);token=f"case{i}_{r['geometry_id']}"
                with np.load(paths[token][0],allow_pickle=False) as a:ours=a['C1'].copy()
                with np.load(paths[token][1],allow_pickle=False) as a:deep=a['prediction'].copy()
                row=json.loads((out/'prediction_rows'/(token+'.json')).read_text());assert sha256(row['prediction_file'])==row['prediction_sha256']
                with np.load(row['prediction_file'],allow_pickle=False) as a:sense=a['prediction'].copy()
                fields.append([gt,ours,deep,sense])
            for view in ['slice','ymean']:
                reduced=[[a[:,32,:] if view=='slice' else a.mean(1) for a in row] for row in fields]
                vmax=max(a.max() for row in reduced for a in row)
                fig,axs=plt.subplots(3,4,figsize=(13,9),layout='constrained')
                for row,g,axes in zip(reduced,['G1','G3','G5'],axs):
                    for a,n,ax in zip(row,['GT64']+MODELS,axes):
                        im=ax.imshow(a.T,origin='lower',extent=[0,.6,0,.6],vmin=0,vmax=vmax,cmap='viridis');ax.set(title=g+' '+n,xlabel='x (m)',ylabel='z (m)')
                fig.colorbar(im,ax=axs.ravel().tolist(),label='CH4 mass fraction');fig.suptitle(f'Val case {i}, draw 0; '+('y = 0.30476 m (height)' if view=='slice' else 'Mean along height y'))
                save(fig,f'fixed_case{i}_{view}')
        write_json(out/'figure_gt_access.json',guard.summary())


def integrity(c,out):
    p=verify_registration(c)
    for name,h in p['official_source']['sha256'].items():assert sha256(REPO/'third_party/senseiver'/name)==h,name
    assert snapshot(c['dataset']['root'])==json.loads((out/'dataset_before.json').read_text())
    for root,meta in json.loads((out/'history_before.json').read_text()).items():assert snapshot(root)==meta,root
    for path,h in p['protected_sources'].items():
        if path!='src/geophy_set3d/models/__init__.py':assert sha256(REPO/path)==h,path
    raw=json.loads((Path(c['comparison']['reference_predictions'])/'inventory_before.json').read_text())['gt128_metadata']
    for path,meta in raw.items():
        st=Path(path).stat();assert dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)==meta
    for ref in [p['reference_checkpoint'],p['deepsets_checkpoint']]:assert sha256(ref['checkpoint'])==ref['sha256']
    sessions=[json.loads(l) for l in (out/'access_sessions.jsonl').read_text().splitlines()]
    assert all(s['forbidden_attempts']==0 and not s['failed'] for s in sessions)
    audit=json.loads((out/'training_access_audit.json').read_text());assert all(audit[k]==0 for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
    rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()]
    assert len(rows)==300 and [r['epoch'] for r in rows]==list(range(300))
    assert all(r['optimizer_steps']==83*(r['epoch']+1) and r['optimizer_steps_this_epoch']==83 for r in rows)
    counts=collections.Counter(json.loads(l)['event'] for l in (out/'inference_events.jsonl').read_text().splitlines());assert counts['forward_started']==counts['saved']==900
    assert json.loads((out/'independent_audit.json').read_text())['passed']
    summary=dict(passed=True,dataset_metadata_unchanged=True,historical_outputs_and_checkpoints_metadata_unchanged=True,GT128_metadata_unchanged_contents_not_read=True,protected_numerical_sources_unchanged=True,epochs=300,optimizer_steps=24900,all_historical_station_digests_equal=all(r['historical_station_digest_equal'] for r in rows),GT_only_after_all_saved=True,training_access=audit,evaluation_sessions=sessions,forwards=dict(main_senseiver=900,selector_senseiver=36,timing_each_model=360,formal_training_batch2=49800,formal_validation_batch2=10800),scope='No final test, GT128, Pool B, extra baselines, multi-seed or posthoc tuning')
    write_json(out/'integrity.json',summary);return summary


def report(c,out,case,stats,training,tails,contrasts,timing):
    complete=json.loads((out/'training_complete.json').read_text());freeze=json.loads((out/'frozen_best.json').read_text());selector=json.loads((out/'selector_replay.json').read_text());smoke=json.loads((out/'engineering_smoke.json').read_text());observer=json.loads((out/'training_process_observer.json').read_text())
    primary=stats[(stats.family=='primary')&(stats.metric=='relative_l1')].set_index('group');aux=stats[(stats.family=='auxiliary')&(stats.metric=='relative_l1')].set_index('group')
    group_rows=[];descriptions=[]
    for g in GROUPS:
        r=primary.loc[g];a=aux.loc[g]
        desc='Ours 更好' if r.ci_high<0 else 'Senseiver 更好' if r.ci_low>0 else '区间跨0，方向不确定'
        descriptions.append(g+'：'+desc)
        group_rows.append([g,f'{r.left_mean:.6f}',f'{r.right_mean:.6f}',f'{a.right_mean:.6f}',f'{r.mean_delta:.6f}',f'[{r.ci_low:.6f}, {r.ci_high:.6f}]',f'{int(r.left_wins)}/{int(r.right_wins)}/{int(r.ties)}',desc])
    aux_rows=[[g,f'{aux.loc[g].mean_delta:.6f}',f'[{aux.loc[g].ci_low:.6f}, {aux.loc[g].ci_high:.6f}]',f'{int(aux.loc[g].left_wins)}/{int(aux.loc[g].right_wins)}/{int(aux.loc[g].ties)}'] for g in GROUPS]
    macro=next(r for r in contrasts if r['family']=='primary' and r['metric']=='relative_l1' and r['comparison']=='G1_G4_macro')
    secondary=[]
    for m in METRICS[1:]:
        for g in GROUPS:
            r=stats[(stats.family=='primary')&(stats.metric==m)&(stats.group==g)].iloc[0]
            secondary.append([g,m,f'{r.left_mean:.7g}',f'{r.right_mean:.7g}',f'{r.mean_delta:.7g}',f'[{r.ci_low:.7g},{r.ci_high:.7g}]',f'{int(r.left_wins)}/{int(r.right_wins)}/{int(r.ties)}'])
    consistency=[]
    for m in ['relative_l2','heldout_relative_l1','observed_relative_l1','peak_value_absolute_error','integrated_concentration_relative_error']:
        part=stats[(stats.family=='primary')&(stats.metric==m)]
        consistency.append(m+'：Ours CI低于0 '+','.join(part[part.ci_high<0].group)+'；Senseiver CI低于Ours '+','.join(part[part.ci_low>0].group)+'；跨0 '+','.join(part[(part.ci_low<=0)&(part.ci_high>=0)].group))
    cost=[]
    for n in MODELS:
        ts=[r for r in timing['rows'] if r['model']==n];times=np.concatenate([r['seconds'] for r in ts]);idn=timing['identities'][n]
        cost.append([n,idn['parameter_count'],f"{idn['fp32_parameter_bytes']/2**20:.3f}",f"{timing['checkpoint_bytes'][n]/2**20:.3f}",f'{np.median(times)*1000:.3f}',f'{np.quantile(times,.9)*1000:.3f}',f"{max(r['peak_allocated_excluding_other_model_bytes'] for r in ts)/2**30:.3f}"])
    cr=[[r['family'],r['comparison'],r['metric'],f"{r['left_mean']:.6f}",f"{r['right_mean']:.6f}",f"{r['mean_delta']:.6f}",f"[{r['ci_low']:.6f},{r['ci_high']:.6f}]"] for r in contrasts if r['metric'] in ['relative_l1','relative_l2','heldout_relative_l1']]
    tr=[[r.group,r.metric,r.model,f'{r.p90:.6f}',f'{r.p95:.6f}',f'{r.maximum:.6f}',r.worst_case_id,int(r.worst_case_index)] for _,r in tails[tails.metric.isin(['relative_l1','relative_l2'])].iterrows()]
    recent=np.mean([r['val']['measurement']['heldout']['relative_l1'] for r in training[-30:]]);prior=np.mean([r['val']['measurement']['heldout']['relative_l1'] for r in training[-60:-30]])
    moments=selector['validation']['output'];collapsed=(moments['std']<=1e-6 or moments['mean']<=1e-6 or moments['fraction_near_one']>=.01 or moments['case_to_case_mean_absolute_difference']<=1e-6)
    convergence='末30轮selector仍改善，有固定预算下未充分收敛迹象' if recent<prior and complete['best_epoch']>270 else '未观察到上述末期持续改善规则；这不证明全局最优或充分收敛'
    recommendation=('独立预注册一次 Ray-Senseiver 的初始化/优化数值诊断，先定位测量训练塌缩原因' if collapsed else '独立预注册一次跨 seed 的 Ours / Ray-Senseiver 同条件重复比较，检验单 seed 结论稳定性')
    text=f'''# Ray-Senseiver-SS 射线积分、自监督适配与三方法冻结比较

实际完整训练 **300 epochs / 24,900 updates**，Ours / DeepSets 保持冻结。{'; '.join(descriptions)}。

G1–G4 macro 3D L1：Ours={macro['left_mean']:.6f}、Senseiver={macro['right_mean']:.6f}，Δ={macro['mean_delta']:.6f}，逐项95%CI=[{macro['ci_low']:.6f},{macro['ci_high']:.6f}]。G5绝对 L1：Ours={primary.loc['G5'].left_mean:.6f}、Senseiver={primary.loc['G5'].right_mean:.6f}。结果为本轮固定配置的实测，不外推原 Senseiver 论文理论能力。

## 1. 机制、官方来源与任务适配

官方论文 Santos et al., *Development of the Senseiver for efficient field reconstruction from sparse observations* (2023)，https://www.nature.com/articles/s42256-023-00746-x ，正文/Methods已读取，CC BY4.0摘录与页面hash见 [paper_source.json](artifacts/ray_senseiver_selfsup_v1/paper_source.json)。官方 https://github.com/OrchardLANL/Senseiver commit `e443eb0ca188b5e83a0e6ce5afb8db7d90393de1`，model.py/network_light.py/positional.py/s_parser.py与BSD-3许可hash见[official_source.json](artifacts/ray_senseiver_selfsup_v1/official_source.json)。原样副本与LANL copyright/disclaimer在third_party/senseiver，改编核心也保留notice。

**Ray-Senseiver-SS 是射线积分、measurement-only 的任务适配，不是原论文空间点传感器、监督场标签实验的直接复现。** 保留 learned latent cross-attention、latent self-attention、残差/LayerNorm/GELU、Fourier位置编码、learned output query和坐标cross-attention解码；输入换成完整射线积分与路径，训练换成原项目自监督目标，最终加sigmoid。不优化/移动传感器，不使用dynamic sensor placement，不含GeometryLifting/OARI/GCA/MRGR或3D卷积decoder。

配置64 latents、latent/preproc64、num_layers3、每block3 self-attention，encoder cross/self heads4，decoder query/preproc64/head4，learned query size1、bands16、dropout0。沿用官方**独立layer_1、共享layer_n重复两次**，不是三套独立block。实际254,913参数，不为匹配Ours参数而削弱宽度；这是预设适配配置，不称官方统一默认或最优配置。[训练前协议](../ray_senseiver_selfsup_v1_protocol_zh.md)，协议commit `098c086`，训练实现commit `d34f97b`。

## 2. 全射线、物理尺度、完整输出与工程检查

每条valid ray包含 y_integral/32284.894580078173；entry=origin+direction*t_near，exit=entry+direction*path_length；entry/exit用2*x/.6−1，unit direction尺度1，length/(sqrt(3)*.6)。entry/exit各16频带 sin/cos(pi*x*f), f=linspace(1,32,16)，共203通道。积分没有被解释成局部浓度。全部有效零测量保留；只排除invalid与padded view，不采样射线/扫描，不学习station编号embedding。全invalid用零虚拟token先验，输出有限。

endpoint-aligned XYZ [0,.6]^3、y为高度，全部262,144 query直接解码，FP32 [B,1,64,64,64]；缓存坐标与96通道Fourier，encoder只调用一次。chunk={c['runtime']['query_chunk_size']}，decoder non-reentrant activation checkpoint={c['runtime']['decoder_checkpoint']}，无detach、插值或query采样。attention仅关闭权重返回，完整保留网络数值机制。

小fixture分块/非分块预测 atol2e−6/rtol2e−5，query/latent/所有decoder参数梯度 atol至3e−5/rtol2e−4检查通过，所有259个测试query收到非零梯度；官方core原注意力路径对比通过。真实最大A5/B5/H2、batch2完整forward/backward/step通过，encoder/preproc/learned latents/decoder均收到有限梯度，H只投影而不输入网络、A/B/H不交。view置换max差={smoke['permutation_max_abs']:.3g}，padding差={smoke['padding_max_abs']:.3g}，ray同步置换/全invalid/零有效值/真实entry-exit单位检查通过。smoke峰值={smoke['peak_gpu_bytes']/2**30:.3f}GiB，仅独立一步，正式fresh初始化。266项完整回归及3项三方法统计检查通过。

## 3. 同协议的实际完整训练、冻结与访问边界

166 train/36 Val，Pool A；自然A3–5/B3–5/H2，Val固定A4/B4/H4原IDs。seed20261011、FP32 effective batch2、AdamW lr2e−4/wd1e−4、clip1、cosine T_max300、83次更新/epoch，实际300/24900，无early stop/结构、lr、频带、loss或horizon搜索。原 `selfsup.objective/project_input/run_epoch/selection_metric` 直接调用：L_hold+.5L_cross+.1L_obs+.05L_cons，原SmoothL1 beta.01、relative.1/epsilon1e−6/mask/reduction完整，P_diff/K96/all valid rays。

实际训练+Val epoch墙钟={complete['total_epoch_seconds']/3600:.3f}h，进程墙钟={observer['wall_seconds']/3600:.3f}h，峰值={complete['peak_gpu_allocated_bytes']/2**30:.3f}GiB。best第{freeze['best_epoch']}轮，由 natural Val 双预测平均 held-out relative L1 唯一选择={complete['best_val_heldout_relative_l1']:.8f}，独立重放={selector['replayed']:.8f}，容差abs1e−8+rel1e−5通过。best SHA256=`{freeze['sha256']}`，checkpoint=`{freeze['checkpoint']}`。

正式初态hash与独立smoke初态一致，未读取smoke/旧/监督权重；本次恢复记录见observer和access审计，只支持本run last.pt完整model/optimizer/scheduler/RNG恢复。完整300轮166cases每轮恰好一次、实际83steps与最终AdamW状态step24900由独立审计核对；历史station digest一致={complete['all_historical_station_digests_equal']}。历史Ours缺少完整逐步optimizer trace，不声称严格随机化训练配对或逐步数值重放。

训练与selector禁读GT64/GT128/foreign checkpoint尝试均0。完成300与best冻结后先selector replay，再900完整体积保存/hash封存，再独立saved-output Val GT64评价（36例）；后者不读取measurement/checkpoint。GT128、Pool B、final test未读。数据/历史两模型checkpoint和输出metadata、冻结数值源码hash均核对，见[integrity.json](artifacts/ray_senseiver_selfsup_v1/integrity.json)与[independent_audit.json](artifacts/ray_senseiver_selfsup_v1/independent_audit.json)。大体积与权重仅在外部独立目录。

## 4. 数值行为与固定预算收敛

best Val输出min/max/mean/std={moments['min']:.6g}/{moments['max']:.6g}/{moments['mean']:.6g}/{moments['std']:.6g}，近1占比={moments['fraction_near_one']:.6g}，case间MA差={moments['case_to_case_mean_absolute_difference']:.6g}。按预定数值规则（mean/std/case差>1e−6且near1<.01）塌缩={collapsed}。这些是数值证据，不能将低测量误差单独视为正确三维重建证明；如果出现塌缩如实保留，不改配置制造优势。

selector前一个30轮均值={prior:.8f}，最后30轮={recent:.8f}，末轮={training[-1]['val']['measurement']['heldout']['relative_l1']:.8f}；{convergence}。同AdamW/lr/预算不保证对所有架构最优。按300轮停止，未追加训练。

![训练曲线](artifacts/ray_senseiver_selfsup_v1/figures/training_measurement_curves.png)

## 5. 同信息、评价身份与主配对终点

全部原36case×25 V4布局=900，input/H IDs及顺序一致；完整measurement、方向、near/path、geometry、view/ray mask、H4、尺度/64³网格/投影器/指标的tensor hashes逐行与冻结DeepSets保存身份一致，并沿已验证Ours/DeepSets物理identity链核对。表示形式不同，不要求Senseiver具有lifting mask；模型实际只编码可获得物理观测。Ours原normal C1与DeepSets已验证900标量直接复用，未重跑它们的全量推理。

十指标定义原样复用 VolumeMetrics 与 relative_projection_metrics。先每case/group平均5布局，再36case bootstrap10000、seed2026101801、percentile逐项95%CI；Δ=Ours−Senseiver，负有利Ours。胜/负/平阈值1e−8。900行/ray/voxel不当独立样本，不算p值，不作多重校正显著性宣称。

{table(group_rows,['组','Ours','Senseiver','DeepSets','主Δ','逐项95%CI','Ours胜/Senseiver胜/平','解读'])}

![五档](artifacts/ray_senseiver_selfsup_v1/figures/group_error_curves.png)
![主CI](artifacts/ray_senseiver_selfsup_v1/figures/paired_delta_ci.png)

辅助比较 Δ=Senseiver−DeepSets，身份独立于主终点：

{table(aux_rows,['组','辅助Δ','逐项95%CI','Senseiver胜/DeepSets胜/平'])}

## 6. 正常精度、困难几何与辅助指标

{table(cr,['比较身份','统计','指标','左方法','右方法','差','逐项95%CI'])}

primary左/右=Ours/Senseiver；auxiliary左/右=Senseiver/DeepSets。macro四组等权；G5−G1每case差值；G5/G1为**总体组均值之比**，每次bootstrap重算均值之比，非case比值均值。结合G5绝对误差判断，不能只按相对退化比排序。

{chr(10).join(consistency)}。

{table(secondary,['组','辅助指标','Ours','Senseiver','Δ','逐项95%CI','Ours胜/Senseiver胜/平'])}

全部三方法900行、180 case/group、100项组配对统计与150项尾部描述见同目录artifacts。MAE/RMSE/peak为无量纲质量分数；integral是voxel sum proxy，不称真实总质量。测量拟合与三维恢复分别解读，主L1、L2、H4、峰值不一致时保留原结论。

## 7. 参数、显存、实际时间与精度成本

{table(cost,['模型','参数','FP32参数MiB','checkpoint MiB','完整前向median ms','P90 ms','allocated峰值GiB'])}

同{timing['gpu']}、{timing['device']}、batch1/FP32、TF32关闭；固定case0/18/35×draw0×G1/G3/G5九输入，每model/input warmup10+同步repeat30，三方法顺序循环交替。每模型360次成本前向（含warmup）。Ours包括内部projector；Senseiver encoder+全部64³查询，包括ray几何/Fourier预处理，没有少query加速口径。eval无梯度故checkpoint不触发，chunk保持{c['runtime']['query_chunk_size']}。allocated峰值扣除另两个同进程常驻模型tensor；Senseiver固定query缓存属于模型运行显存，不能拿参数字节替代。共享GPU、缓存/allocator共存与历史不同时段限制，时序为实测描述，不当独占硬件结果。

新训练墙钟/steps/峰值见第3节；旧两模型原训练墙钟与峰值见保留历史报告，不能严格归因跨时段训练速度。新模型实际参数较Ours更少，完整坐标解码计算模式不同，精度收益/代价需结合上表，不单独归因Ours某个模块。

![精度成本](artifacts/ray_senseiver_selfsup_v1/figures/accuracy_cost.png)

## 8. 尾部与固定重建图

{table(tr,['组','指标','模型','case均值P90','P95','最大','worst ID','Val索引'])}

预先固定九输入，y32=.30476m为水平切片，y为高度、x/z单位m，同时输出沿y均值图；每case三组GT/三方法共享一个显示字段最大值色标。没有按误差挑case或切片，历史预测只读九个固定文件并核验SHA，不重新推理旧模型。

![case0](artifacts/ray_senseiver_selfsup_v1/figures/fixed_case0_slice.png)
![case18](artifacts/ray_senseiver_selfsup_v1/figures/fixed_case18_slice.png)
![case35](artifacts/ray_senseiver_selfsup_v1/figures/fixed_case35_slice.png)

## 9. 结论边界与一个下一阶段任务

本轮回答：保留Senseiver latent/coordinate-query机制，改输入/自监督目标；完整有效ray和完整64³梯度合法；实际300轮/24900步，数值塌缩状态={collapsed}；三方法信息/无场标签/自然selector/冻结评价身份匹配；各档胜负/不确定见第5节；正常/困难/L2/H4与其他辅助一致性见第6节；参数/显存/时间见第3/7节；末期预算迹象见第4节。

这是单seed、历史冻结参考、36例开发Val、射线积分与measurement-only任务适配。不能外推超过原Senseiver全部设置或理论上限；同优化器固定预算不代表其最优训练，不单独归因Ours模块。不根据本轮GT修改任何模型或选择epoch，不打开最终test，历史GO/NO-GO/C0C1结论保留。

**唯一下一阶段建议：{recommendation}。** 本轮仅建议，完成后停止，不自动启动优化/多seed/新基线。

训练入口 scripts/train_ray_senseiver_selfsup.py；独立评价 scripts/evaluate_ray_senseiver.py；报告 scripts/report_ray_senseiver.py；审计 scripts/audit_ray_senseiver.py；独立配置 configs/train_ray_senseiver_selfsup_v1.yaml。外部输出 `{out}`，checkpoint目录独立。代码/协议/实现/标量/图/来源/访问及完整性审计提交Git，大预测/GT/checkpoint不提交。
'''
    (REPO/'docs/reports/ray_senseiver_selfsup_v1_zh.md').write_text(text)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/train_ray_senseiver_selfsup_v1.yaml');a=ap.parse_args();c=load_comparison_config(a.config);out=Path(c['comparison']['output'])
    assert json.loads((out/'gt_complete.json').read_text())['rows']==900
    case=pd.read_csv(out/'case_group_180.csv');stats=pd.read_csv(out/'paired_statistics.csv');tails=pd.read_csv(out/'tails.csv');contrasts=json.loads((out/'contrasts.json').read_text());timing=json.loads((out/'timing.json').read_text());training=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()]
    plots(c,out,case,stats,training,timing);integrity(c,out);report(c,out,case,stats,training,tails,contrasts,timing)
    art=REPO/'docs/reports/artifacts/ray_senseiver_selfsup_v1'
    for p in out.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.log','.sha256','.txt'] and p.name not in ['dataset_before.json','history_before.json']:shutil.copy2(p,art/p.name)
    for name in ['metrics.jsonl','actual_training_inputs.jsonl','inference_events.jsonl','access_sessions.jsonl']:
        with gzip.open(art/(name+'.gz'),'wb') as f:f.write((out/name).read_bytes())
    for name in ['dataset_before.json','history_before.json']:
        with gzip.open(art/(name+'.gz'),'wb') as f:f.write((out/name).read_bytes())
    shutil.copytree(out/'figures',art/'figures',dirs_exist_ok=True)
    write_json(art/'artifact_manifest.json',{str(p.relative_to(art)):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in art.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    print('Three-method Chinese report, figures, scalars and integrity archived',flush=True)

if __name__=='__main__':main()
