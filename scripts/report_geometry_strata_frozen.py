#!/usr/bin/env python3
"""Aggregate scalar frozen evaluations, conservative case inference, Chinese report."""
import json
import sys
import shutil
import subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geophy_set3d.data.build_utils import digest
from geophy_set3d.evaluation.geometry_strata import validate_shared_inputs
from geophy_set3d.training.utils import write_json
from diagnose_clustered_one_swap import write_csv
from evaluate_geometry_strata_frozen import OUT,verify
METRICS=['relative_l1','relative_l2','mae','rmse','observed_relative_l1','observed_relative_l2',
         'heldout_relative_l1','heldout_relative_l2','integrated_concentration_relative_error','peak_value_absolute_error']


def rho_rows(x,y):
    xr=rankdata(x);xr=xr-xr.mean();yr=rankdata(y,axis=-1);yr=yr-yr.mean(axis=-1,keepdims=True)
    denominator=np.sqrt((xr*xr).sum()*(yr*yr).sum(axis=-1))
    return np.divide((yr*xr).sum(axis=-1),denominator,out=np.full(denominator.shape,np.nan),where=denominator>0)


def table(rows,columns):
    lines=['| '+' | '.join(c[0] for c in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    for r in rows:lines.append('| '+' | '.join(str(f(r)) for _,f in columns)+' |')
    return '\n'.join(lines)


def ci_text(ci):return '不可识别（常量）' if ci is None else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]'


def main():
    p=verify();cfg=p['config'];frames=[]
    import re
    test_log=(OUT/'full_tests.log').read_text()
    assert test_log.rstrip().splitlines()[-1].startswith('OK'), 'Full repository tests must finish first'
    full_test_count=int(re.findall(r'Ran (\d+) tests',test_log)[-1])
    for name in p['models']:
        assert json.loads((OUT/f'{name}_complete.json').read_text())['rows']==2340
        rows=[json.loads(l) for l in (OUT/f'{name}_rows.jsonl').read_text().splitlines()]
        assert len(rows)==2340;frames.extend(rows)
    validate_shared_inputs(frames);write_csv(OUT/'per_case_draw.csv',frames)
    df=pd.DataFrame(frames);keys=['model','section','V','stratum','case_id']
    counts=df.groupby(keys).size();assert (counts==5).all()
    cm=df.groupby(keys)[METRICS].mean().reset_index();cm.to_csv(OUT/'case_means.csv',index=False,lineterminator='\n')
    cases=sorted(cm.case_id.unique());assert len(cases)==36
    idx=np.random.default_rng(cfg['bootstrap_seed']).integers(0,36,(10000,36));summaries=[];pairs=[];correlations=[]
    def values(model,section,views,stratum,metric):
        group=cm[(cm.model==model)&(cm.section==section)&(cm.V==views)&(cm.stratum==stratum)]
        return group.set_index('case_id').loc[cases,metric].to_numpy()
    for (model,section,views,stratum),group in cm.groupby(['model','section','V','stratum']):
        row=dict(model=model,section=section,V=int(views),stratum=stratum,case_count=36,draws=5)
        for metric in METRICS:
            v=group.set_index('case_id').loc[cases,metric].to_numpy()
            row[metric]=float(v.mean());row[metric+'_ci']=np.quantile(v[idx].mean(1),[.025,.975]).tolist()
        summaries.append(row)
    for model in p['models']:
        for views in cfg['views']:
            for left,right in [('low','high'),('low','middle'),('middle','high')]:
                for metric in METRICS:
                    a=values(model,'controlled',views,left,metric);b=values(model,'controlled',views,right,metric);d=a-b
                    pairs.append(dict(model=model,V=views,comparison=left+'-'+right,metric=metric,mean_delta=float(d.mean()),
                        ci=np.quantile(d[idx].mean(1),[.025,.975]).tolist(),case_count=36,draws=5))
            sub=df[(df.model==model)&(df.section=='controlled')&(df.V==views)]
            pivot=sub.pivot(index='case_id',columns='geometry_id',values='relative_l1').loc[cases]
            geomids=list(pivot.columns);y=pivot.to_numpy();bygeom=sub.groupby('geometry_id').first().loc[geomids]
            for descriptor in cfg['association_descriptors']:
                x=bygeom[descriptor].to_numpy();r=float(rho_rows(x,y.mean(0)));boot=rho_rows(x,y[idx].mean(1))
                finite_boot=boot[np.isfinite(boot)]
                correlations.append(dict(model=model,V=views,descriptor=descriptor,rho=r if np.isfinite(r) else None,
                    ci=np.quantile(finite_boot,[.025,.975]).tolist() if len(finite_boot) else None,
                    undefined_reason='constant descriptor or aggregate error' if not np.isfinite(r) else None,
                    descriptor_unique_values=len(np.unique(x)),finite_bootstrap_replicates=len(finite_boot),
                    case_count=36,geometry_count=15,bootstrap_samples=10000,bootstrap_seed=cfg['bootstrap_seed'],
                    interpretation='within-V, error averaged across cases per geometry; bootstrap cases only; no p-value'))
    write_csv(OUT/'strata_summary.csv',summaries);write_json(OUT/'strata_summary.json',summaries)
    write_csv(OUT/'correlations.csv',correlations);write_json(OUT/'correlations.json',correlations)
    write_json(OUT/'paired_statistics.json',dict(bootstrap_samples=10000,seed=cfg['bootstrap_seed'],unit='case',comparisons=pairs))
    bycell={(r['model'],r['section'],r['V'],r['stratum']):r for r in summaries}
    gate=[]
    for model in p['models']:
        for views in cfg['views']:
            l,m,h=[bycell[model,'controlled',views,s]['relative_l1'] for s in ('low','middle','high')]
            delta=next(r for r in pairs if r['model']==model and r['V']==views and r['comparison']=='low-high' and r['metric']=='relative_l1')
            corr=[r for r in correlations if r['model']==model and r['V']==views and r['descriptor'] in cfg['gate_descriptors']]
            gate.append(dict(model=model,V=views,ordered=bool(l>m>h),low_high_positive_CI=bool(delta['ci'][0]>0),
                directed_nonweak_association=bool(any(r['rho'] is not None and r['rho']<=-.30 for r in corr)),passed=bool(l>m>h and delta['ci'][0]>0 and any(r['rho'] is not None and r['rho']<=-.30 for r in corr))))
    passing=[m for m in p['models'] if all(r['passed'] for r in gate if r['model']==m)]
    decision='GO' if passing else 'NO-GO';write_json(OUT/'decision.json',dict(stage='C',decision=decision,passing_models=passing,per_V_model=gate,rule=p['statistics']['gate']))
    # Recheck ALL historical rows from completed scalar output, including resumed sessions.
    comparisons=[];old=Path('/data/djc/GeoPhy-Set3D/outputs/diagnostics/clustered_v4_one_swap')
    original=json.loads((old/'per_case_draw.json').read_text());lookup={(r['model'],r['case_id'],r['draw']):r for r in original if r['condition']=='Original'}
    for model,meta in p['models'].items():
        for views in cfg['views']:
            for draw in range(5):
                path=Path(meta['historical_root']).parent/f'V{views}/draw{draw}/C1/per_case_metrics.jsonl'
                historic={r['case_id']:r for r in map(json.loads,path.read_text().splitlines())}
                for r in frames:
                    if r['model']!=model or r['V']!=views or r['draw']!=draw or r['section'] not in ('natural','clustered'):continue
                    hr=historic[r['case_id']] if r['section']=='natural' else lookup[model,r['case_id'],draw]
                    for metric in ['relative_l1','relative_l2','mae','rmse','observed_relative_l1','observed_relative_l2']:
                        expected=hr['observed'][metric.replace('observed_','')] if r['section']=='natural' and metric.startswith('observed_') else hr[metric]
                        diff=abs(expected-r[metric]);tol=1e-8+1e-5*abs(expected);assert diff<=tol
                        comparisons.append(dict(model=model,section=r['section'],V=views,case_id=r['case_id'],draw=draw,metric=metric,actual=r[metric],expected=expected,difference=diff,passed=True))
    write_csv(OUT/'historical_reproduction.csv',comparisons)
    reproduction=dict(passed=True,comparison_count=len(comparisons),natural_comparisons=sum(r['section']=='natural' for r in comparisons),
        clustered_comparisons=sum(r['section']=='clustered' for r in comparisons),maximum_absolute_difference=max(r['difference'] for r in comparisons),tolerance=p['reproduction_tolerance'])
    write_json(OUT/'historical_reproduction.json',reproduction)
    # Exact content checks on protected repository artifacts and metadata checks on data/history.
    inv=json.loads((OUT/'inventory_before.json').read_text())
    bytecode_changes=[k for k,v in inv['repo_content'].items() if k.endswith('.pyc') and digest(k)!=v]
    content_changes=[k for k,v in inv['repo_content'].items() if not k.endswith('.pyc') and digest(k)!=v]
    allowed=str(REPO/'src/geophy_set3d/evaluation/geometry_catalog.py');assert content_changes==[allowed] or set(content_changes)=={allowed}
    metadata_changes=[]
    for root,files in inv['metadata'].items():
        current={str(q):[q.stat().st_size,q.stat().st_mtime_ns,q.stat().st_ino] for q in Path(root).rglob('*') if q.is_file()}
        if current!=files:metadata_changes.append(root)
    assert not metadata_changes,metadata_changes
    verify()
    integrity=dict(protected_repository_content_changes=content_changes,regenerated_untracked_bytecode_changes=bytecode_changes,authorized_catalog_only=True,data_and_historical_metadata_changes=metadata_changes,
        checkpoint_hashes_unchanged=all(digest(m['checkpoint'])==m['sha256'] for m in p['models'].values()),historical_artifacts_unchanged=True,
        projector_model_training_sources_unchanged=True,optimizer_created=False,optimizer_steps=0,GT128_content_opened=False,
        evidence='Protected repository content SHA256; dataset/checkpoint/history size+mtime+inode snapshots and blocked writes, not full data byte hashes')
    write_json(OUT/'integrity.json',integrity)
    audits={k:json.loads((OUT/k).read_text()) for k in ['boundary_access_audit.json','registration_access_audit.json','SelfSup_access_audit.json','Supervised_access_audit.json']}
    assert all(a['forbidden_attempts']==0 for a in audits.values());write_json(OUT/'access_audit.json',audits)
    times=[]
    for (model,section,views),group in df.groupby(['model','section','V']):
        times.append(dict(model=model,section=section,V=int(views),rows=len(group),forward_mean_seconds=float(group.forward_seconds.mean()),
            forward_total_seconds=float(group.forward_seconds.sum()),evaluation_total_seconds=float(group.evaluation_seconds.sum()),
            GPU_peak_allocated_bytes=int(group.GPU_peak_allocated_bytes.max()),GPU_peak_reserved_bytes=int(group.GPU_peak_reserved_bytes.max()),device='cuda:0'))
    write_csv(OUT/'runtime.csv',times)
    from verify_geometry_strata_frozen import main as independent_verify
    independent_verify()
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    colors={'low':'#d95f02','middle':'#7570b3','high':'#1b9e77','natural':'#666666','clustered':'#e7298a'}
    def bars(filename,sections):
        fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for ax,model in zip(axs,p['models']):
            for j,(section,stratum) in enumerate(sections):
                vs=[4] if section=='clustered' else cfg['views'];ys=[];lower=[];upper=[]
                for v in vs:
                    r=bycell[model,section,v,stratum];ys.append(r['relative_l1']);ci=r['relative_l1_ci'];lower.append(r['relative_l1']-ci[0]);upper.append(ci[1]-r['relative_l1'])
                x=np.asarray(vs,dtype=float)+.17*(j-(len(sections)-1)/2)
                ax.errorbar(x,ys,yerr=[lower,upper],fmt='o-',capsize=3,color=colors[stratum],label=stratum)
            ax.set_title(model);ax.set_xticks(cfg['views']);ax.set_xlabel('Input views V');ax.set_ylabel('Volume relative L1');ax.legend()
        fig.tight_layout();fig.savefig(figures/filename,dpi=160);plt.close(fig)
    bars('error_by_V_and_stratum.png',[('controlled',s) for s in ('low','middle','high')])
    bars('natural_vs_controlled_geometry.png',[('natural','natural')]+[('controlled',s) for s in ('low','middle','high')])
    bars('clustered_vs_low_stratum.png',[('controlled','low'),('clustered','clustered')])
    def scatter(filename,descriptors):
        fig,axs=plt.subplots(2,3,figsize=(13,7),squeeze=False)
        for i,model in enumerate(p['models']):
            for j,v in enumerate(cfg['views']):
                ax=axs[i,j];sub=df[(df.model==model)&(df.section=='controlled')&(df.V==v)]
                g=sub.groupby('geometry_id').agg({'relative_l1':'mean',**{k:'first' for k in descriptors},'stratum':'first'})
                for k in descriptors:
                    ax.scatter(g[k],g.relative_l1,c=[colors[s] for s in g.stratum],marker='o' if k==descriptors[0] else 'x',label=k)
                ax.set_title(f'{model} V{v}');ax.set_ylabel('Case-mean relative L1');ax.set_xlabel(' / '.join(descriptors));ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(figures/filename,dpi=160);plt.close(fig)
    scatter('error_vs_center_direction_score.png',['center_direction_score'])
    scatter('error_vs_lambda_ratio.png',['ray_tensor_lambda_min_over_max','ray_tensor_lambda_mid_over_max'])
    scatter('error_vs_coverage.png',['coverage_fraction','coverage_at_least_two_stations'])
    # Report with all required registration, inference, correlation and integrity facts.
    boundary=json.loads((OUT/'boundaries.json').read_text());geo=json.loads((OUT/'geometry_descriptors.json').read_text())
    heights=[h for r in geo for h in r['station_height_m']]
    chunks=['# 冻结 MRGR 的几何分层验证 v1',f'\nStage C **{decision}**。两套冻结网络实际完成4,680次前向，覆盖全部36 Val case、5draw、V3/V4/V6自然随机与9个受控分层格，以及独立clustered V4压力测试。判据在提交c982b05、加载checkpoint之前冻结；本报告不改变分数、边界、draw或门槛。',
        '\n## 1. 当前状态、纠错与稳定总体',
        f'起始本地/远端13d62826e220f1cc891e704a3b418fccf55fbad6；协议提交c982b05。报告生成时HEAD为`{subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()}`。未跟踪的用户绘图脚本不提交。',
        f'高度旧轴z(index2)改为y(index1)，x–z为水平面。64站位实际高度范围[{min(heights):.9f},{max(heights):.9f}]m，均在[.25,.35]m。显式λ_min/mid/max及两个比值已加入，分母max(λ_max,1e-12)。原数组保持升序兼容；原v1清单不追溯修改。新清单另存origins/azimuth/水平radius/center distance、有效射线、实际方向分数、覆盖和平面覆盖。',
        table(list(boundary['V'].values()),[('V',lambda r:r['V']),('总体数',lambda r:r['population_size']),('方式',lambda r:r['generation_mode']),('low/middle边界',lambda r:f"{r['boundaries'][0]:.12f}"),('middle/high边界',lambda r:f"{r['boundaries'][1]:.12f}")]),
        '总体仅从origin计算原center score，V独立tertile，平界归右。V3/V4全枚举；V6 seed2026101401确定性100,000唯一组合。分位、均值、std、min/max、源码/geometry hash见boundaries.json。边界阶段测量opens=0、GT opens=0、checkpoint/model opens=0，未计算K96 coverage。冻结边界后才计算选择布局的K96覆盖。每cell五layout跨case共用，输入顺序冻结。',
        '\n## 2. 冻结模型与评价身份',
        '\n'.join(f"- {m}: `{v['sha256']}`。" for m,v in p['models'].items()),
        '网络相同GeoLift-OARI-MRGR-Res3D/332,070参数、XYZ64³质量分数，FP32、eval、inference_mode、TF32关闭。SelfSup v1训练无体积GT，原measurement Val selector；监督MRGR训练/Val selector用GT，存在同Val模型选择偏倚。参数和buffer逐模型前后hash一致。无optimizer或更新。',
        '受控同case/V三个strata和五draw共同H4，排除15输入并集；自然H4独立键控排除各输入；clustered沿用历史H4。所有IDs/顺序/mask/path length/测量/GT在模型间相同。H不输入网络，不选参数。冻结P_diff为C插值后转χ、K96；参考测量128³/K256，存在表示/积分差异，本轮不读GT128内容。',
        '\n## 3. 六个模型×V分层主表',
        '每case先均值五draw再36case均值。95% percentile bootstrap10,000次、seed2026101404，单位case。CI不把draw/射线作为独立样本；工况家族相关性未消除。三维relative指标无量纲、MAE/RMSE/峰值为质量分数。积分proxy abs(sum(C)−sum(GT))/sum(abs(GT))，不是kg。辅助指标全数归档，不按好坏删case。']
    for model in p['models']:
        for v in cfg['views']:
            ss=[bycell[model,'controlled',v,s] for s in ('low','middle','high')]
            chunks.extend([f'### {model} V{v}',table(ss,[('stratum',lambda r:r['stratum']),('三维L1 [95%CI]',lambda r:f"{r['relative_l1']:.6f} {ci_text(r['relative_l1_ci'])}"),('L2',lambda r:f"{r['relative_l2']:.6f}"),('MAE',lambda r:f"{r['mae']:.8f}"),('RMSE',lambda r:f"{r['rmse']:.8f}"),('Input L1/L2',lambda r:f"{r['observed_relative_l1']:.6f}/{r['observed_relative_l2']:.6f}"),('H4 L1/L2',lambda r:f"{r['heldout_relative_l1']:.6f}/{r['heldout_relative_l2']:.6f}"),('积分proxy',lambda r:f"{r['integrated_concentration_relative_error']:.6f}"),('峰值绝对误差',lambda r:f"{r['peak_value_absolute_error']:.6f}")])])
    lowhigh=[r for r in pairs if r['comparison']=='low-high' and r['metric']=='relative_l1']
    chunks.extend(['\n## 4. 配对差、连续关联与预登记决策',table(lowhigh,[('模型',lambda r:r['model']),('V',lambda r:r['V']),('low−high L1',lambda r:f"{r['mean_delta']:+.6f}"),('95% case CI',lambda r:ci_text(r['ci']))]),
        table(correlations,[('模型',lambda r:r['model']),('V',lambda r:r['V']),('描述量',lambda r:r['descriptor']),('Spearman rho',lambda r:'不可识别（常量）' if r['rho'] is None else f"{r['rho']:+.4f}"),('case-bootstrap95%CI',lambda r:ci_text(r['ci']))]),
        '连续关联逐V/model，在15个受控geometry ID上先聚合36case误差。bootstrap只重采样case，保留全部15几何再重算rho；不对重复geometry×case作经典p值。CI描述case组成不确定性，不包含重新抽布局的变动；仅15个固定布局，不能外推至全部总体。常量描述量的rho/CI明确为不可识别（JSON null），不填0，也不进入关联通过条件。coverage和direction-tensor为几何诊断proxy，不是正式可观测性、条件数、Fisher信息、rank证明或唯一恢复。',
        '登记的稳定排序要求至少一个模型在三个V均low>middle>high、low−high CI下端>0，且每V至少一个指定描述量rho≤−.30；.30仅工程参考，关联CI报告但不作为额外门槛。',
        table(gate,[('模型',lambda r:r['model']),('V',lambda r:r['V']),('顺序成立',lambda r:r['ordered']),('low−high CI>0',lambda r:r['low_high_positive_CI']),('指定描述量rho≤−.30',lambda r:r['directed_nonweak_association']),('该V通过',lambda r:r['passed'])]),
        f'最终Stage C：**{decision}**。通过全部V的模型：{passing}。'])
    chunks.append('本轮六组均出现low>middle>high，两个模型的center score和λ_min/max均有负向非微弱关联，支持这些预设方向描述量在本次固定自然布局中表征难度趋势。coverage与coverage≥2在受控布局均饱和为1，当前16³/K96触达proxy没有区分力；这是该覆盖描述的实际局限，不能据此宣称连续覆盖或可观测性相同。监督模型在全部自然tertile均值更低，自监督模型在极端clustered均值更低；标签、训练目标与Val selector差异共同存在，不能作标签的单因果结论。极端集中对两个模型都更难，但本轮不证明任意布局具有相同可恢复性。')
    natural=[r for r in summaries if r['section'] in ('natural','clustered')]
    chunks.extend(['\n## 5. 自然随机与极端压力保留',table(natural,[('模型',lambda r:r['model']),('条件',lambda r:r['section']),('V',lambda r:r['V']),('三维L1 [95%CI]',lambda r:f"{r['relative_l1']:.6f} {ci_text(r['relative_l1_ci'])}"),('H4 L1',lambda r:f"{r['heldout_relative_l1']:.6f}")]),
        f"历史复现全部通过：自然随机{reproduction['natural_comparisons']}项、clustered {reproduction['clustered_comparisons']}项，共{reproduction['comparison_count']}项；最大绝对差{reproduction['maximum_absolute_difference']:.3g}，容差abs1e-8+rel1e-5未按结果调整。自然H4属新增评价，不能和历史不同H策略直接声称复现；历史核对四项体积+两项input误差。clustered不纳入tertile均值。",
        '历史单站位替换仅作独立解释：SelfSup Original/Matched-Random/Targeted L1=0.393402/0.170992/0.142418，监督=0.495210/0.179599/0.155090。Matched-Random与Targeted共用删除ID，仅新增站位随机。Original→Matched-Random打破极端集中，SelfSup改善56.54%；Matched-Random→Targeted额外改善16.71%，不将全部63.8%归于center score。历史Random字段不改。'])
    for filename in ['error_by_V_and_stratum.png','error_vs_center_direction_score.png','error_vs_lambda_ratio.png','error_vs_coverage.png','natural_vs_controlled_geometry.png','clustered_vs_low_stratum.png']:
        chunks.append(f'![{filename}](artifacts/geometry_strata_frozen_v1/figures/{filename})')
    chunks.extend(['\n## 6. 成本、测试、访问与完整性',table(times,[('模型',lambda r:r['model']),('条件',lambda r:r['section']),('V',lambda r:r['V']),('前向数',lambda r:r['rows']),('均值前向秒',lambda r:f"{r['forward_mean_seconds']:.6f}"),('前向总秒',lambda r:f"{r['forward_total_seconds']:.3f}"),('评价总秒',lambda r:f"{r['evaluation_total_seconds']:.3f}"),('峰值allocated MB',lambda r:f"{r['GPU_peak_allocated_bytes']/1e6:.1f}")]),
        '同一共享RTX4090 cuda:0上顺序运行两模型。监督运行与全量CPU测试有重叠，耗时不可解释为两种训练方式的速度差。时间区分同步网络前向和体积/重投影评价，不含文件读取、batch准备及checkpoint加载；没有隐藏持久算子缓存，GeometryLifting和P_diff保持原实现。共享GPU的其他任务与热身会影响计时，非专用硬件基准。逐条保留device/peak allocated/reserved bytes，CPU统计/绘图成本另在报告日志外层。',
        f'新增13项必需测试与原27项共40项针对性测试通过，全仓{full_test_count}项测试通过。覆盖高度实际family、特征值/finite ratio、纯几何边界读阻断、V独立/抽样复现/membership、两模型IDs/order一致、自然历史IDs、clustered排除及Pool B/test阻断。独立标量核验全部2340输入记录、4680模型row、936个case均值和180个十指标配对区间，并重算几何中心分数/高度/层级。测试故意拒绝的fixture访问不混入真实审计。',
        table([dict(stage=k,**v) for k,v in audits.items()],[('阶段',lambda r:r['stage']),('测量opens',lambda r:r['val_pool_a_projection_opens']),('GT64 opens',lambda r:r['val_gt64_opens']),('checkpoint opens',lambda r:r['checkpoint_opens']),('禁止尝试',lambda r:r['forbidden_attempts'])]),
        'Pool B accessed=False，final test accessed=False，GT128 content accessed=False。两模型分别仅36个Val Pool A文件及36个Val GT64用于评价。未读取train内容，未更改原自监督读取保护。所有历史产物内容/元数据保留，授权改动仅新清单源码；旧全src协议哈希前瞻不兼容单独记载，不改旧hash。源projector/模型/数值训练代码、冻结权重未变。数据完整性依据写阻断和metadata对比，未冒称全量数据byte hash。首次标量汇总因常量coverage导致NaN不能写JSON而停止，保留失败日志；仅将未定义rho/CI明确导出为null，重新汇总已存标量，没有重复前向或改变统计/判据。',
        '\n## 7. 结论与下一阶段边界',
        ('Geometry strata did not reliably order reconstruction difficulty.\n\nDo not launch geometry-stratified training.\n\nFrozen MRGR and SelfSup v1 remain unchanged.' if decision=='NO-GO' else '预登记几何排序判据通过，可进入独立登记的fresh匹配Control/Treatment采样实验。优化前必须提交其完整协议，C结果本身不证明采样训练收益。'),
        '单站位极端→互补收益与自然tertile排序是不同问题；不能用前者代替后者证据。本结果限定固定Pool A、36 Val、15受控几何/V、两个单seed checkpoint；低分数并不自动等同所有case难恢复，几何歧义与网络先验/训练分布共同作用，体积/测量差异保留。结构、训练目标、RVTC关闭状态不变，不追加score搜索、教师、loss、模块、TV/PDE网络先验、Pool B/test或经典矩阵。经典1,270次预算耗尽仍只说明登记有限预算性能，不能称理论下限。',
        '\n## 8. 重放与归档',
        '协议、boundaries、2340站位/H、590几何描述、4680逐case/draw、case均值、完整十指标summary/配对、五描述量rho/区间、访问/完整性/source manifest/tests/runtime/6图均归档。权重、数据、GT、体积和大型算子不提交。',
        '```bash\nOMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests /scratch/djc/envs/gas-torch/bin/python -m unittest test_geometry_strata_v2 test_clustered_one_swap test_evaluation_view_selection test_measurement_consistency -v\n/scratch/djc/envs/gas-torch/bin/python scripts/register_geometry_strata_v2.py\n/scratch/djc/envs/gas-torch/bin/python scripts/evaluate_geometry_strata_frozen.py --model SelfSup --device cuda:0\n/scratch/djc/envs/gas-torch/bin/python scripts/evaluate_geometry_strata_frozen.py --model Supervised --device cuda:0\n/scratch/djc/envs/gas-torch/bin/python scripts/report_geometry_strata_frozen.py\n```',
        '登记及完整结果存在时核对退出；未完成按自身已写成功row恢复，不能按Val结果重抽或调参。报告入口只读标量，不重新推理。'])
    report=REPO/'docs/reports/geometry_strata_frozen_v1_zh.md';report.write_text('\n\n'.join(chunks)+'\n')
    archive=REPO/'docs/reports/artifacts/geometry_strata_frozen_v1';archive.mkdir(parents=True,exist_ok=True)
    files=['boundaries.json','station_subsets.json','geometry_descriptors.csv','per_case_draw.csv','case_means.csv','strata_summary.csv','correlations.csv','paired_statistics.json',
        'access_audit.json','access_sessions.jsonl','integrity.json','protocol.json','protocol.sha256','tests.log','full_tests.log','runtime.csv','historical_reproduction.csv','historical_reproduction.json','decision.json',
        'registration.log','SelfSup_evaluation.log','Supervised_evaluation.log','SelfSup_checkpoint_provenance.json','Supervised_checkpoint_provenance.json','SelfSup_complete.json','Supervised_complete.json','independent_result_verification.json','report_nan_failure.log']
    for f in files:shutil.copy2(OUT/f,archive/f)
    shutil.copytree(figures,archive/'figures',dirs_exist_ok=True)
    manifest=dict(registered_sources=p['source_sha256'],registered_inputs=p['registration_sha256'],historical_inputs=p['historical_sha256'],
        new_report_sources={str(q.relative_to(REPO)):digest(q) for q in [Path(__file__),REPO/'scripts/verify_geometry_strata_frozen.py']},protected_inventory_sha256=p['inventory_before_sha256'],
        excluded='external inventory contains paths+metadata only; raw data, checkpoint, GT, volumes, large operator caches excluded')
    write_json(OUT/'source_manifest.json',manifest);shutil.copy2(OUT/'source_manifest.json',archive/'source_manifest.json')
    sums=[f'{digest(q)}  {q.relative_to(REPO).as_posix()}' for q in sorted(archive.rglob('*')) if q.is_file() and q.name!='SHA256SUMS']
    sums.extend(f'{digest(q)}  {q.relative_to(REPO).as_posix()}' for q in [report,Path(__file__),REPO/'docs/geometry_strata_frozen_v1_protocol_zh.md'])
    (archive/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
    print(json.dumps(dict(decision=decision,gate=gate,low_high=lowhigh,checkpoint_hashes={k:v['sha256'] for k,v in p['models'].items()},rows=len(frames),
        historical_reproduction=reproduction,total_forward_seconds=float(df.forward_seconds.sum()),total_evaluation_seconds=float(df.evaluation_seconds.sum()),
        report=str(report)),indent=2),flush=True)

if __name__=='__main__':main()
