#!/usr/bin/env python3
"""Case-paired statistics, honest geometry/fit associations, and reproducible figures."""
import argparse
import itertools
import json
import shutil
import sys
import time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geophy_set3d.evaluation.geometry_sensitivity import correlation_rows
from geophy_set3d.data.build_utils import digest
from geophy_set3d.training.utils import write_json
from run_geometry_sensitivity import configuration, verify, inventory

METRICS=['relative_l1','relative_l2','mae','rmse','observed_relative_l1','observed_relative_l2',
         'heldout_relative_l1','heldout_relative_l2','integrated_concentration_relative_error','peak_value_absolute_error']
# Frozen before new inference in the registration commit. Signs define quality,
# not fitted weights. All associations are reported, not only the winner.
DESCRIPTORS={
 'center_direction_score':1,'actual_ray_direction_score_sampled':1,
 'ray_tensor_lambda_min_over_max':1,'ray_tensor_lambda_mid_over_max':1,
 'min_pairwise_distance_m':1,'mean_pairwise_distance_m':1,'max_pairwise_distance_m':1,
 'horizontal_hull_area_m2':1,'station_hull_volume_m3':1,'station_hull_surface_area_m2':1,
 'min_center_acute_angle_deg':1,'mean_center_acute_angle_deg':1,
 'station_azimuth_span_deg':1,'center_abs_cosine_redundancy':-1,
 'actual_ray_abs_cosine_redundancy':-1,'coverage_fraction':1,'coverage_at_least_two_stations':1,
 'coarse_sigma_min_raw_ppm_m':1,'coarse_sigma_min_over_max':1,'coarse_condition_number':-1,
 'coarse_retained_condition_number':-1,'coarse_numeric_rank':1,'coarse_effective_rank':1,'coarse_stable_rank':1}
GROUPS=['G1','G2','G3','G4','G5']
COLORS=['#0072B2','#009E73','#E69F00','#D55E00','#CC79A7']


def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);adjusted=np.empty_like(p)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return adjusted


def ci(v):
    a=np.asarray(v);a=a[np.isfinite(a)]
    return np.quantile(a,[.025,.975]).tolist() if len(a) else None


def fit_correlations(x,y,index):
    def residual(v):return v-v.mean(1,keepdims=True)-v.mean(0,keepdims=True)+v.mean()
    def corr(a,b,rank):return float(correlation_rows(a.ravel(),b.ravel(),rank))
    rows=[]
    for scope in ['pooled_descriptive','two_way_centered']:
        a,b=(x,y) if scope=='pooled_descriptive' else (residual(x),residual(y))
        for rank in [False,True]:
            boot=[]
            for batch in np.array_split(index,40):
                for sample in batch:
                    xx,yy=x[sample],y[sample]
                    if scope=='two_way_centered':xx,yy=residual(xx),residual(yy)
                    boot.append(corr(xx,yy,rank))
            rows.append(dict(scope=scope,correlation='Spearman' if rank else 'Pearson',
                coefficient=corr(a,b,rank),case_bootstrap_95ci=ci(boot),p_value=None,
                note='case bootstrap; no 900-independent-row p-value'))
    return rows


def table(rows,fields):
    return '\n'.join(['| '+' | '.join(k for k,_ in fields)+' |','| '+' | '.join('---' for _ in fields)+' |']+
                     ['| '+' | '.join(str(f(r)) for _,f in fields)+' |' for r in rows])


def fmt(v):return '未定义' if v is None else f'{v:.5f}'

def fmtci(v):return '未定义' if v is None else '['+', '.join(f'{a:+.5f}' for a in v)+']'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(REPO/'configs/geometry_sensitivity_v1.yaml'))
    args=parser.parse_args();cfg=configuration(args.config);p=verify(cfg);out=Path(cfg['paths']['output']);start=time.perf_counter()
    assert json.loads((out/'complete.json').read_text())['rows']==900
    records=json.loads((out/'station_subsets.json').read_text())
    rows=[json.loads(l) for l in (out/'rows.jsonl').read_text().splitlines()];assert len(rows)==900
    assert len({(r['case_id'],r['geometry_id']) for r in rows})==900
    expected={(r['case_id'],r['geometry_id']):r for r in records}
    for r in rows:
        e=expected[r['case_id'],r['geometry_id']]
        assert r['station_ids']==e['station_ids'] and r['H_station_ids']==e['H_station_ids']
        assert r['checkpoint_sha256']==cfg['checkpoint_sha256']
    descriptors=json.loads((out/'geometry_layouts.json').read_text())['descriptors']
    gd=pd.DataFrame(descriptors).set_index('geometry_id');df=pd.DataFrame(rows)
    for name in ['station_origins_m','station_height_m','station_distance_to_center_m',*DESCRIPTORS]:df[name]=df.geometry_id.map(gd[name])
    df.to_csv(out/'geometry_sensitivity_results.csv',index=False,lineterminator='\n')
    counts=df.groupby(['case_id','group']).size();assert len(counts)==180 and (counts==5).all()
    cm=df.groupby(['case_id','group'])[METRICS].mean().reset_index();cm.to_csv(out/'case_means.csv',index=False,lineterminator='\n')
    cases=sorted(df.case_id.unique());assert len(cases)==36
    index=np.random.default_rng(cfg['bootstrap_seed']).integers(0,36,(cfg['bootstrap_samples'],36))
    summaries=[];pairs=[]
    for group in GROUPS:
        sub=cm[cm.group==group].set_index('case_id').loc[cases]
        for metric in METRICS:
            y=sub[metric].to_numpy()
            summaries.append(dict(group=group,metric=metric,mean=float(y.mean()),median=float(np.median(y)),
                std=float(y.std(ddof=1)),q25=float(np.quantile(y,.25)),q75=float(np.quantile(y,.75)),
                worst=float(y.max()),worst_case=cases[int(y.argmax())],mean_95ci=ci(y[index].mean(1))))
    for metric in METRICS:
        current=[]
        for left,right in itertools.combinations(GROUPS,2):
            a=cm[cm.group==left].set_index('case_id').loc[cases,metric].to_numpy()
            b=cm[cm.group==right].set_index('case_id').loc[cases,metric].to_numpy();delta=b-a
            pv=float(wilcoxon(delta,method='auto').pvalue) if np.any(delta) else 1.
            current.append(dict(metric=metric,comparison=right+'-'+left,mean_delta=float(delta.mean()),
                median_delta=float(np.median(delta)),case_bootstrap_95ci=ci(delta[index].mean(1)),
                worsened_cases=int((delta>0).sum()),improved_cases=int((delta<0).sum()),tied_cases=int((delta==0).sum()),
                wilcoxon_p=pv,per_case_delta=dict(zip(cases,delta.tolist()))))
        for row,pv in zip(current,holm([r['wilcoxon_p'] for r in current])):row['wilcoxon_holm_p']=float(pv)
        pairs.extend(current)
    pd.DataFrame(summaries).to_csv(out/'group_summary.csv',index=False,lineterminator='\n')
    write_json(out/'paired_statistics.json',pairs)
    geomids=sorted(gd.index);y=df.pivot(index='case_id',columns='geometry_id',values='relative_l1').loc[cases,geomids].to_numpy()
    bygeom=gd.loc[geomids];ym=y.mean(0);boot=y[index].mean(1)
    permutation=np.random.default_rng(cfg['permutation_seed']).random((cfg['permutations'],25)).argsort(1)
    correlations=[]
    for scope in ['all_G1_G5','natural_G1_G4']:
        selected=np.ones(25,bool) if scope=='all_G1_G5' else (bygeom.group!='G5').to_numpy()
        for descriptor in DESCRIPTORS:
            x=bygeom[descriptor].to_numpy(dtype=float)[selected];values=ym[selected]
            valid=np.isfinite(x).all() and np.ptp(x)>0
            for rank in [False,True]:
                if valid:
                    coefficient=float(correlation_rows(x,values,rank));interval=ci(correlation_rows(x,boot[:,selected],rank))
                    perm=permutation if scope=='all_G1_G5' else np.random.default_rng(cfg['permutation_seed']).random((cfg['permutations'],selected.sum())).argsort(1)
                    null=correlation_rows(x,values[perm],rank)
                    pv=float((1+(abs(null)>=abs(coefficient)).sum())/(cfg['permutations']+1))
                else:coefficient=interval=pv=None
                correlations.append(dict(scope=scope,descriptor=descriptor,correlation='Spearman' if rank else 'Pearson',
                    coefficient=coefficient,case_bootstrap_95ci=interval,geometry_permutation_p=pv,
                    status='defined' if valid else 'constant or nonfinite (rank-deficient full coarse condition is undefined)',
                    geometry_count=int(selected.sum()),quality_direction=DESCRIPTORS[descriptor]))
        for kind in ['Pearson','Spearman']:
            current=[r for r in correlations if r['scope']==scope and r['correlation']==kind and r['geometry_permutation_p'] is not None]
            for r,v in zip(current,holm([r['geometry_permutation_p'] for r in current])):r['holm_p']=float(v)
    write_json(out/'geometry_correlations.json',correlations);pd.DataFrame(correlations).to_csv(out/'geometry_correlations.csv',index=False,lineterminator='\n')
    xfit=df.pivot(index='case_id',columns='geometry_id',values='observed_relative_l1').loc[cases,geomids].to_numpy()
    fit=fit_correlations(xfit,y,index)
    for group in GROUPS:
        selection=(bygeom.group==group).to_numpy()
        for kind,rank in [('Pearson',False),('Spearman',True)]:
            xx,yy=xfit[:,selection],y[:,selection]
            coefficient=float(correlation_rows(xx.ravel(),yy.ravel(),rank))
            b=[]
            for ids in index:b.append(float(correlation_rows(xx[ids].ravel(),yy[ids].ravel(),rank)))
            fit.append(dict(scope=group,correlation=kind,coefficient=coefficient,case_bootstrap_95ci=ci(b),p_value=None))
    percase=[]
    for j,case in enumerate(cases):
        percase.append(dict(case_id=case,Pearson=float(correlation_rows(xfit[j],y[j])),Spearman=float(correlation_rows(xfit[j],y[j],True))))
    write_json(out/'measurement_voxel_correlations.json',dict(associations=fit,within_case=percase))
    # Winner is exploratory and GT-informed; only a single predeclared metric.
    candidates=[r for r in correlations if r['scope']=='all_G1_G5' and r['correlation']=='Spearman' and r['coefficient'] is not None and r['coefficient']*r['quality_direction']<0]
    best=sorted(candidates,key=lambda r:(-abs(r['coefficient']),r['descriptor']))[0]
    q=bygeom[best['descriptor']].to_numpy(dtype=float)*best['quality_direction'];score=(q-q.min())/(q.max()-q.min())
    score_info=dict(identity='exploratory GT-informed selection, not a prospective validated score',descriptor=best['descriptor'],
        direction=best['quality_direction'],geometry_min=float(q.min()),geometry_max=float(q.max()),
        coefficient=best['coefficient'],case_bootstrap_95ci=best['case_bootstrap_95ci'],
        selection='largest absolute Spearman with correctly directed sign among registered nonconstant metrics, using Val GT-derived L1',
        not_used_for='groups, network, parameters, historical boundaries, or training')
    write_json(out/'exploratory_score.json',score_info)
    pd.DataFrame(dict(geometry_id=geomids,exploratory_score=score,mean_relative_l1=ym)).to_csv(out/'exploratory_score.csv',index=False,lineterminator='\n')
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'font.family':'DejaVu Sans','pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(figures/(name+'.png'),bbox_inches='tight');fig.savefig(figures/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    def scatter(ax,x,yy,label,group_labels):
        for group,color in zip(GROUPS,COLORS):
            m=np.asarray(group_labels)==group;ax.scatter(np.asarray(x)[m],np.asarray(yy)[m],s=28,color=color,label=group,alpha=.8,edgecolor='white',linewidth=.4)
        if np.ptp(x)>0:
            line=np.polyfit(x,yy,1);ends=np.array([min(x),max(x)]);ax.plot(ends,np.polyval(line,ends),color='#333333',lw=1)
            ax.set_title(f'Pearson r={float(correlation_rows(x,yy)):.3f}; Spearman rho={float(correlation_rows(x,yy,True)):.3f}')
        else:ax.set_title('Constant descriptor: correlation undefined')
        ax.set_xlabel(label);ax.set_ylabel('3D relative L1');ax.grid(alpha=.15)
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,name,label in zip(axes.ravel(),['center_direction_score','mean_center_acute_angle_deg','coarse_sigma_min_over_max','coarse_effective_rank','horizontal_hull_area_m2','min_pairwise_distance_m'],
                            ['Frozen center direction score','Mean acute crossing angle (deg)','Coarse sigma_min / sigma_max (8^3 chi)','Coarse entropy effective rank (8^3 chi)','Horizontal station hull area (m^2)','Minimum station distance (m)']):
        scatter(ax,bygeom[name].to_numpy(float),ym,label,bygeom.group)
    axes[0,0].legend(frameon=False);save(fig,'figure1_geometry_vs_error')
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    arrays=[cm[cm.group==g].set_index('case_id').loc[cases,'relative_l1'].to_numpy() for g in GROUPS]
    box=ax.boxplot(arrays,tick_labels=GROUPS,patch_artist=True,showfliers=True)
    for patch,color in zip(box['boxes'],COLORS):patch.set_facecolor(color);patch.set_alpha(.5)
    for case in range(36):ax.plot(range(1,6),[a[case] for a in arrays],color='gray',alpha=.13,lw=.6)
    ax.set_ylabel('Case mean 3D relative L1 (five layouts)');ax.set_xlabel('Prespecified geometry group');ax.grid(axis='y',alpha=.15);save(fig,'figure2_group_boxplot')
    rx=xfit-xfit.mean(1,keepdims=True)-xfit.mean(0,keepdims=True)+xfit.mean()
    ry=y-y.mean(1,keepdims=True)-y.mean(0,keepdims=True)+y.mean()
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    labels=np.tile(bygeom.group.to_numpy(),36)
    scatter(axes[0],xfit.ravel(),y.ravel(),'Input measurement relative L1',labels)
    scatter(axes[1],rx.ravel(),ry.ravel(),'Two-way centered input relative L1',labels)
    axes[1].set_ylabel('Two-way centered voxel relative L1');axes[0].legend(frameon=False);save(fig,'figure3_measurement_vs_voxel')
    fig,axes=plt.subplots(3,4,figsize=(11,8),constrained_layout=True)
    for row,caseidx in enumerate(cfg['representative_case_indices']):
        volumes=[np.load(out/'case_images'/f'case{caseidx}_{group}.npz') for group in cfg['representative_groups']]
        gt=volumes[0]['GT'];assert all(np.array_equal(gt,v['GT']) for v in volumes)
        arrays=[gt]+[v['prediction'] for v in volumes];vmax=float(gt.max())
        for col,(name,a) in enumerate(zip(['GT',*cfg['representative_groups']],arrays)):
            im=axes[row,col].imshow(a[:,cfg['slice_y_index'],:].T,origin='lower',extent=[0,.6,0,.6],vmin=0,vmax=vmax,cmap='inferno',interpolation='nearest')
            subset=df[(df.case_index==caseidx)&(df.group==name)&(df.draw==cfg['representative_draw'])]
            axes[row,col].set_title(name if name=='GT' else f'{name}: L1={subset.iloc[0].relative_l1:.3f}')
            axes[row,col].set_xlabel('x (m)');axes[row,col].set_ylabel(f'case index {caseidx}; z (m)')
        fig.colorbar(im,ax=axes[row,:].tolist(),label='CH4 mass fraction; shared scale within case',shrink=.85)
    save(fig,'figure4_paired_slices')
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True);scatter(ax,score,ym,'Exploratory GT-informed geometry score',bygeom.group);ax.legend(frameon=False);save(fig,'figure5_exploratory_score')
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for spec in json.loads((out/'spectra.json').read_text()):
        group=gd.loc[spec['geometry_id'],'group'];s=np.array(spec['singular_values_ppm_m']);ax.semilogy(np.arange(1,len(s)+1),np.maximum(s/s[0],1e-12),color=COLORS[GROUPS.index(group)],alpha=.65,lw=.8)
    ax.axhline(cfg['spectral_relative_sigma_tolerance'],ls='--',color='black',lw=1,label='Numerical rank threshold')
    for group,color in zip(GROUPS,COLORS):ax.plot([],[],color=color,label=group)
    ax.set_xlabel('Singular value index (512 coarse nodal unknowns)');ax.set_ylabel('sigma / sigma_max; floor 1e-12 for display');ax.legend(frameon=False,ncol=2);save(fig,'figure6_coarse_spectra')
    audit={name:json.loads((out/name).read_text()) for name in ['registration_access_audit.json','smoke_access_audit.json','full_access_audit.json']}
    assert all(r['forbidden_attempts']==0 for r in audit.values())
    write_json(out/'access_audit.json',audit)
    before=json.loads((out/'inventory_before.json').read_text());now=inventory(cfg)
    assert before==now,'Protected historical/data metadata changed; inspect before reporting'
    integrity=dict(passed=True,protected_dataset_and_history_metadata_unchanged=True,
        registered_source_and_input_hashes_unchanged=True,checkpoint_sha256=cfg['checkpoint_sha256'],
        historical_artifacts_overwritten=False,pool_b_accessed=False,final_test_accessed=False,gt128_content_accessed=False)
    write_json(out/'integrity.json',integrity)
    runtimes=[]
    for group,sub in df.groupby('group'):
        runtimes.append(dict(group=group,forwards=len(sub),forward_mean_seconds=float(sub.forward_seconds.mean()),
            forward_total_seconds=float(sub.forward_seconds.sum()),evaluation_total_seconds=float(sub.evaluation_seconds.sum()),
            peak_gpu_MB=float(sub.GPU_peak_allocated_bytes.max()/1e6)))
    pd.DataFrame(runtimes).to_csv(out/'runtime.csv',index=False,lineterminator='\n')
    l1=[r for r in summaries if r['metric']=='relative_l1'];core={r['group']:r for r in l1}
    allrho=next(r for r in correlations if r['scope']=='all_G1_G5' and r['descriptor']=='center_direction_score' and r['correlation']=='Spearman')
    pair=next(r for r in pairs if r['metric']=='relative_l1' and r['comparison']=='G5-G1')
    geometry_supported=pair['case_bootstrap_95ci'][0]>0 and allrho['coefficient']<-.3
    decision=dict(geometry_dependence_supported=bool(geometry_supported),
        geometry_dominates_network_capacity_proven=False,
        augmentation_sampler_implementation_justified=bool(geometry_supported),
        formal_training_launched=False,curriculum_or_weighting_launched=False,
        reason='Strong fixed-network geometry intervention can motivate a new explicit hard-tail sampler; previous same-q joint sampler NO-GO is retained. Capacity/optimization/loss dominance untested.')
    write_json(out/'decision.json',decision)
    natural=json.loads((Path(cfg['paths']['historical_strata'])/'SelfSup_complete.json').read_text())
    history=pd.DataFrame([json.loads(l) for l in (Path(cfg['paths']['historical_strata'])/'SelfSup_rows.jsonl').read_text().splitlines()])
    hs=history[(history.section=='natural')&(history.V==4)];historical_natural=float(hs.groupby('case_id').relative_l1.mean().mean())
    swap=pd.DataFrame(json.loads((Path(cfg['paths']['historical_swap'])/'per_case_draw.json').read_text()))
    swap=swap[swap.model=='SelfSup'];swapmeans=swap.groupby(['condition','case_id']).relative_l1.mean().groupby('condition').mean().to_dict()
    write_json(out/'historical_context.json',dict(natural_V4_mean_l1=historical_natural,swap_mean_l1=swapmeans,
        source_sha256=p['historical_sha256'],identity='reused completed original-v1 scalar results; separate layout interventions, not new group draws'))
    fulltest=(out/'full_tests.log').read_text();assert fulltest.rstrip().endswith('OK')
    text=['# Geometry Sensitivity Analysis：冻结 SelfSup v1 中文总结',
        f'本轮实际完成900次原SelfSup v1冻结前向：36 Val案例×25固定V4布局（G1–G5各五draw），先完成五条smoke再恢复余895条。checkpoint `{cfg["checkpoint_sha256"]}`，未训练/改网络/损失/投影器。保留上轮同q A/B采样NO-GO。几何依赖结论：**{"支持" if geometry_supported else "证据不足"}**；“几何比容量更主要”没有被证明。',
        '## 1. 五档与测量身份',
        '定义和预算在新推理前提交，见[协议](../geometry_sensitivity_v1_protocol_zh.md)。Pool A全部635,376组合，仅按预生成几何确定q05/q40/q60/q80；G1≥q80、G2[q60,q80)、G3[q40,q60)、G4[q05,q40)、G5<q05且最小方位跨度≤5°。G5是极端压力组，五档不是自然总体等概率代表；排除其余低尾组合，全部case保留。每case/draw H4在五档输入并集外且组间相同，不参与prediction。输入ppm·m、C为XYZ64³质量分数[0,1]，y为高度。',
        '## 2. 案例级分布与配对干预',
        table(l1,[('组',lambda r:r['group']),('mean [95%CI]',lambda r:fmt(r['mean'])+' '+fmtci(r['mean_95ci'])),('median',lambda r:fmt(r['median'])),('sample std',lambda r:fmt(r['std'])),('q25/q75',lambda r:fmt(r['q25'])+'/'+fmt(r['q75'])),('worst',lambda r:fmt(r['worst']))]),
        'worst case IDs完整保存在group_summary.csv；所有十指标的分布和case均值完整归档。先平均五draw，再以36case为统计单位，bootstrap10,000/seed2026101504。工况家族相关性未消除，区间是本案例集合的描述性不确定性。',
        table([r for r in pairs if r['metric']=='relative_l1'],[('差（后档−前档）',lambda r:r['comparison']),('L1差 [95%CI]',lambda r:fmt(r['mean_delta'])+' '+fmtci(r['case_bootstrap_95ci'])),('变差/改善/平',lambda r:f"{r['worsened_cases']}/{r['improved_cases']}/{r['tied_cases']}"),('Wilcoxon Holm p',lambda r:f"{r['wilcoxon_holm_p']:.3g}")]),
        '## 3. 几何与误差：25布局均值及20布局敏感性',
        '关联先在每geometry ID上平均36case误差；Pearson/Spearman CI只bootstrap案例，布局保持固定。置换p在布局级计算，各scope×相关类型跨描述量Holm，不能当900独立观测。下表列全部五档Spearman；Pearson及去除G5的G1–G4关联全数见geometry_correlations.csv/json。多描述量、固定单seed/Pool A和同Val开发样本限制外推。',
        table([r for r in correlations if r['scope']=='all_G1_G5' and r['correlation']=='Spearman'],[('描述量',lambda r:r['descriptor']),('rho [95%CI]',lambda r:fmt(r['coefficient'])+' '+fmtci(r['case_bootstrap_95ci'])),('Holm置换p',lambda r:fmt(r.get('holm_p')))]),
        '覆盖触达proxy可能饱和；若常量则null/未定义，不填0。凸包描述的是站位分布，不是重建域信息充分性。距离/体积/方向之间共线，相关不等于各指标独立因果贡献。',
        '### 算子谱的范围',
        '实际射线、全部valid、K96、三维稀疏线性chi算子，8³=512未知；仅Gram512×512稠密。保存全512奇异值、熵有效秩/稳定秩/数值秩、raw sigma_min和保留子空间condition。阈值1e−6 sigma_max；秩亏整体condition=null，不将保留子空间condition冒称全场条件数。Gram小sigma受舍入影响。冻结P_diff先插值C再非线性转chi；粗chi谱不是其Jacobian，更不是完整64³谱。每布局full64零空间下界见geometry_descriptors.csv，完整64³稀疏视角线性算子必有零空间；粗谱有限condition不能证明唯一恢复。',
        '## 4. Measurement fitting与voxel正确性：不预设弱关联',
        table(fit,[('范围',lambda r:r['scope']),('相关',lambda r:r['correlation']),('系数 [case CI]',lambda r:fmt(r['coefficient'])+' '+fmtci(r['case_bootstrap_95ci']))]),
        '全行pooled仅描述性、不给独立行p值；组内和去除case/layout双向均值用于检验混合几何及case难度是否驱动关系。每case跨25布局关联另存within_case。输入拟合是网络输出经同一P_diff的误差，无任何优化或拟合后选点；H4独立重投影另报。因此弱/强必须按上述范围回答，不能先假定projection consistency与体积精度无关。',
        '## 5. 图与exploratory score',
        f'根据预登记选择规则，候选中最强正确方向单指标为 `{best["descriptor"]}`；Spearman={best["coefficient"]:+.5f}。按当前25布局几何范围minmax，质量方向={best["quality_direction"]:+d}。这是**exploratory、GT-informed**：指标身份依据Val GT误差关联选出，存在选择偏倚，尚未在新布局/独立集验证。不改旧center score、边界或模型，不用于本次组定义/调参/训练；不拟合多指标权重。',
        '代表case预先固定0/18/35、draw0、G1/G3/G5，均同y=32水平切片，统一XYZ轴和每case GT全体积max色标，case之间GT峰值不同所以色标可不同；没有挑最优/最差GT iterate或case。']
    for name in ['figure1_geometry_vs_error','figure2_group_boxplot','figure3_measurement_vs_voxel','figure4_paired_slices','figure5_exploratory_score','figure6_coarse_spectra']:
        text.append(f'![{name}](artifacts/geometry_sensitivity_v1/figures/{name}.png)')
    text.extend(['## 6. 七个问题与下一阶段判断',
        f'1. 普通自然V4历史case平均relative L1={historical_natural:.6f}，本轮G1/G2/G3={core["G1"]["mean"]:.6f}/{core["G2"]["mean"]:.6f}/{core["G3"]["mean"]:.6f}。在当前单seed/Val范围可称相对稳定，不能称已证明跨seed/任意几何稳定。',
        f'2. G5={core["G5"]["mean"]:.6f}，G5−G1={pair["mean_delta"]:+.6f}，95%CI={fmtci(pair["case_bootstrap_95ci"])}，变差{pair["worsened_cases"]}/36case；这是冻结模型的配对几何干预，不是网络新能力。',
        f'3. 本次25布局最强单指标是{best["descriptor"]}，但须结合去除G5的关联、尺度/表示差异和共线性，不能当理论可观测性。',
        '4. 测量与voxel关联见上表，按全局、组内和双向去均值分别解读；好的拟合并不保证唯一正确三维场，亦不能把实际正相关说成“很弱”。',
        f'5. 独立历史单站替换Original/Matched-Random/Targeted L1={swapmeans["Original"]:.6f}/{swapmeans["Random"]:.6f}/{swapmeans["Targeted"]:.6f}；保留原完整来源。两段收益分别是打破极端集中及额外互补指导，不将全部收益归于score。',
        '6. 本轮证实或量化几何敏感性，并未操纵容量、优化或physics loss，不能排除这些因素及网络先验/训练分布。主瓶颈四选一没有可识别的比较证据；准确判断是“几何是当前v1可操作的显著瓶颈之一”，而非“容量不是问题”。',
        '7. 若配对和关联支持几何依赖，有理由实现独立显式normal/moderate/extreme增强sampler并做读取/数值smoke，保留v1。上次same-q方案主要约束A/B联合分布且普通几何质量未提升，其NO-GO保留；新的50/25/25并未被该结果检验。该判据不等于证明增强会成功。正式训练需独立冻结明确匹配预算/协议；当前不自动启动curriculum或hard weighting。',
        '## 7. 成本、保护、验证与交付',
        table(runtimes,[('组',lambda r:r['group']),('前向数',lambda r:r['forwards']),('均值/总前向秒',lambda r:f"{r['forward_mean_seconds']:.5f}/{r['forward_total_seconds']:.2f}"),('评价总秒',lambda r:f"{r['evaluation_total_seconds']:.2f}"),('峰值MB',lambda r:f"{r['peak_gpu_MB']:.1f}")]),
        'GPU为共享RTX4090 cuda:2；同步前向包含网络内部MRGR投影，评价含GT加载及两个P_diff，不含最外层projection读取/batch准备/checkpoint加载。smoke第一条含CUDA冷启动，逐条保留而不隐藏；不可当专用硬件性能基准。算子冷构建、Gram缓存、谱计算单独在operator_cost.json及描述CSV；只用于几何诊断，不加速网络。统计/绘图墙钟另存report_runtime.json。',
        '注册不读measurement/GT/checkpoint内容；实际推理只读36 Val Pool A和36 Val GT64，GT/H在prediction之后用于评价。原训练读取保护未修改。注册、smoke、full访问计数见access_audit.json，禁止尝试均0；源/权重/历史输入byte hash及数据/历史产物metadata核对通过。没有全量GT数据byte hash的夸大声明。未改任何历史数据、checkpoint、配置、数值训练代码、协议或实验结论。未跟踪用户绘图脚本保留且不提交。',
        '新增与已有针对性40项测试通过；完整仓库测试日志见full_tests.log。边界/圆周跨度/凸包轴单位/秩亏/常量关联测试与原RayMatrix伴随、phantom、单位/非线性/物理梯度验证复用；真实全部使用的站位另做float64伴随内积。主CSV900行，180 case×group均值、100个十指标配对、完整相关/谱/固定代表图/审计归档。',
        '源码入口均从config/CLI解析数据与输出路径，没有硬编码部署绝对路径；权重、measurement、GT、体积和大算子缓存仅外部存储不提交。V3/V6新五档、Pool B、最终test、容量/损失改动及curriculum均未运行。',
        '```bash\npython scripts/run_geometry_sensitivity.py --config configs/geometry_sensitivity_v1.yaml --stage register\npython scripts/run_geometry_sensitivity.py --config configs/geometry_sensitivity_v1.yaml --stage smoke --device cuda:2\npython scripts/run_geometry_sensitivity.py --config configs/geometry_sensitivity_v1.yaml --stage full --device cuda:2\npython scripts/report_geometry_sensitivity.py --config configs/geometry_sensitivity_v1.yaml\npython scripts/verify_geometry_sensitivity.py --config configs/geometry_sensitivity_v1.yaml\n```'])
    report=REPO/'docs/reports/geometry_sensitivity_summary.md';report.write_text('\n\n'.join(text)+'\n')
    write_json(out/'report_runtime.json',dict(seconds=time.perf_counter()-start,geometry_forward_seconds=float(df.forward_seconds.sum()),evaluation_seconds=float(df.evaluation_seconds.sum())))
    dest=REPO/'docs/reports/artifacts/geometry_sensitivity_v1';dest.mkdir(exist_ok=True)
    for name in ['geometry_sensitivity_results.csv','case_means.csv','group_summary.csv','paired_statistics.json','geometry_correlations.json','geometry_correlations.csv',
                 'measurement_voxel_correlations.json','exploratory_score.json','exploratory_score.csv','decision.json','historical_context.json','integrity.json','access_audit.json',
                 'checkpoint_provenance.json','smoke_complete.json','complete.json','runtime.csv','report_runtime.json','full_tests.log','tests.log','registration.log','smoke.log','evaluation.log','access_sessions.jsonl']:
        shutil.copy2(out/name,dest/name)
    shutil.copytree(figures,dest/'figures',dirs_exist_ok=True)
    print(json.dumps(dict(report=str(report),groups={g:core[g]['mean'] for g in GROUPS},G5_minus_G1=pair,score=score_info,decision=decision),indent=2))

if __name__=='__main__':main()
