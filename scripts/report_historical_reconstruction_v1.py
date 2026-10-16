#!/usr/bin/env python3
"""Posthoc matched image/shape comparison of frozen historical MRGR and A/B."""
import json
from pathlib import Path

import numpy as np
import analyze_frozen_ab_morphology_v1 as ab
from analyze_historical_reconstruction_v1 import OUT, REFERENCE

ROOT=ab.ROOT
MODELS=('History-C0','History','A','B')


def metrics(p,g):
    pc,ps=ab.moments(p);gc,gs=ab.moments(g)
    result=dict(relative_l1=float(np.abs(p-g).sum()/g.sum()),
        peak_ratio=float(p.max()/g.max()),gt_peak_location_ratio=float(p[np.unravel_index(g.argmax(),g.shape)]/g.max()),
        mass_ratio=float(p.sum()/g.sum()),centroid_distance_mm=float(np.linalg.norm(pc-gc)*1000),
        centroid_delta_xyz_mm=((pc-gc)*1000).tolist(),spread_ratio_xyz=(ps/gs).tolist())
    for fraction in (.01,.1,.5):
        gm=g>=fraction*g.max();pm=p>=fraction*g.max()
        result[f'dice_{fraction}']=float(2*np.logical_and(gm,pm).sum()/(gm.sum()+pm.sum()))
    gm=g>=.5*g.max()
    result['high_concentration_deficit']=float(np.maximum(g-p,0)[gm].sum()/g[gm].sum())
    result['background_predicted_fraction']=float(p[g<.01*g.max()].sum()/p.sum())
    return result


def main():
    complete=ab.read(OUT/'predictions_complete.json');protocol=ab.read(OUT/'protocol.json')
    assert complete['all_predictions_saved_before_gt'] and complete['cases_layouts']==72
    assert complete['protocol_sha256']==ab.sha(OUT/'protocol.json')
    assert complete['manifest_sha256']==ab.sha(OUT/'prediction_manifest.jsonl')
    hist=ab.rows(OUT/'prediction_manifest.jsonl')
    predictions={};archived={};evidence=[]
    for r in hist:
        assert ab.sha(OUT/r['prediction'])==r['sha256']
    for cond in ab.CONDITIONS:
        suffix=Path(cond) if cond=='Val-A-V4' else Path(cond)/'draw0'
        for stage,model in [('C0','History-C0'),('C1','History')]:
            path=REFERENCE/'selfsup_oari_mrgr_v1'/suffix/stage/'per_case_metrics.jsonl'
            archived[model,cond]={r['case_id']:r for r in ab.rows(path)}
            evidence.append(dict(model=model,condition=cond,metrics=str(path),sha256=ab.sha(path)))
        for model,run in ab.RUNS.items():
            directory=Path(ab.read(run/f'evaluation/frozen_model_test/{cond}_prediction.json')['directory'])
            path=directory/'predictions_manifest.jsonl';boundary=ab.read(directory/'prediction_boundary.json')
            assert boundary['all_predictions_saved_before_gt'] and boundary['prediction_manifest_sha256']==ab.sha(path)
            for r in ab.rows(path):
                pred=directory/r['prediction'];assert ab.sha(pred)==r['prediction_sha256']
                predictions[model,cond,r['case_id']]=(pred,r)
            archived[model,cond]={r['case_id']:r for r in ab.rows(directory/'per_case_metrics.jsonl')}
            evidence.append(dict(model=model,condition=cond,manifest=str(path),sha256=ab.sha(path)))
    # All matching, saved prediction identities and model identities checked before GT.
    for r in hist:
        for model in ('A','B'):
            assert predictions[model,r['condition'],r['case_id']][1]['selected_station_ids']==r['station_ids']
        assert archived['History',r['condition']][r['case_id']]['selected_station_ids']==r['station_ids']
    source={r['case_id']:r for r in ab.rows(ab.DATA/'source_index.jsonl') if r['split']=='val'}
    chosen=ab.read(ab.RUNS['A']/'evaluation/frozen_model_test/slice_selection.json')
    examples={r['case_id']:r for r in chosen}
    records=[];fields={};cache={}
    for r in hist:
        case,cond=r['case_id'],r['condition']
        if case not in cache:
            cache[case]=np.load(ab.DATA/source[case]['gt_volume_64_path'],allow_pickle=False).astype(np.float64).squeeze()
        g=cache[case]
        with np.load(OUT/r['prediction'],allow_pickle=False) as saved:
            arrays={'History-C0':saved['C0'].astype(np.float64),'History':saved['C1'].astype(np.float64)}
        for model in ('A','B'):
            arrays[model]=np.load(predictions[model,cond,case][0],allow_pickle=False).astype(np.float64).squeeze()
        for model,p in arrays.items():
            assert p.shape==g.shape==(64,64,64) and np.isfinite(p).all() and p.min()>=0
            m=metrics(p,g)
            # CPU/GPU roundoff allowed; layouts/model/coordinates must reproduce old scores.
            difference=abs(m['relative_l1']-archived[model,cond][case]['relative_l1'])
            assert difference<1e-4,(model,cond,case,difference)
            records.append(dict(case_id=case,condition=cond,model=model,archived_l1_abs_difference=difference,**m))
            if case in examples: fields[case,cond,model]=p
    (OUT/'per_case_morphology.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summaries=[]
    for cond in ab.CONDITIONS:
        for model in MODELS:
            rr=[r for r in records if r['condition']==cond and r['model']==model]
            assert len(rr)==36
            s=dict(condition=cond,model=model,cases=36)
            for k in ('relative_l1','peak_ratio','gt_peak_location_ratio','centroid_distance_mm','dice_0.01','dice_0.1','dice_0.5','high_concentration_deficit','background_predicted_fraction'):
                s['mean_'+k]=float(np.mean([r[k] for r in rr]))
            s['mean_absolute_peak_ratio_error']=float(np.mean([abs(r['peak_ratio']-1) for r in rr]))
            s['mean_mass_absolute_relative_error']=float(np.mean([abs(r['mass_ratio']-1) for r in rr]))
            s['median_spread_ratio_xyz']=np.median([r['spread_ratio_xyz'] for r in rr],axis=0).tolist()
            s['count_peak_underestimated']=sum(r['peak_ratio']<1 for r in rr)
            summaries.append(s)
    def summary(model,cond): return next(r for r in summaries if r['model']==model and r['condition']==cond)
    pairs=[]
    for cond in ab.CONDITIONS:
        lookup={(r['model'],r['case_id']):r for r in records if r['condition']==cond}
        for model in ('History-C0','A','B'):
            pairs.append(dict(condition=cond,model=model,reference='History',
                l1_wins=sum(lookup[model,c]['relative_l1']<lookup['History',c]['relative_l1'] for c in source),
                centroid_wins=sum(lookup[model,c]['centroid_distance_mm']<lookup['History',c]['centroid_distance_mm'] for c in source),
                high_threshold_dice_wins=sum(lookup[model,c]['dice_0.5']>lookup['History',c]['dice_0.5'] for c in source)))
    ab.write(OUT/'summary.json',dict(summary=summaries,paired=pairs,evidence=evidence,
        report_source_sha256=ab.sha(Path(__file__)),
        historical_checkpoint_sha256=protocol['checkpoint']['sha256'],saved_before_gt=complete,
        max_archived_l1_abs_difference=max(r['archived_l1_abs_difference'] for r in records),
        sample_scope='36 known validation cases; original fixed V4 and clustered draw0; one training seed; no causal module attribution',
        definitions=dict(peak='global max ratio; values may occur at different positions',
            gt_peak_location_ratio='predicted concentration at GT argmax divided by GT maximum',
            spatial='equal-weight endpoint nodes; not trapezoidal physical integrals',
            dice='Common threshold for each case/model: 1%,10%,50% of that case GT maximum',
            high_concentration_deficit='sum(max(GT-pred,0)) within GT>=50% GT max / sum(GT) in that region'),
        current_multiscale_checkpoint_read=False,final_test_used=False))
    render(fields,cache,chosen,summary)
    lines=['# 历史 MRGR 与 A/B：匹配重建图像和形态分析','',
        '当前在训模型是历史自监督MRGR的轻量多尺度改进版：保留几何融合、视角交互及一次测量残差反馈，仅在共享解码器增加12通道16³/8³上下文分支。本报告分析历史冻结模型的局限，为训练结束后的图像验收建立参照，不读取或评价正在训练的权重。','',
        '**比较条件：** 36个相同验证案例、普通固定四站与极窄聚簇draw0，逐例严格匹配A/B输入站位及顺序。历史已存900份诊断体场使用其他布局，不能直接横向比较，因此重新用历史冻结第278轮最佳权重在CPU生成72次C0/C1重建；144个场全部保存、哈希封存后，独立进程读取验证GT。没有优化更新、GPU计算或最终test读取。','',
        '| 布局 | 模型 | 三维L1 | 峰值/GT均值 | GT峰位置浓度比 | 质心偏移均值 | 高浓度Dice |',
        '|---|---|---:|---:|---:|---:|---:|']
    for cond in ab.CONDITIONS:
        for model in ('History','A','B'):
            s=summary(model,cond)
            lines.append(f"| {cond} | {model} | {100*s['mean_relative_l1']:.2f}% | {s['mean_peak_ratio']:.3f} | {s['mean_gt_peak_location_ratio']:.3f} | {s['mean_centroid_distance_mm']:.2f} mm | {s['mean_dice_0.5']:.3f} |")
    lines += ['', '高浓度Dice采用各案例GT峰值50%的共同阈值；另有1%、10%阈值见JSON。这里质心/浓度和用等权网格点，正式物理体积分的端点梯形权重定义不同。峰值比的均值可能发生正负抵消，峰值绝对比误差也保存于JSON，不能单独按均值离1远近排名。','',
        '![同案例同站位竖向图像对比](../../outputs/research/historical_reconstruction_morphology_v1/matched_vertical_examples.png)','',
        '图中三个案例严格复用此前A普通布局误差最低、中位、最高选择，每例同时展示普通与聚簇；固定GT峰值z平面，y为高度，共同symlog色标。它们是既有选择，不是为使历史模型显得更好而重新挑例；三张切片不能代替36案例统计。','',
        '![相同近源位置局部放大](../../outputs/research/historical_reconstruction_morphology_v1/source_peak_detail.png)','',
        '上图仅展示此前A普通布局最高误差案例的GT峰值周围7×7个x-z网格，所有模型使用相同位置、相同y和相同线性色标。局部图外的其他峰或伪影需结合全域图，不可据此认定全场峰值均低估。','',
        '![历史反馈前后的高度分布](../../outputs/research/historical_reconstruction_morphology_v1/height_profiles.png)','',
        '**历史反馈前后的定量参照：**','',
        '| 布局 | 阶段 | 三维L1 | 峰值比 | 质心偏移 | 高浓度Dice |',
        '|---|---|---:|---:|---:|---:|']
    for cond in ab.CONDITIONS:
        for model in ('History-C0','History'):
            s=summary(model,cond)
            lines.append(f"| {cond} | {model} | {100*s['mean_relative_l1']:.2f}% | {s['mean_peak_ratio']:.3f} | {s['mean_centroid_distance_mm']:.2f} mm | {s['mean_dice_0.5']:.3f} |")
    lines += ['', 'C0/C1来自同一联合训练冻结模型；阶段间差异是轨迹诊断，不等于重新训练的反馈消融。历史与A/B同时改变了多项结构，也不能把图像差异归因于某一个模块。','',
        '**对多尺度版的验收要求：** 普通布局应同时检查全局L1与高浓度区域，而聚簇应检查峰值位置、质心、方向展宽和异常连通结构。多尺度上下文主要针对空间组织，但不保证自动恢复近源尖峰；若只改善低浓度背景或大轮廓，高浓度Dice仍差，则应另查高分辨率局部特征及监督目标。所有图像使用相同尺度，保留失败例，并在完整训练冻结后再做相同面板评价。','',
        '产物：[汇总及读取审计](../../outputs/research/historical_reconstruction_morphology_v1/summary.json)、[逐案例形态指标](../../outputs/research/historical_reconstruction_morphology_v1/per_case_morphology.jsonl)、[推理入口](../../scripts/analyze_historical_reconstruction_v1.py)、[分析入口](../../scripts/report_historical_reconstruction_v1.py)。']
    (ROOT/'docs/reports/historical_reconstruction_morphology_v1_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summaries,paired=pairs,max_archived_l1_abs_difference=max(r['archived_l1_abs_difference'] for r in records)),indent=2))


def render(fields,cache,chosen,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    models=('GT','History','A','B')
    fig,axes=plt.subplots(6,4,figsize=(13,17),constrained_layout=True)
    for i,selection in enumerate(chosen):
        case=selection['case_id'];g=cache[case];iz=np.unravel_index(g.argmax(),g.shape)[2]
        vmax=max([g.max()]+[fields[case,c,m].max() for c in ab.CONDITIONS for m in models[1:]])
        norm=SymLogNorm(linthresh=g.max()*1e-3,vmin=0,vmax=vmax)
        for j,cond in enumerate(ab.CONDITIONS):
            row=2*i+j
            for col,model in enumerate(models):
                v=g if model=='GT' else fields[case,cond,model]
                im=axes[row,col].imshow(v[:,:,iz].T,origin='lower',extent=(0,.6,0,.6),norm=norm,cmap='magma',interpolation='nearest')
                axes[row,col].set(title=f"{selection['selection']} / {'ordinary' if j==0 else 'cluster'}: {model}",xlabel='x (m)',ylabel='y height (m)')
            fig.colorbar(im,ax=axes[row,:],shrink=.7,label='CH4 mass fraction')
    fig.suptitle('Matched frozen models: fixed GT-peak z plane; shared color scale within each example')
    fig.savefig(OUT/'matched_vertical_examples.png',dpi=150);plt.close(fig)
    case=chosen[-1]['case_id'];g=cache[case];ix,iy,iz=np.unravel_index(g.argmax(),g.shape)
    lo=np.maximum(np.array([ix,iz])-3,0);hi=np.minimum(np.array([ix,iz])+4,64)
    fig,axes=plt.subplots(2,4,figsize=(12,6),constrained_layout=True)
    vals=[g]+[fields[case,c,m] for c in ab.CONDITIONS for m in models[1:]]
    vmax=max(v[lo[0]:hi[0],iy,lo[1]:hi[1]].max() for v in vals)
    for row,cond in enumerate(ab.CONDITIONS):
        for col,model in enumerate(models):
            v=g if model=='GT' else fields[case,cond,model]
            im=axes[row,col].imshow(v[lo[0]:hi[0],iy,lo[1]:hi[1]].T,origin='lower',vmin=0,vmax=vmax,cmap='magma',interpolation='nearest',
                extent=(lo[0]*.6/63,(hi[0]-1)*.6/63,lo[1]*.6/63,(hi[1]-1)*.6/63))
            axes[row,col].set(title=f"{'ordinary' if row==0 else 'cluster'}: {model}",xlabel='x (m)',ylabel='z (m)')
    fig.colorbar(im,ax=axes,shrink=.8,label='CH4 mass fraction; shared linear scale')
    fig.suptitle(f'Same source-peak neighborhood, y={iy*.6/63:.4f} m; prior A highest-error case')
    fig.savefig(OUT/'source_peak_detail.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for col,selection in enumerate(chosen):
        case=selection['case_id'];g=cache[case]
        for row,cond in enumerate(ab.CONDITIONS):
            ax=axes[row,col]
            for model in ('GT','History-C0','History','A','B'):
                v=g if model=='GT' else fields[case,cond,model]
                ax.plot(np.arange(64)*.6/63,v.sum(axis=(0,2))/g.sum(),label=model,linestyle='--' if model=='History-C0' else '-')
            ax.set(title=f"{selection['selection']} / {'ordinary' if row==0 else 'cluster'}",xlabel='y height (m)',ylabel='Fraction per y plane')
            ax.set_xlim(0,.6);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Full-volume height marginals / total GT nodal concentration; complete height range')
    fig.savefig(OUT/'height_profiles.png',dpi=160);plt.close(fig)


if __name__=='__main__': main()
