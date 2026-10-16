#!/usr/bin/env python3
"""CPU-only posthoc morphology of already frozen validation A/B predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/frozen_ab_morphology_v1'
DATA = Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
RUNS = {'A': ROOT / 'outputs/research/scheme_a_budget_v4_wide_seed20261011',
        'B': ROOT / 'outputs/research/scheme_b_perf_v2_formal_seed20261011'}
CONDITIONS = ('Val-A-V4','clustered-V4')


def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v): p.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')


def moments(v):
    # Physical XYZ endpoint grid, y is height. Marginals avoid a dense grid.
    axis = np.linspace(0., .6, 64)
    mass = v.sum()
    marginals = [v.sum(axis=tuple(j for j in range(3) if j != i)) for i in range(3)]
    center = np.array([np.dot(w, axis)/mass for w in marginals])
    spread = np.array([np.sqrt(np.dot(w, (axis-c)**2)/mass) for w,c in zip(marginals,center)])
    return center, spread


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    source = {r['case_id']:r for r in rows(DATA / 'source_index.jsonl') if r['split']=='val'}
    predictions = {}; evidence = []
    # Every input prediction is checked BEFORE opening any validation GT.
    for model,run in RUNS.items():
        complete = read(run / 'evaluation/frozen_model_test/all_predictions_complete.json')
        assert complete['predictions'] == 180
        for cond in CONDITIONS:
            directory = Path(read(run / f'evaluation/frozen_model_test/{cond}_prediction.json')['directory'])
            boundary = read(directory / 'prediction_boundary.json')
            manifest = directory / 'predictions_manifest.jsonl'
            assert sha(manifest) == boundary['prediction_manifest_sha256']
            assert boundary['all_predictions_saved_before_gt']
            records = rows(manifest)
            assert len(records)==36 and {r['case_id'] for r in records} == set(source)
            predictions[model,cond] = {r['case_id']:(directory/r['prediction'],r) for r in records}
            for path,r in predictions[model,cond].values():
                assert sha(path) == r['prediction_sha256']
            evidence.append(dict(model=model, condition=cond, manifest=str(manifest), sha256=sha(manifest)))
    for cond in CONDITIONS:
        for case in source:
            assert predictions['A',cond][case][1]['selected_station_ids'] == predictions['B',cond][case][1]['selected_station_ids']
    selection = read(RUNS['A'] / 'evaluation/frozen_model_test/slice_selection.json')[-1]
    example = selection['case_id']
    records = []; fields = {}
    for case,r in source.items():
        g = np.asarray(np.load(DATA/r['gt_volume_64_path'],allow_pickle=False),dtype=np.float64).squeeze()
        assert g.shape == (64,64,64) and g.min()>=0 and g.sum()>0
        gc,gs = moments(g)
        if case==example: fields['GT'] = g
        for cond in CONDITIONS:
            for model in RUNS:
                path,_ = predictions[model,cond][case]
                p = np.asarray(np.load(path,allow_pickle=False),dtype=np.float64).squeeze()
                assert p.shape == g.shape and np.isfinite(p).all() and p.min()>=0 and p.sum()>0
                pc,ps = moments(p)
                records.append(dict(case_id=case,condition=cond,model=model,
                    relative_l1=float(np.abs(p-g).sum()/g.sum()),
                    peak_ratio=float(p.max()/g.max()),mass_ratio=float(p.sum()/g.sum()),
                    centroid_distance_mm=float(np.linalg.norm(pc-gc)*1000),
                    centroid_delta_xyz_mm=((pc-gc)*1000).tolist(),
                    spread_ratio_xyz=(ps/gs).tolist(),
                    gt_peak_y_index=int(np.unravel_index(g.argmax(),g.shape)[1])))
                if case==example: fields[model,cond] = p
    (OUT/'per_case_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summaries=[]
    for cond in CONDITIONS:
        for model in RUNS:
            rr=[r for r in records if r['condition']==cond and r['model']==model]
            summaries.append(dict(condition=cond,model=model,cases=len(rr),
                mean_peak_ratio=float(np.mean([r['peak_ratio'] for r in rr])),
                median_peak_ratio=float(np.median([r['peak_ratio'] for r in rr])),
                count_peak_underestimated=sum(r['peak_ratio']<1 for r in rr),
                mean_mass_absolute_relative_error=float(np.mean([abs(r['mass_ratio']-1) for r in rr])),
                mean_centroid_distance_mm=float(np.mean([r['centroid_distance_mm'] for r in rr])),
                median_spread_ratio_xyz=np.median([r['spread_ratio_xyz'] for r in rr],axis=0).tolist()))
    write(OUT/'summary.json',dict(scope='Posthoc validation diagnosis; frozen A/B, draw0 only; not a causal ablation or blind test',
        source_sha256=sha(Path(__file__)),prediction_manifests=evidence,summary=summaries,
        example_case=example,example_selection='Reuse the preexisting A ordinary-V4 highest-error case; same case across layouts, not representative of all cases',
        current_training_checkpoint_opened=False,GPU_used=False,final_test_opened=False,
        metric_definitions=dict(peak_ratio='global predicted maximum / global GT maximum, can occur at different positions',
            mass_ratio='sum(prediction)/sum(GT), proportional to integrated concentration on the same uniform grid',
            centroid='concentration-weighted center of full nonnegative field',
            spread='concentration-weighted axis standard deviation; includes plume extent, multiple sources and low-concentration background, not edge blur alone')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    g=fields['GT']; ix,iy,iz=np.unravel_index(g.argmax(),g.shape)
    # Crop the source plane using the GT only, then apply exactly the same ROI.
    support=np.argwhere(g[:,iy,:] >= .01*g[:,iy,:].max())
    lo=np.maximum(support.min(0)-4,0); hi=np.minimum(support.max(0)+5,64)
    extent=(lo[0]*.6/63,(hi[0]-1)*.6/63,lo[1]*.6/63,(hi[1]-1)*.6/63)
    fig,axes=plt.subplots(2,5,figsize=(16,6),constrained_layout=True)
    for row,cond in enumerate(CONDITIONS):
        vals=[g,fields['A',cond],fields['B',cond]]
        crops=[v[lo[0]:hi[0],iy,lo[1]:hi[1]].T for v in vals]
        vmax=max(v.max() for v in crops)
        for col,(label,crop) in enumerate(zip(('GT','A','B'),crops)):
            im=axes[row,col].imshow(crop,origin='lower',extent=extent,vmin=0,vmax=vmax,cmap='magma',interpolation='nearest')
            axes[row,col].set_title(f'{cond}: {label}')
        fig.colorbar(im,ax=axes[row,:3],shrink=.6,label='CH4 mass fraction (shared row scale)')
        differences=[crops[i]-crops[0] for i in (1,2)]
        limit=max(np.abs(v).max() for v in differences)
        for col,label,diff in zip((3,4),('A - GT','B - GT'),differences):
            im=axes[row,col].imshow(diff,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            axes[row,col].set_title(label)
        fig.colorbar(im,ax=axes[row,3:],shrink=.6,label='Signed error (red excess, blue missing)')
    for ax in axes.flat: ax.set_xlabel('x (m)');ax.set_ylabel('z (m)')
    fig.suptitle(f'Frozen A/B, identical source-plane crop at y={iy*.6/63:.4f} m; ordinary-A worst case')
    fig.savefig(OUT/'source_plane_zoom.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
    norm=SymLogNorm(linthresh=g.max()*1e-3,vmin=0,vmax=max(v.max() for v in fields.values()))
    for row,cond in enumerate(CONDITIONS):
        for col,(label,v) in enumerate([('GT',g),('A',fields['A',cond]),('B',fields['B',cond])]):
            im=axes[row,col].imshow(v[:,:,iz].T,origin='lower',extent=(0,.6,0,.6),norm=norm,cmap='magma',interpolation='nearest')
            axes[row,col].set(title=f'{cond}: {label}',xlabel='x (m)',ylabel='y height (m)')
            profile=v.sum(axis=(0,2))/g.sum()
            axes[row,3].plot(np.arange(64)*.6/63,profile,label=label)
        fig.colorbar(im,ax=axes[row,:3],shrink=.7,label='CH4 mass fraction (shared symlog)')
        axes[row,3].set(title='Height marginal / total GT concentration',xlabel='y height (m)',ylabel='Fraction per y plane')
        axes[row,3].legend();axes[row,3].grid(alpha=.2)
    fig.suptitle('Same frozen example: vertical plane through GT peak and full-volume height profiles')
    fig.savefig(OUT/'height_plane_and_profiles.png',dpi=170);plt.close(fig)
    lines=['# 冻结 A/B：重建图像与形态初步诊断','',
        '以前已有方案A的三例GT峰值切片，但缺少系统的空间形态归因。本次仅用已封存预测和已用于开发的验证GT做CPU离线分析；不读取正在训练的多尺度模型、不重训、不使用最终test。两种布局均为draw0，每组36案例，A/B输入站位逐例匹配。历史MRGR本轮未取得相同面板的封存体场，故不作图像层面的历史排名。','',
        '| 布局 | 模型 | 峰值比均值 | 峰值低估例数 | 总量绝对相对误差均值 | 质心距离均值 | XYZ展宽比中位数 |','|---|---|---:|---:|---:|---:|---|']
    for s in summaries:
        spread='/'.join(f'{v:.3f}' for v in s['median_spread_ratio_xyz'])
        lines.append(f"| {s['condition']} | {s['model']} | {s['mean_peak_ratio']:.3f} | {s['count_peak_underestimated']}/36 | {100*s['mean_mass_absolute_relative_error']:.2f}% | {s['mean_centroid_distance_mm']:.2f} mm | {spread} |")
    lines += ['', '峰值比小于1表示预测全局峰值偏低；总量为同一网格上的浓度和。质心是浓度加权位置，展宽是各坐标浓度加权标准差与GT之比。峰值可能发生在不同位置，展宽受多源分布和低浓度背景影响，不能单独当作平滑量。','',
        f'示例固定复用原A普通V4误差最高案例 `{example}`，两布局看相同案例、相同GT定义切面；是失败案例展示，不代表平均表现。近源x-z平面按GT 1%峰值范围加4格边距裁剪，所有模型使用同一裁剪，不能评价裁剪外假阳性。浓度图每行共用色标，误差图每行共用对称色标。','',
        '![近源放大和带符号误差](../../outputs/research/frozen_ab_morphology_v1/source_plane_zoom.png)','',
        '竖向切面固定在GT峰值z坐标，使用共同symlog色标突出低浓度结构；右侧对完整体场沿x/z求和，显示浓度在高度上的分布，以同一GT总量归一化。切面漏掉的结构可能仍出现在全体场高度曲线上。','',
        '![高度分布](../../outputs/research/frozen_ab_morphology_v1/height_plane_and_profiles.png)','',
        '**解释边界：** 单张切面不能证明整体形态、分辨率损失或某个网络模块的因果。峰值低估可能涉及空间平滑、幅值偏差、定位和离散化，应联合局部剖面、多阈值等值面及有符号误差判断。聚簇质心/展宽异常需要对照站位的可观测方向，不能只凭条纹方向断言物理原因。','',
        '**下一步：** 对历史/A/B及完成后的多尺度版，用相同案例和布局统一三正交切面、等值面、沿主轴剖面；在全部36案例报告峰值、质心、方向展宽、多阈值前景重叠及假阳性浓度。当前图像诊断用于提出机制假设，后续以匹配消融验证，不把图片直接当作因果证据。','',
        '产物：[逐案例形态指标](../../outputs/research/frozen_ab_morphology_v1/per_case_metrics.jsonl)、[汇总与定义](../../outputs/research/frozen_ab_morphology_v1/summary.json)、[分析脚本](../../scripts/analyze_frozen_ab_morphology_v1.py)。']
    (ROOT/'docs/reports/frozen_ab_morphology_v1_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()
