#!/usr/bin/env python3
"""Scalar-only final reporting; no dataset, checkpoints or GT reads."""
import argparse,json,sys,shutil,gzip
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from geophy_set3d.evaluation.geometry_augmentation_formal_stats import METRICS,paired,decide,holm,worst
from geophy_set3d.data.build_utils import digest
from geophy_set3d.training.utils import write_json
from run_geometry_augmentation_formal import configuration,verify,ARCHIVE


def table(rows,cols):
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(r.get(k,'')) for k in cols)+' |' for r in rows)+'\n'

def interval(x):return '['+', '.join(f'{a:.6f}' for a in x)+']'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config');a=ap.parse_args();c,out=configuration(a.config);p=verify(c,out);ev=out/'posthoc_evaluation'
    test_log=(out/'full_post_training_tests.log').read_text();assert test_log.rstrip().endswith('OK')
    assert json.loads((ev/'posthoc_complete.json').read_text())['both300_before_any_GT_read']
    rows=json.loads((ev/'per_case_draw.json').read_text());assert len(rows)==5760
    df=pd.DataFrame(rows);keys=['case_id','section','V','stratum','draw'];assert not df.duplicated(['model']+keys).any()
    cd=df[df.model=='Control'].set_index(keys).sort_index();td=df[df.model=='Treatment'].set_index(keys).sort_index();assert cd.index.equals(td.index)
    for k in ['input_station_ids','H_station_ids','geometry_id']:assert cd[k].tolist()==td[k].tolist()
    ids=sorted(df.case_id.unique());assert len(ids)==36
    averaged=df.groupby(['model','case_id','section','stratum'])[list(METRICS)].mean();results=[];case_rows=[];groups={}
    for section in ['natural','development','confirmatory']:
        names=sorted(df.loc[df.section==section,'stratum'].unique())
        for group in names:
            for metric in METRICS:
                x=averaged.xs(('Control',section,group),level=('model','section','stratum'))[metric].reindex(ids).to_numpy()
                y=averaged.xs(('Treatment',section,group),level=('model','section','stratum'))[metric].reindex(ids).to_numpy()
                q=paired(x,y,c['bootstrap_seed'],c['bootstrap_samples']);results.append(dict(section=section,group=group,metric=metric,**q))
                if metric=='relative_l1':
                    groups[(section,group)]=(x,y,q)
                    case_rows.extend(dict(case_id=i,section=section,group=group,Control=float(xx),Treatment=float(yy),delta=float(yy-xx)) for i,xx,yy in zip(ids,x,y))
    confirm={g:groups[('confirmatory',g)] for g in ('G1','G2','G3','G4','G5')}
    nc=np.mean([confirm[g][0] for g in ('G1','G2','G3','G4')],axis=0);nt=np.mean([confirm[g][1] for g in ('G1','G2','G3','G4')],axis=0)
    natural=paired(nc,nt,c['bootstrap_seed'],c['bootstrap_samples']);g5=confirm['G5'][2];decision=decide(g5,natural,c['noninferiority_ratio'],c['partial_max_natural_ratio'],c['partial_min_g5_improvement'])
    gc,gc1=confirm['G5'][0],confirm['G1'][0];gt,gt1=confirm['G5'][1],confirm['G1'][1]
    idx=np.random.default_rng(c['bootstrap_seed']).integers(0,36,(c['bootstrap_samples'],36));dc=gc.mean()/gc1.mean();dt=gt.mean()/gt1.mean()
    dcbs=gc[idx].mean(1)/gc1[idx].mean(1);dtbs=gt[idx].mean(1)/gt1[idx].mean(1)
    degrade=dict(Control=float(dc),Treatment=float(dt),delta=float(dt-dc),delta_ci95=np.quantile(dtbs-dcbs,[.025,.975]).tolist(),
        Control_ci95=np.quantile(dcbs,[.025,.975]).tolist(),Treatment_ci95=np.quantile(dtbs,[.025,.975]).tolist(),
        case_ratio_change=paired(gc/gc1,gt/gt1,c['bootstrap_seed'],c['bootstrap_samples']),absolute_gap=paired(gc-gc1,gt-gt1,c['bootstrap_seed'],c['bootstrap_samples']))
    secondary=[r for r in results if not (r['section']=='confirmatory' and r['group']=='G5' and r['metric']=='relative_l1')]
    ps=[r['wilcoxon_p'] for r in secondary]+[degrade['case_ratio_change']['wilcoxon_p'],degrade['absolute_gap']['wilcoxon_p'],natural['wilcoxon_p']]
    adj=holm(ps)
    for r,q in zip(secondary,adj):r['holm_p']=q
    degrade['case_ratio_change']['holm_p']=adj[-3];degrade['absolute_gap']['holm_p']=adj[-2];natural['holm_p']=adj[-1]
    worsts={arm:{g:worst(confirm[g][0 if arm=='Control' else 1],ids) for g in confirm} for arm in c['arms']}
    for arm in c['arms']:
        ar=df[(df.model==arm)&(df.section=='confirmatory')];row=ar.loc[ar.relative_l1.idxmax()];worsts[arm]['single_layout_max']=dict(case_id=row.case_id,geometry_id=row.geometry_id,relative_l1=float(row.relative_l1))
    train={};curves={}
    for arm in c['arms']:
        rr=[json.loads(l) for l in (out/arm/'metrics.jsonl').read_text().splitlines()];complete=json.loads((out/arm/'complete.json').read_text());assert len(rr)==300 and complete['optimizer_steps']==24900
        curves[arm]=rr;best=min(rr,key=lambda r:r['selector_metric']);grad=[v for r in rr for v in r['train']['gradient_norms']];assert len(grad)==24900
        train[arm]=dict(epochs=300,steps=24900,best_epoch_zero_based=best['epoch'],best_epoch_one_based=best['epoch']+1,best_selector=best['selector_metric'],seconds=float(sum(r['seconds'] for r in rr)),
            gpu_peak_bytes=max(r['GPU_peak_allocated_bytes'] for r in rr),gradient_preclip_max=max(v['preclip'] for v in grad),gradient_postclip_max=max(v['postclip'] for v in grad),
            forward_mean_seconds=float(df[df.model==arm].forward_seconds.mean()),checkpoint_sha256=complete['best_sha256'],access=complete['access'])
    marg=json.loads((out/'registration/station_marginals.json').read_text());freq=[]
    for arm in c['arms']:freq.append(np.sum([v['station_frequency'] for v in marg[arm]['by_mode'].values()],axis=0).astype(float))
    prob=[v/v.sum() for v in freq];shift=dict(total_variation_input_visit_distribution=float(np.abs(prob[0]-prob[1]).sum()/2),
        max_station_probability_absolute_change=float(np.abs(prob[0]-prob[1]).max()),definition='A/B station visits normalized by total visits; separate subset inclusion/exclusion probabilities in station_marginals.json',
        hard_inclusion_range=[min(marg['Treatment']['by_mode']['hard']['inclusion_probability']),max(marg['Treatment']['by_mode']['hard']['inclusion_probability'])])
    stats=dict(GEOMETRY_AUGMENTATION=decision,confirmatory_G5=g5,confirmatory_natural=natural,degradation=degrade,paired_metrics=results,worst_case=worsts,training=train,station_shift=shift,
        bootstrap_seed=c['bootstrap_seed'],bootstrap_samples=10000,n_cases=36,statistical_unit='case after within-group layout mean',paired_inputs_verified=True,secondary_holm_family=len(ps))
    rep=out/'reports';rep.mkdir(exist_ok=True);write_json(rep/'statistics.json',stats);pd.DataFrame(results).to_csv(rep/'paired_metrics.csv',index=False);pd.DataFrame(case_rows).to_csv(rep/'case_group_relative_l1.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for arm,rr in curves.items():
        ax[0].plot(np.arange(1,301),[r['train']['loss']['total'] for r in rr],label=arm);ax[1].plot(np.arange(1,301),[r['selector_metric'] for r in rr],label=arm)
    for x,title in zip(ax,['Train measurement-only loss','Natural Val H4 selector']):x.set_title(title);x.set_xlabel('Epoch');x.legend()
    fig.savefig(rep/'training_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');xx=np.arange(5)
    for j,arm in enumerate(c['arms']):
        vals=[confirm[g][j] for g in confirm];means=np.array([v.mean() for v in vals]);cis=np.array([np.quantile(v[idx].mean(1),[.025,.975]) for v in vals])
        ax.errorbar(xx+(j-.5)*.12,means,yerr=np.stack([means-cis[:,0],cis[:,1]-means]),marker='o',capsize=4,label=arm)
    ax.set_xticks(xx,list(confirm));ax.set_ylabel('Case-mean relative L1');ax.set_title('Independent confirmatory geometry: case bootstrap 95% CI');ax.legend();fig.savefig(rep/'confirmatory_groups.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,9),layout='constrained');delta=np.stack([confirm[g][1]-confirm[g][0] for g in confirm],axis=1);mx=float(np.abs(delta).max());im=ax.imshow(delta,cmap='coolwarm',vmin=-mx,vmax=mx,aspect='auto');ax.set_xticks(range(5),list(confirm));ax.set_yticks(range(36),ids,fontsize=6);ax.set_title('Treatment - Control relative L1 by case');fig.colorbar(im,ax=ax);fig.savefig(rep/'paired_case_heatmap.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.plot(prob[0],label='Control');ax.plot(prob[1],label='Treatment');ax.set_xlabel('Station ID');ax.set_ylabel('A/B input visit fraction');ax.legend();fig.savefig(rep/'station_marginals.png',dpi=160);plt.close(fig)
    sections=[]
    for section in ['natural','development','confirmatory']:
        rr=[]
        for key,(x,y,q) in groups.items():
            if key[0]==section:rr.append(dict(group=key[1],Control=f'{q["Control_mean"]:.6f}',Treatment=f'{q["Treatment_mean"]:.6f}',delta=f'{q["mean_delta"]:.6f}',median=f'{q["median_delta"]:.6f}',CI=interval(q['delta_ci95']),ratio=f'{q["ratio"]:.4f}',wins=f'{q["Treatment_wins"]}/{q["Control_wins"]}/{q["ties"]}',p=f'{q["wilcoxon_p"]:.3g}'))
        sections.append('### '+section+'\n\n'+table(rr,['group','Control','Treatment','delta','median','CI','ratio','wins','p']))
    measurement=[r for r in results if r['section']=='confirmatory' and r['metric'] in ('observed_relative_l1','heldout_relative_l1')]
    practical='非常明显' if g5['improvement']>=.25 else '较强' if g5['improvement']>=.15 else '有意义' if g5['improvement']>=.10 else '轻微' if g5['improvement']>=.05 else '不足5%或恶化'
    g5_measure={r['metric']:r for r in measurement if r['group']=='G5'}
    answers=[
        '未复用：CONTROL_REUSE=REJECTED。历史逐步参数/optimizer/clip后梯度与运行flags缺失，完整numeric equivalence不可证明；保留旧same-q NO-GO。',
        '完全匹配：两臂fresh、同初态/seed/case order/V schedule、各300epochs与24,900steps。唯一主要干预是50/25/25站位采样；station marginal shift包含于机制。',
        '训练/Val/selector审计GT64/GT128、Pool B/test、foreign checkpoint尝试全部为0。独立posthoc只读36例Val GT64，禁读计数为0。',
        f'确认G5 relative L1：Control={g5["Control_mean"]:.6f}、Treatment={g5["Treatment_mean"]:.6f}，delta={g5["mean_delta"]:.6f}。',
        f'改善={100*g5["improvement"]:.2f}%（T/C={g5["ratio"]:.4f}；{practical}）。这是一次预注册干预，不追加搜索。',
        f'case bootstrap95%CI={interval(g5["delta_ci95"])}，完全低于0：{g5["delta_ci95"][1]<0}。',
        f'36例内Treatment wins/Control wins/ties={g5["Treatment_wins"]}/{g5["Control_wins"]}/{g5["ties"]}；先每case平均10layout。',
        f'确认G1–G4 natural macro：C={natural["Control_mean"]:.6f}、T={natural["Treatment_mean"]:.6f}，ratio={natural["ratio"]:.4f}、CI={interval(natural["ratio_ci95"])}，point-rule≤1.03：{natural["ratio"]<=1.03}。CI仅报告，不偷换成CI非劣证明。',
        f'grand-mean G5/G1：{dc:.4f}→{dt:.4f}，delta CI={interval(degrade["delta_ci95"])}。G5−G1变化={degrade["absolute_gap"]["mean_delta"]:.6f}；完整case-ratio诊断另列，定义不混用。',
        f'确认G5 case-group最大L1：{worsts["Control"]["G5"]["maximum"]:.6f}→{worsts["Treatment"]["G5"]["maximum"]:.6f}；p90/p95/median及worst IDs见统计JSON。单layout极端也独立记录。',
        f'确认G5 input relativeL1={g5_measure["observed_relative_l1"]["Control_mean"]:.6f}→{g5_measure["observed_relative_l1"]["Treatment_mean"]:.6f}，delta CI={interval(g5_measure["observed_relative_l1"]["delta_ci95"])}。是否与3D同向应据实际符号判断，拟合更好不单独证明3D恢复更好。',
        f'确认G5共同H4 relativeL1={g5_measure["heldout_relative_l1"]["Control_mean"]:.6f}→{g5_measure["heldout_relative_l1"]["Treatment_mean"]:.6f}，delta CI={interval(g5_measure["heldout_relative_l1"]["delta_ci95"])}；仅最终评价，不参与确认集选参或重建输入。',
        f'station visits分布TV distance={shift["total_variation_input_visit_distribution"]:.4f}，最大单站概率变化={shift["max_station_probability_absolute_change"]:.4f}；hard inclusion范围={shift["hard_inclusion_range"]}。mode条件频率/包含排除概率/拒绝数完整保存，不重平衡。',
        f'GEOMETRY_AUGMENTATION = {decision}。严格使用预注册G5 CI与natural point ratio；PARTIAL不称正式成功。',
        '下一步仅建议独立预注册model-side geometry awareness及四臂ablation：Base、Base+augmentation、Base+geometry-aware、二者结合。无论本轮结果都不在当前Val GT搜索ratio、span、curriculum或loss。本轮不实现新模块或启动新训练。'
    ]
    trows=[dict(arm=arm,epochs=v['epochs'],steps=v['steps'],best_epoch=v['best_epoch_one_based'],selector=f'{v["best_selector"]:.6f}',hours=f'{v["seconds"]/3600:.3f}',forward_ms=f'{v["forward_mean_seconds"]*1000:.2f}',peak_GiB=f'{v["gpu_peak_bytes"]/2**30:.2f}') for arm,v in train.items()]
    mrows=[dict(group=r['group'],metric=r['metric'],Control=f'{r["Control_mean"]:.6f}',Treatment=f'{r["Treatment_mean"]:.6f}',delta=f'{r["mean_delta"]:.6f}',CI=interval(r['delta_ci95'])) for r in measurement]
    report=f'''# 显式 Geometry Augmentation formal v1：匹配实验报告

**GEOMETRY_AUGMENTATION = {decision}**。唯一主要实验变量是训练时A/B几何采样分布。旧same-q实验与original frozen v1保持原结论；本轮不构成geometry比capacity/optimizer/loss更主要的因果证明。

预注册：[正式协议](../geometry_augmentation_formal_v1_protocol_zh.md)，commit36c041c（正式训练前提交，已同步GitHub）。完整源码/config、输入、旧标量哈希见[归档协议](artifacts/geometry_augmentation_formal_v1/protocol.json)。{len(rows)}次真实冻结推理；historical natural/development/confirmatory分开，不把开发集当确认集。

## 指定15项回答

'''+ '\n\n'.join(f'{i+1}. {x}' for i,x in enumerate(answers))+'''

## 前置验证、冻结与访问边界

真实train-mode batch2 AdamW smoke共4step，首步loss重放相同；参数hash/optimizer hash不同，参数最大差约3.28e-7/2.24e-7，optimizer最大差约6.98e-10/4.66e-10。CUDA grid_sample backward非确定性如实登记；10步原函数/标量记录函数差及100步fresh replay详见归档，历史记录不可用不捏造。原replay generator追加时全局step标签多计数，原预注册记录保留，另附control_fresh_replay_index_erratum按实际顺序修正第1/10/100步索引；local step/epoch/hash/损失及实际100次更新均不变，正式训练独立step counter不受影响。GPU2首次OOM（其他用户工作负载进入）在optimizer前退出；改用GPU0/1，保留错误记录，不改batch/超参，不终止他人。

Control/Treatment best仅由原自然Val双A4/B4/H4 measurement selector选取；完成300后SHA冻结，先独立measurement-only评价和selector replay，再以saved-output逐体hash验证进入GT64 evaluator。GT不选epoch、布局、采样边界或判定阈值。确认布局与历史25组合不同，G5≤5°；训练hard≤10°。自然Val selector不采用confirmatory布局。

## 训练预算、时间与收敛

'''+table(trows,['arm','epochs','steps','best_epoch','selector','hours','forward_ms','peak_GiB'])+'''

训练曲线与逐epoch total/hold/cross/observed/consistency、LR、clip前/后norm、best状态已归档；全部完成，无NaN/Inf。时间来自共享RTX4090，每epoch含train+Val+日志，不等于独占GPU速度；网络前向单独同步计时。没有调整预算或按GT挑迭代点。

![训练曲线](artifacts/geometry_augmentation_formal_v1/reports/training_curves.png)

## 配对3D relative L1

36case为统计单位，先同case/group平均draw，bootstrap10,000 seed2026101603。表中wins为T/C/ties；delta=T−C，原始Wilcoxon p列仅描述，secondary统一Holm结果在CSV/JSON，不依据未校正p追加实验。

'''+ '\n\n'.join(sections)+'''

![确认集](artifacts/geometry_augmentation_formal_v1/reports/confirmatory_groups.png)

![案例配对变化](artifacts/geometry_augmentation_formal_v1/reports/paired_case_heatmap.png)

## 全部指标、测量与最坏案例

完整10指标（L1/L2、MAE/RMSE、peak absolute、integral relative proxy、input/H L1/L2）逐组配对均值/median/CI、Wilcoxon与Holm见[paired_metrics.csv](artifacts/geometry_augmentation_formal_v1/reports/paired_metrics.csv)；[statistics.json](artifacts/geometry_augmentation_formal_v1/reports/statistics.json)含worst-case mean/median/p90/p95/max/IDs。peak是CH4质量分数绝对误差，integral是冻结voxel sum proxy相对误差，无量纲，不虚构物理质量单位。

'''+table(mrows,['group','metric','Control','Treatment','delta','CI'])+'''

## station机制、局限与后续

![站位边际](artifacts/geometry_augmentation_formal_v1/reports/station_marginals.png)

hard候选集使站位频率不均匀；natural/moderate/hard条件频率、A/B与3×3联合分布、V频率、inclusion/exclusion probability及joint rejection见登记station_marginals。实际全部epoch case/station digest与计划核对，非事后重新抽样或平衡。

原始评价、bootstrap/决策/Holm实现训练前冻结；汇总、绘图与报告入口在训练期间完成，其SHA在任何冻结模型测量或GT评价前另行登记，不改变分析定义。CI刻画固定seed/固定布局集的case变异，不包含training-seed或geometry-population uncertainty。单一matched seed不能估计training-seed variability。50个确认layout从既有同一Pool A和已知分层规则中独立抽出，排除了旧25组合，但不代表新物理采集域或未知几何分布；36Val曾用于历史研究，不替代最终test。G5极端冗余几何仍可能存在严重信息歧义，布局排序、空间表示与非线性离散化限制继续存在，粗coverage饱和不证明唯一可恢复。共同H只是有限测量约束。augmentation若改善，只说明该干预在冻结架构中的效果；若失败，保留NO-GO，不根据Val GT重试。

历史数据/checkpoint/配置/训练源码及历史结论未修改；用户未跟踪绘图脚本未提交。Pool B、final test、GT128、经典新baseline、新网络模块、ratio/span搜索均未运行。后续只提出model-side建议，完成本轮后停止。
'''
    (REPO/'docs/reports/geometry_augmentation_formal_v1_zh.md').write_text(report)
    # Scalar artifacts only, no predictions/checkpoints/GT.
    for root,names in [(out,['frozen_best.json','full_post_training_tests.log','final_integrity_audit.json','analysis_implementation_manifest.json']),(ev,['per_case_draw.csv','measurement_complete.json','posthoc_complete.json','posthoc_access_audit.json','selector_audit.json','Control_measurement_access_audit.json','Treatment_measurement_access_audit.json']), (out/'registration',['pre_training_tests.log','integrity_pre_training.json','formal_batch_shapes_supplement.json','control_fresh_replay_index_erratum.json'])]:
        for name in names:
            src=root/name;dst=ARCHIVE/src.relative_to(out);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for arm in c['arms']:
        ar=ARCHIVE/arm;ar.mkdir(exist_ok=True)
        for name in ['complete.json','access_audit.json']:shutil.copy2(out/arm/name,ar/name)
        with gzip.open(ar/'metrics.jsonl.gz','wb') as f:f.write((out/arm/'metrics.jsonl').read_bytes())
    dst=ARCHIVE/'reports';dst.mkdir(exist_ok=True)
    for f in rep.iterdir():shutil.copy2(f,dst/f.name)
    write_json(rep/'artifact_hashes.json',{str(f.relative_to(ARCHIVE)):digest(f) for f in ARCHIVE.rglob('*') if f.is_file()})
    shutil.copy2(rep/'artifact_hashes.json',dst/'artifact_hashes.json')
    print(json.dumps(dict(decision=decision,G5=g5,natural=natural,degradation=degrade,training=train),indent=2))
if __name__=='__main__':main()
