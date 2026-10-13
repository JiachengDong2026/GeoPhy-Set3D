#!/usr/bin/env python3
"""Scalar-only matched-sampler efficacy report after registered post-hoc boundary."""
import json,sys,shutil,subprocess,hashlib
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geophy_set3d.data.build_utils import digest
from geophy_set3d.evaluation.geometry_strata import validate_shared_inputs
from geophy_set3d.training.utils import write_json
from diagnose_clustered_one_swap import write_csv
from run_geometry_stratified_matched import configuration,verify
from evaluate_geometry_stratified_matched import freeze
from report_geometry_strata_frozen import ci_text,table,METRICS


def main():
    c,out=configuration();p,frozen=freeze(c,out);ev=out/'evaluation'
    full_test_log=(out/'full_post_training_tests.log').read_text()
    assert full_test_log.rstrip().endswith('OK') and 'Ran 239 tests' in full_test_log
    assert json.loads((ev/'posthoc_complete.json').read_text())['both300_before_any_GT_read']
    rows=json.loads((ev/'per_case_draw.json').read_text());assert len(rows)==4680;validate_shared_inputs(rows)
    df=pd.DataFrame(rows);keys=['model','section','V','stratum','case_id'];assert (df.groupby(keys).size()==5).all()
    cm=df.groupby(keys)[METRICS].mean().reset_index();cm.to_csv(ev/'case_means.csv',index=False,lineterminator='\n')
    cases=sorted(cm.case_id.unique());assert len(cases)==36
    bootstrap=np.random.default_rng(c['bootstrap']['seed']).integers(0,36,(10000,36));summaries=[];pairs=[];macros=[]
    def values(model,section,views,stratum,metric):
        sub=cm[(cm.model==model)&(cm.section==section)&(cm.V==views)&(cm.stratum==stratum)]
        return sub.set_index('case_id').loc[cases,metric].to_numpy()
    def pair_record(section,views,stratum,metric,a,b):
        delta=b-a
        return dict(section=section,V=views,stratum=stratum,metric=metric,Control=float(a.mean()),Treatment=float(b.mean()),
            delta=float(delta.mean()),relative_change=float(delta.mean()/a.mean()),ci=np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist(),
            unit='36case; average5draws first',bootstrap_samples=10000,bootstrap_seed=c['bootstrap']['seed'],
            wins=int((delta<0).sum()),losses=int((delta>0).sum()),ties=int((delta==0).sum()))
    for (model,section,views,stratum),group in cm.groupby(['model','section','V','stratum']):
        row=dict(model=model,section=section,V=int(views),stratum=stratum)
        for metric in METRICS:
            v=group.set_index('case_id').loc[cases,metric].to_numpy();row[metric]=float(v.mean());row[metric+'_ci']=np.quantile(v[bootstrap].mean(1),[.025,.975]).tolist()
        summaries.append(row)
    for section,views,stratum in cm[['section','V','stratum']].drop_duplicates().itertuples(index=False,name=None):
        for metric in METRICS:
            a=values('Control',section,views,stratum,metric);b=values('Treatment',section,views,stratum,metric)
            pairs.append(pair_record(section,int(views),stratum,metric,a,b))
    for section,stratum in [('natural','natural')]+[('controlled',s) for s in ['low','middle','high']]:
        for metric in METRICS:
            av,bv=[np.mean([values(model,section,v,stratum,metric) for v in [3,4,6]],axis=0) for model in ['Control','Treatment']]
            macros.append(pair_record(section,'macro3_4_6',stratum,metric,av,bv))
    low=next(r for r in macros if r['stratum']=='low' and r['metric']=='relative_l1')
    natural=next(r for r in macros if r['section']=='natural' and r['metric']=='relative_l1')
    high=next(r for r in macros if r['stratum']=='high' and r['metric']=='relative_l1')
    rules=dict(natural_not_clearly_worse=bool(natural['ci'][0]<=0),low_improves_direction=bool(low['delta']<0),
        high_not_materially_harmed=not(high['relative_change']>c['material_high_stratum_relative_harm'] and high['ci'][0]>0))
    decision='GO' if all(rules.values()) else 'NO-GO'
    write_json(ev/'decision.json',dict(stage='D',decision=decision,rules=rules,natural_macro=natural,low_macro=low,high_macro=high,registered_gate=p['stage_D_gate']))
    write_json(ev/'paired_statistics.json',dict(per_cell=pairs,macro=macros));write_csv(ev/'paired_statistics.csv',pairs+macros)
    write_json(ev/'summary.json',summaries);write_csv(ev/'summary.csv',summaries)
    training={a:[json.loads(l) for l in (out/a/'metrics.jsonl').read_text().splitlines()] for a in c['arms']}
    selector=json.loads((out/'selector_station_ids.json').read_text());orders=json.loads((out/'case_orders.json').read_text())
    selector_digest=hashlib.sha256(json.dumps(sorted((r['case_id'],r['A'],r['B'],r['H']) for r in selector)).encode()).hexdigest()
    plans={a:json.loads((out/f'{a}_station_plan.json').read_text()) for a in c['arms']}
    distribution=json.loads((out/'sampler_distribution_audit.json').read_text());frequencies=pd.read_csv(out/'sampler_station_frequency.csv')
    training_rows=[]
    for arm,epochs in training.items():
        assert len(epochs)==300 and [r['epoch'] for r in epochs]==list(range(300))
        complete=json.loads((out/arm/'complete.json').read_text());assert complete['optimizer_steps']==24900
        for r in epochs:
            assert r['case_order']==orders[r['epoch']] and r['optimizer_steps']==83*(r['epoch']+1)
            assert r['train']['sample_count']==166 and r['val']['sample_count']==36 and not r['train']['gt_used'] and not r['val']['gt_used']
            assert r['val']['case_station_digest']==training['Control'][r['epoch']]['val']['case_station_digest']
            assert r['val']['case_station_digest']==selector_digest
            epoch_plan=plans[arm][r['epoch']]
            expected_train_digest=hashlib.sha256(json.dumps(sorted((q['case_id'],q['A'],q['B'],q['H']) for q in epoch_plan)).encode()).hexdigest()
            assert r['train']['case_station_digest']==expected_train_digest
            training_rows.append(dict(arm=arm,epoch=r['epoch'],total_loss=r['train']['loss']['total'],train_hold_l1=r['train']['measurement']['heldout']['relative_l1'],
                val_hold_l1=r['val']['measurement']['heldout']['relative_l1'],learning_rate=r['learning_rate'],seconds=r['seconds'],optimizer_steps=r['optimizer_steps'],
                gradient_norm_max=r['train']['gradient_norm_max'],GPU_peak_allocated_bytes=r['GPU_peak_allocated_bytes']))
    for ep in range(300):
        for a,b in zip(plans['Control'][ep],plans['Treatment'][ep]):assert (a['case_id'],len(a['A']),len(a['B']),len(a['H']))==(b['case_id'],len(b['A']),len(b['B']),len(b['H']))
    write_csv(ev/'training_curves.csv',training_rows)
    # All old data/history metadata must still match the Stage C baseline inventory.
    inv=json.loads((Path(c['stage_c'])/'inventory_before.json').read_text());changed=[]
    for root,files in inv['metadata'].items():
        current={str(q):[q.stat().st_size,q.stat().st_mtime_ns,q.stat().st_ino] for q in Path(root).rglob('*') if q.is_file()}
        if current!=files:changed.append(root)
    assert not changed,changed
    protected_changes=[k for k,sha in inv['repo_content'].items() if not k.endswith('.pyc') and digest(k)!=sha]
    assert set(protected_changes)=={str(REPO/'src/geophy_set3d/evaluation/geometry_catalog.py')}
    stage_c_sums=REPO/'docs/reports/artifacts/geometry_strata_frozen_v1/SHA256SUMS'
    for line in stage_c_sums.read_text().splitlines():
        sha,name=line.split('  ',1);assert digest(REPO/name)==sha,name
    verify(c,out)
    access=dict(registration=json.loads((out/'registration_access_audit.json').read_text()),
        training={a:json.loads((out/a/'access_audit.json').read_text()) for a in c['arms']},
        measurement={a:json.loads((ev/f'{a}_measurement_access_audit.json').read_text()) for a in c['arms']},
        posthoc_GT=json.loads((ev/'posthoc_access_audit.json').read_text()))
    for a in [access['registration'],*access['training'].values(),*access['measurement'].values()]:
        assert a['gt_open_attempts']==a['forbidden_npy_attempts']==a['supervised_checkpoint_open_attempts']==a['pool_b_open_attempts']==a['test_or_unknown_npy_attempts']==a['data_write_attempts']==0
    assert access['posthoc_GT']['unique_val_gt64_files']==36 and access['posthoc_GT']['forbidden_attempts']==0
    chronological=(max((out/a/'complete.json').stat().st_mtime_ns for a in c['arms'])<=(out/'frozen_best.json').stat().st_mtime_ns<=
        (ev/'measurement_complete.json').stat().st_mtime_ns<=(ev/'posthoc_complete.json').stat().st_mtime_ns)
    assert chronological
    integrity=dict(same_initial_state=True,only_sampler_scientific_intervention=True,same_case_order=True,same_optimizer_steps=True,same_V_counts_per_case_epoch=True,
        same_objective_and_selector=True,old_data_history_metadata_unchanged=True,source_hashes_unchanged=True,both300_before_GT=True,
        protected_repository_changes=protected_changes,stage_C_artifact_hashes_unchanged=True,
        chronology_verified=chronological,GT_training_validation_selection_opens=0,Pool_B_opens=0,test_opens=0,GT128_content_opens=0,
        best_checkpoints_frozen_before_measurement_and_GT=True,old_checkpoints_unchanged=True)
    write_json(ev/'access_audit.json',access);write_json(ev/'integrity.json',integrity)
    figures=ev/'figures';figures.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for arm,epochs in training.items():
        x=np.arange(1,301);axs[0].plot(x,[r['train']['loss']['total'] for r in epochs],label=arm)
        axs[1].plot(x,[r['val']['measurement']['heldout']['relative_l1'] for r in epochs],label=arm)
        axs[2].plot(x,[r['train']['measurement']['heldout']['relative_l1'] for r in epochs],label=arm)
    for ax,title in zip(axs,['Train original total objective','Original natural Val selector','Train held-out measurement L1']):ax.set_title(title);ax.set_xlabel('Epoch');ax.legend();ax.set_yscale('log')
    fig.tight_layout();fig.savefig(figures/'training_curves.png',dpi=160);plt.close(fig)
    primary=[r for r in macros if r['metric']=='relative_l1']+[r for r in pairs if r['section']=='clustered' and r['metric']=='relative_l1']
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(primary));means=[r['delta'] for r in primary]
    ax.errorbar(x,means,yerr=[[r['delta']-r['ci'][0] for r in primary],[r['ci'][1]-r['delta'] for r in primary]],fmt='o',capsize=4)
    ax.axhline(0,color='black',lw=1);ax.set_xticks(x,[r['stratum'] for r in primary]);ax.set_ylabel('Treatment - Control volume relative L1');fig.tight_layout();fig.savefig(figures/'paired_macro_effects.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3))
    for j,arm in enumerate(c['arms']):
        f=frequencies[frequencies.arm==arm];axs[j].bar(f.station_id,f.A+f.B);axs[j].set_title(arm+' A+B station frequency');axs[j].set_xlabel('Pool A station ID')
    fig.tight_layout();fig.savefig(figures/'sampler_station_frequency.png',dpi=160);plt.close(fig)
    best_epochs={a:min(training[a],key=lambda r:r['val']['measurement']['heldout']['relative_l1'])['epoch']+1 for a in c['arms']}
    duration={a:sum(r['seconds'] for r in epochs) for a,epochs in training.items()}
    forward_totals={a:float(df[df.model==a].forward_seconds.sum()) for a in c['arms']}
    phase_wall_spans=dict(measurement_since_best_freeze_seconds=((ev/'measurement_complete.json').stat().st_mtime_ns-(out/'frozen_best.json').stat().st_mtime_ns)/1e9,
        posthoc_since_measurement_complete_seconds=((ev/'posthoc_complete.json').stat().st_mtime_ns-(ev/'measurement_complete.json').stat().st_mtime_ns)/1e9)
    write_json(ev/'runtime.json',dict(training_and_validation_epoch_seconds=duration,forward_only_seconds=forward_totals,
        file_timestamp_wall_spans=phase_wall_spans,notes='Epoch time excludes checkpoint writes; forward time excludes data setup/evaluation/cache IO. Phase file timestamp spans include launch gaps, compression and evaluation. Shared GPUs.'))
    bycell={(r['model'],r['section'],r['V'],r['stratum']):r for r in summaries}
    selected=[r for r in pairs if r['metric']=='relative_l1']
    report=REPO/'docs/reports/geometry_stratified_selfsup_matched_v1_zh.md'
    paragraphs=['# 自监督几何采样匹配300轮实验 v1',f'\nStage D **{decision}**。两组均实际完成166train×300epochs、24,900次optimizer step；两组best按原无GT自然Val测量selector冻结后，先完成4,680次measurement-only前向，再独立读取36 Val GT64作post-hoc评价。没有按GT修改采样/参数。',
        '\n## 1. 登记与匹配边界','Stage C已验证两个冻结模型在V3/V4/V6的方向描述量排序，见[冻结评价报告](geometry_strata_frozen_v1_zh.md)。训练优化前协议/源码/计划/初态/评价IDs随提交c2d7454先同步GitHub，随后才创建optimizer。',
        f"两组fresh seed20261011初态SHA256 `{p['initial_state_sha256']['Control']}` 完全一致。禁止旧checkpoint warm start。原332,070参数网络、objective、P_diff/K96/all rays/view chunk2、FP32、clip1、AdamW2e-4/wd1e-4、cosine300、effective batch2均相同。每epochcase顺序相同、83step，视角数量逐case/epoch相同。原GT-free训练代码/guard与冻结SelfSup v1不改。",
        '唯一科学干预是A/B联合几何采样：Control原独立3–5自然随机；Treatment均匀q、A/B同q、A/B/H互斥、H2同uniform disjoint policy。两个输入的数量从Control原sampler直接继承。自然tertile本来各约1/3，不能称Treatment提高优质布局占比；同层A/B相关性是干预的一部分，不是所有几何边际严格相同。Extreme clustered只evaluation，未等权加入训练。',
        f"V5独立100,000唯一Pool A组合定义自己的边界`{json.loads((out/'V5_boundaries.json').read_text())['boundaries']}`，V3/V4用Stage B；边界/采样登记测量opens=0、GT opens=0、checkpoint reads=0。",
        table([dict(arm=a,score_mean=d['scores']['mean'],joint=d['AB_stratum_joint_counts']) for a,d in distribution.items()],[('组',lambda r:r['arm']),('平均中心分数',lambda r:f"{r['score_mean']:.8f}"),('A/B stratum3×3频数',lambda r:json.dumps(r['joint']))]),
        '完整per-station A/B/H频数、方位/半径/高度与separation/score直方图在审计文件；这些边际变化原样报告，不后验强制匹配。',
        '\n## 2. Checkpoint选择、实际预算与标签身份',
        table([dict(arm=a,sha=frozen[a]['sha256'],epoch=best_epochs[a],seconds=duration[a],best=min(r['val']['measurement']['heldout']['relative_l1'] for r in training[a])) for a in c['arms']],
            [('组',lambda r:r['arm']),('best SHA256',lambda r:r['sha']),('best epoch',lambda r:r['epoch']),('原Val H L1',lambda r:f"{r['best']:.6f}"),('300轮实测秒',lambda r:f"{r['seconds']:.1f}")]),
        '两组同原Val-A A4/B4/H4固定36case，selector为原dual-input两个预测的平均held-out测量relative L1，不是tertile macro，不读GT。原训练/验证/选择均GT-free，最终GT仅评价；研究路线由此前Stage C含Val GT的几何验证支持，不能称整个研究开发从未使用GT。best可早于300，但必须300全部结束才准posthoc GT；实际完整optimizer步数已逐epoch核验。共享RTX4090 cuda0/1有其他用户任务，计时不是专用硬件基准，也不是只sampler影响速度的因果对照。',
        f'冻结后各2,340次实际前向，Control纯前向合计{forward_totals["Control"]:.3f}秒，Treatment {forward_totals["Treatment"]:.3f}秒。measurement从best冻结至完成的文件时间跨度{phase_wall_spans["measurement_since_best_freeze_seconds"]:.3f}秒，独立posthoc从measurement完成至GT评价完成跨度{phase_wall_spans["posthoc_since_measurement_complete_seconds"]:.3f}秒；跨度包含启动间隔、评价和压缩/存取缓存，纯前向时间不含这些成本。训练epoch时间包括训练/Val而不含checkpoint写入，完整内存/时间曲线及逐forward秒数归档。',
        '![训练曲线](artifacts/geometry_stratified_selfsup_matched_v1/figures/training_curves.png)',
        '\n## 3. 相同输入的完整分层/自然/压力配对',
        '每case五draw先均值，再36case均值。所有station IDs/order/H/mask/测量和GT一样。差值Treatment−Control，负为改善；case bootstrap10,000、seed2026101404、percentile95%CI，不把draw/射线作独立样本。以下主L1，完整十指标均值/区间与配对在summary/paired_statistics.csv。',
        table(selected,[('section',lambda r:r['section']),('V',lambda r:r['V']),('stratum',lambda r:r['stratum']),('Control L1',lambda r:f"{r['Control']:.6f}"),('Treatment L1',lambda r:f"{r['Treatment']:.6f}"),('T−C',lambda r:f"{r['delta']:+.6f}"),('95%case CI',lambda r:ci_text(r['ci'])),('胜/负/平',lambda r:f"{r['wins']}/{r['losses']}/{r['ties']}")]),
        '\n## 4. 完整三维及测量辅助指标',
        table(summaries,[('组',lambda r:r['model']),('条件',lambda r:f"{r['section']}/V{r['V']}/{r['stratum']}"),('三维L2',lambda r:f"{r['relative_l2']:.6f}"),('MAE/RMSE',lambda r:f"{r['mae']:.8f}/{r['rmse']:.8f}"),('Input L1/L2',lambda r:f"{r['observed_relative_l1']:.6f}/{r['observed_relative_l2']:.6f}"),('H4 L1/L2',lambda r:f"{r['heldout_relative_l1']:.6f}/{r['heldout_relative_l2']:.6f}"),('积分proxy',lambda r:f"{r['integrated_concentration_relative_error']:.6f}"),('峰值误差',lambda r:f"{r['peak_value_absolute_error']:.6f}")]),
        'MAE/RMSE与峰值是质量分数，relative指标无量纲；积分proxy沿原 abs(sum(C)−sum(GT))/sum(abs(GT))并非kg。固定P_diff插值C后转χ/K96，对参考128³/K256测量仍有表示/积分误差；本轮不读GT128内容、不改变投影器。H只评价。',
        '\n## 5. 预登记macro判据与结论',table(primary,[('macro/压力',lambda r:r['stratum']),('Control L1',lambda r:f"{r['Control']:.6f}"),('Treatment L1',lambda r:f"{r['Treatment']:.6f}"),('T−C',lambda r:f"{r['delta']:+.6f}"),('相对变化',lambda r:f"{100*r['relative_change']:+.2f}%"),('case95%CI',lambda r:ci_text(r['ci']))]),
        'macro按每case先平均V3/V4/V6，再case配对；自然/low/middle/high各自独立宏平均，不把clustered混入low。判据：自然CI下端≤0；low均值下降；high若相对增加>5%且CI下端>0则材料性受损。5%为事前工程尺度，clustered只是secondary。',
        f"最终规则结果：`{json.dumps(rules)}`；Stage D **{decision}**。不因为极端压力改善而忽略普通场景损害，不据posthoc GT改变边界、sampler、score或checkpoint。",
        '本次自然macro误差相对增加约1.50%、low增加约1.97%，相应配对区间全正，均支持受损；middle约+0.40%、high约+1.13%且区间含0，尚不确定。clustered约+7.80%、区间全正，同样受损。两个best均在第283轮由原测量selector选定，所以上述差异不来自一组换用层级selector。Stage C的方向描述量关联成立，而同层A/B采样干预未带来收益；这是两个不同命题，不因训练NO-GO改写描述量或边界。',
        '一种解释假设是同层配对减少了训练中跨几何信息条件的组合，使原consistency所遇到的样本关系改变；本轮没有机制性梯度干预，不能把受损唯一归因于该假设。原自然单输入tertile边际本就近似均衡，不能将此结果包装为“增加低信息布局训练的普遍失败”，也不支持自动搜索新weights或score。',
        '![macro配对](artifacts/geometry_stratified_selfsup_matched_v1/figures/paired_macro_effects.png)',
        '\n## 6. 审计、测试、完整性与解释局限',
        '优化前10项新增匹配测试通过，覆盖同初态/工序/步数/loss/完整V计划、唯一sampler变化、A/B/H不交、A/B同q、clustered排除及同selector；前置全仓229项通过。优化后10项再次通过，最终全仓239项通过；另独立重算4,680条row、936个case均值及170项十指标配对/macro区间，全部匹配，并重放全部49,800个Control sample验证原sampler完全一致。再次核验600条epoch日志、每组24,900step、所有计划视角数量、实际train/Val站位digest。运行中原metric float转换警告属于原实现的诊断标量，不修改数值代码，不当作求解失败。',
        f"训练GT64/GT128 opens=0、Pool B=0、test=0、foreign weights=0，原历史/数据metadata对比通过。模型新checkpoint只保存在独立外部实验目录。两组complete→frozen_best→measurement_complete→posthoc_complete时间顺序验证`{chronological}`；measurement阶段GT opens=0，posthoc单独打开36GT64，仅评价已存输出。完整访问日志、hash和阶段证据归档。",
        '本次单matched seed/两个新checkpoint，不能外推普遍采样收益；自然与tertile共享Val工况，case bootstrap不消除家族相关性；原measurement selector也用同Val。Stage C描述量有效不保证训练采样干预有效。A/B同层改变联合分布与consistency条件，即使两组loss不变，也须避免把结果解释为简单增加low训练占比。V3/V4/V6只在Pool A和Val评价，Pool B/test未打开。监督历史模型身份、原SelfSup v1、RVTC关闭及经典有限预算结论全部保留。',
        '![站位边际](artifacts/geometry_stratified_selfsup_matched_v1/figures/sampler_station_frequency.png)',
        '\n## 7. 交付与停止',
        '协议、V5边界、计划/source/initial hashes、station分布审计、selector/evaluation IDs、训练曲线/耗时/step、best SHA、measurement与独立posthoc GT审计、逐case/draw、case均值、十指标配对/宏区间、关键图和完整性均归档。数据、GT、权重、输出体积与大型完整采样计划仍只在外部，Git保留小标量/证明文件。',
        f'Stage D结果为{decision}，本轮工作结束。维持原冻结架构及历史基线，不自动进行v2、multi-seed、新score/教师/loss/模块/先验、经典重跑、Pool B或最终test。']
    report.write_text('\n\n'.join(paragraphs)+'\n')
    archive=REPO/'docs/reports/artifacts/geometry_stratified_selfsup_matched_v1';archive.mkdir(exist_ok=True)
    for f in ['case_means.csv','summary.csv','paired_statistics.csv','paired_statistics.json','decision.json','training_curves.csv','access_audit.json','integrity.json','per_case_draw.csv',
        'measurement_complete.json','posthoc_complete.json','Control_measurement_access_audit.json','Treatment_measurement_access_audit.json','posthoc_access_audit.json','runtime.json','independent_result_verification.json']:
        shutil.copy2(ev/f,archive/f)
    for f in ['frozen_best.json','Control_training.log','Treatment_training.log','post_training_tests.log','full_post_training_tests.log','independent_verification.log','measurements_evaluation.log','gt_evaluation.log','post_training_pipeline.log']:
        shutil.copy2(out/f,archive/f)
    for arm in c['arms']:
        for f in ['complete.json','access_audit.json']:shutil.copy2(out/arm/f,archive/f'{arm}_{f}')
        shutil.copy2(out/arm/'metrics.jsonl',archive/f'{arm}_metrics.jsonl')
    shutil.copytree(figures,archive/'figures',dirs_exist_ok=True)
    manifest=dict(registered_sources=p['source_sha256'],registered_inputs=p['input_sha256'],report_source_sha256=digest(__file__),verification_source_sha256=digest(REPO/'scripts/verify_geometry_stratified_matched.py'),
        frozen_best=frozen,posthoc_rows_sha256=digest(ev/'per_case_draw.json'),protocol_commit='c2d7454',excluded='all projection/GT/checkpoints/prediction volumes and large station plans stay external')
    write_json(archive/'source_manifest.json',manifest)
    sums=[f'{digest(q)}  {q.relative_to(REPO).as_posix()}' for q in sorted(archive.rglob('*')) if q.is_file() and q.name!='SHA256SUMS']
    sums.extend(f'{digest(q)}  {q.relative_to(REPO).as_posix()}' for q in [report,Path(__file__),REPO/'scripts/verify_geometry_stratified_matched.py',REPO/'docs/geometry_stratified_selfsup_matched_v1_protocol_zh.md'])
    (archive/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
    print(json.dumps(dict(decision=decision,rules=rules,primary=primary,checkpoint_hashes=frozen,initial_states_identical=True,only_sampler_differs=True,
        GT_training_opens=0,Pool_B_opens=0,test_opens=0,both300_before_posthoc_GT=True,training_seconds=duration,report=str(report)),indent=2))

if __name__=='__main__':main()
