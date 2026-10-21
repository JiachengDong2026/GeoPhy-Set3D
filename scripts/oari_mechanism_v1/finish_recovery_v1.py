"""Case-level summaries, independent checks and Chinese interpretation."""
from runtime import *
import math
import shutil

A_KEYS=('bias_rms','content_rms','rms_ratio','std_ratio','key_centered_bias_rms','no_bias_to_bias_JS_mean','no_bias_to_bias_L1_mean','entropy_delta_mean')
C_KEYS=('attention_l1_mean','attention_l2_mean','attention_js_mean','attention_entropy_delta_mean','bias_change_mean_abs','bias_change_rms','context_mean_abs_change','context_relative_change','voxel_attention_context_pearson')


def numeric(value):return None if value in (None,'') else float(value)


def summarize_small():
    bias=csv_read(BASE/'per_case_bias.csv');attention=csv_read(BASE/'attention_change_statistics.csv');case_rows=[];panels=[]
    for condition in CONDITIONS:
        ids=sorted({r['case_id'] for r in bias if r['condition']==condition})
        assert len(ids)==3
        for head in ('ALL_HEADS',0,1,2,3):
            for mode in ('A','P1','P2'):
                raw=bias if mode=='A' else attention;keys=A_KEYS if mode=='A' else C_KEYS
                selected=[r for r in raw if r['condition']==condition and (mode=='A' or r['experiment']==mode) and (head=='ALL_HEADS' or int(r['head'])==head)]
                for key in keys:
                    values=[]
                    for case in ids:
                        subset=[r for r in selected if r['case_id']==case];assert len(subset)==(4 if head=='ALL_HEADS' else 1)
                        vv=[numeric(r[key]) for r in subset];value=None if any(v is None for v in vv) else float(np.mean(vv))
                        # ALL_HEADS RMS/ratios pool head moments within each case;
                        # JS/entropy averages and mean head-wise correlations are descriptive.
                        if head=='ALL_HEADS' and mode=='A':
                            def rms(k):return float(np.sqrt(np.mean([float(r[k])**2 for r in subset])))
                            if key in ('bias_rms','content_rms','key_centered_bias_rms'):value=rms(key)
                            elif key=='rms_ratio':value=rms('bias_rms')/rms('content_rms')
                            elif key=='std_ratio':
                                def std(prefix):
                                    mean=np.mean([float(r[prefix+'_mean']) for r in subset])
                                    return float(np.sqrt(max(np.mean([float(r[prefix+'_std'])**2+float(r[prefix+'_mean'])**2 for r in subset])-mean**2,0)))
                                value=std('bias')/std('content') if std('content')>0 else None
                        values.append(value);case_rows.append(dict(diagnostic='A' if mode=='A' else 'C',case_id=case,condition=condition,draw=0,experiment=mode,head=head,metric=key,value=value))
                    valid=all(v is not None for v in values);a=np.array(values,dtype=float) if valid else None
                    idx=np.random.default_rng(SEED).integers(0,3,(20000,3))
                    panels.append(dict(diagnostic='A' if mode=='A' else 'C',condition=condition,experiment=mode,head=head,metric=key,n_cases=3,
                        mean=float(a.mean()) if valid else None,median=float(np.median(a)) if valid else None,
                        ci95=np.quantile(a[idx].mean(1),[.025,.975]).tolist() if valid else None,case_ids=ids,case_values=values,
                        scope='fixed3case sample only; not all35cases; voxel pairs not independent samples',exploratory=True))
    return panels,case_rows


def independent_check(summary):
    # B reconstructed from original absolute sealed scalars rather than newly
    # created attenuation CSV; subtract stages after grouping independent paths.
    raw=csv_read(HIST.BASE/'per_case_metrics.csv');cases=verified(HIST.BASE/'cases.json')
    table={}
    for r in raw:
        if r['experiment'] in ('A','P1','P2'):table.setdefault((r['case_id'],r['condition'],r['stage'],r['experiment']),[]).append(r)
    idx=np.random.default_rng(SEED).choice(35,(20000,35),replace=True);weights=np.stack([(idx==i).mean(1) for i in range(35)],1)
    maxdiff=0.
    for panel in summary['B_panels']:
        arrays={key:[] for key in ('delta_C0','delta_C1','attenuation','absolute_attenuation')}
        for case in cases:
            dd=[]
            for stage in ('C0','C1'):
                rr=table[case,panel['condition'],stage,panel['experiment']];bb=table[case,panel['condition'],stage,'A'];key=panel['metric']
                if any(r[key]=='' for r in rr+bb):dd.append(None)
                else:dd.append(sum(float(r[key]) for r in rr)/len(rr)-sum(float(r[key]) for r in bb)/len(bb))
            arrays['delta_C0'].append(dd[0]);arrays['delta_C1'].append(dd[1])
            arrays['attenuation'].append(None if None in dd else dd[0]-dd[1]);arrays['absolute_attenuation'].append(None if None in dd else abs(dd[0])-abs(dd[1]))
        # abs(mean intervention delta) differs from mean(abs(delta)) across
        # permutations. Verify the absolute auxiliary directly with scalar pairs.
        for ci,case in enumerate(cases):
            values=[]
            for r in table[case,panel['condition'],'C0',panel['experiment']]:
                counterpart=next(x for x in table[case,panel['condition'],'C1',panel['experiment']] if (x['draw'],x['permutation_index'])==(r['draw'],r['permutation_index']))
                b0=next(x for x in table[case,panel['condition'],'C0','A'] if x['draw']==r['draw']);b1=next(x for x in table[case,panel['condition'],'C1','A'] if x['draw']==r['draw'])
                key=panel['metric']
                values.append(None if any(x[key]=='' for x in (r,counterpart,b0,b1)) else abs(float(r[key])-float(b0[key]))-abs(float(counterpart[key])-float(b1[key])))
            arrays['absolute_attenuation'][ci]=None if None in values else sum(values)/len(values)
        for key,vv in arrays.items():
            expected=panel[key+'_case_values']
            if None in vv:assert panel[key+'_ci95'] is None;continue
            a=np.array(vv);np.testing.assert_allclose(a,expected,rtol=1e-9,atol=1e-12)
            ci=np.quantile(weights@a,[.025,.975]);np.testing.assert_allclose(ci,panel[key+'_ci95'],rtol=1e-9,atol=1e-12)
            maxdiff=max(maxdiff,abs(a.mean()-panel[key+'_mean']))
    # Recompute case bootstrap using a multinomial-weight representation.
    idx3=np.random.default_rng(SEED).choice(3,(20000,3),replace=True);w3=np.stack([(idx3==i).mean(1) for i in range(3)],1)
    for panel in summary['AC_panels']:
        values=panel['case_values']
        if None in values:assert panel['ci95'] is None;continue
        a=np.array(values);np.testing.assert_allclose(np.quantile(w3@a,[.025,.975]),panel['ci95'],rtol=1e-10,atol=1e-12)
        assert abs(a.mean()-panel['mean'])<1e-12
    complete=verified(BASE/'probe_complete.json');verify_hashes(complete['summary_files'])
    assert sha(BASE/'prediction_manifest.jsonl')==complete['manifest_sha256']
    for r in rows(BASE/'prediction_manifest.jsonl')+rows(BASE/'failed_attempt1_prediction_manifest.jsonl'):
        assert sha(BASE/r['prediction'])==r['sha256'] and sha(r['historical_prediction_path'])==r['historical_sha256']
        with np.load(BASE/r['prediction'],allow_pickle=False) as new,np.load(r['historical_prediction_path'],allow_pickle=False) as old:
            for stage in ('C0','C1'):
                assert hashlib.sha256(new[stage].tobytes()).hexdigest()==r['output_tensor_sha256'][stage]
                delta=np.abs(new[stage].astype(np.float64)-old[stage]);assert float(delta.max())<=1e-5 and float(delta.mean()/(np.abs(old[stage]).mean()+1e-6))<=1e-4
    freeze(BASE/'independent_verification.json',dict(at=now(),passed=True,B_panels=len(summary['B_panels']),AC_panels=len(summary['AC_panels']),
        B_method='independent regrouping from archived absolute scalars, subtract paired stages, weighted vs indexed bootstrap',
        AC_method='case-weight bootstrap; independent actual-QKV softmax and SciPy JS/entropy in probe; prediction bytes/arrays crosschecked again',
        sampled_forward_outputs_checked=57,fields_checked=114,max_B_scalar_disagreement=maxdiff,GT_read=False,GT128_read=False))


def report(summary):
    def ap(condition,mode,key):return next(p for p in summary['AC_panels'] if (p['condition'],p['experiment'],p['head'],p['metric'])==(condition,mode,'ALL_HEADS',key))
    def bp(condition,mode,metric):return next(p for p in summary['B_panels'] if (p['condition'],p['experiment'],p['metric'])==(condition,mode,metric))
    def f(v):return '未定义' if v is None else f'{v:.6g}'
    def interval(v,scale=1):return '未定义' if v is None else '['+', '.join(f'{x*scale:+.6g}' for x in v)+']'
    complete=verified(BASE/'probe_complete.json');manifest=rows(BASE/'prediction_manifest.jsonl')
    maximum=max(v['max_absolute'] for r in manifest for v in r['historical_comparison'].values())
    lines=['# Uniform Fusion MRGR：OARI三项低成本机制诊断','',f'完成时间：{now()}。','',
      '模型为正式Multi-scale MRGR + Uniform Fusion，seed20261011、epoch285、342725参数。checkpoint SHA256：',f'`{HIST.EXPECTED}`。',
      '本轮无训练、反向传播、优化器更新、模型结构变更、检查点重选或新检查点。未读取GT64或GT128；历史封存指标直接复用。全部结果是已使用测试集上的探索性机制证据，不是独立测试确认。','',
      '## 审计与新增计算边界','',
      'B复用既有35案例×17布局的A/P1/P2指标、原始配对差值和预测manifest，共5670组干预/输入、11340行C0/C1。没有重新评分或生成B预测。已有no_ray_relations训练消融只作为已登记历史背景，不重新训练或重算。',
      '历史仅保存偏置绝对均值摘要，没有attention/content logits/context张量，不能回答A/C。故固定案例索引0、17、34，在普通V4、spread-V4、V3、V6、clustered-V4各draw0采样，A/P1/P2共45次冻结前向。P1只用每视角数的第一条历史固定排列；未据结果选择案例或置换。A/C仅有3个固定案例，不能代表全部35例，draw0也不能代表五draw平均。','',
      '只读hook捕获实际QKV、返回的attention/bias和context，返回None，不改变正式计算。矩阵及时转CPU，每输入汇总后释放，没有在GPU累计整套attention。P1/P2复用原诊断Intervention：P1只重排关系张量两个视角轴，P2只从第一仿射输出移除W_rel·r，保留W_obs·o+b，不是物理角度中性替换。',
      f'全部45次输出与相同案例/布局/draw/干预的历史预测通过一致性核验：最大绝对差{maximum:.8g}，阈值1e-5及相对平均绝对差1e-4。所有参数冻结、无grad、eval；输入哈希和权重哈希不变；P1/P2的features/geometry/visibility/observability及QKV与A一致。','',
      '## A：偏置与内容logits的实际尺度','',
      '实际mask为visibility AND view_mask，softmax轴为候选key视角。所有尺度/分布统计只含有效query/key pair；排除padding与全无效体素。CSV分别保留每案例、head、ordered pair和self-pair标记的均值、标准差、RMS、绝对均值、带符号及绝对P50/P90/P95/P99。',
      '下表先在每案例内汇总4个head的二阶矩，再报告3个固定案例的描述性均值；内容logits为QKᵀ/√8。key中心化偏置去掉每个query上对有效key共同的偏移，该共同偏移不改变softmax。原始偏置大不一定意味着注意力改变大。','',
      '| 布局（draw0） | bias RMS | 内容RMS | RMS比 | std比 | key中心化bias RMS | 无bias→有bias JS | 注意力L1 | entropy变化(nat) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        vals=[ap(condition,'A',k)['mean'] for k in A_KEYS]
        lines.append('| '+condition+' | '+' | '.join(f(v) for v in vals)+' |')
    lines += ['', '数值异常与恢复：首次尝试在12次前向后停止。SciPy Jensen-Shannon距离在近零负舍入值上取平方根导致验证NaN；模型输出及历史一致性没有失败。12份预测及失败日志保留，另登记恢复版本，改用SciPy rel_entr直接求JS并将不低于−1e-12的舍入负值截为0；不修改模型、输入、干预、案例或统计方案。由于attention汇总未落盘，恢复完成45次采样；总前向57次，未隐去额外成本。', '', '原始bias和内容logits幅度、key中心化幅度与softmax JS/L1应联合解释；本报告不设任意“有效/无效”阈值。关系输入的逐pair均值/标准差及与bias的Pearson相关系数保存在bias_scale_statistics.csv；这些空间相关仅描述共同几何变化，不识别独立因果效应。self-pair等常量通道的相关系数未定义并保留。',
      '', '## C：射线关系干预是否改变注意力','',
      '| 布局（3例/draw0） | 干预 | attention L1 | attention L2 | JS(nat) | entropy变化 | bias平均绝对变化 | context平均绝对变化 | 平均逐头体素相关 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for mode in ('P1','P2'):
            keys=('attention_l1_mean','attention_l2_mean','attention_js_mean','attention_entropy_delta_mean','bias_change_mean_abs','context_mean_abs_change','voxel_attention_context_pearson')
            lines.append('| '+condition+' | '+mode+' | '+' | '.join(f(ap(condition,mode,k)['mean']) for k in keys)+' |')
    lines += ['', '每个head、每个有效query的L1/L2/JS与entropy变化分别汇总；每个输入视角的平均权重前值/后值/差值向量和head信息保存在attention_change_statistics.csv。注意力变化与同位置context变化的相关在每案例、head内部计算；表中为这些逐头系数的描述性均值，不把体素当作独立重复。context相对变化与逐例分布完整保留。',
      '内容QKV完全不变而attention/bias改变，支持这些冻结权重的注意力分配依赖显式关系输入。注意力改变不等于重建收益，也不能据少量案例推出所有35例或新几何上的结论。P1不等于整个输入几何错配，P2也不等于重新训练移除OARI。',
      '', '## B：C0到C1的指标差值是否缩小（全部35例）','',
      '差值=干预指标−对应A指标；attenuation=delta_C0−delta_C1。先平均同案例/draw内置换，再平均原实际draw，再以35案例配对bootstrap20000次（seed20261023）。普通/spread各1draw，V3/V6/clustered各5draw；置换和draw均不充当样本。',
      '下面L1单位为百分点。区间是探索性、未校正案例区间，不估计训练种子总体方差，也不修改原14项比较家族。对负差值或Dice等指标，正的signed attenuation不普遍等同于性能改善；另保存绝对差幅度的变化，避免将符号当作收益。','',
      '| 布局 | 干预 | delta C0 (pp) | delta C1 (pp) | attenuation (pp)及95%CI | 中位attenuation (pp) |','|---|---|---:|---:|---|---:|']
    for condition in CONDITIONS:
        for mode in ('P1','P2'):
            p=bp(condition,mode,'relative_l1')
            lines.append(f'| {condition} | {mode} | {p["delta_C0_mean"]*100:+.6f} | {p["delta_C1_mean"]*100:+.6f} | {p["attenuation_mean"]*100:+.6f} {interval(p["attenuation_ci95"],100)} | {p["attenuation_median"]*100:+.6f} |')
    lines += ['', '| 布局 | 干预 | L2 attenuation | Dice50 attenuation | 质心距离attenuation mm |','|---|---|---:|---:|---:|']
    for condition in CONDITIONS:
        for mode in ('P1','P2'):lines.append('| '+condition+' | '+mode+' | '+' | '.join(f(bp(condition,mode,k)['attenuation_mean']) for k in ('relative_l2','dice50','centroid_distance_mm'))+' |')
    normal1=bp('Val-A-V4','P1','relative_l1');normal2=bp('Val-A-V4','P2','relative_l1')
    lines += ['', f'普通V4的L1干预差值从C0到C1缩小：P1平均attenuation {normal1["attenuation_mean"]*100:+.6f}个百分点，P2为{normal2["attenuation_mean"]*100:+.6f}个百分点。其他布局与其他指标的方向必须分别看表及逐例数据，不能泛化成所有指标都改善。',
      '这些既有输出支持“后续精化削弱了部分最终指标上的干预差异”。历史未保存可比较的整套中间残差和反馈更新量，本轮没有为B重新实现反馈或重跑预测；不能证明Physics Residual Feedback是唯一补偿来源。',
      '', '## 解释与结构改进判断','',
      '偏置/attention结果检验冻结分支是否工作，训练消融检验重新学习后的增量性能，两者不同。普通V4既有no_ray_relations训练消融C1约+0.147个百分点且案例区间跨零，clustered约+2.647个百分点；而冻结P1/P2在clustered小幅改善，仍应同时保留，不能把训练差值解释为推理时关系通道的直接贡献。',
      'GeometryLifting、geometry embedding、可观测性和物理反馈提供了其他几何通路，信息冗余或后续补偿都是与现有证据相容的解释。当前证据不足以认定整个OARI无效或应删除，也不足以证明某个新结构一定更好；更适合限制论文中对显式关系直接贡献的表述。若进一步改进，应先明确要区分的机制并单独预登记，而不是仅根据注意力变化幅度增加复杂度。本轮不启动任何新训练或结构搜索。',
      '', '## 资源、核验与产物','',
      f'采样使用GPU `{complete["gpu_uuid"]}`，通过原UUID排他锁及间隔15秒双空闲检查。GeometryLifting两个任务的GPU/进程/batch/训练协议均未改变，资源和前后状态记录在resource_audit.json。统计仅用2CPU线程；没有终止其他任务。',
      '独立核验包括：原始QKV的float64 CPU QK/softmax复算与实际attention一致；每head最多1024条固定有效query的SciPy JS/entropy复核；B从历史绝对指标重新分组并用权重形式重采样；45份恢复预测及12份首次尝试预测与历史数组再次逐一核验。',
      'A已完成（45次前向与C共用）；B已完成（全部复用既有指标，无前向/GT）；C已完成（固定3例×5布局draw0最小采样）。未完成项：历史整套残差/反馈更新量不存在，因此没有唯一补偿来源分析；没有新增35例attention全套或跨种子泛化结论。',
      '', '[审计计划](artifacts/mrgr_oari_mechanism_diagnostics_v1/audit_plan_zh.md) · [协议](artifacts/mrgr_oari_mechanism_diagnostics_v1/protocol.json) · [偏置分布](artifacts/mrgr_oari_mechanism_diagnostics_v1/bias_scale_statistics.csv) · [attention变化](artifacts/mrgr_oari_mechanism_diagnostics_v1/attention_change_statistics.csv) · [C0/C1衰减统计](artifacts/mrgr_oari_mechanism_diagnostics_v1/c0_c1_attenuation_statistics.csv) · [逐例结果](artifacts/mrgr_oari_mechanism_diagnostics_v1/per_case_results.csv) · [完整摘要](artifacts/mrgr_oari_mechanism_diagnostics_v1/summary.json) · [验收](artifacts/mrgr_oari_mechanism_diagnostics_v1/acceptance.json)。',
      '[历史冻结诊断](mrgr_geometry_utilization_diagnostic_v1_zh.md) · [历史解释补充](mrgr_geometry_utilization_v1_interpretation_zh.md) · [均匀融合训练消融](mrgr_uniform_ablations_v1_zh.md)。',
      '', f'原始采样预测与日志：`{BASE}`；复现代码：`scripts/oari_mechanism_v1/`。大型数组不入Git，历史产物没有覆盖。']
    assert not REPORT.exists();REPORT.write_text('\n'.join(lines)+'\n')


def main():
    recovery=verified(BASE/'numerical_recovery_v1.json');verify_hashes(recovery['sources'])
    backend();p=verified(BASE/'protocol.json');verify_hashes(p['sources']);completed=verified(BASE/'probe_complete.json');assert completed['completed']
    small,rows_small=summarize_small();b=verified(BASE/'attenuation_summary.json')
    combined=[dict(diagnostic='B',**r) for r in csv_read(BASE/'per_case_attenuation.csv')]+rows_small
    csv_write(BASE/'per_case_results.csv',combined)
    summary=dict(at=now(),model=p['model'],A_status='completed: minimal new frozen sampling',B_status='completed: existing sealed scalar reuse only',C_status='completed: same45forwards asA',
        B_panels=b['panels'],AC_panels=small,bootstrap_seed=SEED,resamples=20000,B_n_cases=35,AC_n_fixed_cases=3,
        new_forwards=57,valid_recovery_forwards=45,initial_attempt_forwards=12,training_runs=0,GT64_read=False,GT128_read=False,independent_confirmation=False,
        restrictions=['A/C limited to3fixedcases anddraw0','no seed population uncertainty','attention change not performance benefit','feedback not isolated'])
    freeze(BASE/'summary.json',summary);independent_check(summary)
    before=verified(BASE/'audit.json')['training_state'];after=read(TRAIN_BASE/'status.json')
    freeze(BASE/'resource_audit.json',dict(at=now(),training_before=before,training_after=after,resources_after=resources(),
        tasks=[read(p) for p in (BASE/'scheduling').glob('*.json')],training_processes_signalled=False,training_GPU_or_protocol_modified=False,
        raw_snapshots=str(BASE/'scheduling/resources.jsonl')))
    report(summary)
    failures={p.name for p in BASE.glob('*failure*.json')};assert failures=={'probe_failure.json','controller_failure.json'},failures
    verify_hashes(p['sources']);assert sha(p['model']['checkpoint'])==HIST.EXPECTED
    freeze(BASE/'acceptance.json',dict(at=now(),completed=True,A_completed=True,B_completed=True,C_completed=True,
        model_id='MRGR-UniformFusion-v1',seed=20261011,epoch=285,parameters=342725,checkpoint_sha256=HIST.EXPECTED,
        new_forwards=57,valid_recovery_forwards=45,initial_attempt_forwards=12,A_C_fixed_cases=3,A_C_draw=0,A_C_layouts=5,B_cases=35,B_layouts=17,B_forwards=0,GT64_read=False,GT128_read=False,
        training_runs=0,backward_calls=0,optimizer_updates=0,new_checkpoints=0,model_structure_changed=False,
        predictions_match_history=True,independent_statistics_verified=True,training_uninterrupted=True,independent_confirmation=False,
        unavailable=['historical full residual/feedback-update tensors; no unique compensation-source attribution'],anomalies=['initial probe stopped after12 forwards when SciPy JS square root returned NaN for roundoff-scale negative JS; outputs/logs preserved; separately registered direct-KL numerical verification recovery completed45 forwards'],
        evidence={str(x):sha(x) for x in [*BASE.glob('*.json'),*BASE.glob('*.csv'),BASE/'prediction_manifest.jsonl',REPORT] if x.name!='status.json'}))
    export();print('ACCEPTED all3 OARI low-cost frozen diagnostics; GeometryLifting training continues',flush=True)


def export():
    ARTIFACT.mkdir(exist_ok=False)
    for path in [*BASE.glob('*.json'),*BASE.glob('*.csv'),*BASE.glob('*.md'),BASE/'prediction_manifest.jsonl']:
        if path.name=='status.json':continue
        shutil.copy2(path,ARTIFACT/path.name)
        side=path.with_suffix(path.suffix+'.sha256')
        if side.exists():shutil.copy2(side,ARTIFACT/side.name)
    dest=ARTIFACT/'execution_logs';dest.mkdir()
    for path in (BASE/'scheduling').iterdir():
        if path.suffix in ('.json','.log'):shutil.copy2(path,dest/path.name)
    freeze(ARTIFACT/'export_manifest.json',dict(at=now(),files={str(p.relative_to(PROJECT)):sha(p) for p in ARTIFACT.rglob('*') if p.is_file()},large_arrays_on_scratch=True))


if __name__=='__main__':
    with Guard():main()
