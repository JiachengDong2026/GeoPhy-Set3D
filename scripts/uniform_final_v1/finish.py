"""Independent volume verification, paired statistics, figures and final audit bundle."""
from runtime import *
from scope import effective_runs, scope_evidence, deferred_runs, evaluation_stage
import csv
import gzip
import math
import shutil
import numpy as np
from evaluate import TestGuard
METRICS=('volume.relative_l1','volume.relative_l2','morphology.gt_peak_location_ratio',
    'morphology.peak_ratio','morphology.absolute_peak_ratio_error','morphology.centroid_distance_mm',
    'morphology.dice_0.1','morphology.dice_0.5','measurement.observed.relative_l1',
    'measurement.observed.relative_l2','measurement.heldout.relative_l1','measurement.heldout.relative_l2')
ARTIFACT=PROJECT/'docs/reports/artifacts/mrgr_uniform_final_v1'

def scalar(r,key):
    group,leaf=key.split('.',1)
    if group=='measurement':
        label,leaf=leaf.split('.',1);return r[group][label][leaf]
    return r[group][leaf]

def all_methods():return verified(BASE/'final_lock.json')['methods']+verified(BASE/'reuse_methods.json')
def root_for(m):return Path(m.get('prediction_root',BASE/'evaluation/test'/m['key']))

def merge():
    rr=rows(BASE/'evaluation/test/metrics.jsonl')
    references=verified(BASE/'reuse_methods.json')
    paths={m['source_metrics'] for m in references}
    for path in paths:
        keys={m['key'] for m in references if m['source_metrics']==path}
        rr.extend(r for r in rows(path) if r['method'] in keys)
    lookup={(r['method'],r['stage'],r['condition'],r['draw'],r['case_id']):r for r in rr}
    assert len(lookup)==len(rr)
    expected=sum(1190 if m['kind']=='neural' else 595 for m in all_methods())
    assert len(rr)==expected,(len(rr),expected)
    jsonl(BASE/'analysis/all_metrics.jsonl',rr)
    return rr,lookup

def independent(lookup):
    source={r['case_id']:r for r in rows(DATA/'source_index.jsonl') if r['split']=='test'}
    computed={};errors={};count=0
    with TestGuard(gt_splits=['test']) as guard:
        gt={c:np.load(DATA/r['gt_volume_64_path'],allow_pickle=False).squeeze().astype(np.float64) for c,r in source.items()}
        coords=np.linspace(0,.6,64)
        def center(v):return np.array([np.dot(v.sum(axis=tuple(j for j in range(3) if j!=i)),coords)/v.sum() for i in range(3)])
        for m in all_methods():
            root=root_for(m)
            for record in rows(root/'manifest.jsonl'):
                path=root/record['prediction'];assert sha(path)==record['sha256'];g=gt[record['case_id']]
                with np.load(path,allow_pickle=False) as z:
                    assert set(z.files)==({'C0','C1'} if m['kind']=='neural' else {'C1'})
                    for stage in z.files:
                        p=z[stage].astype(np.float64)
                        assert p.shape==(64,)*3 and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                        key=(m['key'],stage,record['condition'],record['draw'],record['case_id']);r=lookup[key]
                        vals={'volume.relative_l1':float(np.abs(p-g).sum()/max(np.abs(g).sum(),1e-6)),
                            'volume.relative_l2':float(np.sqrt(np.square(p-g).sum()/max(np.square(g).sum(),1e-12))),
                            'morphology.peak_ratio':float(p.max()/g.max()),
                            'morphology.absolute_peak_ratio_error':float(abs(p.max()/g.max()-1)),
                            'morphology.gt_peak_location_ratio':float(p[np.unravel_index(g.argmax(),g.shape)]/g.max())}
                        if p.sum()>0:vals['morphology.centroid_distance_mm']=float(np.linalg.norm(center(p)-center(g))*1000)
                        for fraction in (.1,.5):
                            pm=p>=fraction*g.max();gm=g>=fraction*g.max()
                            vals['morphology.dice_'+str(fraction)]=float(2*(pm&gm).sum()/(pm.sum()+gm.sum()))
                        for k,v in vals.items():
                            expected=scalar(r,k)
                            assert math.isclose(v,expected,rel_tol=1e-8,abs_tol=1e-9),(key,k,v,expected)
                            errors[k]=max(errors.get(k,0.),abs(v-expected))
                        computed[key]=vals['volume.relative_l1'];count+=1
            print('Independent verification',m['key'],count,flush=True)
        assert not guard.denied
    freeze(BASE/'analysis/independent_fields.json',dict(at=now(),passed=True,fields=count,
        metrics_max_abs_difference=errors,audit=guard.summary(),GT128_read=False))
    return computed

def expressions(mm):
    available={m['key'] for m in mm}
    uniform=[('uniform_fusion_seed20261011',1/4),('uniform_fusion_seed20261019',1/4),
        ('uniform_seed20261020',1/4),('uniform_seed20261021',1/4)]
    final=[]
    def add(label,terms,primary=True):
        assert all(k in available for k,stage,w in terms),label
        final.append(dict(comparison=label,terms=[dict(method=k,stage=stage,weight=w) for k,stage,w in terms],primary_family=primary))
    for key in ('MRGR','LR4E4','NNLS-NL','Tikhonov','TV','SART'):
        add('uniform4seed_case_mean-minus-'+key,[(k,'C1',w) for k,w in uniform]+[(key,'C1',-1)])
    gca=[('uniform_fusion_seed20261011','C1',.5),('uniform_fusion_seed20261019','C1',.5),('MRGR','C1',-.5),('multiscale_seed20261019','C1',-.5)]
    add('uniform2seed-minus-matchedGCA2seed',gca)
    for arm in ('no_feedback','zero_residual','no_ray_relations','no_consistency','single_scale'):
        seeds=[seed for seed in ABLATION_SEEDS if f'uniform_{arm}_seed{seed}' in available]
        # Do not substitute a single-seed result for a registered two-seed
        # primary comparison. Preserve its unperformed slot in the family of14.
        if len(seeds)==2:
            terms=[]
            for seed in seeds:
                terms.extend([(f'uniform_{arm}_seed{seed}','C1',.5),(f'uniform_fusion_seed{seed}','C1',-.5)])
            add(arm+'-minus-uniform2seed',terms)
        for seed in seeds:
            add(arm+f'-minus-uniform_seed{seed}',[(f'uniform_{arm}_seed{seed}','C1',1),(f'uniform_fusion_seed{seed}','C1',-1)],False)
    add('pruned_descriptor-minus-uniform2seed',[(f'{arm}_seed{seed}','C1',w) for seed in ABLATION_SEEDS for arm,w in [('uniform_pruned_observability',.5),('uniform_fusion',-.5)]])
    promoted=[s for s in NEW_SEEDS if f'uniform_feedback_oari_seed{s}' in available]
    if promoted:
        terms=[(f'{arm}_seed{s}','C1',w/len(promoted)) for s in promoted for arm,w in [('uniform_feedback_oari',1),('uniform',-1)]]
        add('feedbackOARI2seed-minus-matchedUniform2seed' if len(promoted)==2 else 'feedbackOARI_single_seed_exploratory',terms,len(promoted)==2)
        for seed in promoted:
            add(f'feedbackOARI-minus-uniform_seed{seed}',[(f'uniform_feedback_oari_seed{seed}','C1',1),(f'uniform_seed{seed}','C1',-1)],False)
    for seed in ABLATION_SEEDS:
        g='MRGR' if seed==20261011 else 'multiscale_seed20261019'
        add(f'uniform-minus-GCA_seed{seed}',[(f'uniform_fusion_seed{seed}','C1',1),(g,'C1',-1)],False)
        add(f'pruned_descriptor-minus-uniform_seed{seed}',[(f'uniform_pruned_observability_seed{seed}','C1',1),(f'uniform_fusion_seed{seed}','C1',-1)],False)
    for m in mm:
        if m['kind']=='neural':add(m['key']+'_C1-minus-C0',[(m['key'],'C1',1),(m['key'],'C0',-1)],False)
    assert sum(e['primary_family'] for e in final)<=14
    return final

def stats(mm,lookup,computed):
    cases=verified(BASE/'registration.json')['cases']['test'];layouts=verified(BASE/'layouts.json')['test']
    panels=[(s['condition'],[s['draw']]) for s in layouts]+[(c,list(range(5))) for c in ('V3','V6','clustered-V4')]
    vectors={};summary=[];csvrows=[];worst=[]
    for m in mm:
        for stage in (('C0','C1') if m['kind']=='neural' else ('C1',)):
            for condition,draws in panels:
                metrics={};individual=[]
                for case in cases:
                    row=dict(method=m['key'],arm=m.get('arm',m['key']),seed=m.get('seed'),stage=stage,
                        condition=condition,draws=','.join(map(str,draws)),case_id=case)
                    for metric in METRICS:
                        vals=[scalar(lookup[m['key'],stage,condition,d,case],metric) for d in draws]
                        row[metric]=None if any(v is None for v in vals) else float(np.mean(vals))
                    csvrows.append(row);individual.append(row)
                for metric in METRICS:
                    array=np.array([r[metric] if r[metric] is not None else np.nan for r in individual]);vectors[m['key'],stage,condition,tuple(draws),metric]=array
                    finite=array[np.isfinite(array)]
                    metrics[metric]=dict(valid_cases=len(finite),mean=float(finite.mean()) if len(finite) else None,
                        median=float(np.median(finite)) if len(finite) else None,p90=float(np.quantile(finite,.9)) if len(finite) else None,max=float(finite.max()) if len(finite) else None)
                summary.append(dict(method=m['key'],arm=m.get('arm'),seed=m.get('seed'),stage=stage,condition=condition,draws=draws,metrics=metrics,historical=m['historical']))
                worst.extend(sorted(individual,key=lambda r:r['volume.relative_l1'],reverse=True)[:3])
    write(BASE/'analysis/expressions.json',expressions(mm))
    indices=np.random.default_rng(20261020).integers(0,35,(20000,35))
    weights=np.stack([(indices==i).mean(1) for i in range(35)],1);pairs=[];checks=0
    for condition,draws in [('Val-A-V4',[0]),('spread-V4',[0]),('V3',list(range(5))),('V6',list(range(5))),('clustered-V4',list(range(5)))]:
        for e in expressions(mm):
            for metric in METRICS:
                delta=sum(t['weight']*vectors[t['method'],t['stage'],condition,tuple(draws),metric] for t in e['terms'])
                finite=np.isfinite(delta);primary=e['primary_family'] and condition=='Val-A-V4' and metric=='volume.relative_l1'
                result=dict(comparison=e['comparison'],condition=condition,draws=draws,metric=metric,primary=primary,
                    primary_family_comparisons=14,terms=e['terms'],valid_pairs=int(finite.sum()),
                    case_differences={c:float(v) if np.isfinite(v) else None for c,v in zip(cases,delta)})
                if finite.all():
                    boot=weights@delta;ci=np.quantile(boot,[.025,.975]).tolist()
                    simultaneous=np.quantile(boot,[.025/14,1-.025/14]).tolist() if primary else None
                    result.update(delta=float(delta.mean()),ci95=ci,simultaneous_ci=simultaneous,
                        lower=int((delta<-1e-8).sum()),ties=int((np.abs(delta)<=1e-8).sum()),higher=int((delta>1e-8).sum()))
                    if metric=='volume.relative_l1':
                        independent_delta=np.array([sum(t['weight']*np.mean([computed[t['method'],t['stage'],condition,d,c] for d in draws]) for t in e['terms']) for c in cases])
                        np.testing.assert_allclose(delta,independent_delta,rtol=1e-10,atol=1e-10)
                        q=[.025,.975]+([.025/14,1-.025/14] if primary else [])
                        expected=ci+(simultaneous if primary else [])
                        np.testing.assert_allclose(np.quantile(independent_delta[indices].mean(1),q),expected,rtol=1e-10,atol=1e-10);checks+=1
                else:result.update(delta=None,ci95=None,simultaneous_ci=None,lower=None,ties=None,higher=None)
                pairs.append(result)
    for filename,rr in [('per_case_metrics.csv',csvrows),('worst_cases.csv',worst)]:
        with (BASE/'analysis'/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
    jsonl(BASE/'analysis/paired_statistics.jsonl',pairs)
    compact=[{k:v for k,v in p.items() if k not in ('case_differences','terms')} for p in pairs]
    s=dict(at=now(),cases=35,layouts=17,methods=len(mm),panels=summary,paired=compact,
        statistics=verified(BASE/'design.json')['evaluation']['statistics'],scope_evidence=scope_evidence(BASE),
        missing_primary_comparisons=14-sum(p['primary'] for p in compact),
        independent_confirmation=False,previously_used_test=True)
    write(BASE/'analysis/summary.json',s)
    freeze(BASE/'analysis/independent_verification.json',dict(at=now(),passed=True,fields=len(computed),
        L1_pairings_and_intervals_verified=checks,source_field_sha256=sha(BASE/'analysis/independent_fields.json'),
        tolerances=dict(field_rtol=1e-8,field_atol=1e-9,paired_rtol=1e-10,paired_atol=1e-10)))
    return s

def figures(mm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm,Normalize
    available={m['key'] for m in mm}
    oari_seed=next((s for s in NEW_SEEDS if f'uniform_feedback_oari_seed{s}' in available),NEW_SEEDS[0])
    chosen=['MRGR','LR4E4','uniform_fusion_seed20261011',f'uniform_seed{oari_seed}',
        f'uniform_feedback_oari_seed{oari_seed}','NNLS-NL','Tikhonov','TV','SART']
    chosen=[k for k in chosen if any(m['key']==k for m in mm)]
    paths={}
    for m in mm:
        if m['key'] not in chosen:continue
        root=root_for(m)
        for r in rows(root/'manifest.jsonl'):paths[m['key'],r['condition'],r['draw'],r['case_id']]=root/r['prediction']
    source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')};cases=verified(BASE/'registration.json')['cases']['test'];files=[]
    with TestGuard(gt_splits=['test']) as guard:
        for index in (0,17,34):
            case=cases[index];gt=np.load(DATA/source[case]['gt_volume_64_path'],allow_pickle=False).squeeze()
            z=np.unravel_index(gt.argmax(),gt.shape)[2]
            for condition in ('Val-A-V4','clustered-V4'):
                fields=[('GT',gt)]
                for name in chosen:
                    with np.load(paths[name,condition,0,case],allow_pickle=False) as archive:fields.append((name,archive['C1']))
                for mapping in ('asinh','linear'):
                    norm=AsinhNorm(linear_width=float(gt.max())*.02,vmin=0,vmax=float(gt.max())) if mapping=='asinh' else Normalize(vmin=0,vmax=float(gt.max()))
                    fig,axes=plt.subplots(len(fields),2,figsize=(10,2.4*len(fields)),constrained_layout=True)
                    for row,(name,a) in enumerate(fields):
                        for col,plane in enumerate((a[:,:,z].T,a[:,32,:].T)):
                            im=axes[row,col].imshow(plane,origin='lower',extent=[0,600,0,600],cmap='magma',norm=norm)
                            axes[row,col].set_title(f'{name} | peak/GT={a.max()/gt.max():.3f}',fontsize=8)
                            axes[row,col].set_xlabel('x (mm)');axes[row,col].set_ylabel('y (mm)' if col==0 else 'z (mm)')
                    axes[0,0].set_title(f'GT-peak z={z}; XY',fontsize=9);axes[0,1].set_title('Fixed center y=32; XZ',fontsize=9)
                    fig.colorbar(im,ax=axes,label=f'Mass fraction; shared {mapping}; above GT peak saturates',shrink=.7)
                    fig.suptitle(f'Used-test fixed case{index}: {case}\n{condition} draw0 (numeric clustered results average5draws)',fontsize=8)
                    path=BASE/'analysis'/f'fixed_case_{index}_{condition}_{mapping}.png'
                    fig.savefig(path,dpi=130)
                    if mapping=='asinh':fig.savefig(path.with_suffix('.pdf'));files.append(path.with_suffix('.pdf'))
                    plt.close(fig);files.append(path)
        assert not guard.denied
    write(BASE/'analysis/figure_audit.json',dict(at=now(),files={str(p):sha(p) for p in files},audit=guard.summary(),
        selection=verified(BASE/'design.json')['evaluation']['figures']))

def write_reports(s):
    root=PROJECT/'docs/reports';a='artifacts/mrgr_uniform_final_v1/'
    def panel(method,condition,stage='C1'):
        return next(p['metrics'] for p in s['panels'] if p['method']==method and p['condition']==condition and p['stage']==stage and len(p['draws'])==(1 if condition=='Val-A-V4' else 5))
    def result_table(methods):
        lines=['| 模型 | 普通V4 L1 | 聚簇V4 L1 | 聚簇L2 | 聚簇质心mm | 聚簇GT峰处比 | 聚簇峰比绝对误差 | 聚簇Dice50 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name in methods:
            x=panel(name,'Val-A-V4');y=panel(name,'clustered-V4')
            lines.append(f'| {name} | {100*x["volume.relative_l1"]["mean"]:.4f}% | {100*y["volume.relative_l1"]["mean"]:.4f}% | {100*y["volume.relative_l2"]["mean"]:.4f}% | {y["morphology.centroid_distance_mm"]["mean"]:.4f} | {y["morphology.gt_peak_location_ratio"]["mean"]:.4f} | {y["morphology.absolute_peak_ratio_error"]["mean"]:.4f} | {y["morphology.dice_0.5"]["mean"]:.4f} |')
        return lines
    mm=all_methods();design=verified(BASE/'design.json');stage=evaluation_stage(BASE)
    stage_note='本阶段按用户要求，先完成两种子均匀融合与两种子反馈OARI的三维配对评价；7个尚未启动的骨干消融暂缓，3个第二种子此前取消。暂缓任务不自动恢复，反馈OARI是否作为最终模型等待用户审阅，本报告不作自动晋级。'
    main=['uniform_fusion_seed20261011','uniform_fusion_seed20261019','uniform_seed20261020','uniform_seed20261021']
    model_report=['# MRGR-UniformFusion 正式候选身份与训练记录','',
        '主模型只替换GCA融合，保留描述符、射线关系、集合交互、多尺度、损失与原始测量残差反馈。部署标识为MRGR-UniformFusion-v1；固定seed20261011、第285轮，不根据本轮测试结果改选。全部参数可训练，逐张量名称/形状/参数数目见parameter_inventory。','',
        '有效视角权重为m/max(Σm,1)，m=visibility AND view_mask；无效特征先屏蔽，NaN无效视角不污染输出；全无效位置精确零。C0/C1共用一个decoder，两次前向而非两个独立模型；实际中间张量及decoder调用见同设备benchmark。','',
        '统一原训练配置：300轮/24900更新，FP32、TF32关闭、batch2、AdamW2e-4/wd1e-4、cosine300、梯度裁剪1，原A/B/H采样、loss和p99.5归一化不改。新增两种子从20轮完整断点续训，和OARI匹配恢复路径；前缀不另算预算。所有种子按全程验证投影H4最低值选轮，无体积GT。','',
        '| 种子 | 参数 | 最佳轮 | 检查点SHA256 | 来源 |','|---|---:|---:|---|---|']
    catalog=[]
    for name in main:
        m=next(m for m in mm if m['key']==name);catalog.append(m)
        model_report.append(f'| {m["seed"]} | {m["parameters"]} | {m["best_epoch"]} | `{m["checkpoint_sha256"]}` | {"历史复用" if m["historical"] else "本轮新训练"} |')
    model_report+=['',f'[正式配置](../../configs/mrgr_uniform_fusion_v1.json) · [固定身份]({a}model_identity.json) · [所有种子权重清单]({a}model_catalog.json) · [参数清单]({a}parameter_inventory.json) · [实际前向形状/成本]({a}benchmark.json) · [逐轮日志与选轮核验]({a}training_audit.json)。',
        '', '大型权重保留model_catalog记录的服务器路径，Git不提交权重或预测数组。推理必须使用正式配置登记的uniform工厂；原config.model.name只是未改动的历史基础构造器，直接build_model会遗漏均匀融合替换。统一复现入口见scripts/uniform_final_v1/README.md。']
    (root/'mrgr_uniform_final_model_v1_zh.md').write_text('\n'.join(model_report)+'\n')
    write(BASE/'model_catalog.json',dict(at=now(),main_model_id='MRGR-UniformFusion-v1',deployment_seed=20261011,
        deployment_checkpoint_sha256=design['main_inference_checkpoint'],seeds=catalog,selection='each seed H4 only; deployment seed fixed pretraining; no test ranking'))
    overview=['# MRGR-UniformFusion：'+('四项训练的三维配对评价（最终模型待定）' if stage else '最终候选冻结与补充评价总报告'),'',f'本阶段评价完成时间：{now()}（北京时间）。',
        '', '**数据边界：本轮沿用此前多次使用的35例，属于补充评价。数据审计未发现额外独立案例；均匀融合方向是在已用测试结果之后确定，不能称为新独立确认。原始MRGR、历史测试及报告均保留。**','',
        '部署候选固定为MRGR-UniformFusion、seed20261011、第285轮、342725参数；检查点SHA256为`64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc`。该身份在新训练前指定，不根据本轮测试排名调整；四种子均单独报告，种子均值不代表集成。','',
        '## 主模型与经典算法','']+result_table(main+['MRGR','LR4E4','NNLS-NL','Tikhonov','TV','SART'])
    if stage:
        overview[2:2]=[stage_note,'',f'[本阶段范围及暂缓清单]({a}evaluation_stage_amendment_v1.json)','']
    overview+=['','## 唯一主要终点与同时区间','',
        '普通固定V4的C1三维相对L1为唯一主要终点。下面差值按比较名称方向计算，单位为百分点，区间为预登记14项Bonferroni同时案例bootstrap区间；即使个别OARI未晋级，也不缩小校正项数。方向选择及数据复用限制仍使结果具有补充、探索性性质。','',
        '| 比较 | L1差百分点 | 同时区间 |','|---|---:|---|']
    for p in s['paired']:
        if not p['primary']:continue
        lo,hi=p['simultaneous_ci'];overview.append(f'| {p["comparison"]} | {100*p["delta"]:+.4f} | [{100*lo:+.4f}, {100*hi:+.4f}] |')
    overview+=['','## C0到C1：共享解码器的阶段轨迹','',
        '| 模型 | 普通C0 L1 | 普通C1 L1 | 聚簇C0 L1 | 聚簇C1 L1 |','|---|---:|---:|---:|---:|']
    for name in main:
        values=[panel(name,c,stage)['volume.relative_l1']['mean'] for c,stage in [('Val-A-V4','C0'),('Val-A-V4','C1'),('clustered-V4','C0'),('clustered-V4','C1')]]
        overview.append('| '+name+' | '+' | '.join(f'{100*v:.4f}%' for v in values)+' |')
    overview+=['','四种子主模型的普通/聚簇均值分别为'+
        '/'.join(f'{100*np.mean([panel(name,c)["volume.relative_l1"]["mean"] for name in main]):.4f}%' for c in ('Val-A-V4','clustered-V4'))+
        '；四个种子均值间的样本标准差分别为'+
        '/'.join(f'{100*np.std([panel(name,c)["volume.relative_l1"]["mean"] for name in main],ddof=1):.4f}个百分点' for c in ('Val-A-V4','clustered-V4'))+
        '。这只是四个已登记种子的描述性离散程度，不是训练种子总体置信区间。']
    overview+=['','## 证据、复现及结论边界','',
        '历史复用：原始MRGR两种子、LR4E4、均匀融合两种子、四种经典算法，以及均匀融合＋描述符结构删除两种子。本阶段新增训练数为'+str(len(effective_runs(BASE)))+'，实际纳入清单见范围修订与训练审计。所有复用预测按595条布局清单逐文件校验，新增预测全部封存后才读取本轮GT64；GT128未读。','',
        '经典算法使用历史固定预算和参数，NNLS-NL/TV/SART多为200步预算耗尽，不能当作精确最优解。SART采用节点摩尔分数的线性离散算子，统一投影评分仍使用原非线性P(C)。不对异设备历史计时作严格速度排名。','',
        ('本阶段五项新骨干消融全部暂缓，没有对应三维结论；只保留已核验的GCA/均匀融合和描述符结构删除历史证据。原14项主要比较的校正分母不变，五项骨干消融比较缺失。' if stage else
         '消融的条件效应见专项报告，不能用C0/C1替代独立训练。无反馈减少12512参数；零残差保留参数但不计算反馈投影，相应输入权重无任务梯度；单尺度参数匹配但计算与感受野改变。按用户缩短排期要求，测试前取消了无射线关系、无一致性、单尺度的第二种子，这三项仅给单种子探索性案例区间；原14项校正家族不缩小，对应三项双种子比较缺失。单/双种子的案例区间均不覆盖训练随机性的总体不确定性。'),'',
        '反馈OARI为独立支线，不合并主模型。缺少全新案例确认，也没有匹配新增容量的非OARI模块对照，即使多种子同向改善，也不足以将全部收益归因于OARI注意力机制或自动晋级。峰值、质心、Dice与全场L1均完整报告。','',
        ('下一步等待用户根据完整L1、峰值、定位和种子间差异讨论最终候选；不自动将OARI设为最终模型，不自动恢复暂缓实验。当前35例已使用，候选选择不能当作新的独立确认。' if stage else
         '建议冻结本轮可复现的均匀融合候选及补充结果，不继续用同一测试集搜索。论文可报告当前分布内精度与几何敏感性，独立泛化声明仍需新的未使用案例/外部数据；OARI纳入未来主模型须另行独立确认。'),'',
        '[模型身份与配置](mrgr_uniform_final_model_v1_zh.md) · [测试协议](../mrgr_uniform_final_v1_protocol_zh.md) · [消融报告](mrgr_uniform_final_ablations_v1_zh.md) · [OARI支线](mrgr_uniform_oari_v1_zh.md) · [资源调度](mrgr_uniform_final_resources_v1_zh.md)','',
        f'[完整统计]({a}summary.json) · [逐案例CSV]({a}per_case_metrics.csv) · [失败/高误差案例]({a}worst_cases.csv) · [全部原始指标gzip]({a}all_metrics.jsonl.gz) · [独立重算]({a}independent_verification.json) · [验收]({a}acceptance.json)','',
        '## 预登记固定切面','', 'XY位于GT峰的z；XZ固定y索引32。全方法共用GT浓度幅度和asinh色标，不逐模型归一化；线性PNG与asinh PDF完整保留。聚簇图为draw0，表中为五draw案例内均值。','']
    raw_path=BASE/'analysis/all_metrics.jsonl'
    raw=rows(raw_path) if raw_path.exists() else []
    rates=[]
    for m in mm:
        selected=[r for r in raw if r['method']==m['key'] and r['stage']=='C1']
        rates.append(dict(method=m['key'],reconstructions=len(selected),expected=595,
            numerical_failures=sum(r.get('solver_status') in ('line_search_failed','nonfinite') for r in selected),
            budget_exhausted=sum(r.get('solver_status')=='budget_exhausted' for r in selected),
            objective_stable=sum(r.get('solver_status')=='objective_stable' for r in selected)))
    write(BASE/'analysis/failure_rates.json',rates)
    # Keep failure/budget details alongside raw worst-case records, including
    # all175 clustered draw-level reconstructions rather than dropping them.
    budget=['','## 经典算法停止状态','', '| 方法 | 数值失败 | 预算耗尽 | 稳定停止 |','|---|---:|---:|---:|']
    for r in rates:
        if r['method'] in ('NNLS-NL','Tikhonov','TV','SART'):
            budget.append(f'| {r["method"]} | {r["numerical_failures"]} | {r["budget_exhausted"]} | {r["objective_stable"]} |')
    budget+=['',f'[全部方法的完整性与停止统计]({a}failure_rates.json)。未按误差或是否预算耗尽删除案例。']
    overview+=budget
    for index in (0,17,34):
        for condition in ('Val-A-V4','clustered-V4'):
            overview.append(f'![固定案例{index} {condition}]({a}fixed_case_{index}_{condition}_asinh.png)\n')
    (root/'mrgr_uniform_final_v1_zh.md').write_text('\n'.join(overview)+'\n')
    ablations=['# 均匀融合骨干上的匹配消融','',
        '全部单因素实验使用原始数据/损失/采样/300轮预算；一致性消融只将对应权重置零。种子20261011/20261019的均匀融合对照复用历史已核验产物；新训练来自相同初始构造和数据流，CUDA原子累加存在跨执行非确定性。描述符结构删除复用已完成的D组。','']
    if stage:
        ablations+=[stage_note,'','## 暂缓的新消融（未训练、未评价）','']
        ablations.extend('- `'+Path(n).name+'`' for n in deferred_runs(BASE))
        ablations+=['',f'[阶段调整及未启动证明]({a}evaluation_stage_amendment_v1.json)','',
            '以下结果表仅为已有均匀融合和描述符删除证据，不代表暂缓消融已经完成。','']
    elif scope_evidence(BASE):
        ablations+=['测试前按用户要求缩短排期：无反馈、零残差保留20261011/20261019双种子；无射线关系、无一致性、单尺度只保留预登记的首个种子20261011，未启动的20261019取消。保留原始登记，未按训练表现或测试结果选择保留种子。后三项为单种子探索性消融，不能证明跨训练种子稳定性；原14项主要比较的校正分母不变，对应三个双种子比较标为未完成。',
            '',f'[种子范围变更记录]({a}seed_scope_amendment_v1.json)','']
    ablations+=result_table([m['key'] for m in mm if m.get('arm') in ('uniform_fusion','uniform_no_feedback','uniform_zero_residual','uniform_no_ray_relations','uniform_no_consistency','uniform_single_scale','uniform_pruned_observability')])
    ablations+=['','## 两种子案例配对差值','', '| 因素（消融−均匀融合） | 布局 | L1差百分点 | 未校正95%案例区间 |','|---|---|---:|---|']
    for p in s['paired']:
        if p['metric']!='volume.relative_l1' or p['condition'] not in ('Val-A-V4','clustered-V4'):continue
        if not p['comparison'].endswith('minus-uniform2seed'):continue
        lo,hi=p['ci95'];ablations.append(f'| {p["comparison"]} | {p["condition"]} | {100*p["delta"]:+.4f} | [{100*lo:+.4f}, {100*hi:+.4f}] |')
    single_arms=[arm for arm in ('no_ray_relations','no_consistency','single_scale')
        if sum(m.get('arm')=='uniform_'+arm for m in mm)==1]
    if single_arms:
        ablations+=['','## 单种子探索性配对差值','',
            '与同种子的历史均匀融合对照配对，95%区间仅反映35案例差异，不包含训练种子不确定性；不替代原登记的双种子主要比较。','',
            '| 因素及种子 | 布局 | L1差百分点 | 未校正95%案例区间 |','|---|---|---:|---|']
        for p in s['paired']:
            if p['metric']!='volume.relative_l1' or p['condition'] not in ('Val-A-V4','clustered-V4'):continue
            if not any(p['comparison'].startswith(arm+'-minus-uniform_seed') for arm in single_arms):continue
            lo,hi=p['ci95'];ablations.append(f'| {p["comparison"]} | {p["condition"]} | {100*p["delta"]:+.4f} | [{100*lo:+.4f}, {100*hi:+.4f}] |')
    ablations+=['','## 对机制解释的支持程度','']
    for p in s['paired']:
        if not p['primary'] or not p['comparison'].endswith('minus-uniform2seed'):continue
        lo,hi=p['simultaneous_ci']
        conclusion='普通V4主要终点的同时区间支持完整均匀融合版本误差更低' if lo>0 else '普通V4主要终点的同时区间支持该消融版本误差更低，不能声称原模块有益' if hi<0 else '普通V4主要终点同时区间跨零，效应尚不确定'
        cluster=next(q for q in s['paired'] if q['comparison']==p['comparison'] and q['condition']=='clustered-V4' and q['metric']=='volume.relative_l1')
        ablations.append(f'{p["comparison"]}：{conclusion}；聚簇探索性L1差为{100*cluster["delta"]:+.4f}个百分点。该判断受容量/计算差异和训练种子数量限制。\n')
    ablations+=['','旧GCA骨干上的历史消融不当作本骨干的同配置消融。无反馈删除编码器/适配器且不计算反馈投影，容量减少12512；零残差保留容量但跳过反馈投影，计算成本不同；无射线关系将前三关系通道置零，保留描述符；无一致性只改权重；单尺度只改空间拓扑；描述符删除为实际结构删除384参数。名义参数相同不表示有效容量与梯度路径相同。',
        '',f'[逐种子、逐案例、全部指标及C0/C1]({a}per_case_metrics.csv)；[完整差值及区间]({a}paired_statistics.jsonl.gz)；[同设备成本]({a}benchmark.json)。两种子不足以估计训练种子总体不确定性。']
    (root/'mrgr_uniform_final_ablations_v1_zh.md').write_text('\n'.join(ablations)+'\n')
    oari=['# 均匀融合＋反馈OARI：独立研究','',
        'Control为均匀融合MRGR；Treatment只在残差适配器中间加入独立OARI，描述符、损失、多尺度与数据流不变。新增10980参数，总参数353705。两个新种子各先20轮，以验证投影H4及工程/成本门槛判断，再按原cosine300、完整断点状态继续到300轮；最终仅按全部300轮验证H4选轮。未使用测试GT晋级。','',
        '本轮20轮门槛只排除明显工程失败：H4不超过匹配Control的1.10倍、成本不超过2倍、峰值显存不超过20GiB，完整检查通过。它不要求短程三维指标改善，也不沿用历史GCA支线的三维筛选条件。两个独立研究任务的规则与授权边界分别保留。','']
    oari+=result_table([m['key'] for m in mm if m.get('arm') in ('uniform','uniform_feedback_oari')])
    oari+=['','## C0到C1的误差及峰值轨迹','',
        'C0/C1使用同一训练模型的共享解码器，属于阶段诊断。峰比绝对误差为abs(max(C)/max(GT)−1)，与GT峰坐标处的浓度比不同。','',
        '| 模型 | 布局 | C0 L1 | C1 L1 | C0峰比绝对误差 | C1峰比绝对误差 |','|---|---|---:|---:|---:|---:|']
    for m in mm:
        if m.get('arm') not in ('uniform','uniform_feedback_oari'):continue
        for condition in ('Val-A-V4','clustered-V4'):
            x=panel(m['key'],condition,'C0');y=panel(m['key'],condition,'C1')
            oari.append(f'| {m["key"]} | {condition} | {100*x["volume.relative_l1"]["mean"]:.4f}% | {100*y["volume.relative_l1"]["mean"]:.4f}% | {x["morphology.absolute_peak_ratio_error"]["mean"]:.6f} | {y["morphology.absolute_peak_ratio_error"]["mean"]:.6f} |')
    oari+=['','| 比较 | 布局 | 指标 | 差值 | 未校正95%案例区间 |','|---|---|---|---:|---|']
    for p in s['paired']:
        if not p['comparison'].startswith('feedbackOARI') or p['condition'] not in ('Val-A-V4','clustered-V4') or p['metric'] not in ('volume.relative_l1','volume.relative_l2','morphology.absolute_peak_ratio_error','morphology.centroid_distance_mm','morphology.dice_0.5'):continue
        if p['delta'] is None:continue
        lo,hi=p['ci95'];oari.append(f'| {p["comparison"]} | {p["condition"]} | {p["metric"]} | {p["delta"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] |')
    seed_l1=[p for p in s['paired'] if p['comparison'].startswith('feedbackOARI-minus-uniform_seed') and p['metric']=='volume.relative_l1' and p['condition'] in ('Val-A-V4','clustered-V4')]
    if len(seed_l1)==4 and all(p['delta']<0 for p in seed_l1):
        recommendation='两个种子的普通/聚簇L1均为改善方向，值得在新的独立案例上继续确认；仍需核对峰值与定位取舍，暂不晋级主模型。'
    elif len(seed_l1)<4:recommendation='未获得两种子的完整正式比较，当前证据不足以建议扩大；保留工程筛查失败/未晋级原因，暂不晋级主模型。'
    else:recommendation='两个种子、两个关键布局的L1方向不完全一致，当前收益证据混合；不据此扩大无约束搜索或晋级主模型，未来验证需预登记新独立案例。'
    aggregate=next((p for p in s['paired'] if p['comparison']=='feedbackOARI2seed-minus-matchedUniform2seed' and p['condition']=='clustered-V4' and p['metric']=='morphology.absolute_peak_ratio_error'),None)
    if aggregate is not None and aggregate['delta'] is not None:
        recommendation+=f'聚簇峰比绝对误差的两种子平均差为{aggregate["delta"]:+.6f}，'+('存在峰值恢复退化取舍。' if aggregate['delta']>0 else '该项平均值未恶化，但不能据此证明非劣性。')
    oari+=['', '表中L1/L2差值使用0–1比例，质心为mm，其他指标为无量纲；完整主要终点同时区间见总报告。',
        '',f'[筛查门槛与逐种子结果]({a}training_complete.json)；[训练身份]({a}training_audit.json)；[同设备推理成本]({a}benchmark.json)。',
        '',recommendation,'', '没有额外独立案例，两个训练种子和案例区间仍不足以证明训练随机性下普遍稳定优势；新增容量的非注意力匹配对照仍缺失。所有不利指标保留。']
    if stage:oari+=['',stage_note]
    (root/'mrgr_uniform_oari_v1_zh.md').write_text('\n'.join(oari)+'\n')
    events=rows(BASE/'scheduling/events.jsonl');tasks=[]
    for p in sorted((BASE/'scheduling').glob('*.json')):
        r=read(p)
        if 'key' in r:tasks.append(r)
    write(BASE/'analysis/resource_tasks.json',dict(at=now(),tasks=tasks,events=events,
        policy=design['resource_policy'],external_processes_signalled=False,
        raw_snapshots=str(BASE/'scheduling/resources.jsonl')))
    res=['# 本轮GPU调度与恢复记录','',
        '每次分配均采用连续两次间隔至少15秒的空闲检测，并在UUID共享flock获取后再次核查；门槛为利用率≤5%、已用显存<100MiB、可用≥20000MiB且无计算进程。worker继承锁FD，控制器退出不会提前释放运行任务的设备锁。初始并行2项，完成前三轮及主机资源审计后才可扩展至4项。没有对其他任务发信号。','',
        '| 任务 | GPU UUID | 开始 | 结束 | 状态 |','|---|---|---|---|---|']
    if stage:res[2:2]=[stage_note,'']
    for r in tasks:res.append(f'| {r["key"]} | {r["gpu_uuid"]} | {r["started_at"]} | {r.get("finished_at","")} | {r["state"]} |')
    res+=['','工程预检查曾修正测试期望：原零残差对照有意跳过P(C0)并返回None，首版检查误要求其有投影张量。只修复检查，不改变模型/训练协议；失败日志保留。其他异常及恢复如发生，均见机器资源任务表和原始scratch记录，不覆盖历史产物。',
        '',f'[任务/资源事件完整表]({a}resource_tasks.json)；原始每15秒进程/显存/磁盘快照保留scratch。GPU历史计时不作跨设备速度排名，同设备测量单列。']
    (root/'mrgr_uniform_final_resources_v1_zh.md').write_text('\n'.join(res)+'\n')

def publish(s):
    ARTIFACT.mkdir(parents=True,exist_ok=True)
    chosen=['design.json','repository_audit.json','model_identity.json','model_catalog.json','data_audit.json','independent_case_search.json','reuse_methods.json','reuse_audit.json','parameter_inventory.json',
        'training_complete.json','training_audit.json','benchmark.json','registration.json','layouts.json',
        'final_lock.json','reuse_prediction_seal.json','checks/cpu.json','checks/gpu.json','checks/statistics_synthetic.json','checks/pruned_benchmark_dispatch.json',
        'analysis/summary.json','analysis/per_case_metrics.csv','analysis/worst_cases.csv','analysis/failure_rates.json',
        'analysis/independent_fields.json','analysis/independent_verification.json','analysis/expressions.json',
        'analysis/figure_audit.json','analysis/resource_tasks.json','evaluation/test/all_predictions_sealed.json',
        'evaluation/test/gt_access.json']
    evidence={}
    chosen.extend(str(p.relative_to(BASE)) for p in (BASE/'checks').glob('statistics_and_reports_synthetic*.json'))
    chosen.extend(str(Path(p).relative_to(BASE)) for p in scope_evidence(BASE))
    for name in chosen:
        source=BASE/name;target=ARTIFACT/Path(name).name
        if target.exists():assert sha(target)==sha(source),target
        else:shutil.copy2(source,target)
        evidence[str(target.relative_to(PROJECT))]=dict(sha256=sha(target),origin=str(source),origin_sha256=sha(source))
    # Preserve the actual preflight failure and controller-only reload records.
    # Reconstructed debugging source snapshots are not treated as original bytes.
    extras=[BASE/'checks/failures.jsonl',BASE/'checks/seed_scope_amendment_v1.json',
        BASE/'checks/evaluation_stage_amendment_v1.json',BASE/'scheduling/cpu_checks.log',
        *sorted((BASE/'scheduling').glob('controller_reload_*.json'))]
    for source in extras:
        if not source.exists():continue
        target=ARTIFACT/'execution_audit'/source.name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(source,target)
        evidence[str(target.relative_to(PROJECT))]=dict(sha256=sha(target),origin=str(source),origin_sha256=sha(source))
    for name in ('all_metrics.jsonl','paired_statistics.jsonl'):
        source=BASE/'analysis'/name;target=ARTIFACT/(name+'.gz')
        with target.open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:z.write(source.read_bytes())
        assert sha(source)==hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest()
        evidence[str(target.relative_to(PROJECT))]=dict(sha256=sha(target),origin=str(source),origin_sha256=sha(source),transform='lossless gzip')
    for source in (BASE/'analysis').glob('fixed_case_*'):
        target=ARTIFACT/source.name;shutil.copy2(source,target)
        evidence[str(target.relative_to(PROJECT))]=dict(sha256=sha(target),origin=str(source),origin_sha256=sha(source))
    for name in effective_runs(BASE):
        target=Path(name)
        for leaf in ('protocol.json','metrics.jsonl','summary.json','boundary_audit.json','prefix_complete.json','environment.json','gt_access_audit.json','events.jsonl'):
            source=target/leaf
            if not source.exists():continue
            dest=ARTIFACT/'training'/target.name/leaf;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            evidence[str(dest.relative_to(PROJECT))]=dict(sha256=sha(dest),origin=str(source),origin_sha256=sha(source))
    for source in (BASE/'gates').glob('*.json'):
        dest=ARTIFACT/'gates'/source.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest)
        evidence[str(dest.relative_to(PROJECT))]=dict(sha256=sha(dest),origin=str(source),origin_sha256=sha(source))
    reports=[PROJECT/'docs/reports'/n for n in ('mrgr_uniform_final_v1_zh.md','mrgr_uniform_final_ablations_v1_zh.md','mrgr_uniform_oari_v1_zh.md','mrgr_uniform_final_resources_v1_zh.md','mrgr_uniform_final_model_v1_zh.md')]
    sources=verified(BASE/'final_lock.json')['sources']
    for path,digest in sources.items():assert sha(path)==digest,path
    for p in reports:evidence[str(p.relative_to(PROJECT))]=dict(sha256=sha(p),transform='generated report')
    # Recompute every exported panel mean from wide CSV, rather than trusting export.
    with (ARTIFACT/'per_case_metrics.csv').open() as f:rr=list(csv.DictReader(f))
    means={}
    for r in rr:
        key=(r['method'],r['stage'],r['condition'],r['draws'])
        means.setdefault(key,[]).append(r)
    checked=0
    for p in s['panels']:
        records=means[p['method'],p['stage'],p['condition'],','.join(map(str,p['draws']))]
        assert len(records)==35
        for metric in METRICS:
            vals=[float(r[metric]) for r in records if r[metric]!='']
            expected=p['metrics'][metric]['mean']
            assert len(vals)==p['metrics'][metric]['valid_cases']
            if vals:assert math.isclose(float(np.mean(vals)),expected,rel_tol=1e-12,abs_tol=1e-12)
            checked+=1
    freeze(BASE/'acceptance.json',dict(at=now(),completed=True,supplementary_used35test=True,
        result_scope='four_run_priority_3D_evaluation' if evaluation_stage(BASE) else 'full_registered_evaluation',
        overall_work_complete=not bool(deferred_runs(BASE)),deferred_ablation_runs=deferred_runs(BASE),
        final_OARI_model_decision='awaiting_user_review',scope_evidence=scope_evidence(BASE),
        independent_confirmation=False,GT128_read=False,all_new_training_before_current_test_GT=True,
        no_test_tuning=True,independently_verified_fields=read(BASE/'analysis/independent_fields.json')['fields'],
        exported_panel_means_verified=checked,main_model_id='MRGR-UniformFusion',
        main_checkpoint_sha256=verified(BASE/'design.json')['main_inference_checkpoint'],
        OARI_merged_into_main=False,evidence={str(BASE/'final_lock.json'):sha(BASE/'final_lock.json'),
            str(BASE/'analysis/independent_verification.json'):sha(BASE/'analysis/independent_verification.json'),
            str(PROJECT/'docs/reports/mrgr_uniform_final_v1_zh.md'):sha(PROJECT/'docs/reports/mrgr_uniform_final_v1_zh.md')}))
    shutil.copy2(BASE/'acceptance.json',ARTIFACT/'acceptance.json')
    evidence[str((ARTIFACT/'acceptance.json').relative_to(PROJECT))]=dict(sha256=sha(ARTIFACT/'acceptance.json'),origin=str(BASE/'acceptance.json'))
    write(ARTIFACT/'export_manifest.json',dict(at=now(),files=evidence,exported_panel_means_verified=checked,
        notes='exact copies except explicitly lossless gzip; weights/prediction arrays remain scratch; source hashes include independent OARI worktree'))

def main():
    BASE.joinpath('analysis').mkdir(exist_ok=True)
    assert not (BASE/'acceptance.json').exists(),'Accepted result is immutable'
    lock=verified(BASE/'final_lock.json');verify_hashes(lock['sources'])
    assert read(BASE/'benchmark.json')['source_sha256']==sha(SCRIPT/'benchmark.py')
    metrics,lookup=merge();computed=independent(lookup)
    summary=stats(all_methods(),lookup,computed);figures(all_methods());write_reports(summary);publish(summary)
    print('Full supplementary evaluation and independent acceptance completed',flush=True)

if __name__=='__main__':main()
