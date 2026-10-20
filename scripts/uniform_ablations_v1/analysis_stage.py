"""Reuse original 14-slot case statistics; publish into a new immutable report."""
from stage import *
import csv
import gzip
import math
import shutil
import tempfile
import numpy as np

ARTIFACT = PROJECT/'docs/reports/artifacts/mrgr_uniform_ablations_v1'
REPORT = PROJECT/'docs/reports/mrgr_uniform_ablations_v1_zh.md'
PROGRESS = PROJECT/'docs/reports/mrgr_uniform_ablations_v1_progress_zh.md'


def engine():
    import finish as original
    original.BASE=BASE
    original.TestGuard=TestGuard
    original.scope_evidence=scope_evidence
    original.effective_runs=lambda *a:run_paths()
    original.evaluation_stage=lambda *a:None
    original.deferred_runs=lambda *a:[]
    return original


def methods():
    return verified(BASE/'final_lock.json')['methods']+verified(BASE/'reuse_methods.json')


def benchmark():
    import benchmark as original
    original.BASE=BASE
    original.main()


def checks():
    """Nonconstant synthetic cases exercise original bootstrap and new reports."""
    a=engine(); mm=[]
    for n in run_paths():
        p=verified(Path(n)/'protocol.json')
        mm.append(dict(key=Path(n).name,kind='neural',arm=p['arm'],seed=p['seed'],
                       parameters=p['parameters'],historical=False,best_epoch=300))
    mm+=verified(BASE/'reuse_methods.json')
    assert len(mm)==22
    saved=a.BASE
    with tempfile.TemporaryDirectory(prefix='uniform_ablation_checks_',dir='/scratch/djc') as temp:
        root=Path(temp);(root/'analysis').mkdir()
        cases=[f'case{i}' for i in range(35)]
        freeze(root/'registration.json',dict(cases={'test':cases}))
        freeze(root/'layouts.json',verified(BASE/'layouts.json'))
        freeze(root/'design.json',verified(BASE/'design.json'))
        freeze(root/'analysis/independent_fields.json',dict(synthetic=True))
        lookup={};computed={}
        for i,m in enumerate(mm):
            for stage in (('C0','C1') if m['kind']=='neural' else ('C1',)):
                for s in verified(BASE/'layouts.json')['test']:
                    for j,c in enumerate(cases):
                        x=.1+i*.002+j*j*.00001*(1+i*.1)+s['draw']*.001+(stage=='C0')*.02
                        row=dict(volume=dict(relative_l1=x,relative_l2=x*2),
                            morphology=dict(gt_peak_location_ratio=.7,peak_ratio=.8,
                                absolute_peak_ratio_error=.2,centroid_distance_mm=x*10,
                                **{'dice_0.1':.8,'dice_0.5':.6}),
                            measurement={k:dict(relative_l1=x,relative_l2=x) for k in ('observed','heldout')})
                        key=m['key'],stage,s['condition'],s['draw'],c
                        lookup[key]=row;computed[key]=x
        a.BASE=root
        try:
            summary=a.stats(mm,lookup,computed)
            expr=a.expressions(mm)
            assert sum(p['primary'] for p in summary['paired'])==11
            assert summary['missing_primary_comparisons']==3
            for arm in ('no_ray_relations','no_consistency','single_scale'):
                assert not any(e['comparison']==arm+'-minus-uniform2seed' for e in expr)
                assert not next(e for e in expr if e['comparison']==arm+'-minus-uniform_seed20261011')['primary_family']
            for p in summary['paired']:
                assert p['primary_family_comparisons']==14
                assert not p['primary'] or p['simultaneous_ci'] is not None
            # Clustered5draws reduce to35case units; directly check nonlinear
            # case patterns, means, and fixed seed bootstrap against raw fixture.
            example=next(p for p in summary['paired'] if p['comparison']=='no_feedback-minus-uniform2seed' and p['condition']=='clustered-V4' and p['metric']=='volume.relative_l1')
            terms=next(e for e in expr if e['comparison']==example['comparison'])['terms']
            delta=np.array([sum(t['weight']*np.mean([computed[t['method'],t['stage'],'clustered-V4',d,c] for d in range(5)]) for t in terms) for c in cases])
            indices=np.random.default_rng(20261020).integers(0,35,(20000,35))
            np.testing.assert_allclose(example['ci95'],np.quantile(delta[indices].mean(1),[.025,.975]),atol=1e-12)
            # Render report only in temporary output, never publish fixtures.
            reports(summary,mm,root/'report.md',synthetic=True)
            text=(root/'report.md').read_text()
            assert '不是独立测试确认' in text and '14' in text and '单种子' in text
            assert all(line==line.rstrip() for line in text.splitlines())
        finally:a.BASE=saved
    freeze(BASE/'checks/statistics_synthetic.json',dict(at=now(),passed=True,methods=22,
        primary_available=11,family_denominator=14,missing_two_seed_comparisons=3,
        nonconstant_case_fixture=True,clustered_draws_averaged_before_bootstrap=True,
        independent_indexed_bootstrap=True,new_report_rendered=True,real_GT_read=False,
        sources=source_identity()))
    print('PASS nonconstant synthetic22method/35case/17layout statistics and report',flush=True)


def reports(s,mm,path=REPORT,synthetic=False):
    def panel(key,condition,stage='C1'):
        candidates=[p for p in s['panels'] if p['method']==key and p['condition']==condition and p['stage']==stage]
        return next(p for p in candidates if len(p['draws'])==(5 if condition=='clustered-V4' else 1))['metrics']
    def num(x,digits=4): return '未定义' if x is None else f'{x:.{digits}f}'
    def ci(values,scale=100):
        return '未定义' if values is None else '['+', '.join(f'{scale*x:+.4f}' for x in values)+']'
    a='artifacts/mrgr_uniform_ablations_v1/'
    lines=['# 均匀融合 MRGR：正式骨干消融补充评价','',
        '本阶段沿用已使用的35例、17布局，是补充配对评价，不是独立测试确认。GT128禁止读取。',
        '主模型固定为MRGR-UniformFusion-v1，seed20261011、第285轮、342725参数，checkpoint SHA256：',
        '`64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc`。消融结果不用于重选模型或轮次。',
        '反馈OARI研究停止；仅保留历史封存指标，以维持原14项比较族，不新增其训练或推理。','',
        '## 范围、复用与协议','',
        '复用已核验的均匀融合seed20261011/20261019对照、描述符结构删除双种子、其他历史参照及已完成的seed20261020/20261021统计。',
        '全面搜索scratch、历史研究目录和各工作树输出后，七个目标任务只有登记、没有正式checkpoint，因此本阶段补跑七项；短程工程检查和原GCA骨干消融均不替代正式任务。',
        '每项300轮、24900更新，FP32/TF32关闭、batch2、AdamW lr2e-4/wd1e-4、cosine300、clip1。仅按完整训练期间验证投影H4严格最小值选轮；训练禁止GT64/GT128和测试投影。',
        '全部七项训练完成、checkpoint冻结、7×595组新增C0/C1预测统一封存后，才读取本阶段授权GT64。',
        '无反馈删除编码器/适配器并跳过反馈投影，减少12512参数；零残差保留名义参数但残差置零且跳过反馈投影，梯度/计算路径不同；不能把两者等同为纯容量匹配消融。',
        '无射线关系只清零前三关系通道、保留描述符与维度；无一致性仅将权重0.05改为0；单尺度沿原登记空间拓扑，保持宽度/参数，感受野和计算不同；描述符删除为真实结构删除384参数。','',
        '## 逐种子完整结果','',
        '普通为固定Val-A-V4，聚簇先对每案例五draw平均，再对35案例汇总。L1/L2为百分比，定位为mm，其他量无量纲。','',
        '| 模型 | 布局 | C0 L1% | C1 L1% | C1 L2% | GT峰坐标比 | 峰比绝对误差 | 质心mm | Dice50 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    relevant=[m for m in mm if m.get('arm') in ('uniform_fusion','uniform_pruned_observability') or m['key'] in TASKS]
    for m in relevant:
        for condition in ('Val-A-V4','clustered-V4'):
            p=panel(m['key'],condition);p0=panel(m['key'],condition,'C0')
            values=[num(p0['volume.relative_l1']['mean']*100),num(p['volume.relative_l1']['mean']*100),
                    num(p['volume.relative_l2']['mean']*100)]
            values += [num(p[k]['mean']) for k in ('morphology.gt_peak_location_ratio','morphology.absolute_peak_ratio_error',
                                                  'morphology.centroid_distance_mm','morphology.dice_0.5')]
            lines.append('| '+m['key']+' | '+condition+' | '+' | '.join(values)+' |')
    lines+=['','## 配对差值与预登记校正','',
        '差值方向为“消融−同种子均匀融合”，L1正值表示完整模型误差更低。以案例为单位bootstrap20000次，种子20261020。',
        '原14项Bonferroni分母保持不变。按此前测试前的种子缩减登记，无反馈、零残差双种子汇总补齐；无射线关系、无一致性和单尺度各仅seed20261011，95%区间属于单种子探索性证据，不替代原三个双种子主要比较。',
        '原14项中现可计算11项，3项双种子比较仍不执行；这是原种子范围变更的规定，不新增种子、不缩小校正分母。案例区间不估计训练种子总体不确定性。','',
        '| 比较 | 布局 | L1差百分点 | 95%案例区间 | 原14项同时区间（仅主要终点） |',
        '|---|---|---:|---|---|']
    factors=('no_feedback','zero_residual','no_ray_relations','no_consistency','single_scale','pruned_descriptor')
    for p in s['paired']:
        if p['condition'] not in ('Val-A-V4','clustered-V4') or p['metric']!='volume.relative_l1':continue
        if not p['comparison'].startswith(factors):continue
        lines.append(f'| {p["comparison"]} | {p["condition"]} | {num(p["delta"]*100) if p["delta"] is not None else "未定义"} | {ci(p["ci95"])} | {ci(p["simultaneous_ci"]) if p["primary"] else "探索性，未作主要终点校正"} |')
    lines+=['','## 各因素的贡献与取舍','']
    for arm in factors:
        label=arm+'-minus-uniform2seed'
        ordinary=next((p for p in s['paired'] if p['comparison']==label and p['condition']=='Val-A-V4' and p['metric']=='volume.relative_l1'),None)
        if ordinary is None:
            label=arm+'-minus-uniform_seed20261011'
            ordinary=next(p for p in s['paired'] if p['comparison']==label and p['condition']=='Val-A-V4' and p['metric']=='volume.relative_l1')
        bounds=ordinary['simultaneous_ci'] if ordinary['primary'] else ordinary['ci95']
        cluster=next(p for p in s['paired'] if p['comparison']==label and p['condition']=='clustered-V4' and p['metric']=='volume.relative_l1')
        conclusion='区间跨零，主要终点收益证据不足' if bounds[0]<=0<=bounds[1] else '完整模块误差较低' if bounds[0]>0 else '消融版本误差较低，不能声称该模块改善主要终点'
        lines.append(f'{arm}：普通V4差{ordinary["delta"]*100:+.4f}个百分点，{conclusion}；聚簇V4差{cluster["delta"]*100:+.4f}个百分点，方向'+('一致。' if ordinary['delta']*cluster['delta']>=0 else '不一致，存在布局依赖。'))
        for cond in ('Val-A-V4','clustered-V4'):
            changes=[]
            for metric,labelmetric in [('morphology.absolute_peak_ratio_error','峰比绝对误差'),('morphology.centroid_distance_mm','定位mm'),('morphology.dice_0.5','Dice50')]:
                p=next(q for q in s['paired'] if q['comparison']==label and q['condition']==cond and q['metric']==metric)
                changes.append(labelmetric+'差'+num(p['delta'])+'、95%区间'+ci(p['ci95'],1))
            lines.append(cond+'：'+'；'.join(changes)+'。')
        seedpairs=[p for p in s['paired'] if p['comparison'].startswith(arm+'-minus-uniform_seed') and p['condition']=='Val-A-V4' and p['metric']=='volume.relative_l1']
        if seedpairs:
            lines.append('逐种子普通V4差：'+'；'.join(p['comparison'].split('_seed')[-1]+f' {p["delta"]*100:+.4f}个百分点' for p in seedpairs)+'。'+('仅单种子，不能推断跨种子稳定性。' if len(seedpairs)==1 else '两个固定种子的案例均值不能估计训练种子总体区间。'))
        lines.append('机制解释须结合上面的容量、有效梯度路径与计算差异；统计关联不足以独立证明内部物理机制。\n')
    lines+=['## 状态、成本与完整性','',
        '| 新任务 | 轮次/更新 | 选轮 | 参数 | checkpoint SHA256 |','|---|---|---:|---:|---|']
    for m in mm:
        if m['key'] not in TASKS:continue
        lines.append(f'| {m["key"]} | 300 / 24900 | {m["best_epoch"]} | {m["parameters"]} | {m.get("checkpoint_sha256","synthetic")} |')
    if not synthetic:
        b=verified(BASE/'benchmark.json')
        lines+=['','同一GPU、同一开发案例、batch1/V4/FP32，3次预热、10次同步C0+C1前向；不包含载入/I/O，不作跨设备训练速度排名。','',
            '| 方法 | 参数 | 中位前向ms | 峰值已分配MiB |','|---|---:|---:|---:|']
        for r in b['results']:
            lines.append(f'| {r["method"]} | {r["parameters"]} | {r["median_seconds"]*1000:.3f} | {r["peak_allocated_bytes"]/2**20:.2f} |')
        lines+=['','数值失败、零输出、预算耗尽和高误差案例全部保留。预定L1≥50%和≥100%阈值只用于描述高误差率，不排除案例。','',
            '| 模型 | C1重建数 | L1≥50% | L1≥100% | 数值失败 |','|---|---:|---:|---:|---:|']
        for r in read(BASE/'analysis/failure_rates.json'):
            lines.append(f'| {r["method"]} | {r["reconstructions"]} | {r["l1_ge_0.5"]} | {r["l1_ge_1"]} | {r["numerical_failures"]} |')
        tasks=[read(p) for p in sorted((BASE/'scheduling').glob('*.json')) if 'key' in read(p)]
        lines+=['','| 任务 | GPU UUID | PID | 开始（北京时间） | 结束 | 状态 |','|---|---|---:|---|---|---|']
        for r in tasks:lines.append(f'| {r["key"]} | {r["gpu_uuid"]} | {r["pid"]} | {r["started_at"]} | {r.get("finished_at", "")} | {r["state"]} |')
    lines+=['',f'[复用审计]({a}reuse_audit.json)；[阶段登记]({a}stage_registration.json)；[训练审计]({a}training_audit.json)；[全案例指标]({a}per_case_metrics.csv)；[完整配对统计]({a}paired_statistics.jsonl.gz)；[最差案例]({a}worst_cases.csv)；[独立重算]({a}independent_verification.json)；[成本]({a}benchmark.json)；[资源]({a}resource_tasks.json)。',
        '',f'原始大文件、checkpoint、预测及资源快照保留在 `{BASE}`。可复现命令：',
        '```bash',f'/scratch/djc/envs/gas-torch/bin/python {HERE}/stage.py controller','```',
        '已有接收产物禁止覆盖；恢复前检查失败记录、任务锁、完整模型/优化器/调度器/RNG/步数。','',
        '所有次要指标、C0/C1、17布局逐draw和案例内五draw均值均随CSV/原始指标导出。单种子有利结果或个别次要指标不能支持模块普遍有效的论文表述。']
    Path(path).write_text('\n'.join(lines)+'\n')


def figures(mm):
    """Fixed case0/17/34, both layouts; GT-shared scales, ablation rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm,Normalize
    chosen=[m for m in mm if m['key']=='uniform_fusion_seed20261011' or (m['key'] in TASKS and m['seed']==20261011) or m['key']=='uniform_pruned_observability_seed20261011']
    cases=verified(BASE/'registration.json')['cases']['test']
    source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')}
    a=engine();files={}
    with TestGuard(gt_splits=['test']) as guard:
        for index in (0,17,34):
            case=cases[index];gt=np.load(DATA/source[case]['gt_volume_64_path'],allow_pickle=False).squeeze()
            z=np.unravel_index(gt.argmax(),gt.shape)[2]
            for condition in ('Val-A-V4','clustered-V4'):
                fields=[('GT',gt)]
                for m in chosen:
                    root=a.root_for(m);r=next(r for r in rows(root/'manifest.jsonl') if r['case_id']==case and r['condition']==condition and r['draw']==0)
                    with np.load(root/r['prediction'],allow_pickle=False) as archive:fields.append((m['key'],archive['C1']))
                for mapping in ('asinh','linear'):
                    norm=AsinhNorm(linear_width=float(gt.max())*.02,vmin=0,vmax=float(gt.max())) if mapping=='asinh' else Normalize(vmin=0,vmax=float(gt.max()))
                    fig,axes=plt.subplots(len(fields),2,figsize=(10,len(fields)*2.4),constrained_layout=True)
                    for i,(name,field) in enumerate(fields):
                        for j,plane in enumerate((field[:,:,z].T,field[:,32,:].T)):
                            im=axes[i,j].imshow(plane,origin='lower',extent=[0,600,0,600],cmap='magma',norm=norm)
                            axes[i,j].set_title(f'{name} | peak/GT={field.max()/gt.max():.3f}',fontsize=8)
                            axes[i,j].set_xlabel('x (mm)');axes[i,j].set_ylabel('y (mm)' if j==0 else 'z (mm)')
                    fig.colorbar(im,ax=axes,label=f'Mass fraction; shared GT {mapping} scale; saturates above GT peak',shrink=.65)
                    fig.suptitle(f'Used-test case{index} {condition} draw0 | peak-z XY / fixed center-y XZ',fontsize=9)
                    path=BASE/'analysis'/f'fixed_case_{index}_{condition}_{mapping}.png'
                    fig.savefig(path,dpi=130);files[str(path)]=sha(path)
                    if mapping=='asinh':fig.savefig(path.with_suffix('.pdf'));files[str(path.with_suffix('.pdf'))]=sha(path.with_suffix('.pdf'))
                    plt.close(fig)
        assert not guard.denied
    freeze(BASE/'analysis/figure_audit.json',dict(at=now(),files=files,audit=guard.summary(),selection='fixed indices0/17/34, ordinary and clustered draw0; sharedGT asinh/linear'))


def finish():
    assert not (BASE/'acceptance.json').exists(),'Accepted new-stage result is immutable'
    lock=verified(BASE/'final_lock.json');verify_hashes(lock['sources']);verify_hashes(lock['evidence'])
    a=engine();BASE.joinpath('analysis').mkdir(exist_ok=True)
    rr,lookup=a.merge()
    computed=a.independent(lookup)
    s=a.stats(methods(),lookup,computed)
    assert len(rr)==23800 and len(computed)==23800
    assert s['missing_primary_comparisons']==3
    failures=[]
    for m in methods():
        selected=[r for r in rr if r['method']==m['key'] and r['stage']=='C1']
        assert len(selected)==595
        failures.append(dict(method=m['key'],reconstructions=len(selected),
            numerical_failures=sum(r.get('solver_status') in ('line_search_failed','nonfinite') for r in selected),
            budget_exhausted=sum(r.get('solver_status')=='budget_exhausted' for r in selected),
            zero_output=sum(r['morphology']['peak_ratio']==0 for r in selected),
            l1_ge_0_5=sum(r['volume']['relative_l1']>=.5 for r in selected),
            l1_ge_1=sum(r['volume']['relative_l1']>=1 for r in selected)))
        # JSON names with decimal punctuation match the report's declared cutoff.
        failures[-1]['l1_ge_0.5']=failures[-1].pop('l1_ge_0_5')
    write(BASE/'analysis/failure_rates.json',failures)
    tasks=[read(p) for p in sorted((BASE/'scheduling').glob('*.json')) if 'key' in read(p)]
    write(BASE/'analysis/resource_tasks.json',dict(at=now(),tasks=tasks,
        events=rows(BASE/'scheduling/events.jsonl'),raw_snapshots=str(BASE/'scheduling/resources.jsonl'),
        external_processes_signalled=False))
    figures(methods());reports(s,methods())
    # Independently verify exported CSV means, including the 5draw case panels.
    with (BASE/'analysis/per_case_metrics.csv').open() as f:export=list(csv.DictReader(f))
    grouped={}
    for row in export:grouped.setdefault((row['method'],row['stage'],row['condition'],row['draws']),[]).append(row)
    checked=0
    for p in s['panels']:
        group=grouped[p['method'],p['stage'],p['condition'],','.join(map(str,p['draws']))];assert len(group)==35
        for metric in a.METRICS:
            vals=[float(r[metric]) for r in group if r[metric]!='']
            assert len(vals)==p['metrics'][metric]['valid_cases']
            if vals:assert math.isclose(np.mean(vals),p['metrics'][metric]['mean'],rel_tol=1e-12,abs_tol=1e-12)
            checked+=1
    assert sha(verified(BASE/'model_identity.json')['main']['checkpoint'])==verified(BASE/'design.json')['main_inference_checkpoint']
    freeze(BASE/'acceptance.json',dict(at=now(),completed=True,overall_work_complete=True,
        seven_task_states={k:'newly_completed_and_verified' for k in TASKS},
        new_training=7,reused_methods=15,independently_verified_fields=len(computed),
        exported_panel_means_verified=checked,primary_comparisons=11,family_denominator=14,
        unavailable_original_two_seed_comparisons=3,supplementary_used35test=True,
        independent_confirmation=False,GT128_read=False,all_new_predictions_before_current_GT=True,
        main_checkpoint_sha256=verified(BASE/'design.json')['main_inference_checkpoint'],
        feedback_OARI_restarted=False,OARI_merged_into_main=False,
        evidence={str(BASE/'final_lock.json'):sha(BASE/'final_lock.json'),str(REPORT):sha(REPORT),
                  str(BASE/'analysis/independent_verification.json'):sha(BASE/'analysis/independent_verification.json')}))
    publish()


def publish():
    ARTIFACT.mkdir(parents=True,exist_ok=True);evidence={}
    chosen=[p for p in BASE.glob('*.json') if p.name not in ('status.json',)]
    chosen += list((BASE/'checks').glob('*.json'))
    chosen += [p for p in (BASE/'analysis').glob('*') if p.is_file() and p.name!='all_metrics.jsonl' and p.name!='paired_statistics.jsonl']
    chosen += [BASE/'evaluation/test'/n for n in ('all_predictions_sealed.json','gt_access.json')]
    chosen += list((BASE/'scheduling').glob('*.json'))
    for n in run_paths():
        chosen += [Path(n)/leaf for leaf in ('protocol.json','summary.json','metrics.jsonl','boundary_audit.json',
                    'gt_access_audit.json','environment.json','events.jsonl') if (Path(n)/leaf).exists()]
    for p in chosen:
        dest=ARTIFACT/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==sha(p),dest
        else:shutil.copy2(p,dest)
        evidence[str(dest.relative_to(PROJECT))]=dict(sha256=sha(dest),origin=str(p))
        sidecar=p.with_suffix(p.suffix+'.sha256')
        if sidecar.exists():shutil.copy2(sidecar,dest.with_suffix(dest.suffix+'.sha256'))
    for name in ('all_metrics.jsonl','paired_statistics.jsonl'):
        p=BASE/'analysis'/name;dest=ARTIFACT/'analysis'/(name+'.gz')
        with dest.open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:f.write(p.read_bytes())
        assert hashlib.sha256(gzip.decompress(dest.read_bytes())).hexdigest()==sha(p)
        evidence[str(dest.relative_to(PROJECT))]=dict(sha256=sha(dest),origin=str(p),transform='lossless gzip')
    # Flat aliases match report links; canonical subdirectory hierarchy retained.
    for p in list(ARTIFACT.rglob('*')):
        if p.is_file() and p.parent!=ARTIFACT and not p.is_relative_to(ARTIFACT/'training') and not p.is_relative_to(ARTIFACT/'scheduling'):
            dest=ARTIFACT/p.name
            if not dest.exists():shutil.copy2(p,dest)
    write(ARTIFACT/'export_manifest.json',dict(at=now(),files=evidence,large_arrays_and_checkpoints_in_scratch=True))


def sync():
    assert verified(BASE/'acceptance.json')['completed']
    allow=['scripts/uniform_ablations_v1',str(REPORT.relative_to(PROJECT)),str(PROGRESS.relative_to(PROJECT)),str(ARTIFACT.relative_to(PROJECT))]
    env=dict(os.environ)
    subprocess.run(['git','diff','--check','--',*allow],cwd=PROJECT,env=env,check=True)
    subprocess.run(['git','add','--',*allow],cwd=PROJECT,env=env,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=PROJECT,text=True).splitlines()
    assert all(any(p==x or p.startswith(x+'/') for x in allow) for p in staged),staged
    if staged:
        subprocess.run(['git','diff','--cached','--check'],cwd=PROJECT,env=env,check=True)
        subprocess.run(['git','commit','-m','Complete registered uniform MRGR backbone ablations on matched test35'],cwd=PROJECT,env=env,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
    result=subprocess.run(['git','push','origin','HEAD:main'],cwd=PROJECT,env=env,text=True,capture_output=True)
    evidence=dict(at=now(),commit=commit,push_exit=result.returncode,stdout=result.stdout,stderr=result.stderr)
    write(BASE/'github_sync.json',evidence)
    assert result.returncode==0,evidence
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT,text=True).split()[0]
    assert remote==commit
    evidence.update(remote_main=remote,verified=True);write(BASE/'github_sync.json',evidence)
    print('GitHub sync verified',commit,flush=True)
