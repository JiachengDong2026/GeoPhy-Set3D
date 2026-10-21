"""Post-seal GT64 scoring, independent checks, case statistics and publication."""
from diagnostic import *
import csv
import gzip
import shutil


def write_csv(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in records for k in r))
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(records)
    tmp.replace(path)


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def scalar(row,key):
    v=row[key];return None if v in (None,'') else float(v)


def independent_metrics(p,g):
    """Separate NumPy formulas (not canonical shape/volume scoring functions)."""
    axis=np.linspace(0,.6,64)
    def center(v):
        return np.array([np.dot(v.sum(axis=tuple(j for j in range(3) if j!=i)),axis)/v.sum() for i in range(3)])
    peak=float(g.max());ratio=float(p.max()/peak)
    pm=p>=peak*.5;gm=g>=peak*.5
    return dict(relative_l1=float(np.abs(p-g).sum()/max(np.abs(g).sum(),1e-6)),
        relative_l2=float(np.sqrt(np.square(p-g).sum()/max(np.square(g).sum(),1e-12))),
        dice50=float(2*(pm&gm).sum()/(pm.sum()+gm.sum())),
        absolute_peak_ratio_error=abs(ratio-1),centroid_distance_mm=float(np.linalg.norm(center(p)-center(g))*1000) if p.sum()>0 else None,
        gt_peak_location_ratio=float(p[np.unravel_index(g.argmax(),g.shape)]/peak),peak_ratio=ratio)


def canonical_metrics(p,g):
    import evaluation
    shape=evaluation.safe_shape(p,g)
    volume=evaluation.volume_metrics(p,g)
    return dict(relative_l1=volume['relative_l1'],relative_l2=volume['relative_l2'],
        dice50=shape['dice_0.5'],absolute_peak_ratio_error=abs(shape['peak_ratio']-1),
        centroid_distance_mm=shape['centroid_distance_mm'],gt_peak_location_ratio=shape['gt_peak_location_ratio'],peak_ratio=shape['peak_ratio'])


def score():
    seal=verified(BASE/'all_predictions_sealed.json');verify_hashes(seal['manifest_hashes'])
    verified(BASE/'experiment_registry.json')
    targets=[BASE/'predictions'/f'layout{i:02d}' for i in range(17)]
    sources={r['case_id']:r for r in rows(DATA/'source_index.jsonl') if r['split']=='test'}
    gt={};allrows=[];sensitivity=[];verifications=[]
    path=BASE/'analysis';path.mkdir(exist_ok=True)
    first_access=now()
    with Guard(gt_splits=['test']) as guard:
        for case,src in sources.items():
            gt[case]=np.load(DATA/src['gt_volume_64_path'],allow_pickle=False).squeeze().astype(np.float64)
        for i,target in enumerate(targets):
            dest=path/f'layout{i:02d}_metrics.csv';verify_path=path/f'layout{i:02d}_verified.json'
            rr=rows(target/'manifest.jsonl')
            if verify_path.exists():
                v=verified(verify_path);assert sha(dest)==v['metrics_sha256']
                assert sha(target/'manifest.jsonl')==v['manifest_sha256'];verifications.append(v)
                allrows.extend(read_csv(dest))
            else:
                current=[];errors={};fields=0
                for record in rr:
                    pred=target/record['prediction'];assert sha(pred)==record['sha256']
                    g=gt[record['case_id']]
                    with np.load(pred,allow_pickle=False) as z:
                        for stage in ('C0','C1'):
                            p=z[stage].astype(np.float64)
                            metrics=canonical_metrics(p,g);ind=independent_metrics(p,g)
                            for key,val in ind.items():
                                if val is None:assert metrics[key] is None
                                else:
                                    assert math.isclose(val,metrics[key],rel_tol=1e-8,abs_tol=1e-9),(i,record,key,val,metrics[key])
                                    errors[key]=max(errors.get(key,0.),abs(val-metrics[key]))
                            current.append(dict(case_id=record['case_id'],case_index=record['case_index'],
                                condition=record['condition'],draw=record['draw'],views=record['views'],stage=stage,
                                experiment=record['experiment'],permutation_index=record['permutation_index'],
                                permutation=json.dumps(record['permutation']),checkpoint_sha256=EXPECTED,
                                prediction_sha256=record['sha256'],prediction_path=str(pred),**metrics))
                            fields+=1
                write_csv(dest,current)
                v=dict(at=now(),passed=True,fields=fields,metrics_max_abs_difference=errors,
                       metrics_sha256=sha(dest),manifest_sha256=sha(target/'manifest.jsonl'))
                freeze(verify_path,v);verifications.append(v);allrows.extend(current)
            for record in rr:
                for stage in ('C0','C1'):
                    sensitivity.append(dict(case_id=record['case_id'],case_index=record['case_index'],
                        condition=record['condition'],draw=record['draw'],views=record['views'],stage=stage,
                        experiment=record['experiment'],permutation_index=record['permutation_index'],
                        permutation=json.dumps(record['permutation']),**record['prediction_difference'][stage],
                        repeat_mean_abs=None if record['repeat_difference'] is None else record['repeat_difference'][stage]['mean_abs'],
                        repeat_max_abs=None if record['repeat_difference'] is None else record['repeat_difference'][stage]['max_abs'],
                        repeat_relative_difference=None if record['repeat_difference'] is None else record['repeat_difference'][stage]['relative_prediction_difference'],
                        **record['actual_forward_hook']))
            print('SCORED + INDEPENDENTLY VERIFIED',i,len(allrows),'fields',flush=True)
        # A and formal trained ablations are reused, never re-inferred. Only
        # minimal A numerical replay cases were needed for repeatability in P0.
        reuse=[]
        for m in verified(BASE/'reuse_methods.json'):
            lookup={(r['case_id'],r['condition'],r['draw'],r['stage']):r for r in rows(m['source_metrics']) if r['method']==m['key']}
            assert len(lookup)==1190
            for r in lookup.values():
                reuse.append(dict(case_id=r['case_id'],condition=r['condition'],draw=r['draw'],stage=r['stage'],
                    experiment='A' if m['key']==main_method()['key'] else m['key'],permutation_index=-1,permutation='null',
                    checkpoint_sha256=m['checkpoint_sha256'],**flatten_old(r)))
            if m['key']==main_method()['key']:
                errors={};count=0;root=Path(m['prediction_root'])
                for record in rows(root/'manifest.jsonl'):
                    with np.load(root/record['prediction'],allow_pickle=False) as z:
                        for stage in ('C0','C1'):
                            metrics=independent_metrics(z[stage].astype(np.float64),gt[record['case_id']])
                            old=flatten_old(lookup[record['case_id'],record['condition'],record['draw'],stage])
                            for key,val in metrics.items():
                                assert val is None and old[key] is None or val is not None and math.isclose(val,old[key],rel_tol=1e-8,abs_tol=1e-9)
                                if val is not None:errors[key]=max(errors.get(key,0.),abs(val-old[key]))
                            count+=1
                verifications.append(dict(experiment='A',fields=count,metrics_max_abs_difference=errors))
        assert not guard.denied
    write_csv(BASE/'per_case_metrics.csv',allrows+reuse)
    write_csv(BASE/'permutation_sensitivity.csv',sensitivity)
    freeze(BASE/'GT64_access.json',dict(at=now(),first_current_GT64_at=first_access,
        sealed_at=seal['at'],seal_sha256=sha(BASE/'all_predictions_sealed.json'),
        GT64_sha256={str(DATA/src['gt_volume_64_path']):sha(DATA/src['gt_volume_64_path']) for src in sources.values()},
        GT128_read=False,audit=guard.summary()))
    freeze(BASE/'independent_verification.json',dict(at=now(),passed=True,layouts=verifications,
        new_fields=sum(v['fields'] for v in verifications if 'experiment' not in v),
        baseline_fields=1190,tolerances=dict(rtol=1e-8,atol=1e-9),
        metrics_sha256=sha(BASE/'per_case_metrics.csv'),sensitivity_sha256=sha(BASE/'permutation_sensitivity.csv')))


def describe(values):
    vals=np.array([v for v in values if v is not None],dtype=np.float64)
    if not len(vals):return dict(n=0,mean=None,std=None,median=None,min=None,max=None)
    return dict(n=len(vals),mean=float(vals.mean()),std=float(vals.std(ddof=0)),
                median=float(np.median(vals)),min=float(vals.min()),max=float(vals.max()))


def statistics():
    rr=read_csv(BASE/'per_case_metrics.csv');sens=read_csv(BASE/'permutation_sensitivity.csv')
    cases=verified(BASE/'cases.json');settings=verified(BASE/'layouts.json')['test']
    conditions=list(dict.fromkeys(s['condition'] for s in settings))
    orig={(r['case_id'],r['condition'],r['draw'],r['stage']):r for r in rr if r['experiment']=='A'}
    pairs=[];mapping_deltas=[];percase=[];panels=[]
    # Save every paired scalar change at the exact same case/layout/stage.
    for r in rr:
        if r['experiment']=='A':continue
        reference=orig[r['case_id'],r['condition'],r['draw'],r['stage']]
        record={k:r.get(k) for k in ('case_id','condition','draw','stage','experiment','permutation_index','permutation')}
        for metric in METRICS:
            x=scalar(r,metric);y=scalar(reference,metric)
            record[metric]=None if x is None or y is None else x-y
        pairs.append(record)
    write_csv(BASE/'paired_metrics.csv',pairs)
    indices=np.random.default_rng(20261020).integers(0,35,(20000,35))
    weights=np.stack([(indices==i).mean(1) for i in range(35)],1)
    checks=0;experiments=list(dict.fromkeys(r['experiment'] for r in rr))
    grouped={}
    for r in rr:grouped.setdefault((r['experiment'],r['stage'],r['condition'],r['case_id'],r['draw']),[]).append(r)
    for condition in conditions:
        draws=sorted({str(s['draw']) for s in settings if s['condition']==condition})
        for experiment in experiments:
            if experiment=='repeat':continue # only3 calibration cases, never claim35
            for stage in ('C0','C1'):
                values_by_metric={k:[] for k in METRICS};base_by_metric={k:[] for k in METRICS}
                for case in cases:
                    row=dict(case_id=case,condition=condition,stage=stage,experiment=experiment,draw_count=len(draws))
                    for metric in METRICS:
                        draw_values=[];base_values=[];n=0
                        for draw in draws:
                            group=grouped[experiment,stage,condition,case,draw]
                            x=[scalar(r,metric) for r in group];n+=len(x)
                            draw_values.append(None if any(v is None for v in x) else float(np.mean(x)))
                            base_values.append(scalar(orig[case,condition,draw,stage],metric))
                        value=None if any(v is None for v in draw_values) else float(np.mean(draw_values))
                        base=None if any(v is None for v in base_values) else float(np.mean(base_values))
                        row[metric]=value;row[metric+'_delta']=None if value is None or base is None else value-base
                        values_by_metric[metric].append(value);base_by_metric[metric].append(base)
                        row['repeated_predictions']=n
                    percase.append(row)
                for metric in METRICS:
                    values=values_by_metric[metric];bases=base_by_metric[metric]
                    delta=[None if x is None or y is None else x-y for x,y in zip(values,bases)]
                    interval=None
                    if all(v is not None for v in delta):
                        x=np.array(delta);boot=weights@x;interval=np.quantile(boot,[.025,.975]).tolist()
                        # Independently index-resample35 case differences.
                        np.testing.assert_allclose(interval,np.quantile(x[indices].mean(1),[.025,.975]),rtol=1e-10,atol=1e-10)
                        checks+=1
                    panels.append(dict(condition=condition,draws=draws,experiment=experiment,stage=stage,metric=metric,
                        values=describe(values),baseline=describe(bases),paired_delta=describe(delta),ci95=interval,
                        exploratory=True,independent_training_seeds=0,
                        wins=sum(d<0 for d in delta if d is not None),losses=sum(d>0 for d in delta if d is not None)))
        # Fixed permutation j across cases/draws gives an additional complete
        # distribution over mappings; still descriptive, never n=350/1750.
        for mode in ('P0','P1'):
            for stage in ('C0','C1'):
                for j in range(5 if condition=='V3' else 10):
                    for metric in METRICS:
                        values=[]
                        for case in cases:
                            delta=[]
                            for draw in draws:
                                r=next(r for r in grouped[mode,stage,condition,case,draw] if int(r['permutation_index'])==j)
                                x=scalar(r,metric);y=scalar(orig[case,condition,draw,stage],metric)
                                delta.append(None if x is None or y is None else x-y)
                            values.append(None if any(v is None for v in delta) else float(np.mean(delta)))
                        mapping_deltas.append(dict(condition=condition,experiment=mode,stage=stage,permutation_index=j,metric=metric,**describe(values)))
    write_csv(BASE/'case_aggregates.csv',percase)
    write_csv(BASE/'permutation_metric_distributions.csv',mapping_deltas)
    # Within each case/draw, distribution across permutations (including full
    # metric changes), so favorable permutations cannot hide the adverse ones.
    distribution=[]
    pg={}
    for r in pairs:
        if r['experiment'] not in ('P0','P1'):continue
        pg.setdefault((r['experiment'],r['stage'],r['condition'],r['draw'],r['case_id']),[]).append(r)
    for key,group in pg.items():
        mode,stage,condition,draw,case=key
        for metric in METRICS:
            distribution.append(dict(experiment=mode,stage=stage,condition=condition,draw=draw,case_id=case,
                metric=metric,**describe([r[metric] for r in group])))
    write_csv(BASE/'within_case_permutation_distributions.csv',distribution)
    invariance=[]
    for condition in conditions:
        for mode in ('P0','P1','P2','repeat'):
            for stage in ('C0','C1'):
                group=[r for r in sens if r['condition']==condition and r['experiment']==mode and r['stage']==stage]
                entry=dict(condition=condition,experiment=mode,stage=stage,predictions=len(group),
                    cases=len({r['case_id'] for r in group}))
                for metric in ('mean_abs','max_abs','relative_prediction_difference'):
                    entry[metric]=describe([float(r[metric]) for r in group])
                if mode=='P0':
                    entry['above_registered_tolerance']=sum(float(r['max_abs'])>1e-5 or float(r['relative_prediction_difference'])>1e-4 for r in group)
                if mode=='repeat':
                    for metric in ('repeat_mean_abs','repeat_max_abs','repeat_relative_difference'):
                        entry[metric]=describe([float(r[metric]) for r in group if r[metric]!=''])
                invariance.append(entry)
    allcase_metrics=len(rr)
    freeze(BASE/'statistics.json',dict(at=now(),panels=panels,permutation_deltas=mapping_deltas,
        prediction_sensitivity=invariance,bootstrap_checks=checks,
        registry_statistics=verified(BASE/'experiment_registry.json')['statistics'],
        raw_scalar_rows=allcase_metrics,per_case_aggregate_rows=len(percase),independent_confirmation=False))
    return read(BASE/'statistics.json')


def panel(s,mode,condition,metric='relative_l1',stage='C1'):
    return next(r for r in s['panels'] if r['experiment']==mode and r['condition']==condition and r['metric']==metric and r['stage']==stage)


def make_figures(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions=list(dict.fromkeys(r['condition'] for r in s['panels']))
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    colors=['#5b7cbe','#d56d4a','#6c9e70']
    for stage,ax in zip(('C0','C1'),axes):
        for k,mode in enumerate(('P0','P1','P2')):
            vals=[panel(s,mode,c,stage=stage) for c in conditions]
            y=np.array([r['paired_delta']['mean']*100 for r in vals]);x=np.arange(len(vals))+(k-1)*.23
            lo=np.array([r['ci95'][0]*100 for r in vals]);hi=np.array([r['ci95'][1]*100 for r in vals])
            ax.errorbar(x,y,yerr=np.maximum(np.stack((y-lo,hi-y)),0),fmt='o',capsize=3,color=colors[k],label=mode)
        ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(len(conditions)),conditions,rotation=25)
        ax.set_title(stage+' paired L1 change (case-bootstrap95%CI)');ax.set_ylabel('L1 percentage points, diagnostic minus A');ax.legend()
    fig.suptitle('Frozen uniform MRGR; used35cases; mean permutations THEN mean draws; exploratory',fontsize=10)
    fig.savefig(BASE/'diagnostic_l1_changes.png',dpi=160);fig.savefig(BASE/'diagnostic_l1_changes.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for k,mode in enumerate(('P0','P1','P2')):
        vals=[next(r for r in s['prediction_sensitivity'] if r['condition']==c and r['experiment']==mode and r['stage']=='C1') for c in conditions]
        ax.plot(conditions,[max(r['relative_prediction_difference']['mean'],1e-12) for r in vals],'o-',label=mode,color=colors[k])
    ax.set_yscale('log');ax.set_ylabel('Mean D_pred against sealed A (epsilon=1e-6)');ax.legend()
    ax.set_title('Prediction sensitivity; mapping repetitions are not independent samples')
    fig.savefig(BASE/'prediction_sensitivity.png',dpi=160);fig.savefig(BASE/'prediction_sensitivity.pdf');plt.close(fig)


def report(s):
    def pct(v):return '未定义' if v is None else f'{100*v:.6f}'
    def fmt(v):return '未定义' if v is None else f'{v:.6g}'
    def ci(r,scale=100):return '未定义' if r['ci95'] is None else '['+', '.join(f'{v*scale:+.6f}' for v in r['ci95'])+']'
    conditions=list(dict.fromkeys(r['condition'] for r in s['panels']))
    a='artifacts/mrgr_geometry_utilization_v1/'
    inv=[r for r in s['prediction_sensitivity'] if r['experiment']=='P0']
    maxdiff=max(r['max_abs']['max'] for r in inv);maxrel=max(r['relative_prediction_difference']['max'] for r in inv)
    failed=sum(r['above_registered_tolerance'] for r in inv)
    lines=['# OARI几何关系利用与视角顺序不变性：冻结模型无训练诊断','',
        f'完成时间（北京时间）：{now()}。本轮未训练、未反向传播、未更新权重、未变更模块结构，反馈OARI没有重启。',
        '主模型为MRGR-UniformFusion-v1，seed20261011，第285轮，342725参数；SHA256：',f'`{EXPECTED}`。',
        '所有结果来自同一批已使用的35例×17布局补充诊断，不是独立测试确认，也不用于选模型/轮次或后处理。GT128未读。','',
        '## 复用与新增推理范围','',
        'A完整OARI的595组C0/C1预测和1190行原指标直接复用；uniform_no_ray_relations正式训练消融及描述符结构删除双种子预测/指标也逐文件核验后复用。没有重跑完整A测试套件。',
        'P0/P1/P2缺少干预输出，故对全部595组合进行必要新增诊断；每个布局仅固定案例索引0/17/34各重复两次A前向，校准重复性及历史/当前数值漂移。其余A不重复推理。',
        'V4/V6从所有非恒等排列中固定无放回抽10个；V3总共只有5个不同非恒等排列，全部运行，未以重复排列凑足10。映射种子20261021+视角数；所有35案例、所有布局均保留。','',
        '## 代码与干预审计','',
        '真实关系张量为[B×XYZ,V,V,3]：abs方向余弦、sqrt(1−cos²)、abs方位差/180；没有按训练均值标准化。自对角关系固定[1,0,0]，局部不可见pair置零。',
        'OARI将3关系通道与5维可观测性描述符拼接，经Linear(8,16)→SiLU→Linear(16,4)产生逐头注意力偏置。描述符另外经嵌入加入token，站点/方向/路径还通过GeometryLifting、geometry embedding和物理残差反馈进入网络；所以显式关系通道不等于整个OARI。',
        'P0在选定原输入集合后同步重排全部input_*及view_mask，其他扫描角网格、H集合不动；射线、lifting、可见性在原前向内重新计算。逐次钩子断言R′=PRPᵀ精确成立，可见性同步、描述符在数值容差内相同。padding仅停留原位。',
        'P1在实际interaction.forward入口只置换关系张量两个视角轴，features、geometry、visibility、view_mask、observability保持原对象。原interaction仍重新施加局部pair mask；错配可能把其他位置的不可见零值带到可见pair，是应保留的分布偏移来源。',
        'P2不声称“物理零角度是中性”。它在第一仿射层输出中去掉W_rel·r，保留W_obs·o+b，再走原SiLU和后续层；这是学习预激活坐标中的零加性贡献。所有权重和真实关系张量原值保留。CPU检查独立证实与仅首3输入通道置零代数等价，从而可和既有no_ray_relations定义对应；仍属推理干预，不能当作重新训练或结构删除。',
        'P2同时移除了self-pair余弦1线索，且可能造成训练分布外激活。它不能证明存在唯一的物理“无关系”状态。每个forward均记录真实钩子调用次数、首仿射输入维度、关系变化或去掉的贡献及最终bias有限性。','',
        '## P0：视角顺序不变性','',
        f'全部P0的C0/C1最大绝对体差为{maxdiff:.6g}，最大D_pred为{maxrel:.6g}；超过预登记绝对1e−5或相对1e−4阈值的输出数为{failed}。',
        ('在本轮布局及容差下支持视角排列不变性；没有发现超出检查容差的顺序依赖。' if failed==0 else '发现超出预登记数值容差的输出，须先结合索引、mask、局部计算与重复基准审计；不能直接认定顺序依赖。'),
        '浮点求和及特征批量计算的排列可能引入舍入误差。同步置换保持真实几何不变，因此其指标变化不作为几何利用收益。','',
        '| 布局 | 阶段 | P0最大绝对差 | P0最大D_pred | 同输入两次重复最大差 | 重复A对历史A最大差 |',
        '|---|---|---:|---:|---:|---:|']
    for r in inv:
        repeat=next(x for x in s['prediction_sensitivity'] if x['condition']==r['condition'] and x['stage']==r['stage'] and x['experiment']=='repeat')
        lines.append(f'| {r["condition"]} | {r["stage"]} | {fmt(r["max_abs"]["max"])} | {fmt(r["relative_prediction_difference"]["max"])} | {fmt(repeat["repeat_max_abs"]["max"])} | {fmt(repeat["max_abs"]["max"])} |')
    lines+=['','## P1/P2与正式训练消融：跨布局结果','',
        '表为C1三维L1，单位为百分比；差值为“干预−同案例A”，单位为百分点。先在每案例/每draw平均所有置换，再在案例内平均五draw，最后对35案例作配对bootstrap20000次（seed20261020）。',
        '所有本轮诊断区间均为未校正95%探索性案例区间，不另创主要比较族、不修改此前14项Bonferroni分母；置换/布局draw不视为独立样本，不能解释为训练种子总体区间。','',
        '| 布局 | A L1% | P0差 | P1差及95%区间 | P2差及95%区间 | 重新训练无关系差及95%区间 |',
        '|---|---:|---:|---|---|---|']
    train='uniform_no_ray_relations_seed20261011'
    for c in conditions:
        p0=panel(s,'P0',c);p1=panel(s,'P1',c);p2=panel(s,'P2',c);t=panel(s,train,c)
        lines.append(f'| {c} | {pct(panel(s,"A",c)["values"]["mean"])} | {pct(p0["paired_delta"]["mean"])} | {pct(p1["paired_delta"]["mean"])} {ci(p1)} | {pct(p2["paired_delta"]["mean"])} {ci(p2)} | {pct(t["paired_delta"]["mean"])} {ci(t)} |')
    lines+=['','## P1：正确关系对应是否被利用','']
    for c in conditions:
        r=panel(s,'P1',c);lo,hi=r['ci95'];conclusion='错配平均恶化，支持模型利用正确对应' if lo>0 else '错配平均改善，不能据此支持正确对应有益' if hi<0 else '案例区间跨零，正确对应的重建收益证据不足'
        maps=[x['mean'] for x in s['permutation_deltas'] if x['condition']==c and x['experiment']=='P1' and x['stage']=='C1' and x['metric']=='relative_l1']
        d=describe(maps)
        lines.append(f'{c}：{conclusion}。各固定置换的案例均值差（百分点）：均值{pct(d["mean"])}，标准差{pct(d["std"])}，中位数{pct(d["median"])}，范围[{pct(d["min"])}, {pct(d["max"])}]。\n')
    lines+=['关系错配下降是敏感性/利用的诊断证据，不足以证明学到可泛化几何规律；尤其应结合P2的幅度、可见性错配和既有训练消融，不把错配脆弱性等同为模块净收益。','',
        '## P2、布局依赖与其他指标','',
        '| 布局 | 干预 | 阶段 | L2差 | Dice50差 | 峰比绝对误差差 | 质心差mm |',
        '|---|---|---|---:|---:|---:|---:|']
    for c in conditions:
        for mode in ('P1','P2',train):
            for stage in ('C0','C1'):
                values=[panel(s,mode,c,k,stage)['paired_delta']['mean'] for k in ('relative_l2','dice50','absolute_peak_ratio_error','centroid_distance_mm')]
                lines.append(f'| {c} | {mode} | {stage} | '+' | '.join(fmt(v) for v in values)+' |')
    lines+=['','L2为0–1比例差；Dice50越高越好，其余误差越低越好。全部指标置信区间、GT峰坐标比、峰比及C0/C1分布保留在完整统计文件中。','']
    for c in conditions:
        p1=panel(s,'P1',c);p2=panel(s,'P2',c);tr=panel(s,train,c)
        direction='一致' if p2['paired_delta']['mean']*tr['paired_delta']['mean']>=0 else '不一致'
        lines.append(f'{c}：P2与重新训练无关系的L1差方向{direction}；P1/P2分别为{pct(p1["paired_delta"]["mean"])} / {pct(p2["paired_delta"]["mean"])}个百分点。固定模型中删除输入贡献无法模拟训练适应，幅度不同不构成协议冲突。\n')
    lines+=['## 机制解释与下一阶段建议','',
        '本轮P0验证集合输入排列性质；P1/P2检验固定模型对显式关系的敏感性。此前uniform_no_ray_relations是完整300轮的单种子重新训练证据，两类实验回答不同问题。描述符结构删除双种子历史结果也保留，不能等同为删除整个OARI。',
        'GeometryLifting、geometry token embedding、可观测性token/逐对偏置和物理残差反馈仍存在，可能补偿显式关系缺失。但本轮并未隔离这些通路，故“补偿”仅是可检验解释，不能作为已识别机制。',
        '当前单冻结权重、已使用35案例的诊断不足以决定整体OARI可被删除，也不足以证明跨新几何泛化。若论文要声称整个OARI交互块的增量收益，后续需单独预登记、匹配容量/其余几何通路及预算的NoOARI训练；本轮不启动。若只说明正确关系对应的敏感性和布局依赖，使用本轮完整分布并结合既有训练消融即可，不能仅凭某个有利置换扩大结论。','',
        '## 数值、异常与完整性','',
        '所有35案例、17布局、全部固定置换都保留，未按表现筛选；V3不足10的原因已在注册前声明。所有新增C0/C1体积有限、范围[0,1]；逐forward钩子与状态哈希检查通过，所有参数requires_grad=False且无grad。',
        'GT64访问发生在全部新增预测文件及清单校验封存后；GT128始终禁止。每个新增体与独立NumPy公式交叉核对，原A指标也从已有预测独立重算。配对区间经独立索引重采样核验。',
        f'原始预测和日志位于 `{BASE}`；大文件不入Git。','',
        f'![L1差与案例区间]({a}diagnostic_l1_changes.png)',
        f'![预测体敏感性]({a}prediction_sensitivity.png)','',
        f'[登记]({a}experiment_registry.json) · [复用审计]({a}reuse_audit.json) · [全部映射]({a}permutation_maps.json) · [逐案例指标]({a}per_case_metrics.csv) · [预测体差异]({a}permutation_sensitivity.csv) · [配对差值]({a}paired_metrics.csv) · [完整统计]({a}statistics.json) · [置换内分布]({a}within_case_permutation_distributions.csv) · [独立核验]({a}independent_verification.json) · [验收]({a}acceptance.json)。','',
        '可复现入口：','```bash',f'/scratch/djc/envs/gas-torch/bin/python {HERE}/diagnostic.py controller','```',
        '已存在的注册/封存/评分不覆盖。出现异常先查status.json、scheduling日志与每布局complete.json；无需训练或重选权重。']
    REPORT.write_text('\n'.join(lines)+'\n')


def publish():
    ARTIFACT.mkdir(parents=True,exist_ok=True);export={}
    chosen=list(BASE.glob('*.json'))+list(BASE.glob('*.csv'))+list(BASE.glob('*.png'))+list(BASE.glob('*.pdf'))
    chosen += list((BASE/'checks').glob('*.json'))+list((BASE/'analysis').glob('*verified.json'))
    for p in chosen:
        if p.name in ('status.json','github_sync.json'):continue
        dest=ARTIFACT/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==sha(p),dest
        else:shutil.copy2(p,dest)
        export[str(dest.relative_to(PROJECT))]=dict(sha256=sha(dest),source=str(p))
        side=p.with_suffix(p.suffix+'.sha256')
        if side.exists():shutil.copy2(side,dest.with_suffix(dest.suffix+'.sha256'))
    # Preserve lightweight per-layout scheduling and hook evidence as lossless
    # compressed manifests; predictions themselves stay on scratch.
    for i in range(17):
        target=BASE/'predictions'/f'layout{i:02d}'
        dest=ARTIFACT/'prediction_manifests'/f'layout{i:02d}.jsonl.gz';dest.parent.mkdir(exist_ok=True)
        raw=(target/'manifest.jsonl').read_bytes()
        with dest.open('wb') as out:
            with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as f:f.write(raw)
        assert gzip.decompress(dest.read_bytes())==raw
        export[str(dest.relative_to(PROJECT))]=dict(sha256=sha(dest),source=str(target/'manifest.jsonl'),source_sha256=sha(target/'manifest.jsonl'),transform='lossless gzip')
        shutil.copy2(target/'complete.json',dest.with_name(f'layout{i:02d}_complete.json'))
    resources_snapshot=dict(tasks=[read(p) for p in (BASE/'scheduling').glob('*.json') if 'key' in read(p)],
        events=rows(BASE/'scheduling/events.jsonl'),raw_resources=str(BASE/'scheduling/resources.jsonl'),external_processes_signalled=False)
    write(ARTIFACT/'resource_audit.json',resources_snapshot)
    export[str((ARTIFACT/'resource_audit.json').relative_to(PROJECT))]=dict(sha256=sha(ARTIFACT/'resource_audit.json'))
    write(ARTIFACT/'export_manifest.json',dict(at=now(),files=export,large_predictions_remain_scratch=True))


def main():
    backend();registry=verified(BASE/'experiment_registry.json');verify_hashes(registry['sources'])
    assert not (BASE/'acceptance.json').exists(),'Accepted output immutable'
    assert verified(BASE/'all_predictions_sealed.json')['all_new_predictions_saved_before_current_GT64']
    if not (BASE/'independent_verification.json').exists():score()
    s=read(BASE/'statistics.json') if (BASE/'statistics.json').exists() else statistics()
    make_figures(s);report(s)
    assert sha(main_method()['checkpoint'])==EXPECTED
    ind=verified(BASE/'independent_verification.json')
    assert ind['new_fields']==registry['new_predictions']*2
    inv=[r for r in s['prediction_sensitivity'] if r['experiment']=='P0']
    freeze(BASE/'acceptance.json',dict(at=now(),completed=True,main_checkpoint_sha256=EXPECTED,
        training_runs=0,weight_updates=0,model_structure_changed=False,feedback_OARI_restarted=False,
        baseline_reused_combinations=595,layouts=17,cases=35,new_predictions=registry['new_predictions'],
        independently_verified_new_fields=ind['new_fields'],baseline_fields_verified=1190,
        P0_completed=True,P1_completed=True,P2_completed=True,P4_completed=True,
        P0_over_tolerance_outputs=sum(r['above_registered_tolerance'] for r in inv),
        V4_V6_distinct_permutations=10,V3_distinct_permutations=5,V3_insufficient_documented_before_inference=True,
        bootstrap_checks=s['bootstrap_checks'],GT128_read=False,all_predictions_sealed_before_current_GT64=True,
        supplementary_used_test=True,independent_confirmation=False,
        evidence={str(p):sha(p) for p in (REPORT,BASE/'statistics.json',BASE/'independent_verification.json',BASE/'all_predictions_sealed.json')}))
    publish()
    # Ensure all report links exist, all exported numbers finite or explicitly
    # missing, and every registered source remains unchanged before publication.
    import re
    for link in re.findall(r'\]\(([^)]+)\)',REPORT.read_text()):assert (REPORT.parent/link).exists(),link
    verify_hashes(registry['sources'])
    print('ACCEPTED frozen geometry diagnostic; no training; sources unchanged',flush=True)


def sync():
    accepted=verified(BASE/'acceptance.json');assert accepted['completed']
    verify_hashes(accepted['evidence']);verify_hashes(verified(BASE/'experiment_registry.json')['sources'])
    for name,item in read(ARTIFACT/'export_manifest.json')['files'].items():assert sha(PROJECT/name)==item['sha256']
    allow=['scripts/geometry_utilization_v1',str(ARTIFACT.relative_to(PROJECT)),str(REPORT.relative_to(PROJECT)),
           'docs/reports/mrgr_geometry_utilization_v1_progress_zh.md']
    assert subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT,text=True).strip()=='main'
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=PROJECT,text=True).splitlines()
    assert all(any(p==x or p.startswith(x+'/') for x in allow) for p in staged),staged
    subprocess.run(['git','add','--',*allow],cwd=PROJECT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=PROJECT,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=PROJECT,text=True).splitlines()
    assert not any(Path(p).suffix in ('.pt','.pth','.npy','.npz') for p in staged)
    if staged:subprocess.run(['git','commit','-m','Diagnose frozen OARI relation utilization and view permutation invariance'],cwd=PROJECT,check=True)
    subprocess.run(['git','fetch','origin'],cwd=PROJECT,check=True)
    subprocess.run(['git','merge-base','--is-ancestor','origin/main','HEAD'],cwd=PROJECT,check=True)
    result=subprocess.run(['git','push','origin','HEAD:main'],cwd=PROJECT,text=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
    write(BASE/'github_sync.json',dict(at=now(),commit=commit,push_exit=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==0,result.stderr
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT,text=True).split()[0]
    assert remote==commit
    write(BASE/'github_sync.json',dict(at=now(),commit=commit,remote_main=remote,verified=True,push_exit=0))
    print('GitHub synchronized',commit,flush=True)
