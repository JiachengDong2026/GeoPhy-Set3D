"""Paper tables/figures and honest live Chinese status, without test-time selection."""
from common import *
import argparse
import csv
import numpy as np

REPORT=ROOT/'docs/reports/mrgr_paper_final_experiments_zh.md'

def table(summary):
    lines=['| 布局（逐例聚合） | 方法 | 阶段 | 三维L1 | 三维L2 | L1中位数 | L1 P90 | L1最大 |','|---|---|---|---:|---:|---:|---:|---:|']
    for p in summary['panels']:
        if p['stage']!='C1' or len(p['draws']) not in ([1] if p['condition'] in ('Val-A-V4','spread-V4') else [5]):continue
        m=p['metrics'];a=m['volume.relative_l1'];b=m['volume.relative_l2']
        lines.append(f"| {p['condition']} | {p['method']} | {p['stage']} | {a['mean']:.4%} | {b['mean']:.4%} | {a['median']:.4%} | {a['p90']:.4%} | {a['max']:.4%} |")
    return lines

def secondary_tables(summary):
    selected=[p for p in summary['panels'] if p['stage']=='C1' and ((p['condition']=='Val-A-V4' and p['draws']==[0]) or (p['condition']=='clustered-V4' and len(p['draws'])==5))]
    lines=['','### 次要形态与峰值（探索性）','', '| 布局 | 方法 | 质心mm | 峰比 | 峰比绝对误差 | GT峰处比 | Dice10 | Dice50 | 背景占比 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def f(v):return '未定义' if v is None else f'{v:.4f}'
    for p in selected:
        m=p['metrics'];values=[m['morphology.'+k]['mean'] for k in ('centroid_distance_mm','peak_ratio','absolute_peak_ratio_error','gt_peak_location_ratio','dice_0.1','dice_0.5','background_predicted_fraction')]
        lines.append('| '+p['condition']+' | '+p['method']+' | '+' | '.join(f(v) for v in values)+' |')
    lines+=['','### 输入/额外H4投影与成本','', '| 布局 | 方法 | 输入L1 | 输入L2 | H4 L1 | H4 L2 | 重建秒中位数 | 峰Torch显存MiB（最大） |','|---|---|---:|---:|---:|---:|---:|---:|']
    for p in selected:
        m=p['metrics'];values=[m['measurement.'+label+'.'+k]['mean'] for label in ('observed','heldout') for k in ('relative_l1','relative_l2')]
        peak=m['peak_allocated_bytes']['max'];seconds=m['reconstruct_seconds']['median']
        lines.append('| '+p['condition']+' | '+p['method']+' | '+' | '.join('未记录' if v is None else f'{v:.4%}' for v in values)+f' | {seconds:.5f} | '+('历史未记录' if peak is None else f'{peak/2**20:.1f}')+' |')
    lines+=['','五draw面板的计时/显存先按案例平均再汇总；逐次真实最大显存和冷启动成本请见逐例manifest。历史神经模型开发计时与新经典计时边界不同，不作严格速度排名；最终测试使用同一新入口重新计时。','',
        '### 聚簇五draw分别报告（C1三维L1）','', '| 方法 | draw0 | draw1 | draw2 | draw3 | draw4 |','|---|---:|---:|---:|---:|---:|']
    for method in sorted({p['method'] for p in summary['panels']}):
        ps=[next(p for p in summary['panels'] if p['method']==method and p['stage']=='C1' and p['condition']=='clustered-V4' and p['draws']==[d]) for d in range(5)]
        lines.append('| '+method+' | '+' | '.join(f'{p["metrics"]["volume.relative_l1"]["mean"]:.4%}' for p in ps)+' |')
    return lines

def export(base):
    rr=rows(base/'metrics.jsonl');summary=read(base/'summary.json')
    import evaluation as ev
    flat=[]
    for r in rr:flat.append(dict(method=r['method'],stage=r['stage'],case_id=r['case_id'],condition=r['condition'],draw=r['draw'],**ev.scalar_metrics(r)))
    with (base/'per_case_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in flat))));writer.writeheader();writer.writerows(flat)
    with (base/'paired_comparisons.csv').open('w') as f:
        keys=['condition','metric','reference','primary','mean_MRGR','mean_reference','delta','ci95','simultaneous_ci','wins','ties','losses'];writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(summary['paired'])
    failures=[]
    for method in sorted({r['method'] for r in rr}):
        for condition in ('Val-A-V4','clustered-V4'):
            group=[r for r in rr if r['method']==method and r['stage']=='C1' and r['condition']==condition and r['draw']==0]
            for rank,r in enumerate(sorted(group,key=lambda r:r['volume']['relative_l1'],reverse=True)[:3],1):
                failures.append(dict(method=method,condition=condition,rank=rank,case_id=r['case_id'],L1=r['volume']['relative_l1'],L2=r['volume']['relative_l2'],GT_peak_location_ratio=r['morphology']['gt_peak_location_ratio'],centroid_mm=r['morphology']['centroid_distance_mm'],Dice50=r['morphology']['dice_0.5'],solver_status=r.get('solver_status')))
    with (base/'worst_cases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(failures[0]));w.writeheader();w.writerows(failures)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[p for p in summary['panels'] if p['stage']=='C1' and p['condition']=='Val-A-V4']
    fig,ax=plt.subplots(figsize=(max(8,len(panels)*.9),4.5),layout='constrained');x=np.arange(len(panels))
    ax.bar(x,[p['metrics']['volume.relative_l1']['mean']*100 for p in panels]);ax.set_xticks(x,[p['method'] for p in panels],rotation=35,ha='right');ax.set(ylabel='3D relative L1 (%)',title=f'{summary["split"]}: ordinary fixed V4, case means');ax.grid(axis='y',alpha=.2);fig.savefig(base/'ordinary_l1.png',dpi=160);plt.close(fig)
    # Fixed indices, common GT-peak z slice and colorbar per case; never best-case selection.
    split=summary['split'];cases=verified(OUT/'registration.json')['cases'][split];indices=[0,len(cases)//2,len(cases)-1];source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')}
    seal=verified(base/'all_predictions_sealed.json');methods=seal['methods'];manifest_lookup={}
    for mp in seal['manifests']:
        target=Path(mp).parent
        for r in rows(mp):manifest_lookup[target.name if 'classical' not in str(target) else target.parent.name,r['condition'],r['draw'],r['case_id']]=(target/r['prediction'],r['sha256'])
    # Full method table stays available; figures have legible columns grouped by method.
    groups=[methods[i:i+4] for i in range(0,len(methods),4)]
    for group_index,group in enumerate(groups):
        fig,axes=plt.subplots(6,1+len(group),figsize=(3.2*(1+len(group)),16),layout='constrained',squeeze=False)
        with AccessGuard(gt_splits=[split]) as guard:
            for ci,index in enumerate(indices):
                case=cases[index];gt=np.load(DATA/source[case]['gt_volume_64_path'],allow_pickle=False).squeeze();iz=np.unravel_index(gt.argmax(),gt.shape)[2]
                for j,condition in enumerate(('Val-A-V4','clustered-V4')):
                    fields=[('GT',gt)]
                    for m in group:
                        path,h=manifest_lookup[m['key'],condition,0,case];assert sha(path)==h
                        with np.load(path,allow_pickle=False) as z:fields.append((m['key'],z['C1'].copy()))
                    for col,(name,field) in enumerate(fields):
                        im=axes[2*ci+j,col].imshow(field[:,:,iz].T,origin='lower',cmap='magma',vmin=0,vmax=float(gt.max()),extent=(0,.6,0,.6),aspect='equal')
                        axes[2*ci+j,col].set(title=f'{name}\n{condition} case-index={index}',xlabel='x (m)',ylabel='y (m)')
                    fig.colorbar(im,ax=axes[2*ci+j,:].tolist(),shrink=.75,label='CH4 mass fraction')
        fig.savefig(base/f'fixed_reconstructions_{group_index}.png',dpi=140);plt.close(fig)
    write(base/'figure_audit.json',dict(at=now(),case_indices=indices,method_order=[m['key'] for m in methods],conditions=['Val-A-V4','clustered-V4 draw0'],slice='GT argmax z; same linear scale [0, GT max] within each case',selection_by_results=False))

def progress():
    manifest=verified(OUT/'model_manifest.json');stage=read(OUT/'pipeline_status.json') if (OUT/'pipeline_status.json').exists() else dict(state='development_in_progress')
    lines=['# GeoPhy-Set3D：论文最终实验汇总','',f'更新时间：{now()}（北京时间）。当前状态：`{stage["state"]}`。', '',
        '**整项任务尚未验收完成。** 只有全部正式训练、全方法预测封存、独立最终评分与复核均完成后才标记完成。','',
        '## 已完成的审计、冻结与正确性检查','',
        '本地main，起始提交55b9a1fa05e45230928cb9c02640923d0f76b906；历史跟踪源码未修改，既有产物保留；本任务未提交或推送。远程代理失效，无法核实远程HEAD，不能宣称与远程一致。', '',
        '数据清单核实为166训练、36开发验证、35保留测试。已逐项复现历史17布局的36例输入及H4顺序。主模型保持原第278轮，LR4E4保持独立第291轮，未用新结果替换主模型。', '',
        '| 模型 | epoch | 参数 | SHA256 |','|---|---:|---:|---|']
    for m in manifest['models']:lines.append(f'| {m["key"]} | {m["best_epoch"]} | {m["parameters"]} | `{m["checkpoint_sha256"]}` |')
    lines+=['','[执行计划与风险](mrgr_paper_final_execution_plan_v1_zh.md)；[统一测试协议](../mrgr_final_evaluation_protocol_v1_zh.md)；[模型冻结清单](../../outputs/research/mrgr_paper_final_v1/model_manifest.json)；[机器注册](../../outputs/research/mrgr_paper_final_v1/registration.json)。','',
        '真实物理前向对质量分数先插值再做非线性摩尔分数转换。同算子经典比较为NNLS-NL、Tikhonov及平滑各向同性TV；按用户要求在最终测试前追加节点摩尔分数SART，明确其离散算子差异，所有投影评价统一使用原始P(C)。[SART补充协议与进度](mrgr_paper_final_sart_supplement_v1_zh.md)。', '',
        '数值检查覆盖输入形状、有限值、非负范围、真实前向及梯度、正则项离散、更新、停止、CPU精确迭代恢复、GPU跨进程训练恢复、GT隔离和封存损坏拒绝。CUDA grid_sample反向存在原子累加非确定性，恢复比较采用rtol1e-4/atol1e-6，RNG/计数/采样身份精确。首次数值测试的严格GPU梯度等值断言失败、日志变量遮蔽错误均保留；修复后通过，不把程序退出码作为唯一证据。', '',
        '经典恢复检查另发现“已收敛checkpoint重启多走一步”的边界错误。已归档最初源码并追加不可变implementation_amendment_v2登记，修复仅影响恢复入口；72次不间断校准不受影响，参数候选与选择规则不变，未接触最终测试。', '',
        '## 经典算法：仅开发阶段','']
    selected=OUT/'classical/selected.json'
    if selected.exists():
        s=verified(selected);lines+=['72次有限校准先整批预测封存再读取6个训练案例GT选择配置；每算法两候选、两个布局，无测试调参。','', '| 方法 | 选中候选 | 训练校准L1 | 固定预算 |','|---|---:|---:|---|']
        for m,v in s['selected'].items():lines.append(f'| {m} | {v["candidate"]} | {v["mean_l1"]:.4%} | `{json.dumps(v["config"],ensure_ascii=False)}` |')
    classical=OUT/'evaluation/classical_dev'
    if (classical/'summary.json').exists():
        s=read(classical/'summary.json');lines+=['','以下为36例开发验证，五draw在每案例内先平均，不能称最终测试。','']+table(s)
        lines+=secondary_tables(s)
        lines+=['',f'记录的求解失败数：{len(s["failures"])}；预算耗尽单列为有限计算结果，不等价达到最优。', '',
            '[全部开发指标/配对区间](../../outputs/research/mrgr_paper_final_v1/evaluation/classical_dev/summary.json)；[逐例CSV](../../outputs/research/mrgr_paper_final_v1/evaluation/classical_dev/per_case_metrics.csv)。']
    else:
        for method in ('NNLS-NL','Tikhonov','TV'):
            path=OUT/'classical/val'/method/'frozen/manifest.jsonl';n=len(rows(path)) if path.exists() else 0;lines.append(f'- {method}：开发预测 {n}/612。')
    sart_status=OUT/'sart/pipeline_status.json'
    if sart_status.exists():
        ss=read(sart_status);p=OUT/'classical/val/SART/frozen/manifest.jsonl';n=len(rows(p)) if p.exists() else 0
        lines+=['',f'SART补充开发状态：`{ss["state"]}`，验证预测{n}/612。CPU稀疏计算不占用正在训练的GPU；CPU峰RSS与GPU显存分别记录，不直接比较设备速度。']
        if (OUT/'evaluation/sart_dev/summary.json').exists():lines+=['']+table(read(OUT/'evaluation/sart_dev/summary.json'))
    lines+=['','## 独立训练消融','',
        '单尺度与完整多尺度参数完全匹配（344678），所有上下文卷积留在32³，因此FLOPs和有效感受野仍改变，不能归因于唯一抽象概念。无反馈模型实际删除分支（332166参数），容量同步下降必须说明；其C0=C1是该模型输出接口定义，不能冒称两种独立模型。一致性置零对照仍独立训练，其余损失保持一致。20261011/20261019两种子配对；第二种子完整多尺度作为参考，首种子复用原主模型。', '',
        '| 变体 | seed | 轮次 | 最优验证H | 状态 |','|---|---:|---:|---:|---|']
    from module_registry_v2 import ablation_arms
    for arm in (*ablation_arms(),'multiscale'):
        for seed in (20261011,20261019):
            if arm=='multiscale' and seed==20261011:continue
            p=OUT/'training'/f'{arm}_seed{seed}'/'status.json'
            if p.exists():s=read(p);lines.append(f'| {arm} | {seed} | {s.get("epoch",s.get("epochs"))}/300 | {s.get("best_val_H",float("nan")):.6f} | {s["state"]} |')
            else:lines.append(f'| {arm} | {seed} | 0/300 | — | 待调度 |')
    extension=OUT/'module_ablations_v2/registration.json'
    if extension.exists():
        lines+=['','[OARI、GCA与同容量反馈消融追加协议](mrgr_paper_final_module_ablations_v2_zh.md)：追加4组×2种子×300轮，正式测试扩为21方法、20个主要终点对照、12,495次重建与22,610个阶段场。全部新增训练及开发复核通过后才关闭最终方法清单。']
        statefile=OUT/'module_ablations_v2/status.json'
        if statefile.exists():lines += ['',f'追加控制器状态：`{read(statefile)["state"]}`。']
    costs=OUT/'architecture_costs.json'
    if costs.exists():
        lines+=['','固定V4实际调用的卷积/线性MACs（仅部分算术成本，不含提升/插值/注意力矩阵乘法/投影等，不称总FLOPs）：','', '| 变体 | 参数 | Conv/Linear MACs |','|---|---:|---:|']
        for row in read(costs)['records']:lines.append(f'| {row["arm"]} | {row["parameters"]} | {row["conv_linear_MACs_fixed_V4"]:,} |')
        lines+=['','既有consistency=0.01短筛查为负，不复跑；本次置零且完整300轮回答一致性项是否有独立贡献，不用于再次选优主模型。']
    val=OUT/'evaluation/val'
    if (val/'summary.json').exists():lines+=['','### 完整开发评价（含消融）','']+table(read(val/'summary.json'))
    for label,path in [('开发',val/'ablation_paired.json'),('最终测试',OUT/'evaluation/test/ablation_paired.json')]:
        if path.exists():
            lines+=['',f'### {label}：同训练种子的消融配对','', '| 消融 | seed | 布局 | 消融−完整L1（百分点） | 案例95%CI（百分点） |','|---|---|---|---:|---|']
            for row in read(path)['results']:
                if row['metric']=='relative_l1' and row['condition'] in ('Val-A-V4','clustered-V4'):
                    ci=np.array(row['ci95_case'])*100;lines.append(f'| {row["arm"]} | {row["seed"]} | {row["condition"]} | {row["delta_ablation_minus_full"]*100:.4f} | [{ci[0]:.4f},{ci[1]:.4f}] |')
            lines+=['','正差值表示消融误差更大；两种子均值仅为描述性结果，区间仍只反映案例变化，不能充当训练种子置信区间。']
    lines+=['','## 35例最终测试','']
    final=OUT/'evaluation/test'
    if (final/'summary.json').exists() and (final/'independent_verification.json').exists():
        s=read(final/'summary.json');audit=read(final/'gt_access.json');lines+=['以下为最终锁定后的35例测试；所有方法595次预测已完整封存后，独立评分进程才读取GT64。GT128始终禁止。','']+table(s)
        lines+=secondary_tables(s)
        lines+=['','### 唯一主要终点的配对比较','', '| 对照 | MRGR−对照L1（百分点） | 95%CI（百分点） | Bonferroni区间（百分点） | 胜/平/负 |','|---|---:|---|---|---|']
        for p in s['paired']:
            if p['primary']:
                ci=np.array(p['ci95'])*100;adj=np.array(p['simultaneous_ci'])*100;lines.append(f'| {p["reference"]} | {p["delta"]*100:.4f} | [{ci[0]:.4f},{ci[1]:.4f}] | [{adj[0]:.4f},{adj[1]:.4f}] | {p["wins"]}/{p["ties"]}/{p["losses"]} |')
        lines+=['',f'GT64首次评分读取时间：{audit["first_GT64_at"]}；全方法封存时间：{audit["seal_at"]}。', '',
            '[最终逐例CSV](../../outputs/research/mrgr_paper_final_v1/evaluation/test/per_case_metrics.csv)；[全部统计](../../outputs/research/mrgr_paper_final_v1/evaluation/test/summary.json)；[独立复核](../../outputs/research/mrgr_paper_final_v1/evaluation/test/independent_verification.json)。']
    else:lines+=['尚未执行最终GT评分。不得从当前开发结果推断35例测试表现；最终权重、所有算法与消融及评价代码全部冻结后，才启动正式预测与封存。']
    lines+=['','## 解释边界与未完成事项','',
        '三维定位、峰值恢复、投影误差与整体L1可能不同步，需保留所有负结果；平滑TV采用有依据的可微近似，不能称非平滑TV的精确最优解。经典算法固定200步预算和零初始化，稀疏布局欠定，其表现同时受先验与预算影响。两种子不足以精确估计训练随机性，案例bootstrap仅描述案例层面变化。未补充噪声/标定误差、全新几何域或额外训练数据。', '',
        '所有GPU均按实时空闲状态以UUID绑定；GPU0的其他用户任务不干扰。每任务设备、开始/结束、退出码和资源快照位于`scheduling/`，每例时间/峰值显存在manifest，训练逐轮成本在metrics。恢复和软件错误记录保留，不覆盖历史成果。', '',
        f'当前流水线状态：`{stage["state"]}`；未完成项目以训练表与预测/复核文件实际存在性为准，不将排队或运行中写成完成。']
    if stage['state']=='complete':
        lines[4]='**整项任务已完成自动复核，详细验收见final_acceptance.json。**'
    from costs import main as cost_ledger
    cost_ledger()
    lines+=['','## 固定案例图、误差图与失败案例索引','']
    for scope in ('classical_dev','sart_dev','val','test'):
        base=OUT/'evaluation'/scope
        if (base/'ordinary_l1.png').exists():
            prefix='../../outputs/research/mrgr_paper_final_v1/evaluation/'+scope+'/'
            lines+=[f'![{scope}普通V4误差]({prefix}ordinary_l1.png)','',f'[{scope}各方法最高误差案例（用于失败分析，不用于选模型）]({prefix}worst_cases.csv)。','']
            for image_path in sorted(base.glob('fixed_reconstructions_*.png')):lines+=[f'![{scope}预定固定案例]({prefix}{image_path.name})','']
    lines+=['固定图像使用预先登记的清单索引和GT峰切面；相同案例各方法共享线性色标[0,GT峰值]，超出范围会饱和，须结合峰幅数字阅读。切面不能证明三维拓扑或模块因果。']
    if (OUT/'evaluation/classical_slices_v2/figure_audit.json').exists():
        lines+=['','### 原始MRGR与四种经典算法的清晰切面对比','',
            '[统一线性/asinh/独立形状归一化三种显示及高清PDF](mrgr_paper_final_classical_slices_v2_zh.md)。以下两图对每案例的全部方法采用同一asinh色标，颜色仍表示实际质量分数；峰值比另标，不把形状归一化当作幅值恢复。','']
        for condition in ('Val-A-V4','clustered-V4'):
            lines += [f'![{condition}六方法清晰切面对比](../../outputs/research/mrgr_paper_final_v1/evaluation/classical_slices_v2/{condition}_shared_asinh_overview.png)','']
    lines+=['','[统一任务成本与退出码表](../../outputs/research/mrgr_paper_final_v1/task_costs.csv)。']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(REPORT,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['progress','export']);p.add_argument('--base');a=p.parse_args();progress() if a.action=='progress' else export(Path(a.base))
