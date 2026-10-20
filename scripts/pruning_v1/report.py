"""Chinese status/final report and fixed development slices; never reads test."""
from runtime import *
import shutil
import numpy as np

LABELS={'multiscale':'A 完整 MRGR','uniform_fusion':'B 均匀融合','pruned_observability':'C 删除描述符','uniform_pruned_observability':'D 双重简化','no_observability':'零输入诊断'}

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    methods=verified(BASE/'reuse_methods.json')+verified(BASE/'methods.json');lookup={};allowed={}
    for m in methods:
        if m['arm']=='no_observability':continue
        root=Path(m.get('prediction_root',BASE/'evaluation/val'/m['key']))
        manifest=Path(m.get('manifest',root/'manifest.jsonl'))
        for r in rows(manifest):
            lookup[m['arm'],m['seed'],r['condition'],r['draw'],r['case_id']]=root/r['prediction']
            allowed[str((root/r['prediction']).resolve())]='sealed_development_prediction'
    source={r['case_id']:r for r in rows(DATA/'source_index.jsonl') if r['split']=='val'}
    cases=verified(BASE/'registration.json')['cases']['val'];files=[]
    with DevGuard(gt_splits=['val']) as guard:
        guard.allowed.update(allowed)
        for seed in SEEDS:
            for index in (0,18,35):
                case=cases[index];g=np.load(DATA/source[case]['gt_volume_64_path'],allow_pickle=False).squeeze();y=np.unravel_index(g.argmax(),g.shape)[1]
                fig,axes=plt.subplots(5,2,figsize=(9,17),constrained_layout=True);norm=AsinhNorm(linear_width=float(g.max())*.02,vmin=0,vmax=float(g.max()))
                for col,condition in enumerate(('Val-A-V4','clustered-V4')):
                    fields=[('GT',g)]
                    for arm in ('multiscale','uniform_fusion',*ARMS):
                        with np.load(lookup[arm,seed,condition,0,case],allow_pickle=False) as z:fields.append((LABELS[arm].split()[0],z['C1']))
                    for row,(name,a) in enumerate(fields):
                        image=axes[row,col].imshow(a[:,y,:].T,origin='lower',extent=[0,600,0,600],cmap='magma',norm=norm)
                        axes[row,col].set_title(f'{name} | {condition} draw0 | peak/GT={a.max()/g.max():.3f}')
                        axes[row,col].set_xlabel('x (mm)');axes[row,col].set_ylabel('z (mm)')
                fig.colorbar(image,ax=axes,label='Mass fraction; shared asinh scale; saturation above GT peak',shrink=.7)
                fig.suptitle(f'Development case index {index}, seed {seed}; fixed GT peak y={y}; {case}',fontsize=9)
                path=BASE/'analysis'/f'fixed_case_{index}_seed{seed}.png';fig.savefig(path,dpi=180);fig.savefig(path.with_suffix('.pdf'));plt.close(fig);files.append(path)
        assert not guard.denied
    write(BASE/'analysis/figure_audit.json',dict(at=now(),audit=guard.summary(),files={str(p):sha(p) for p in files},selection='predeclared indices0/18/35; clustered draw0 only for slices; numerical aggregate uses all5 draws'))
    return files

def main():
    design=verified(BASE/'design.json');audit=verified(BASE/'reuse_audit.json') if (BASE/'reuse_audit.json').exists() else None
    summary=read(BASE/'analysis/summary.json') if (BASE/'analysis/summary.json').exists() else None
    complete=summary is not None and (BASE/'analysis/independent_verification.json').exists() and (BASE/'benchmark.json').exists()
    lines=['# GCA 均匀融合替换与可观测性描述符结构删除：开发实验 v1','',f'更新：{now()}。独立 worktree：`{PROJECT}`；分支 `experiment/gca-uniform-observability-pruning-v1`；基线 `{design["git_head"]}`。',
        '', '**范围：166 训练案例与 36 个已用开发案例；全部比较为探索性结果。原始多尺度主模型及既有 35 例正式测试记录保持冻结。本轮不读取 test 投影、test GT64 或任何 GT128。**',
        '', '## 预登记与固定协议','',
        '检验有效视角均匀融合能否替代 GCA，以及局部方向多样性五维描述符能否真正删除；不预设简化有益。A/B 复用已完成产物，C/D 各两个固定种子完整 300 轮，不能按中期结果提前结束。',
        '', '固定多尺度编码器、OARI 射线关系和集合交互、掩码、两阶段反馈、P(C)、采样、数据划分及 17 布局；FP32、TF32 关闭、batch2、AdamW 2e-4/wd1e-4、cosine300、梯度裁剪1、4个持久 fork worker。每任务24,900更新。只按验证 H4 相对L1严格最小值选轮。',
        '', '| 组 | 融合 | 描述符 | 参数 | 训练 |','|---|---|---|---:|---|',
        '| A | GCA | 保留 | 344678 | 复用两种子 |','| B | 有效视角平均 | 保留 | 342725 | 复用两种子 |',
        '| C | GCA | 结构删除 | 344294 | 两种子各300轮 |','| D | 有效视角平均 | 结构删除 | 342341 | 两种子各300轮 |',
        '', '## 删除边界与初始化','',
        '新增 `pruned_models.py`：删除 `observability` 和 `interaction.observability_embedding`；token_projection 输入56→48（删除256权重），pairwise_bias首层输入8→3（删除80权重），嵌入层删除48参数，总计删除384参数。删除方向张量3×3、特征值分解、五维描述符、有效视角分数与8维嵌入拼接；方向向量仍用于余弦绝对值、正弦和归一化方位差三种射线关系。',
        '', '同种子先构造完整模型，再截取token前48列及偏置首层前三列。保留全部偏置及其他同名张量，逐元素核对；不把零描述符嵌入偏置折叠进投影。因此 C 与“描述符输入置零”是不同结构。新建线性层时恢复CPU RNG，避免改变其他初始化流。',
        '', 'PrunedMRGR 通过 MRO 只替换 stage0；原 MRGR.forward 函数原样继承，C0→原P(C0)→有符号残差→共享融合/解码→C1 不变。均匀融合在C0/C1共享使用，有效特征求和除有效数（下界1），全无效输出与权重精确为零，删除1953个GCA参数。',
        '', f'[初始化完整对应清单]({BASE}/initialization_manifest.json)；[预登记与源码SHA256]({BASE}/design.json)。',
        '', '## 既有产物复用核查','']
    if audit:
        lines+=['| 模型 | 轮/更新 | 最佳轮 | 验证H4 L1 | 开发预测 | 初始化/源码/检查点/预测哈希 |','|---|---|---:|---:|---:|---|']
        for r in audit['runs']:lines.append(f'| {r["key"]} | 300/24900 | {r["best_epoch"]} | {100*r["best_val_H"]:.4f}% | {r["predictions"]} | 通过 |')
        lines+=['','六个既有模型均复用，不重复训练。两组no_observability保留参数和描述符计算，仅输入置零，作为诊断对照，不能证明真正删除无损。']
    else:lines+=['核查待完成；无产物被默认为有效。']
    lines+=['',f'[完整复用证据]({BASE}/reuse_audit.json)。','', '## 验收与队列','',
        '训练前检查：均匀融合重排/NaN无效视角/全无效/梯度；真实64³形状与范围；与全宽零通道代数参照逐张量对照；原前向与有符号残差一致；禁止调用eigvalsh；射线关系仍有效；完整自监督损失两步反传；每种新结构独立进程2轮不中断与1+1续训对照。',
        '', '恢复验收涵盖模型、最佳模型、AdamW、scheduler、Python/NumPy/CPU/CUDA/loader RNG、epoch采样与训练计数；整数/RNG精确相等，浮点rtol1e-4/atol1e-6（CUDA grid_sample backward原子累加非确定性）。失败日志保留并阻断正式任务。',
        '', f'验收：{"通过" if (BASE/"checks/passed.json").exists() else "尚未全部通过，正式训练被门控"}。[验收记录]({BASE}/checks/passed.json)；[失败证据]({BASE}/checks/failures.jsonl)。',
        '', '调度每15秒重新查询UUID、显存、利用率、GPU进程及所属用户、主机内存与磁盘；连续两次无计算进程、显存占用<100MiB、空闲≥20000MiB、利用率≤5%后，启动前再查。每卡仅一项任务，以UUID绑定cuda:0；不对外部进程发信号。可用卡增加时自动并行。CPU线程受限，大数组/权重保存在scratch。',
        '', '| 新任务 | 状态 | 最佳轮 | 当前轮 | 日志 |','|---|---|---:|---:|---|']
    for run in design['runs']:
        path=Path(run);s=read(path/'status.json') if (path/'status.json').exists() else {};task=path.name
        lines.append(f'| {task} | {s.get("state","待验收/排队")} | {s.get("best_epoch","—")} | {s.get("epoch",0)}/300 | [日志]({BASE}/scheduling/{task}.log) |')
    lines+=['',f'[实时流水线状态]({BASE}/pipeline_status.json)；[实时队列]({BASE}/queue_status.json)；[GPU与进程资源轨迹]({BASE}/scheduling/resources.jsonl)；[启动/退出记录]({BASE}/scheduling/events.jsonl)。',
        '', '## 开发结果与配对统计','',
        '普通V4为Val-A-V4；聚簇V4先在同案例内平均五draw，再跨36案例汇总。下表为C1三维相对L1。17组逐布局L1/L2、各例中位/P90/最大、峰比及绝对峰比误差、GT峰处比、等权质心mm、Dice10/50和输入/H4投影误差均保存在完整表。物理体积分质心与等权节点形态质心分别保留。',
        '', '| 组 | 种子 | 普通V4 L1 | 聚簇V4五draw L1 |','|---|---:|---:|---:|']
    if summary:
        for arm in ('multiscale','uniform_fusion',*ARMS,'no_observability'):
            for seed in SEEDS:
                def panel(condition):return next(r for r in summary['panels'] if r['method']==f'{arm}_seed{seed}' and r['stage']=='C1' and r['condition']==condition and r['draws']==([0] if condition=='Val-A-V4' else list(range(5))))
                a=panel('Val-A-V4');b=panel('clustered-V4')
                lines.append(f'| {LABELS[arm]} | {seed} | {100*a["metrics"]["volume.relative_l1"]["mean"]:.4f}% | {100*b["metrics"]["volume.relative_l1"]["mean"]:.4f}% |')
        paired=rows(BASE/'analysis/paired_statistics.jsonl')
        lines+=['','| 比较 | 布局 | 两种子案例均值差(百分点) | 95%案例区间 | 低/平/高 |','|---|---|---:|---|---|']
        for r in paired:
            if r['seed']!='two_seed_case_mean' or r['metric']!='volume.relative_l1':continue
            if not (r['condition']=='Val-A-V4' or r['condition']=='clustered-V4' and len(r['draws'])==5):continue
            lo,hi=[100*x for x in r['ci95']]
            lines.append(f'| {r["comparison"]} | {r["condition"]} | {100*r["delta"]:+.4f} | [{lo:+.4f},{hi:+.4f}] | {r["negative"]}/{r["tie"]}/{r["positive"]} |')
        lines+=['','差值按后者减前者的表达式明确计算，负值表示L1下降；交互为(D−C)−(B−A)。各单种子与案例严格配对，bootstrap20000次/seed20261019。两种子均值只作描述；案例区间不覆盖训练种子总体方差，未经多重校正，不能据此证明普遍提升或等效。',
            '',f'[逐案例完整CSV]({BASE}/analysis/per_case_metrics.csv)；[逐布局与分位统计]({BASE}/analysis/summary.json)；[全部配对区间与案例差值]({BASE}/analysis/paired_statistics.csv)；[独立重算]({BASE}/analysis/independent_verification.json)。']
    else:
        for arm in ('multiscale','uniform_fusion',*ARMS):lines.append(f'| {LABELS[arm]} | 两种子 | 待统一汇总 | 待统一汇总 |')
        lines+=['','新结构尚未全部训练并评分，不作结构优劣判断。']
    if (BASE/'benchmark.json').exists():
        bench=verified(BASE/'benchmark.json');lines+=['','## 实际计算成本','',bench['boundary'],'',bench['MAC_scope'],'',
            '| 模型 | 参数 | Conv/Linear MACs | 稳态中位ms | P90 ms | 推理峰MiB |','|---|---:|---:|---:|---:|---:|']
        for r in bench['results']:lines.append(f'| {r["method"]} | {r["parameters"]} | {r["learned_conv_linear_MACs"]} | {1000*r["median_seconds"]:.3f} | {1000*r["p90_seconds"]:.3f} | {r["peak_allocated_bytes"]/2**20:.1f} |')
        lines+=['','历史开发推理计时边界不同，不用于速度排名；上表在同一空闲GPU、相同入口warmup10/重复40重新测量。','',
            '| 新任务 | 完整训练小时 | 逐轮最大峰MiB | 最佳轮 |','|---|---:|---:|---:|']
        for run in design['runs']:
            p=Path(run);s=read(p/'summary.json');rr=rows(p/'metrics.jsonl');lines.append(f'| {p.name} | {s["elapsed_seconds"]/3600:.3f} | {max(r["peak_allocated_bytes"] for r in rr)/2**20:.1f} | {s["best_epoch"]} |')
    lines+=['','## 局限与建议','']
    if complete:
        paired=rows(BASE/'analysis/paired_statistics.jsonl')
        for label,name in [('B-A','GCA替换'),('C-A','描述符结构删除'),('D-B','均匀融合基础上的描述符删除')]:
            parts=[]
            for condition,draws in [('Val-A-V4',[0]),('clustered-V4',list(range(5)))]:
                r=next(r for r in paired if r['comparison']==label and r['seed']=='two_seed_case_mean' and r['condition']==condition and r['draws']==draws and r['metric']=='volume.relative_l1')
                parts.append(f'{condition}的L1变化{100*r["delta"]:+.4f}个百分点')
            lines.append(f'{name}：'+ '；'.join(parts)+'。这些是条件于两种子的开发观察。')
            lines.append('')
        lines+=['推荐仍保留冻结A作为本轮论文主模型。B/C/D是否进入下一阶段，须结合完整峰值/定位和逐例尾部指标；可以进入独立数据上的确认实验，但本轮不支持替换正式主模型或宣称删除无显著损害。尚无预登记非劣界值，两种子与已用开发集不足以证明等效；如指标间有取舍，必须同时报告。',
            '', '完整失败率、最差五例与全部负结果见summary；可视化固定索引0/18/35，不按新模型表现挑例。聚簇图只显示draw0，不能代表五draw聚合性能。']
        for p in figures():lines+=['',f'![预登记开发切面 {p.stem}]({p})']
    else:lines+=['当前无完整C/D结果，暂不能决定GCA是否值得替换、描述符是否值得删除。四项300轮训练及后续评分、配对分析、独立复核由持久队列顺序推进；完成前不把任务登记或排队写成完成。']
    lines+=['','## 复现与文件管理','',f'解释器：`/scratch/djc/envs/gas-torch/bin/python`。工作目录：`{PROJECT}`。启动：`python scripts/pruning_v1/pipeline.py`。现有controller运行时互斥锁会阻止重复启动；失败不会自动重试或覆盖，先检查失败证据。',
        '',f'源码：`scripts/pruning_v1/`。产物根目录：`{BASE}`。权重、预测数组、全部标量与SHA保存在scratch，不提交Git。源码与小型报告留在独立worktree；通过后的小型证据在`docs/reports/artifacts/gca_pruning_v1/`。']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(lines)+'\n')
    if complete:
        evidence=REPORT.parent/'artifacts/gca_pruning_v1';evidence.mkdir(parents=True,exist_ok=True)
        for name in ('design.json','checks/passed.json','benchmark.json','analysis/independent_verification.json'):
            data=read(BASE/name)
            if name=='design.json':data={k:v for k,v in data.items() if k!='sources'}
            if name=='analysis/independent_verification.json':data={k:v for k,v in data.items() if k!='audit'}
            write(evidence/Path(name).name,data)
        write(evidence/'comparison.json',dict(panels=[p for p in summary['panels'] if p['stage']=='C1'],statistics=summary['statistics']))
    print(REPORT,flush=True)

if __name__=='__main__':main()
