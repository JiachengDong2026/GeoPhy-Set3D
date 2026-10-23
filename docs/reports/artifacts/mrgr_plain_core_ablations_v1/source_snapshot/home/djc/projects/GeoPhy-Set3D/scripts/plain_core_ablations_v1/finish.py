"""Chinese final report, live experiment tracker, lightweight audit export."""
from common_stage import *
import shutil
import gzip


def create_report():
    stats=verified(BASE/'statistics.json')['summaries'];bench=verified(BASE/'benchmark.json');models=verified(BASE/'model_manifest.json')
    access=verified(BASE/'GT64_access.json');check=verified(BASE/'independent_verification.json');audit=verified(BASE/'audit.json')
    def find(arm,seed,condition,stage='C1',metric='relative_l1'):
        return next(r for r in stats if r['model']==f'{arm}_seed{seed}' and r['condition']==condition and r['stage']==stage and r['metric']==metric and r['draws']==([0] if condition=='Val-A-V4' else list(range(5))))
    lines=['# 普通集合交互最终模型：双种子核心模块探索性消融正式报告','',f'生成时间：{now()}。','',
        '六项消融各完成300轮、24900已提交更新；只复用两份普通交互完整基线，未重训基线、OARI或其他历史消融。35案例此前已用于选择与探索，本结果不是独立测试确认。两个训练种子分别配对，案例区间不能解释为训练种子总体区间。','',
        '## 模块贡献与一致性','',
        '| 模块 | 种子 | 布局 | 完整C1 L1(%) | 消融C1 L1(%) | Δ均值(pp) | Δ中位数(pp) | 案例95%区间(pp) | 增/降/不变 |','|---|---:|---|---:|---:|---:|---:|---|---|']
    labels={GEOMETRY:'GeometryLifting',SINGLE:'多尺度MRGR',NO_PRF:'PRF'}
    for arm in labels:
        for seed in (20261011,20261019):
            for condition in ('Val-A-V4','clustered-V4'):
                r=find(arm,seed,condition)
                lines.append(f"| {labels[arm]} | {seed} | {condition} | {100*r['baseline_mean']:.4f} | {100*r['treatment_mean']:.4f} | {100*r['mean']:+.4f} | {100*r['median']:+.4f} | [{100*r['ci95'][0]:+.4f}, {100*r['ci95'][1]:+.4f}] | {r['increased']}/{r['decreased']}/{r['unchanged']} |")
    lines.append('')
    limited=[]
    for arm,label in labels.items():
        rs=[find(arm,s,c) for s in (20261011,20261019) for c in ('Val-A-V4','clustered-V4')]
        positive=sum(r['mean']>1e-8 for r in rs);clear=sum(r['ci95'][0]>0 for r in rs)
        if positive==4:
            text=f'四个配对均值都支持保留该模块，{clear}/4个案例区间完全高于零。两个固定种子和两布局的方向一致；贡献的强弱仍须结合辅助指标与成本，不能外推训练种子总体。'
        elif positive==0:
            text='四个配对均值均未支持保留模块的L1收益，本轮不能将其作为已获验证的精度核心创新。案例区间跨零也不构成等效性证明。';limited.append(label)
        else:
            text=f'只有{positive}/4个配对均值支持保留模块，存在种子或布局依赖，不能声称稳定独立收益。';limited.append(label)
        lines.append(f'**{label}**：{text}')
        a=[find(arm,20261011,c)['mean'] for c in ('Val-A-V4','clustered-V4')]
        b=[find(arm,20261019,c)['mean'] for c in ('Val-A-V4','clustered-V4')]
        lines.append(f'种子20261011的普通/聚簇ΔL1为{100*a[0]:+.4f}/{100*a[1]:+.4f}pp；种子20261019为{100*b[0]:+.4f}/{100*b[1]:+.4f}pp。')
    lines+=['', '核心创新的表述应限于这里实际显示的收益：方向一致且辅助指标/成本合理的模块可以作为有探索性实验支持的设计；'+('、'.join(limited)+'仍需限定其精度主张。' if limited else '三个模块均有同向描述性支持，但仍不能称独立确认。'),
        '若需要进一步验证不一致项，最小建议是先审阅逐案例和两个固定种子差异，再单独预登记一个额外配对训练种子或新的独立案例验证；本轮不自动启动额外训练、不搜索超参数。','',
        '## 正式单因素定义与训练协议','',
        '所有消融采用同种子fresh普通交互工厂。保留参数和buffer的初始值、构造后CPU RNG均逐项严格匹配，未从训练后基线加载初始化。原CUDA grid_sample backward可能使用非确定性原子操作，故不声称训练轨迹逐位一致。',
        'GeometryLifting直接复用正式固定探测平面挤出定义，主/残差两次lifting共同替换，保持几何描述符和可见性；它同时改变覆盖和深度结构。单尺度直接复用SingleScaleContext，全部原层/通道保留，在32³计算，不增加通道适配。PRF实际删除residual_encoder/residual_adapter，唯一接口适配为调用普通交互stage0，网络C0=C1；训练objective内全部原测量和一致性约束保留。',
        '166训练/36验证、batch2、300轮/24900更新、AdamW lr2e-4/wd1e-4、cosine300、FP32/TF32off、clip1、原损失/预处理/采样/worker不变。每epoch提交last，best只按完整300轮验证投影H4严格最小值，不用测试GT选轮。','',
        '## 模型与可复算身份','',
        '| 模型 | 参数 | best epoch | best SHA256 |','|---|---:|---:|---|']
    for m in models:lines.append(f"| {m['key']} | {m['parameters']} | {m['best_epoch']} | `{m['checkpoint_sha256']}` |")
    lines+=['',f"配置和全部源码hash见 `{BASE}/training/*/protocol.json`；父协议SHA256 `{sha(BASE/'protocol.json')}`。历史模型未覆盖，基线hash再次核验。",'',
        '## 评价与完整配对统计','',
        '普通Val-A-V4 draw0；clustered-V4五draw先在案例内取均值。另列聚簇draw0敏感性表。Δ=消融−同种子完整模型，误差正为变差，Dice正为改善，峰幅比本身不具有单调优劣方向。每行35案例，20000次配对bootstrap、seed20261026、未校正的均值百分位95%区间；中位数、SD与|Δ|≤1e-8数值不变计数同时报告。两个种子不合并，draw不充当独立案例，无预登记等效界限。',
        '直接使用正式volume_metrics/safe_shape，64³ XYZ端点[0,.6]³m、原[0,1]质量分数、float64，无逐例重新归一化。Dice50阈值为0.5×GTmax；峰位置并列取首节点。梯形积分质心和普通节点质心两种定义分别保留。',
        f'逐案例逐draw C0/C1全指标：`{BASE}/metrics.csv`；逐案例聚合配对：`{BASE}/paired_case_metrics.csv`；全部360个指标/阶段/布局配对汇总：`{BASE}/summary.csv` 和 `statistics.json`。','',
        '| 模块 | 种子 | 布局 | 阶段 | 指标 | 完整均值 | 消融均值 | 平均差 | 中位数差 | 95%区间 | 增/降/不变 |','|---|---:|---|---|---|---:|---:|---:|---:|---|---|']
    for r in stats:
        if r['metric']=='D' or r['draws']!=([0] if r['condition']=='Val-A-V4' else list(range(5))):continue
        arm=next(a for a in labels if r['model']==f"{a}_seed{r['seed']}")
        lines.append(f"| {labels[arm]} | {r['seed']} | {r['condition']} | {r['stage']} | {r['metric']} | {r['baseline_mean']:.6f} | {r['treatment_mean']:.6f} | {r['mean']:+.6f} | {r['median']:+.6f} | [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}] | {r['increased']}/{r['decreased']}/{r['unchanged']} |")
    lines+=['','## 参数与推理成本','',bench['protocol'],'',
        '| 模型 | 参数 | Δ参数 | 平均(ms) | Δ时间(ms) | 中位数(ms) | 峰值分配(MiB) |','|---|---:|---:|---:|---:|---:|---:|']
    costs={r['key']:r for r in bench['records']}
    for m in models:
        r=costs[m['key']];ref=costs[f"{PLAIN}_seed{m['seed']}"]
        lines.append(f"| {m['key']} | {r['parameters']} | {r['parameters']-ref['parameters']:+d} | {1000*r['mean_seconds']:.3f} | {1000*(r['mean_seconds']-ref['mean_seconds']):+.3f} | {1000*r['median_seconds']:.3f} | {r['peak_allocated_bytes']/1024**2:.2f} |")
    lines+=['','计时九次来自同GPU的三个固定验证输入，不是独立模型性能样本。参数相同不代表计算量相同，单尺度32³卷积的实际成本见表。','',
        '## 训练执行与资源记录','',
        '| 实验 | GPU UUID | 开始 | 结束 | 引擎耗时(s) | epoch / 已提交更新 | 峰值分配(MiB) |','|---|---|---|---|---:|---|---:|']
    events=rows(BASE/'scheduling/events.jsonl')
    for a,s in TASKS:
        target=BASE/'training'/f'{a}_seed{s}';summary=read(target/'summary.json');log=rows(target/'metrics.jsonl')
        launches=[e for e in events if e.get('key')==target.name]
        starts=[e for e in launches if e['state']=='running'];ends=[e for e in launches if e['state']=='complete']
        lines.append(f"| {target.name} | {', '.join(dict.fromkeys(e['gpu_uuid'] for e in starts))} | {starts[0]['started_at']} | {ends[-1]['finished_at']} | {summary['elapsed_seconds']:.2f} | 300 / 24900 | {max(r['peak_allocated_bytes'] for r in log)/1024**2:.2f} |")
    incidents=list((BASE/'recovery').glob('*.json')) if (BASE/'recovery').exists() else []
    lines+=['',f'异常/恢复登记文件数：{len(incidents)}。资源双快照、GPU利用率、显存与进程归属见 `scheduling/resources.jsonl`，任务起止与重试见 `scheduling/events.jsonl`；所有故障日志保留，不静默重试。第一批四项完成并验收后才允许PRF第二批；独占UUID锁FD由worker继承，外部工作流上下文空档不用于抢占。没有终止其他用户进程。','',
        '## 访问边界、复核与交付','',
        f"1260组新预测与420组复用预测（3360个C0/C1字段）均在首次本轮GT64访问前统一封存。仅访问35份授权GT64，hash与历史一致；GT128始终禁用。{access['archived_fields_verified']}个历史基线字段重新计分最大差{access['maximum_archived_difference']:.3g}。独立计数权重与排序插值算法复核{check['summary_vectors']}个统计向量，最大差{check['maximum_difference']:.3g}。",
        f'逐案例预测身份、输入hash及checkpoint hash见 `{BASE}/prediction_manifest.json`，大数组和权重保留scratch；仓库轻量证据在 `docs/reports/artifacts/{BASE.name}/`，追踪表在 `{TRACKER.name}`。',
        '正式验收核对六项预算、单因素定义、基线保护、案例/布局/draw对齐、数值范围、预测文件hash、可复算统计、资源与历史保护。结论严格限于两个固定种子、两布局及已使用35案例，不强行得到预期方向。']
    assert not REPORT.exists();REPORT.write_text('\n'.join(lines)+'\n')


def main():
    if (BASE/'acceptance.json').exists():assert verified(BASE/'acceptance.json')['completed'];return
    verify_hashes(verified(BASE/'protocol.json')['sources']);mm=methods()
    for m in verified(BASE/'references.json'):assert sha(m['checkpoint'])==EXPECTED[m['seed']]==m['checkpoint_sha256']
    assert verified(BASE/'independent_verification.json')['passed']
    if not REPORT.exists():create_report()
    update_tracker();ART.mkdir(parents=True,exist_ok=True)
    for p in BASE.rglob('*'):
        if not p.is_file() or p.suffix in ('.pt','.npz','.npy','.lock') or p.name in ('controller_launch.log',):continue
        target=ART/p.relative_to(BASE);target.parent.mkdir(parents=True,exist_ok=True)
        if p.name=='resources.jsonl':
            with p.open('rb') as f,gzip.open(str(target)+'.gz','wb') as g:shutil.copyfileobj(f,g)
        else:shutil.copy2(p,target)
    exports={str(p.relative_to(PROJECT)):sha(p) for p in ART.rglob('*') if p.is_file()}
    exports.update({str(p.relative_to(PROJECT)):sha(p) for p in HERE.glob('*.py')})
    freeze(BASE/'acceptance.json',dict(at=now(),completed=True,training_runs=6,epochs_each=300,updates_each=24900,
        new_predictions=1260,reused_predictions=420,test_cases=35,GT128_read=False,independent_confirmation=False,
        independent_statistics_verified=True,baselines_unchanged=True,foreign_processes_signalled=0,
        report=str(REPORT),report_sha256=sha(REPORT),tracker=str(TRACKER),exports=exports))
    for suffix in ('','.sha256'):shutil.copy2(str(BASE/'acceptance.json')+suffix,str(ART/'acceptance.json')+suffix)
    print('COMPLETE',REPORT,flush=True)


if __name__=='__main__':main()
