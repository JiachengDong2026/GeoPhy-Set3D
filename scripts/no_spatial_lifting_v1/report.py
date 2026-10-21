"""Chinese result report, artifact export, and evidence-based final acceptance."""
from runtime import *
import csv
import gzip
import re
import shutil


def finish():
    protocol=verified(BASE/'protocol.json');verify_hashes(protocol['sources'])
    lock=verified(BASE/'evaluation_lock.json');verify_hashes(lock['sources'])
    metrics=verified(BASE/'metric_verification.json');ind=verified(BASE/'independent_statistics_verification.json')
    assert metrics['passed'] and ind['passed'] and verified(BASE/'engineering_checks.json')['passed']
    s=verified(BASE/'statistics.json');benchmark=verified(BASE/'benchmark.json')
    def panel(seed,condition,stage='C1',metric='relative_l1'):
        return next(p for p in s['panels'] if (p['seed'],p['condition'],p['stage'],p['metric'])==(seed,condition,stage,metric))
    def fmt(v,n=4):return '未定义' if v is None else f'{v:.{n}f}'
    def interval(v,scale=1):return '未定义' if v is None else '['+', '.join(f'{x*scale:+.4f}' for x in v)+']'
    main=panel('two_fixed_seed_mean_descriptive','Val-A-V4');cluster=panel('two_fixed_seed_mean_descriptive','clustered-V4')
    same=all(panel(seed,'Val-A-V4')['mean_delta']>0 for seed in SEEDS)
    lines=['# Uniform Fusion MRGR：GeometryLifting 重新训练结构消融','',f'完成时间：{now()}。',
        '**完整对照和消融模型均采用 Multi-scale MRGR + Uniform Fusion，均为342725个可训练参数。未使用GCA学习式融合。**','',
        '本实验在已使用的正式测试集上提供补充机制证据，不是独立测试确认。唯一主要终点为普通固定Val-A-V4的C1三维相对L1；clustered-V4为预指定重要次要结果，其余布局及辅助指标为探索性。',
        '', '## 正式对照身份与配对', '',
        '身份来自最终模型配置、最终model_identity/model_catalog、模型决策记录、实际检查点及训练/测试产物，未从旧诊断名称推断。正式登记的seed20261011检查点也曾用于此前诊断；文件相同，但本次正式身份由最终登记独立确认。旧GCA epoch278、344678参数模型不作为本次对照。',
        '', '| 类型 | 种子 | 最佳epoch | 参数 | checkpoint SHA256 |','|---|---:|---:|---:|---|']
    for label,mm in [('完整Uniform Fusion',verified(BASE/'controls.json')),('消融Uniform Fusion',lock['methods'])]:
        for m in mm:lines.append(f'| {label} | {m["seed"]} | {m["best_epoch"]} | {m["parameters"]} | `{m["checkpoint_sha256"]}` |')
    lines += ['', '两个正式完整对照各有300轮、24900次更新，既有1190份封存预测全部复用，没有重新训练或重复完整对照正式推理。两个消融分别新训练300轮、24900次更新，验证H4投影相对L1严格最低选轮，不使用测试GT选轮。',
        '', '## 唯一替代方案与解释范围', '',
        '消融模块将每视角二维特征双线性缩放到32×32（align_corners=True），将探测器水平轴固定对应世界Y、垂直轴对应世界Z，再沿世界X广播，得到[B,V,C,32,32,32]。同一固定约定用于所有案例、视角和种子，不根据结果搜索轴向。主干32通道与残差16通道两次GeometryLifting调用均替换。',
        '特征值的生成不使用逐体素角度采样网格；继承的angular_grid保持原样，仅在该替代模块中用于保留visibility，同时继续供原始geometry_descriptor使用。OARI、几何描述符、射线关系、可观测性、路径长度输入、均匀融合、多尺度解码、物理投影、残差编码/适配和一致性损失保留。反馈的特征映射是同一预登记消融因素的第二处应用。',
        '相较全局池化，该方案保留32×32二维局部变化，且不增删任何参数；但仍改变采样覆盖/密度、插值位置，并在几何嵌入前施加深度恒定结构。因此结果只解释为该空间映射替代方案的影响，不能量化纯粹单算子的因果净贡献，也不是移除所有几何信息。候选比较和限制在训练前封存。',
        '', '## 工程、初始化与训练边界', '',
        '官方最终构造器scripts/uniform_final_v1/uniform_models.py:model_for(config,uniform,device)在其原始runtime绑定下调用，再隔离替换无参数lifting类。两个种子全部共享状态张量逐一相同，分别匹配历史完整Uniform Fusion初始化哈希；没有新增参数或额外随机初始化。UniformFusion本身无参数，工程检查验证有效视角权重及融合输出，而不虚构融合参数梯度。',
        '两种子真实CUDA两步自监督前向/损失/反向通过；所有输出/梯度有限，C0/C1在[0,1]，第二步被检查的编码器、OARI、多尺度上下文、残差编码器与适配器都有非零梯度。零初始化门控的第一步内部零梯度按原架构记录。两次提升输出与独立resize/repeat逐元素一致，visibility、descriptor、ray relations和observability与原几何计算逐元素一致。',
        '每种子分别完成跨进程连续2轮vs1轮+恢复至2轮检查（4个训练/2个验证案例、4个persistent workers）；模型、最佳模型、optimizer、scheduler、Python/NumPy/Torch/CUDA/loader RNG及采样摘要核验通过。整数/RNG精确相同，浮点容差rtol1e-4、atol1e-6。',
        '正式训练保留166训练/36验证案例、FP32、batch2、AdamW lr2e-4/wd1e-4、cosine300、梯度裁剪1、原A/B/H采样及全部损失。原CUDA grid_sample反向可能有原子累积非确定性；配对初始化及短程恢复通过不等于宣称所有长程CUDA轨迹逐位确定。',
        '训练/工程只允许训练与验证投影，禁止测试数组、GT64、GT128及历史测试指标。审计阶段曾读取历史评分表首条LR参考行以识别格式，已登记；该行不用于结构、训练或选轮。新增模型训练期间未读测试指标。全部1190份新增预测、2380个C0/C1字段生成并统一封存后才授权GT64评分；GT128始终禁止。',
        '', '## 配对主要结果与重要次要结果', '',
        '差值=消融误差−同种子完整模型误差；相对L1的正值表示完整模型更好。对clustered等布局先在同案例内平均5draw，再以35个案例为统计单位。报告每个固定种子及同案例两固定种子的描述性均值；配对bootstrap20000次，seed20261019。两种子、5draw不扩充样本量。95%区间是案例重采样区间，不能代替训练种子总体方差。',
        '保留历史均匀融合消融14比较家族不变；额外列出的14分母区间只是保守参考，不声称本新增分析被纳入原家族或补做独立确认。',
        '', '| 布局 | 种子 | 完整C1 L1% | 消融C1 L1% | 差值比例 | 差值百分点及95%区间 | 中位差百分点 | 恶化案例 |',
        '|---|---|---:|---:|---:|---|---:|---:|']
    for condition in ('Val-A-V4','clustered-V4'):
        for seed in (*SEEDS,'two_fixed_seed_mean_descriptive'):
            p=panel(seed,condition)
            lines.append(f'| {condition} | {seed} | {p["full_mean"]*100:.4f} | {p["ablation_mean"]*100:.4f} | {p["mean_delta"]:+.6f} | {p["mean_delta"]*100:+.4f} {interval(p["ci95"],100)} | {p["median_delta"]*100:+.4f} | {p["increased_cases"]}/35 |')
    lines += ['', '| 普通V4主要终点种子 | 原14分母保守参考区间（百分点） |','|---|---|']
    for seed in (*SEEDS,'two_fixed_seed_mean_descriptive'):lines.append(f'| {seed} | {interval(panel(seed,"Val-A-V4")["conservative_family14_interval"],100)} |')
    lines += ['', '## 全部布局、C0/C1与辅助指标（探索性）', '',
        '| 布局 | 阶段 | 两固定种子描述性完整L1% | 消融L1% | 差值百分点及95%区间 | L2差 | Dice50差 | 峰比绝对误差差 | 质心距离差mm |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for stage in ('C0','C1'):
            p=panel('two_fixed_seed_mean_descriptive',condition,stage)
            aux=[panel('two_fixed_seed_mean_descriptive',condition,stage,m)['mean_delta'] for m in ('relative_l2','dice50','absolute_peak_ratio_error','centroid_distance_mm')]
            lines.append(f'| {condition} | {stage} | {p["full_mean"]*100:.4f} | {p["ablation_mean"]*100:.4f} | {p["mean_delta"]*100:+.4f} {interval(p["ci95"],100)} | '+' | '.join(fmt(v) for v in aux)+' |')
    lines += ['', 'Dice50使用GT峰值50%作为两者共同阈值；峰值误差为|pred_peak/GT_peak−1|；质心为质量加权XYZ中心，单位mm。全部案例、全部draw、两个阶段和两个种子均保存，未删除失败或不利案例。未定义指标保留为空并使相应完整面板区间未定义，不静默丢样本。',
        '', '## 计算量、耗时与资源', '',
        '| 架构（均Uniform Fusion） | 参数 | Conv/Linear部分MACs | B1/V4 C0+C1平均ms | 中位ms |','|---|---:|---:|---:|---:|']
    for r in benchmark['results']:lines.append(f'| {r["arm"]} | {r["parameters"]} | {r["partial_Conv_Linear_MACs"]} | {r["mean_seconds"]*1000:.3f} | {r["median_seconds"]*1000:.3f} |')
    lines += ['', f'同GPU `{benchmark["gpu_uuid"]}`（{benchmark["gpu_name"]}），10次预热、40次计时，排除I/O，包含物理反馈。这里为初始化模型的架构计时。部分MACs仅计Conv2d/Conv3d/Linear，未计投影器、grid_sample/resize、attention矩阵乘法、归一化及逐元素运算，不能称总FLOPs。',
        '原始lifting每通道每视角进行32768个几何采样；替代为1024个平面插值位置加广播。两者参数相同、上述Conv/Linear部分相同，但采样及数据移动开销不同。资源按UUID排他锁、间隔15秒双空闲检查、主机内存/磁盘/负载门槛调度；没有终止或抢占其他任务。',
        '', '| 种子 | GPU UUID | 训练秒 | 最佳H4 |','|---|---|---:|---:|']
    for seed in SEEDS:
        summary=read(run_protocol(seed)/'summary.json');env=read(run_protocol(seed)/'environment.json')
        lines.append(f'| {seed} | `{env["gpu_uuid"]}` | {summary["elapsed_seconds"]:.1f} | {summary["best_val_H"]:.7f} |')
    conclusion=('两个固定匹配种子普通V4均出现正向误差差值，支持该重新训练条件下完整空间映射的性能贡献。' if same else '两个固定匹配种子普通V4方向未全部为误差恶化，不能宣称一致的映射收益；应按完整配对结果解释。')
    lines += ['', '## 结论与限制', '', conclusion,
        f'两固定种子描述性普通V4 C1差值为{main["mean_delta"]*100:+.4f}个百分点，clustered-V4为{cluster["mean_delta"]*100:+.4f}个百分点；布局影响差异为描述性比较，未据此追加干预或改变选轮。',
        '冻结错误映射敏感性与本次重新训练替代回答不同问题。现有OARI、几何描述符、visibility、路径和物理反馈仍可提供补偿；仅凭整体误差变化不能定位各通路的补偿量，也不能声称完全失去几何能力。C0/C1结果描述反馈后的总体表现，未新增反馈消融。',
        '本次仅两个固定种子、同一已使用测试集、单一固定平面替代方案。它不单独证明模型学到可泛化几何规律，不估计随机训练种子总体方差，也不能推出GeometryLifting是唯一有效几何通路。二维采样覆盖、密度和深度结构的额外差异应随论文结论同时陈述。',
        '', '## 验证、异常与产物', '',
        f'工程检查、两任务完整训练、1190份新增预测封存、4760个基准/消融字段的5项独立NumPy指标核验、{ind["panels_checked"]}组独立统计核验通过。未定义辅助指标记录数：{len(metrics["undefined"])}。全部异常与失败文件均检查，验收未跳过任何阶段。',
        '[正式Uniform Fusion模型报告](mrgr_uniform_final_model_v1_zh.md) · [正式训练与测试报告](mrgr_uniform_final_v1_zh.md) · [已有均匀融合模块消融](mrgr_uniform_ablations_v1_zh.md) · [旧GCA冻结映射诊断](mrgr_lifting_mapping_v1_zh.md)。',
        '', '[不可变协议](artifacts/mrgr_no_spatial_lifting_v1/protocol.json) · [历史身份审计](artifacts/mrgr_no_spatial_lifting_v1/historical_audit.json) · [候选方案比较](artifacts/mrgr_no_spatial_lifting_v1/candidate_comparison.json) · [初始化审计](artifacts/mrgr_no_spatial_lifting_v1/initialization_audit.json) · [工程检查](artifacts/mrgr_no_spatial_lifting_v1/engineering_checks.json) · [逐案例全部指标](artifacts/mrgr_no_spatial_lifting_v1/per_case_metrics.csv) · [逐案例配对差](artifacts/mrgr_no_spatial_lifting_v1/paired_case_metrics.csv) · [完整配对统计](artifacts/mrgr_no_spatial_lifting_v1/statistics.json) · [独立统计复核](artifacts/mrgr_no_spatial_lifting_v1/independent_statistics_verification.json) · [验收](artifacts/mrgr_no_spatial_lifting_v1/acceptance.json)。',
        '', f'大型预测、检查点、训练/资源原始日志：`{BASE}`。代码入口：`scripts/no_spatial_lifting_v1/`。未覆盖历史预测、模型清单或正式默认实现。']
    assert not REPORT.exists();REPORT.write_text('\n'.join(lines)+'\n')
    failures=list(BASE.glob('*failure*.json'));assert not failures,failures
    audit=verified(BASE/'historical_audit.json');verify_hashes(audit['checkpoint_and_training_evidence'])
    for m in protocol['controls']:assert sha(Path(m['checkpoint']))==m['checkpoint_sha256']
    files=[p for p in BASE.glob('*.json') if p.name!='status.json']+[p for p in BASE.glob('*.csv')]+[REPORT]
    freeze(BASE/'acceptance.json',dict(at=now(),completed=True,fusion_full='Uniform Fusion',fusion_ablation='Uniform Fusion',parameters=342725,
        seeds=list(SEEDS),formal_training_tasks=2,epochs_per_seed=300,updates_per_seed=24900,new_predictions=1190,new_fields=2380,reused_control_predictions=1190,
        engineering_passed=True,initialization_exactly_paired=True,selection='validation projection H4 only',GT128_read=False,
        predictions_globally_sealed_before_GT64=True,independent_metrics_verified=True,independent_statistics_verified=True,
        supplementary_used_test=True,independent_confirmation=False,primary=s['primary'],anomalies=[],
        limitations=['single fixed planar sampling/extrusion alternative changes sampling density/coverage and depth structure','two fixed seeds cannot estimate population seed variance','remaining geometry paths may compensate'],
        evidence={str(p):sha(p) for p in files}))
    export()
    for link in re.findall(r'\]\(([^)]+)\)',REPORT.read_text()):assert (REPORT.parent/link).exists(),link
    print('ACCEPTED retrained Uniform Fusion planar-lifting ablation',flush=True)


def export():
    ARTIFACT.mkdir(exist_ok=False);exported={}
    for path in [*BASE.glob('*.json'),*BASE.glob('*.csv')]:
        if path.name=='status.json':continue
        target=ARTIFACT/path.name;shutil.copy2(path,target)
        side=path.with_suffix(path.suffix+'.sha256')
        if side.exists():shutil.copy2(side,target.with_suffix(target.suffix+'.sha256'))
        exported[str(target.relative_to(PROJECT))]=dict(source=str(path),sha256=sha(target))
    for seed in SEEDS:
        root=BASE/'evaluation'/str(seed);dst=ARTIFACT/'prediction_manifests'/str(seed);dst.mkdir(parents=True)
        for name in ('manifest.jsonl','layout_audits.jsonl'):
            data=(root/name).read_bytes();target=dst/(name+'.gz')
            with target.open('xb') as f:
                with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g:g.write(data)
            assert gzip.decompress(target.read_bytes())==data
            exported[str(target.relative_to(PROJECT))]=dict(source=str(root/name),sha256=sha(target),source_sha256=sha(root/name),transform='lossless gzip')
        shutil.copy2(root/'complete.json',dst/'complete.json');shutil.copy2(root/'complete.json.sha256',dst/'complete.json.sha256')
        train=run_protocol(seed);dest=ARTIFACT/'training'/str(seed);dest.mkdir(parents=True)
        for name in ('protocol.json','protocol.json.sha256','summary.json','environment.json','metrics.jsonl','gt_access_audit.json','boundary_audit.json'):
            shutil.copy2(train/name,dest/name);exported[str((dest/name).relative_to(PROJECT))]=dict(source=str(train/name),sha256=sha(dest/name))
    dest=ARTIFACT/'scheduling';dest.mkdir()
    for path in (BASE/'scheduling').glob('*.json'):shutil.copy2(path,dest/path.name)
    for name in ('engineering_attempt1.log','evaluation.log','scoring.log'):
        path=BASE/'scheduling'/name
        if path.exists():shutil.copy2(path,dest/name)
    freeze(ARTIFACT/'export_manifest.json',dict(at=now(),files=exported,large_predictions_checkpoints_on_scratch=True,
        raw_resource_log=str(BASE/'scheduling/resources.jsonl'),external_processes_signalled=False))
