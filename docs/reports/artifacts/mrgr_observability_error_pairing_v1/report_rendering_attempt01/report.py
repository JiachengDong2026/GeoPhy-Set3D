"""Chinese evidence report and standalone plots from verified scalar results."""
from analyze import *


def run():
    assert not REPORT.exists()
    verify=verified(ART/'independent_verification.json');assert verify['passed']
    data=verified(ART/'statistics.json');summary=data['summary'];correlations=data['correlations']
    models=verified(ART/'models.json');patterns=verified(ART/'case_patterns.json')
    pairs=list(csv.DictReader((ART/'paired_case_metrics.csv').open()))
    cases=verified(ART/'cases.json')
    def find(condition,model,metric='relative_l1',stage='C1'):
        return next(r for r in summary if (r['condition'],r['model'],r['metric'],r['stage'])==(condition,model,metric,stage))
    def ci(r,mult=1):return f"[{r['ci95_low']*mult:+.4f}, {r['ci95_high']*mult:+.4f}]"
    # Explicitly exploratory, paired contrast requested by the layout question;
    # never replace the pre-locked two per-layout primary estimates.
    tokens=[r for r in pairs if r['model']=='obs_token' and r['stage']=='C1']
    lookup={(r['case_id'],r['condition']):r for r in tokens}
    delta=[float(lookup[c,'clustered-V4']['delta_relative_l1'])-float(lookup[c,'Val-A-V4']['delta_relative_l1']) for c in cases]
    contrast=stats(delta,np.random.default_rng(SEED).integers(0,35,(RESAMPLES,35)))
    freeze(ART/'layout_contrast_exploratory.json',dict(at=now(),
        status='post-scoring exploratory requested layout contrast; not a preregistered primary endpoint',
        contrast='within-case clustered obs_token delta L1 minus ordinary delta L1',
        seed=SEED,resamples=RESAMPLES,**contrast,case_deltas=dict(zip(cases,delta))))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
    for ax,condition in zip(axes,CONDITIONS):
        rr=[lookup[c,condition] for c in cases]
        ax.scatter([100*float(r['D']) for r in rr],[100*float(r['delta_relative_l1']) for r in rr],s=30,alpha=.8)
        ax.axhline(0,color='gray',linewidth=1)
        ax.set(title=condition,xlabel='C1 prediction change D (%)')
        for i,r in enumerate(rr):ax.annotate(str(i),(100*float(r['D']),100*float(r['delta_relative_l1'])),fontsize=6,xytext=(3,2),textcoords='offset points')
    axes[0].set_ylabel('C1 relative L1 change (percentage points)')
    fig.tight_layout();fig.savefig(ART/'D_vs_error.png',dpi=180);fig.savefig(ART/'D_vs_error.pdf');plt.close(fig)
    lines=['# OARI 可观测性描述符：冻结干预与真实重建误差配对验证','',
        '本阶段为已使用的35个测试案例上的探索性机制分析，不是独立测试确认。所有完整对照与结构删除模型均采用 **Multi-scale MRGR + Uniform Fusion**。',
        '', '## 主要判断','',
        '**A．冻结模型依赖是否有益：** `obs_token` 在聚簇V4使C1相对L1平均增加1.8151个百分点，30/35案例增加，案例bootstrap区间完全在零以上；普通V4平均仅增加0.0217个百分点，区间跨零，未建立清晰平均恶化。不能把普通V4区间跨零解释为严格等效。',
        '**B．预测变化与误差变化：** 聚簇V4的预测变化更大，且与L1恶化具有中等案例级正相关；仍有6个案例D≥5%而|ΔL1|≤0.5个百分点。D和ΔE不是同一量，也不能由D直接判断性能。',
        '**C．冻结与重新训练：** 普通V4结构删除的两个种子平均均略差；聚簇V4结构删除在两个种子方向仍相反。冻结seed20261011的明显恶化不能外推为描述符结构在重新训练后必有稳定独立收益。',
        '**D．机制解释：** 证据支持“该冻结模型确实使用描述符token输入，且该路径对聚簇V4的L1有益”；对于跨训练模型的独立性能贡献，现有证据仍不稳定。剩余路径可补偿是相容解释，但训练随机性、移除结构改变优化以及冻结置零造成分布偏移也相容，尚不能区分。不能断言信息难以替代。',
        '**E．是否需下一实验：** 回答本轮限定问题不需要新训练。若要排除“当前冻结依赖仅限seed20261011”，可另行登记在已有seed20261019完整模型上做相同obs_token探针（两布局各35案例，至多70次冻结前向）。这只能检查第二种子是否同向，仍不能独自证明补偿机制；本轮没有执行该追加实验。',
        '', '## 审计、授权与执行边界','',
        '审计前工作树干净，HEAD为`450a4ecf4ef24702d0db2113fe69b4c01a138661`。真实路径从正式登记与历史验收定位，未根据模型config名称猜构造器。4个checkpoint均核验文件SHA256、epoch、config、训练protocol哈希、参数量、无体GT训练标记，并通过官方模型工厂strict load，实际fusion为UniformFusion，eval且参数无梯度；模型不执行forward。',
        '现有授权是阶段限定的：OARI v2不可变protocol明确GT64_read=false；历史结构消融与正式评估授权不能自动扩展。本轮先停止GT评分并完成无GT审计，用户随后明确回复“授权本阶段限定的 GT64 离线评分”。授权单独记入GT64_authorization.json，未改写旧协议。',
        '随后固定本阶段protocol、输入与预测清单并封存，才读取35份授权GT64，每份只加载一次（另做文件SHA256读取）。GT哈希与历史授权记录逐一一致；GT128未读。新增训练、反传、优化器更新、检查点选择、调参、模型前向均为0；CPU两线程，不占用GPU。',
        '覆盖70个输入（35案例×两布局×draw0），复用490份不同预测文件、980份C0/C1字段：A/B/C/D共280份，加full seed19与pruned两种子210份；full seed11即A，不重复计数。全部文件哈希、输出shape/有限性/[0,1]、checkpoint封存链、案例、站点顺序与draw一致。CPU重新构造70个测量输入，哈希与原OARI记录完全一致，未读目标数组。',
        '全部预测及4个checkpoint可直接复用，无输入/预测/指标缺失；预计及实际新增推理均为0。',
        '', '| 模型 | 种子 | epoch | 参数 | checkpoint SHA256 |', '|---|---:|---:|---:|---|']
    for m in models:lines.append(f"| {m['key']} | {m['seed']} | {m['best_epoch']} | {m['parameters']} | `{m['checkpoint_sha256']}` |")
    lines += ['', '完整模型官方工厂为`scripts/uniform_final_v1/uniform_models.py:model_for(config,"uniform","cpu")`；删除模型使用登记的`scripts/pruning_v1/pruned_models.py:model_for(config,"uniform_pruned_observability","cpu")`。完整权重路径、配置、封存链见models.json及provenance_chain.json。',
        '', '## 干预及指标定义','',
        '严格复用OARI v2已保存的干预输出与实现：B=obs_token仅移除observability_embedding首线性层五维输入贡献，保留learned bias和SiLU；C=obs_bias仅去掉pairwise_bias首仿射中描述符列3..7的贡献，保留偏置；D=obs_both同时执行。A为原始完整模型。原特征、几何、visibility、view_mask和观测性输入张量身份全部相同。token路径还直接进入token残差，不能将其全部作用称为attention改变。',
        '冻结B保留结构与全部参数；重新训练的结构消融同时删除描述符模块和两处相关输入结构，少384参数，二者不是完全相同的操作。结构删除可能改变优化过程，冻结输入屏蔽也可能形成训练外组合。',
        '指标直接调用正式`scripts/paper_final_v1/evaluation.py`导入的volume_metrics与safe_shape；按原float64、原始[0,1]浓度质量分数计算，不按每案例重新归一化。',
        '', '| 指标 | 正式定义及单位 |', '|---|---|',
        '| 相对L1 | Σ|p−g| / max(Σ|g|,1e−6)，无量纲 |',
        '| 相对L2 | sqrt(Σ(p−g)² / max(Σg²,1e−12))，无量纲 |',
        '| 峰幅比/绝对比例误差 | max(p)/max(g)，以及|该比值−1|；无量纲 |',
        '| 峰值绝对误差 | |max(p)−max(g)|，质量分数；并非mm或相对比例 |',
        '| 峰位置误差 | 两argmax的欧氏距离×600/63 mm；并列用XYZ字典序首点 |',
        '| centroid_error_mm | 正式volume_metrics按0..0.6m XYZ端点网格的梯形积分权重求质心，再取距离mm |',
        '| centroid_distance_mm | 正式形态学metrics按普通节点浓度权重求质心，距离mm；与上一指标分开保存 |',
        '| Dice50 | 双方共用0.5×GT最大值阈值；不是volume的固定绝对阈值Dice |',
        '| 预测变化D | Σ|B−A|/(Σ|A|+1e−6)，无量纲。另保留历史max分母约定的D_legacy，逐份重算吻合 |',
        '', 'ΔE=干预或结构删除误差−匹配完整模型误差。误差指标正值表示恶化；Dice差值正值表示改善；峰幅比本身无单调优劣方向。绝对比例差乘100即百分点。所有结构删除D使用匹配种子的完整模型为分母。',
        '', '## 配对统计方案','',
        '主分析全用draw0，两布局各35案例。历史聚簇五draw均值没有混入本轮。每个指标在同案例内先配对，以案例为单位bootstrap20,000次，种子20261025，百分位95%区间估计平均配对差；标准差为样本SD(ddof=1)。区间均为未经多重比较校正的探索性案例区间，不是训练种子总体区间。',
        '数值“基本不变”为|Δ|≤1e−8，沿用既有配对计数约定；不是临床或实用等效界限。两个训练种子分别报告，不扩充n。相关性使用35个案例的Pearson与Spearman，仅作探索性描述；差值恒定时相关无定义，输出null和原因，不伪造0。',
        '', '## 主要终点：obs_token的C1相对L1','',
        '| 布局 | A均值 | B均值 | 平均Δ绝对比例 | 平均Δ(pp)与95%区间 | Δ中位数(pp) | ΔSD(pp) | 增加/降低/不变 |', '|---|---:|---:|---:|---|---:|---:|---|']
    for condition in CONDITIONS:
        r=find(condition,'obs_token')
        lines.append(f"| {condition} | {r['baseline_mean']:.8f} | {r['intervention_mean']:.8f} | {r['mean']:+.8f} | {100*r['mean']:+.4f} {ci(r,100)} | {100*r['median']:+.4f} | {100*r['sd']:.4f} | {r['increased']}/{r['decreased']}/{r['unchanged']} |")
    lines += ['', f"普通A/B均值分别为8.5928%/8.6145%，聚簇为32.8316%/34.6467%。另作明确标记的评分后探索性布局配对对比：聚簇Δ减普通Δ={100*contrast['mean']:+.4f}pp，95%区间{ci(contrast,100)}pp；这是同案例差的差，不由两个区间是否重叠推断布局差异。",
        '', '## 预测变化、真实误差及案例分布','',
        '| 布局 | C1 D均值/中位数/SD(%) | D最小/25%/75%/最大(%) | D与ΔL1 Pearson / Spearman |', '|---|---|---|---|']
    for condition in CONDITIONS:
        d=find(condition,'obs_token','D');c=next(r for r in correlations if r['condition']==condition and r['model']=='obs_token' and r['stage']=='C1' and r['metric']=='relative_l1')
        lines.append(f"| {condition} | {100*d['mean']:.4f}/{100*d['median']:.4f}/{100*d['sd']:.4f} | {100*d['minimum']:.4f}/{100*d['q25']:.4f}/{100*d['q75']:.4f}/{100*d['maximum']:.4f} | {c['pearson']:.4f} / {c['spearman']:.4f} |")
    lines += ['', '![预测变化与真实误差变化，数字为固定案例序号](artifacts/mrgr_observability_error_pairing_v1/D_vs_error.png)',
        '', 'D≥5%且|ΔL1|≤0.5pp的描述性标记在读取GT前固定，普通0例、聚簇6例。它不是正式等效性检验。全部35案例及其编号、指标、差值保存在paired_case_metrics.csv，不只保留这些案例。聚簇6例如下：',
        '', '| 案例 | D(%) | ΔL1(pp) |', '|---|---:|---:|']
    pattern=next(r for r in patterns if r['model']=='obs_token' and r['condition']=='clustered-V4' and r['stage']=='C1')
    for case in pattern['large_D_small_L1_cases']:
        r=lookup[case,'clustered-V4'];lines.append(f"| {case} | {100*float(r['D']):.4f} | {100*float(r['delta_relative_l1']):+.4f} |")
    lines += ['', '在主C1的两布局中，按预设D≤1%、峰比例绝对误差增加≥0.1或峰位置误差增加≥一网格步(9.5238mm)的标记，没有对应案例。这不排除较小幅度变化，也不是对所有潜在峰值风险的否定。C0和辅助干预的全部标记见case_patterns.json。',
        '', '## C0与辅助指标：有益作用不是所有指标一致改善','',
        '| 布局 | 阶段 | 指标 | A均值→B均值 | 平均差(原单位) | 95%区间 |', '|---|---|---|---:|---:|---|']
    for condition in CONDITIONS:
        for stage in ('C0','C1'):
            for metric in METRICS:
                r=find(condition,'obs_token',metric,stage)
                lines.append(f"| {condition} | {stage} | {metric} | {r['baseline_mean']:.6f}→{r['intervention_mean']:.6f} | {r['mean']:+.6f} | {ci(r)} |")
    lines += ['', '聚簇C0相对L1差为+4.2331pp，C1为+1.8151pp，与下游步骤减轻输出扰动相容；这不能单独把减轻归因为反馈的因果补偿。聚簇C1质心距离反而平均降低约0.9241mm，L2与Dice50的差区间跨零，峰位置均值变化受少数离散argmax跳转影响。因此本报告限定“L1有益”，不称描述符对所有重建质量都有净收益。',
        '', '## 辅助路径与重新训练结构删除','',
        '| 布局 | 比较 | C1 ΔL1(pp) | 95%区间(pp) | 增加/降低/不变 |', '|---|---|---:|---|---|']
    for condition in CONDITIONS:
        for model in ('obs_bias','obs_both','uniform_pruned_observability_seed20261011','uniform_pruned_observability_seed20261019'):
            r=find(condition,model)
            lines.append(f"| {condition} | {model} | {100*r['mean']:+.4f} | {ci(r,100)} | {r['increased']}/{r['decreased']}/{r['unchanged']} |")
    lines += ['', '聚簇obs_bias的L1平均变化为−0.0272pp，35案例均降低，幅度很小；obs_both约+1.7881pp，与obs_token接近。两路径可非线性相互作用，不能将差值直接当成可加的独立贡献或把token全部归因于注意力。',
        '结构删除的普通V4两个种子分别+0.1299/+0.1691pp；聚簇draw0分别+0.6554/−1.6975pp。此处不是历史五draw平均+1.6902/−0.9806pp，后者未纳入本轮统计。不能根据某一个种子案例区间不跨零推断跨训练种子的稳定贡献。',
        '', '## 核验、异常与产物','',
        f"独立核验通过：用重采样案例计数矩阵乘配对差，以及独立排序插值计算bootstrap均值区间；与主结果最大绝对差{verify['maximum_statistical_difference']:.3g}。独立向量L1范数重算全部D，最大差{verify['maximum_D_difference']:.3g}。560份正式历史输出的全部所需指标重算差为0。全部预测在评分前及核验时分别复核SHA256。",
        '异常保留两项：首次审计发现仓库导出的reuse_methods.json没有SHA侧文件，改从有侧文件的scratch权威原件验证，日志保存在同级mrgr_observability_error_pairing_v1_attempt01；统计首次JSON导出遇到恒定差导致相关NaN，保留原始脚本、CSV与日志，登记statistics_export_amendment.json，仅改成无定义null。未变更指标、假设、干预、随机种子、重采样方案，未追加GT读取或模型前向。',
        '复现入口和产物目录：`scripts/observability_error_pairing_v1/README.md`、`docs/reports/artifacts/mrgr_observability_error_pairing_v1/`。protocol.json、GT64_authorization.json、prediction_manifest.json、predictions_sealed.json、checkpoint_audit.json、case_metrics.csv、paired_case_metrics.csv、summary.csv、statistics.json、correlations.csv、GT64_access.json、independent_verification.json、acceptance.json共同构成证据链。原预测不复制，不覆盖旧实验。',
        '本诊断仅说明当前冻结模型在这两种输入配置下对可观测性描述符输入的依赖及其对应L1影响；不能单独证明模型学到可泛化的几何规律，也不能证明结构不可替代。']
    REPORT.write_text('\n'.join(lines)+'\n')
    freeze(ART/'report_identity.json',dict(at=now(),report=str(REPORT),report_sha256=sha(REPORT),
        code_sha256=sha(HERE/'report.py'),matplotlib=matplotlib.__version__))
    files={str(p.relative_to(ROOT)):sha(p) for p in ART.rglob('*') if p.is_file()}
    files.update({str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')})
    files[str(REPORT.relative_to(ROOT))]=sha(REPORT)
    freeze(ART/'acceptance.json',dict(at=now(),complete=True,protocol_sha256=sha(ART/'protocol.json'),
        authorized_GT64_scoring=True,case_count=35,draws=[0],layouts=list(CONDITIONS),
        checkpoint_count=4,all_models_UniformFusion=True,reused_predictions=490,scored_fields=980,
        new_forwards=0,training=False,backward=False,optimizer_updates=0,GT128_read=False,
        independent_verification_passed=True,exploratory_used_test=True,independent_confirmation=False,
        report=str(REPORT),report_sha256=sha(REPORT),artifacts=files))
    print(REPORT)


if __name__=='__main__':
    backend();run()
