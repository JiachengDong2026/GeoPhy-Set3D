"""Report all planned contrasts; never promote a model or select checkpoints."""
from stage import *
import csv


def create_report():
    assert verified(BASE/'independent_verification.json')['passed']
    stat=verified(BASE/'statistics.json');rr=stat['summaries'];bench=verified(BASE/'benchmark.json')
    models=verified(BASE/'model_manifest.json');access=verified(BASE/'GT64_access.json')
    def row(model,condition,draws,stage='C1',metric='relative_l1'):
        return next(r for r in rr if (r['model'],r['condition'],r['draws'],r['stage'],r['metric'])==(model,condition,draws,stage,metric))
    def interval(r,factor=100):return f"[{factor*r['ci95'][0]:+.4f}, {factor*r['ci95'][1]:+.4f}]"
    def direction(x):return '增加' if x>1e-8 else '降低' if x < -1e-8 else '基本不变'
    mainplain=[row(f'{PLAIN}_seed{s}',c,ds) for s in (20261011,20261019) for c,ds in [('Val-A-V4',[0]),('clustered-V4',list(range(5)))]]
    if all(r['mean']>0 and r['ci95'][0]>0 for r in mainplain):
        recommendation='两种子、两布局的普通交互−OARI平均L1差均为正，且案例区间均在零以上；在本轮限定条件下，支持保留OARI。仍不能把两个固定种子的结果称为训练种子总体的稳定性确认。'
    elif all(r['mean']<=0 for r in mainplain):
        recommendation='普通交互在两种子、两布局的平均L1均未劣于OARI；若成本也降低，可优先考虑更简单模型作为后续论文候选。但未预登记性能等效界限，区间跨零不能证明严格等效，且辅助指标仍需共同评估；本阶段不自动替换正式模型。'
    else:
        recommendation='普通交互与OARI的平均L1优势未在两种子、两布局中形成统一强证据；不能据此声称OARI专门结构具有稳定独立增益。若普通交互成本更低，可列为简化候选；尚不足以直接宣布其等效或全面替代OARI，本阶段保留正式模型身份。'
    auxiliary=[]
    for metric in ('relative_l2','absolute_peak_ratio_error','centroid_error_mm','Dice50'):
        contrasts=[row(f'{PLAIN}_seed{s}',c,ds,metric=metric) for s in (20261011,20261019)
                   for c,ds in [('Val-A-V4',[0]),('clustered-V4',list(range(5)))]]
        oriented=[r['mean']*(-1 if metric=='Dice50' else 1) for r in contrasts]
        auxiliary.append(f"{metric}的四个配对比较中，均值方向支持OARI/普通交互/数值持平为"
                         f"{sum(v>1e-8 for v in oriented)}/{sum(v < -1e-8 for v in oriented)}/{sum(abs(v)<=1e-8 for v in oriented)}")
    cost={r['key']:r for r in bench['records']}
    cost_text='；'.join(f"种子{s}的普通交互/OARI平均耗时比{cost[f'{PLAIN}_seed{s}']['mean_seconds']/cost[f'uniform_fusion_seed{s}']['mean_seconds']:.3f}，"
                       f"峰值分配显存比{cost[f'{PLAIN}_seed{s}']['peak_allocated_bytes']/cost[f'uniform_fusion_seed{s}']['peak_allocated_bytes']:.3f}"
                       for s in (20261011,20261019))
    lines=['# OARI 四项定向实验：普通集合交互、关系输入第二种子与冻结token复核','',
        '本报告比较同一已使用35测试案例上的探索性结果，不是独立测试确认。所有完整对照与训练变体都采用 **Uniform Fusion**；正式模型、历史checkpoint、原预测与历史协议均未改写。',
        '', '## 结论与论文建议','',recommendation,'',
        '辅助终点核对：'+'；'.join(auxiliary)+'。这些方向计数只是四个已登记比较的描述，不是四个独立样本，也不替代各自区间。',
        cost_text+'。是否优先简化须同时考虑上面的L1、辅助终点与成本；本轮没有预登记等效界限，不据单项优势宣称全面替代。','',
        '普通集合交互仍接收原4维几何描述符及正确GeometryLifting，因此它检验OARI专门设计的增益，不是去掉全部几何信息的对照。冻结token实验检验已训练模型的当前依赖，不能直接等同重新训练后的结构贡献。',
        '', '## 实现差异与工程检查','',
        '独立变体uniform_plain_set_interaction仅替换C0前交互：GeometryConditionedSetInteraction接收features、geometry、visibility和view_mask，跳过OARI射线关系、局部描述符、可观测性token与pairwise bias。原Encoder、GeometryLifting、几何描述符、多尺度decoder、Uniform Fusion及Physics Residual Feedback保留。原MRGR.forward通过C3 MRO调用新的PlainSetStage0，反馈代码逐字未改。',
        '从官方工厂构造同种子新鲜模型后，形状兼容的交互子层保留其新鲜初始化；48→32 token_projection独立初始化，不加载56→32不兼容参数，构造恢复RNG。复制不涉及任何训练checkpoint。共享前置模块与解码/反馈的初始化逐张量一致。训练后共享模块自然可能学出不同权重，未要求其训练后输出相同。',
        '普通交互共342209参数，对照342725，减少516（约0.151%）。这是完整交互设计替换，参数量并未严格匹配，不能把所有差值归为某一几何通道的唯一因果贡献。no_ray_relations仍为342725参数，完全沿用首仿射输入前三关系通道置零定义，保留描述符与learned bias。',
        'CPU门槛在两个种子验证共享初始化、第一次GeometryLifting和几何描述符逐元素相同；mask中的NaN隔离、padding不变、排列等变、全无效交互输出为零、完整模型有限/[0,1]、接口与梯度通过。普通attention没有伪造OARI bias。GPU smoke使用独立新鲜模型与单个正式双输入batch，完整损失反传和两次优化通过；其状态没有用于正式训练。',
        '', '## 训练、数据与调度','',
        '三项训练均完整300轮、24900次提交更新，166训练/36验证案例，batch2、AdamW lr2e-4/wd1e-4、cosine300、clip1、FP32、TF32关闭；完全复用原正式训练引擎和损失系数。资源中断时未提交轮的部分计算不保留，按既定epoch-commit机制重放，因此24900指最终轨迹的提交更新数。仅以完整训练期间验证投影H4严格最小值选轮，不按GT选checkpoint，不提前按结果停轮。',
        '启动前GPU1/2被其他用户使用，只在通过UUID锁、间隔15秒双快照、进程/显存/利用率/主机检查的卡上启动任务。沿用初始2并发、首两训练各≥3轮且主机资源充足后最多4并发规则；按每项单GPU规则调度，锁FD由worker继承。GPU1折间短暂无CUDA上下文，第三训练取得锁并启动后，另一用户已启动但延迟初始化的下一折任务建立CUDA上下文，发生短暂共卡。检测后仅中断/清理本方第三训练及其数据worker，保留完整第4轮last.pt与全部日志；排除GPU1/2的外部折间空档，待原两训练完成后在GPU0/3按原引擎恢复相同随机数、优化器和调度器状态，继续至完整300轮，没有终止其他用户进程。实际时间线、进程/锁、恢复校验见recovery及scheduling；中断轮重放属于既定epoch-commit恢复，未选择性丢弃已提交轮。',
        '三训练模型各普通draw0 35份＋聚簇draw0..4 175份，共630新增预测。seed20261019冻结obs_token仅两布局draw0共70新增预测，另有两个固定工程no-op前向。复用完整两种子、no-ray种子11各210份，以及此前obs_token种子11的70份，共700历史预测。共1400不同预测文件、2800 C0/C1字段。',
        '新冻结模型为正式uniform_fusion_seed20261019，严格验证checkpoint哈希与官方工厂加载；直接载入原OARI v2 Intervention实现，移除描述符首仿射的五维输入贡献，保留learned bias/SiLU，未训练。先搜索可核验的seed19输出，缺失才补算；两布局首案例no-op与历史C0/C1在登记容差内一致。所有输入、参数和checkpoint哈希保持不变，始终eval/无梯度。',
        '本轮用户明确授权在既有35案例范围评估本阶段结果。三项训练完成、选定checkpoint冻结且全部新增预测统一封存后，才首次读取本阶段GT64；只加载35份GT64，其SHA256与既有记录一致，GT128始终禁止。',
        '', '| 模型 | epoch | 参数 | SHA256 |','|---|---:|---:|---|']
    for m in models:lines.append(f"| {m['key']} | {m['best_epoch']} | {m['parameters']} | `{m['checkpoint_sha256']}` |")
    lines += ['', '## 统计与指标','',
        '差值=处理误差−同种子完整模型误差，正值表示误差增加；绝对比例差×100即百分点。两种子分别报告，每个区间统计单位均35案例。聚簇正式训练比较先在每案例内平均5个draw，再进行20000次配对bootstrap，固定seed20261026，百分位95%区间估计平均差；draw0单独分析，不与五draw平均混用。区间未经多重校正，均为探索性案例区间，不能解释为训练种子总体不确定性。SD使用ddof1；|Δ|≤1e-8仅表示数值基本不变，不是性能等效界限。',
        '直接使用原正式评估的volume_metrics/safe_shape，原始[0,1]质量分数、XYZ端点网格0..0.6m、float64，无每案例重新归一化。相对L1分母max(Σabs(GT),1e-6)，相对L2平方分母max(ΣGT²,1e-12)再开方；峰幅比=max(pred)/max(GT)，峰比例绝对误差=abs(比值−1)，峰值绝对误差为质量分数差。峰位置距离mm、并列峰采用首个XYZ节点。Dice50双方共用0.5×GTmax阈值。保留梯形积分质心centroid_error_mm与普通节点质心centroid_distance_mm，两者不混用。Dice差正值表示改善，峰幅比本身无单调优劣方向。',
        '', '## 普通集合交互与OARI：正式训练配对','',
        '| 种子 | 布局 | draw | OARI C1 L1(%) | 普通交互(%) | ΔL1(pp) | 95%区间(pp) | 中位数(pp) | 增/降/不变 |',
        '|---|---|---|---:|---:|---:|---|---:|---|']
    for s in (20261011,20261019):
        for c,ds in [('Val-A-V4',[0]),('clustered-V4',list(range(5)))]:
            r=row(f'{PLAIN}_seed{s}',c,ds)
            lines.append(f"| {s} | {c} | {','.join(map(str,ds))} | {100*r['baseline_mean']:.4f} | {100*r['treatment_mean']:.4f} | {100*r['mean']:+.4f} | {interval(r)} | {100*r['median']:+.4f} | {r['increased']}/{r['decreased']}/{r['unchanged']} |")
    lines.append('')
    for c,ds in [('Val-A-V4',[0]),('clustered-V4',list(range(5)))]:
        a,b=[row(f'{PLAIN}_seed{s}',c,ds) for s in (20261011,20261019)]
        interpretation=('两种子的均值均支持OARI误差更低' if a['mean']>0 and b['mean']>0 else
                        '两种子的均值均支持普通交互误差更低' if a['mean']<0 and b['mean']<0 else
                        '两个种子没有形成一致的均值变化方向')
        lines.append(f"{c}：替换OARI后，两种子C1 L1分别{direction(a['mean'])}{abs(100*a['mean']):.4f}和{direction(b['mean'])}{abs(100*b['mean']):.4f}个百分点；{interpretation}，证据强度以各自案例区间为限。")
    lines += ['', '普通交互替换后的方向是否一致，须分别看上表四个对比；不能以其中最有利一行替代全局判断。下方同时列出C0与所有辅助指标，并与成本一起评估。',
        '', '## 射线关系输入：第二种子是否复现','',
        '| 种子 | 布局 | draw | no-ray−完整C1 L1(pp) | 95%区间(pp) | 中位数(pp) | 增/降/不变 |',
        '|---|---|---|---:|---|---:|---|']
    for s in (20261011,20261019):
        for c,ds in [('Val-A-V4',[0]),('clustered-V4',list(range(5)))]:
            r=row(f'uniform_no_ray_relations_seed{s}',c,ds)
            lines.append(f"| {s} | {c} | {','.join(map(str,ds))} | {100*r['mean']:+.4f} | {interval(r)} | {100*r['median']:+.4f} | {r['increased']}/{r['decreased']}/{r['unchanged']} |")
    a=row('uniform_no_ray_relations_seed20261011','clustered-V4',list(range(5)))
    b=row('uniform_no_ray_relations_seed20261019','clustered-V4',list(range(5)))
    relation_interpretation=('两个种子的均值均支持保留射线关系输入；其收益的描述性方向在第二种子复现，但证据强度仍以各自案例区间为限。' if a['mean']>0 and b['mean']>0 else
        '两个种子的均值均为移除射线关系后误差降低，本轮不支持关系输入具有聚簇独立收益。' if a['mean']<0 and b['mean']<0 else
        '未得到一致方向复现，不能声称关系输入有稳定的聚簇独立收益。')
    lines += ['', f"聚簇五draw平均下，两个种子的no-ray差值方向分别为{direction(a['mean'])}/{direction(b['mean'])}。"+relation_interpretation+
        '两个固定种子不足以确认训练种子总体的稳定性。',
        '', '## 冻结obs_token：两个种子仅draw0对齐','',
        '| 种子 | 布局 | ΔC1 L1(pp) | 95%区间(pp) | 中位数(pp) | 增/降/不变 | C1 D均值(%) |',
        '|---|---|---:|---|---:|---|---:|']
    for s in (20261011,20261019):
        for c in ('Val-A-V4','clustered-V4'):
            r=row(f'obs_token_seed{s}',c,[0]);d=row(f'obs_token_seed{s}',c,[0],metric='D')
            lines.append(f"| {s} | {c} | {100*r['mean']:+.4f} | {interval(r)} | {100*r['median']:+.4f} | {r['increased']}/{r['decreased']}/{r['unchanged']} | {100*d['mean']:.4f} |")
    a=row('obs_token_seed20261011','clustered-V4',[0]);b=row('obs_token_seed20261019','clustered-V4',[0])
    lines += ['', ('第二种子聚簇C1 L1的平均差与第一种子同为正，且其95%案例区间也在零以上，支持在这两个冻结模型中复现有益token依赖。' if a['mean']>0 and b['ci95'][0]>0 else
        '第二种子的聚簇冻结token效应未达到与第一种子相同的明确恶化证据；应保留均值、区间和方向计数，不把预测改变本身称为性能恶化。'),
        '冻结obs_token仅屏蔽token首仿射输入贡献，保留pairwise描述符路径和全部训练权重；普通集合交互则重新训练且同时移除了专门关系/描述符结构。两者不能用作完全相同的干预解释。即使冻结恶化而普通交互接近，也只能与补偿解释相容，不能排除训练随机性、优化变化或冻结干预的分布偏移。',
        '', '## 固定硬件推理成本','',bench['protocol'],'',
        '| 模型 | 参数 | 平均前向(ms) | 中位数(ms) | SD(ms) | 峰值分配显存(MiB) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in bench['records']:lines.append(f"| {r['key']} | {r['parameters']} | {1000*r['mean_seconds']:.3f} | {1000*r['median_seconds']:.3f} | {1000*r['sd_seconds']:.3f} | {r['peak_allocated_bytes']/1024**2:.2f} |")
    lines += ['', '九次计时来自同一GPU上的三个固定验证输入，不视作独立模型性能样本；GPU UUID与逐次计时保存在benchmark.json。普通交互跳过特征值分解和关系构造，实际节省以上表为准；只有成本降低并且误差、峰值、位置、Dice及两个种子都可接受，才有依据优先选择更简单模型。没有预登记等效容差，因此不以“不显著”作为等效证据。',
        '', '## 全部C0/C1辅助终点（差值原单位）','',
        '| 模型 | 布局/draw | 阶段 | 指标 | 完整均值 | 处理均值 | 平均差 | 中位数差 | SD差 | 95%区间 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in rr:
        if r['metric']=='D':continue
        # All registered draw protocols, including draw0 counterparts, are kept.
        lines.append(f"| {r['model']} | {r['condition']}/{','.join(map(str,r['draws']))} | {r['stage']} | {r['metric']} | {r['baseline_mean']:.6f} | {r['treatment_mean']:.6f} | {r['mean']:+.6f} | {r['median']:+.6f} | {r['sd']:.6f} | {interval(r,1)} |")
    verification=verified(BASE/'independent_verification.json')
    lines += ['', '## 审计、限制与交付','',
        f"所有{access['archived_fields_verified']}份可与历史核对的C0/C1字段，其全部指标最大绝对差为{access['maximum_archived_difference']:.3g}。独立实现用案例计数权重、逐draw差先平均及独立排序插值复核bootstrap，最大差为{verification['maximum_difference']:.3g}；全部1400预测和全部checkpoint哈希再次核验。",
        '首次登记后在静态检查中发现smoke目标返回volumes是tuple，修正为直接迭代，旧登记完整保留在mrgr_oari_targeted_v1_registration_attempt01；正式训练前重新登记，未改变任何训练预算、损失或正式权重。另有上述GPU折间启动竞争与第三训练的受控恢复，原失败日志、退出码、last.pt身份和恢复证据完整保留；不能把该运行称为从未中断。其余异常若有保留在controller/status和scheduling日志中，不静默替换模型、精度或数据。',
        '产物包含：独立设计、训练protocol/config/结构/源代码哈希、CPU/GPU工程检查、完整300轮日志与checkpoint SHA256、GPU资源和锁记录、1400份预测身份与文件哈希、统一封存、35份GT64访问审计、逐案例逐draw指标、按案例配对统计、同卡成本比较、独立核验及acceptance.json。大数组/checkpoint保留scratch，仓库导出其身份和全部分析证据，不复制或覆盖历史大文件。',
        '本轮不自动改变论文正式模型或重选检查点。最终结论限于两个固定训练种子、两类布局与已使用35案例；不能单独证明模型掌握可泛化几何规律或某专门模块不可替代。']
    assert not REPORT.exists()
    REPORT.write_text('\n'.join(lines)+'\n')
    print(REPORT,flush=True)
