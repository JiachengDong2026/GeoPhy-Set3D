#!/usr/bin/env python3
"""Render evidence-only Chinese RVTC diagnosis and a proposed, unexecuted A/B protocol."""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def figures(output, summary):
    folder=output/'figures';folder.mkdir(exist_ok=True)
    rows=read_csv(output/'historical_epoch_metrics.csv')
    epochs=[int(row['epoch']) for row in rows]
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    axes[0].plot(epochs,[float(row['teacher_discrepancy']) for row in rows],color='#3b6caa',label='Within-epoch changing weights')
    axes[0].scatter([0,81],[summary['history']['fixed_parameter_evaluations'][key]['teacher_discrepancy'] for key in ('initial','final')],
                    color='#c94f32',label='Fixed weights (historical)')
    axes[0].set(xlabel='Epoch (81 marks final fixed evaluation)',ylabel='Normalized student–teacher discrepancy');axes[0].legend(fontsize=8)
    for key,label in (('sparse_H_l1','Students'),('teacher_H_l1','Rich teacher')):
        axes[1].plot(epochs,[float(row[key]) for row in rows],label=label)
    axes[1].axhline(1,color='grey',ls=':',label='Zero-volume baseline');axes[1].set(yscale='log',xlabel='Epoch',ylabel='H relative L1');axes[1].legend(fontsize=8)
    for key,label in (('student_mean','Student mean'),('teacher_mean','Teacher mean'),('teacher_std','Teacher std')):
        axes[2].plot(epochs,[float(row[key]) for row in rows],label=label)
    axes[2].set(yscale='log',xlabel='Epoch',ylabel='Predicted volume scale (no GT)');axes[2].legend(fontsize=8)
    fig.suptitle('Original gate: agreement, measurement fit and output scale describe different things')
    fig.savefig(folder/'historical_gate_dynamics.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),layout='constrained')
    groups=('gate_H','gate_U','Val-A_H')
    for ax,metric in zip(axes,('relative_l1','relative_l2')):
        for i,key in enumerate(groups):
            values=summary['measurement'][key][metric];mean=values['teacher_minus_student_mean'];lo,hi=values['paired_mean_delta_percentile_95ci']
            ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color='#3b6caa')
        ax.axhline(0,color='grey',ls=':');ax.set_xticks(range(3),('Trained H (n=8)','New U (n=8)','Val-A H (n=36)'))
        ax.set_ylabel('Teacher minus mean student '+metric)
    fig.suptitle('Paired case bootstrap 95% intervals; negative favors the teacher')
    fig.savefig(folder/'independent_teacher_measurements.png',dpi=170);plt.close(fig)
    grads=read_csv(output/'gradient_per_case.csv');fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),layout='constrained')
    for i,cohort in enumerate(('gate','Val-A')):
        selected=[row for row in grads if row['cohort']==cohort]
        axes[0].scatter([i]*len(selected),[float(row['cosine_measurement_consistency']) for row in selected],alpha=.7,s=20)
        axes[1].scatter([i]*len(selected),[float(row['consistency_to_measurement_sum_norm']) for row in selected],alpha=.7,s=20)
    axes[0].axhline(0,color='grey',ls=':');axes[0].set(ylabel='cos(weighted measurement, consistency)',ylim=(-1.05,1.05))
    axes[1].set_ylabel('Weighted consistency / measurement gradient norm')
    for ax in axes:ax.set_xticks([0,1],('Gate cases','Val-A cases'))
    fig.suptitle('Final checkpoint: unclipped gradients per case; no parameter update')
    fig.savefig(folder/'shared_parameter_gradients.png',dpi=170);plt.close(fig)


def proposed_protocol(output, summary):
    protocol=dict(status='DRAFT / NOT EXECUTED / requires registration before any future run',
        purpose='Separate rich-view measurement training benefit from the additional effect of volumetric teacher consistency',
        original_v2_no_go_retained=True,baseline_retained='SelfSup v1',
        scope='One matched limited engineering contrast; no formal300, GT, Pool B, final test, v3, EMA, added module/prior or sweep',
        arms=dict(A=dict(rich_branch=True,teacher_measurement_weights=dict(hold=1.,observed=.1),teacher_consistency_weight=0.),
                  B=dict(rich_branch=True,teacher_measurement_weights=dict(hold=1.,observed=.1),teacher_consistency_weight=.1)),
        shared=dict(model='Frozen GeoLift-OARI-MRGR-Res3D',parameters=332070,initialization='fresh random in both arms; verify identical state SHA256',
            seed=20261011,train_cases='Same original gate8; exact frozen case_station_protocol.json',
            train_views='same epoch0 A/B/H/T in both arms',epochs=80,optimizer_steps=320,
            batch_size=1,gradient_accumulation=2,precision='fp32',optimizer='AdamW',lr=.0002,weight_decay=.0001,
            gradient_clip_norm=1.,scheduler='none, matching the original gate',
            sparse_loss_weights=dict(hold=1.,cross=.5,observed=.1),
            projector=dict(samples=96,view_chunk=2,all_rays=True,output_resolution=64),
            data_order='identical original dataset order; no shuffle',evaluation_checkpoint='fixed final epoch80 in both arms; no best-checkpoint search',
            independent_evaluation='same fixed U4 for gate8 and Val-A A4/B4/H4/T8 lists; H/U never network inputs or validation updates'),
        engineering=dict(criteria=['all losses/outputs/raw gradients finite', 'zero forbidden access attempts',
            'unchanged parameter count/projector/shape; permutation and padding within original 1e-5 tolerance; all-invalid finite',
            'no saturation/constant-output collapse under original numerical checks',
            'correct detach, student output gradient direction and teacher measurement gradients; clipped optimization logged separately'],
            consistency_window_decrease='diagnostic only; not a runtime pass/fail condition',
            basis='numerical correctness and data-access rules; reuse original numerical tolerances, no threshold fitted to this diagnostic'),
        efficacy=dict(primary='B minus A mean sparse held-out relative L1 on the same Val-A36 final checkpoints',
            secondary=['paired relative L2','same-case U4 prediction error','teacher-versus-student measurements',
                       'within-arm absolute discrepancy and teacher denominator per case','unclipped shared gradient geometry'],
            uncertainty='paired case bootstrap 95% interval; no ray/view pseudo-replication; one seed, exploratory',
            interpretation='negative B-A supports incremental independent-measurement benefit; interval containing zero is inconclusive; positive signals harm',
            statistical_basis='directional benefit around zero (identity/no additional gain), conventional 95% paired uncertainty; no data-fitted magnitude threshold',
            veto='dynamic student-teacher discrepancy alone cannot decide method effectiveness',
            voxel_claims='Not tested by this GT-free contrast; cannot establish 3D distillation accuracy. Any later full method evaluation requires separately registered formal training and the original post-freeze GT boundary.'),
        required_before_execution=['review and register this draft, exact input IDs, evaluation rules and unchanged initialization/sampler/step checks',
            'freeze source/config hashes; preserve all old experiments; record every proposed rule before optimization'],
        continuation='Do not launch formal training or replace v1 from this diagnosis; use the limited contrast only to obtain the missing A/B incremental-benefit evidence')
    (output/'proposed_control_protocol.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False,allow_unicode=True))


def report(output, summary):
    h=summary['history'];fixed=h['fixed_parameter_evaluations'];minimum=h['minimum_training_discrepancy'];last=h['epoch80_training']
    replay=summary['final_checkpoint_replay']
    val_grad=summary['gradients']['Val-A']['cohort_mean_loss_gradient']
    val_projection=val_grad['cosine_measurement_consistency']*val_grad['consistency_to_measurement_sum_norm']
    table='| 统计口径 | 师生差异 D | 学生 H relative L1 | 教师 H relative L1 |\n|---|---:|---:|---:|\n'
    for label,row in (('轮内前10轮均值',None),('轮内后10轮均值',None),('初始固定参数（历史记录）',fixed['initial']),
                      ('第9轮轮内统计（D最低）',minimum),('第80轮轮内统计',last),('最终固定参数（历史记录）',fixed['final'])):
        if row is None:
            rows=read_csv(output/'historical_epoch_metrics.csv')[:10] if label.startswith('轮内前') else read_csv(output/'historical_epoch_metrics.csv')[-10:]
            d=h['first10_mean'] if label.startswith('轮内前') else h['last10_mean']
            s=np.mean([float(x['sparse_H_l1']) for x in rows]);t=np.mean([float(x['teacher_H_l1']) for x in rows])
        else:d=row['teacher_discrepancy'];s=row['sparse_H_l1'];t=row['teacher_H_l1']
        table+=f'| {label} | {d:.8f} | {s:.8f} | {t:.8f} |\n'
    measures='| 评估域 | n | 指标 | 平均学生 | 教师 | 教师−学生均值 | 中位数 | 教师胜率 | 配对均值95%CI |\n|---|---:|---|---:|---:|---:|---:|---:|---|\n'
    aliases=dict(gate_H='原 gate H（训练使用过）',gate_U='gate 新站位 U（已见案例）',**{'Val-A_H':'Val-A H（未见案例）'})
    for key,group in summary['measurement'].items():
        for metric in ('relative_l1','relative_l2'):
            r=group[metric];lo,hi=r['paired_mean_delta_percentile_95ci']
            measures+=f"| {aliases[key]} | {group['case_count']} | {metric} | {r['student_mean']:.6f} | {r['teacher_mean']:.6f} | {r['teacher_minus_student_mean']:.6f} | {r['teacher_minus_student_median']:.6f} | {r['teacher_win_rate']:.1%} | [{lo:.6f}, {hi:.6f}] |\n"
    worse='| 评估域 | 较差案例（L1排序，最多列3个；全量在summary/CSV） | 教师−学生 L1 |\n|---|---|---:|\n'
    for key,group in summary['measurement'].items():
        for row in group['relative_l1']['worse_cases'][:3]:worse+=f"| {aliases[key]} | {row['case_id']} | {row['teacher_minus_student']:+.6f} |\n"
    norms='| 案例组 | ||平均g学生|| | ||平均g教师|| | ||平均g一致性|| | ||平均g测量合计|| | 一致性/测量合计 |\n|---|---:|---:|---:|---:|---:|\n'
    cosines='| 案例组 | cos(学生,教师) | cos(学生,一致性) | cos(教师,一致性) | cos(测量合计,一致性) |\n|---|---:|---:|---:|---:|\n'
    percase='| 案例组 | cos(测量,一致性)均值 / 中位数 | 负余弦案例 | 一致性/测量范数比中位数 / 最大值 |\n|---|---|---|---|\n'
    for key,group in summary['gradients'].items():
        r=group['cohort_mean_loss_gradient'];stats=group['per_case_statistics'];c=stats['cosine_measurement_consistency'];ratio=stats['consistency_to_measurement_sum_norm']
        norms+=f"| {key} | {r['student_norm']:.6f} | {r['teacher_norm']:.6f} | {r['consistency_norm']:.6f} | {r['measurement_sum_norm']:.6f} | {r['consistency_to_measurement_sum_norm']:.2%} |\n"
        cosines+=f"| {key} | {r['cosine_student_teacher']:+.6f} | {r['cosine_student_consistency']:+.6f} | {r['cosine_teacher_consistency']:+.6f} | {r['cosine_measurement_consistency']:+.6f} |\n"
        percase+=f"| {key} | {c['mean']:+.6f} / {c['median']:+.6f} | {round(c['negative_fraction']*group['case_count'])}/{group['case_count']} | {ratio['median']:.2%} / {ratio['max']:.2%} |\n"
    report=f'''# SelfSup-MRGR v2（RVTC）失败原因诊断

**原 v2 工程 gate NO-GO 保持不变。** 本轮仅做无GT、无参数更新的诊断，未重训gate、未启动正式300轮、未创建v3。本报告与原门槛、原结果报告相互独立，不追溯改判。

审阅基准与开始时远端/本地HEAD均为 `52fab4fc6f40b60992e44386e81d304ed51a010d`，无更新差异需要合并。使用实际现有checkpoint `{summary['checkpoint']}`；SHA256：`{summary['checkpoint_sha256']}`。模型仍为332,070参数。诊断完成时间：{summary['completed_at']}（Asia/Shanghai）。

## 1. 判断与证据分级

- **已确认事实：** 未定位到v2的联合构造、measurement配对、detach或学生输出梯度方向实现错误。原门槛确实失败，但它检验的是动态归一化师生差异的窗口趋势，不是数值可运行性，也不是独立测量或三维有效性。最确定的问题是门槛解释范围过宽；这不改变按原协议得到的NO-GO。
- **已确认事实：** 新U和Val-A上教师有小幅平均测量优势。Val-A的L1胜率仅20/36（55.56%），16个案例更差，L1均值CI上界非常接近0；不能称逐案例稳定可靠的教师。L2平均优势相对更清楚。
- **已确认事实：** 在最终checkpoint，gate总体测量/一致性平均梯度同向，Val-A总体明显反向；一致性权重虽仅0.1，其梯度并非零，但没有范数支配测量项的证据。
- **解释假设：** 共享权重与持续变化的目标可能使一致性项阻碍部分未见案例的测量拟合，是需要A/B对照检验的风险。Val-A从未参与参数更新，不能把验证集梯度冲突直接当作历史gate训练失败的原因。
- **尚未验证：** 蒸馏相对于“仅增加富视角测量训练”的额外收益、实际三维体积准确性、整个训练期的梯度冲突轨迹及跨seed/视角稳定性。无法将本次证据唯一归因为教师不足或优化冲突。

## 2. 历史失败判据独立复核

完整读取并交叉核对80条 `gate_metrics.jsonl`、原 `gate_summary.json` 和80条 `teacher_training_diagnostics.csv`；CSV每项都与对应JSON一致。

{table}

**已确认事实：** `run_epoch(..., optimizer)`在每个sample前向后记录loss/预测，并每两sample执行一次优化（batch1/accumulation2）。每轮8个sample，包含4次更新；轮内均值混合多个不同参数状态，且同一共享网络的教师也在改变。`initial`/`final`来自无optimizer的固定参数整组评估。两者不能混用，第80轮0.098435/0.540274不是轮末checkpoint的0.092839/0.519349。

原门槛精确实现为“末10轮轮内D均值小于首10轮轮内D均值”，现为0.099864>0.036064，因此**预设工程判据未通过**。它不能证明detach失败、优化不收敛、教师泛化失败或体积更差。固定参数D从0.103018降至0.092839仅提供另一种诊断视角，不替换原判据。第9轮D极低而H误差9.566079远高于零体积基线1，说明单独追求低D可能奖励共同的测量不正确解。

**尚未验证/缺失：** 初始和中间checkpoint未被原gate入口保存。初始固定参数值是可信历史日志，本轮没有伪造/重新初始化一个“初始checkpoint”；只能复评实际存在的最终 `gate.pt`。本轮最终重放D={replay['teacher_discrepancy']:.8f}、学生H L1={replay['student_H_l1']:.8f}、教师H L1={replay['teacher_H_l1']:.8f}，与原fixed-final记录一致（rtol1e-5、atol1e-7）。

## 3. 分母和输出尺度

D_i=mean|C_student−C_T| / (mean|C_T|+1e-6)，先按sample归一化再平均。历史教师mean从初始固定参数{fixed['initial']['teacher_output']['mean']:.8f}变为最终{fixed['final']['teacher_output']['mean']:.8f}；第9轮为{minimum['teacher_mean']:.8f}，第80轮为{last['teacher_mean']:.8f}。第9/80轮教师std为{minimum['teacher_std']:.8f}/{last['teacher_std']:.8f}，反映体积尺度与空间变化同时改变。完整每轮学生/教师H L1/L2、mean/std和教师max保存于 `historical_epoch_metrics.csv`。

**解释假设：** 早期背景量级降低、后来空间结构变化与共享教师移动，都可能影响归一化D趋势；没有保存逐案例体积，不能唯一拆分历史变化的来源。

**不可恢复：** 历史日志只有平均比值和平均输出矩，不能恢复逐案例分母与绝对差异的配对或它们的协方差。E[r]E[d]一般不等于E[rd]，不能用“平均比值×平均分母”补算初始或第9轮的绝对体积差异。本轮只在真实最终checkpoint逐case直接计算了分子/分母：gate8平均绝对师生差异{replay['absolute_voxel_disagreement_mean']:.10f}、平均教师分母{replay['teacher_denominator_mean']:.10f}，并在CSV保存每个sample的原始量。这不是GT体积误差。

![历史动态](artifacts/selfsup_mrgr_v2_failure_diagnosis/figures/historical_gate_dynamics.png)

## 4. 富视角教师独立测量检验

**本轮实际执行：** gate8使用原冻结A/B/H/T，每case增加U4。U由按ID排序的Pool A剩余站位，以case-ID blake2b与seed2026101201构成SeedSequence，无放回确定性抽取；排除A∪B∪H，不按结果选站。Val-A使用全部36例，原sampler固定seed20261011/epoch0、A4/B4/H4/T8；与gate8 case IDs无交集。A/B/T是唯一网络输入，H/U仅投影目标。本轮没有将U、Val-A或其梯度用于更新参数。

“学生”是两个预测误差的平均 .5(error(C_A)+error(C_B))，不是对体积或投影做ensemble。三者针对完全相同的目标有效射线。relativeL1/L2沿用冻结实现，K96、all rays、view_chunk2。教师−学生为负表示教师更好。

{measures}

区间使用10,000次配对case bootstrap，seed2026101202；同一个case整行重采样，不把射线或多视角当独立样本。未作多重比较校正；这些是单checkpoint探索性区间，不能当最终显著性/跨seed确认，也未处理同一物理工况家族的额外相关性。

{worse}

完整52条域×case测量结果、A/B各自误差、每case分子/分母与输出统计均在 `measurement_per_case.csv`；全部44个case的A/B/H/T/U站位、seed、checkpoint哈希在 `case_station_protocol.json`。L1/L2的全部较差case均列于 `summary.json`。

**解释边界：** 原gate H已经参与训练loss，是样本内拟合证据；U是已见案例的新站位证据；Val-A才是未见案例证据。U与Val-A均有平均优势，但Val-A逐case优势弱且不普遍。即使独立测量改善，也不能证明富视角教师三维更准确。本轮没有读取任何GT。

![教师独立测量](artifacts/selfsup_mrgr_v2_failure_diagnosis/figures/independent_teacher_measurements.png)

## 5. 共享参数梯度：与detach检验分开

按原权重计算S=L_hold+.5L_cross+.1L_obs，T=L_hold_teacher+.1L_obs_teacher，D=.1L_teacher；M=S+T。每case通过 `autograd.grad` 计算全部共享参数的梯度向量；无optimizer、无`.backward()`累积、无裁剪、无参数/缓冲区更新。所有范数使用应用原权重后的原始梯度，余弦用双精度累积。Gate8与Val-A36全部计算，没有按梯度结果筛案例。

下两表是**先平均case梯度向量，再计算范数/余弦**，不是平均单case范数或平均余弦。

{norms}

{cosines}

再单独给出逐case分布，避免总体平均掩盖局部冲突：

{percase}

**已确认事实：** 全44例教师输出对一致性loss的梯度为零，学生输出梯度与(student−detached_teacher)的内积为正，教师measurement输出梯度非零。前两项只证明detach及输出方向正确；本次共享参数检查进一步发现Val-A上25/36例一致性与测量负余弦。Gate仅2/8例负余弦，总体向量同向。完整S/T/D两两余弦、范数、相对大小及branch检查在 `gradient_per_case.csv`。

**解释假设：** 最终共享网络在已见/未见案例上的梯度几何不同，可能存在蒸馏泛化风险。没有证据支持“一致性梯度过强并支配测量项”：总体一致性/测量范数为gate9.92%、Val-A4.21%，逐case最大约30.75%/32.42%。小范数反向项仍可能有影响，但影响强度不能由一个余弦直接推出。作为原始欧氏梯度的一阶方向计算，Val-A总体g_D在g_M上的投影/||g_M||为{val_projection:.2%}，且所有case的||g_D||/||g_M||<1；因此不能声称当前一致性项使合梯度整体反转测量下降方向。这个几何计算没有执行更新，也不等同于带矩估计/预条件的实际AdamW步。原gate checkpoint未保存optimizer矩状态，实际更新方向无法由现有权重恢复。

**尚未验证：** 这些只是最终checkpoint的局部一阶性质。Val-A梯度本轮未、历史gate也未用于更新；不能将其等同为实际训练冲突。缺失中间checkpoint，无法追溯冲突是否在训练期出现或造成什么后果；本轮也未做扰动参数/optimizer.step来声称因果机制。

![共享梯度](artifacts/selfsup_mrgr_v2_failure_diagnosis/figures/shared_parameter_gradients.png)

## 6. 下一步建议：仅一个有限对照，待登记、未执行

**结论：** 保留v1和原v2 NO-GO；目前没有充分证据启动正式训练或冻结RVTC为候选。独立教师存在小幅平均优势，却不逐case稳定；一致性存在未见案例局部冲突风险，却无全训练期因果证据。值得补充的唯一最小证据是“相同富视角测量loss下，关/开原一致性”的匹配A/B对照，不增加任何模块或搜索权重。

- 对照A：同一富视角分支与1.0 heldout/.1 observed教师测量loss，teacher consistency=0。
- 对照B：完全相同的富视角测量loss，开启原teacher consistency=.1。
- 两组均fresh random，seed20261011，验证初始state哈希一致；相同原gate8、固定epoch0 A/B/H/T、顺序无shuffle、80epochs×4steps=320updates、FP32、batch1/accumulation2、AdamW lr2e-4/wd1e-4、clip1、无scheduler（匹配原gate）；其余loss、projector和网络完全冻结。
- 使用同一预登记U和Val-A清单，固定epoch80最终checkpoint；不挑best epoch、不按结果重抽站位。该对照区分“富视角测量训练的收益”与“体积一致性项的额外收益”。它只能检验额外独立测量收益，仍不能单独确认三维蒸馏准确性。

新门槛建议拆开：**工程可运行性**检查有限数值、访问审计、正确gradient分支、既有集合安全/非塌缩检查，沿用原数值容差；D窗口变化仅记录诊断。**方法有效性**用同一Val-A case上B−A的稀疏H relativeL1作为主量，relativeL2/U为辅助，配对case95%CI表达不确定性；零为“无额外收益”的自然参照，未拟合本次数据的容易通过幅度阈值。负效应且区间整体小于0提供有限测量增益支持；区间含0应记不确定，明确反向则停止蒸馏路线。任何后续体积结论仍需另行预登记完整正式训练、measurement-only选模和冻结后GT边界。

完整待执行草案在 `proposed_control_protocol.yaml`，status为DRAFT/NOT EXECUTED。所有新规则与ID/source/config哈希必须在下一次优化前登记；不追溯替换旧门槛。本轮没有执行A/B任一组。

## 7. 审计、测试和不可变性

**本轮实际执行：** 9项针对性测试，覆盖历史口径、U的确定性排除、配对case重采样、冻结权重分解、零范数与反向余弦、精确checkpoint白名单/禁写、GT/Pool B阻断、输出detach与共享冲突的区别、H/U不输入网络及不可恢复量的防误算。单元测试使用明确fixture，不代替本轮44例真实实验。历史179项测试仅引用，未冒称本轮重跑。完整本轮日志 `targeted_tests.log`。

初版诊断入口把独立YAML误交给训练专用config loader，在加载checkpoint前即因缺少model键退出；已改为独立YAML解析，原错误日志保存在外部 `entry_config_error.log`。这是本轮诊断入口的问题，不是原v2实现问题；没有训练或权重操作。成功诊断的源码SHA256另存 `diagnostic_source_manifest.json`。

```json
{json.dumps(summary['gt_access'],indent=2,ensure_ascii=False)}
```

诊断使用0个worker，继承原V2ReadGuard所有限制并进一步把checkpoint限制为精确 `gate.pt` 且禁止写。`audit_hook_inherited_by_workers`为继承字段，当前不表示使用了worker。实际允许投影文件44个，全部Pool A；checkpoint打开计数包括前后hash与一次torch.load。

```json
{json.dumps(summary['integrity'],indent=2,ensure_ascii=False)}
```

原v2比较目录/仓库归档/gate日志逐文件SHA256未变，checkpoint SHA256前后未变，所有参数与缓冲区逐tensor相等，数据集仅做metadata完整性对比（不打开GT作hash）。模型、projector和原训练源码hash均保持冻结。没有有监督checkpoint、Pool B或最终test访问。

## 8. 复现与归档

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests \\
  /scratch/djc/envs/gas-torch/bin/python -m unittest discover -s tests -p 'test_rvtc_failure_diagnostics.py' -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
  /scratch/djc/envs/gas-torch/bin/python scripts/diagnose_selfsup_mrgr_v2_failure.py --config configs/diagnose_selfsup_mrgr_v2_failure.yaml
/scratch/djc/envs/gas-torch/bin/python scripts/report_selfsup_mrgr_v2_diagnosis.py
```

复评必须把独立config的output改为新的目录，保留现有诊断证据；缺失gate checkpoint会明确报错，不用合成数据替代，也不训练来补齐。渲染入口只读取结果。归档目录为 `docs/reports/artifacts/selfsup_mrgr_v2_failure_diagnosis/`，包含SHA256清单；不提交模型权重或数据。
'''
    (REPO/'docs/reports/selfsup_mrgr_v2_failure_diagnosis.md').write_text(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/data/djc/GeoPhy-Set3D/outputs/diagnostics/selfsup_mrgr_v2_failure_v1'))
    args=parser.parse_args();summary=json.loads((args.output/'summary.json').read_text())
    assert summary['completed'] and summary['integrity']['optimizer_steps']==0
    proposed_protocol(args.output,summary);figures(args.output,summary);report(args.output,summary)


if __name__=='__main__':main()
