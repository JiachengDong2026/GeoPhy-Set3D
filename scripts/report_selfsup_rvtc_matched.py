#!/usr/bin/env python3
"""Render and archive completed matched RVTC measurements; never optimize."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_selfsup_rvtc_matched import configs,verify,guard_for,checkpoints,read_lines,write_csv
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import write_json
from geophy_set3d.data.build_utils import digest
from geophy_set3d.training.selfsup_v2 import audit_is_clean


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def validate_results(pair,base,root,summary):
    p=verify(pair,base,root);states=[];ckpts={};g=guard_for(root,base)
    with g:
        for arm in ('Control','Treatment'):
            initial=torch.load(root/'checkpoints'/arm/'initial.pt',map_location='cpu',weights_only=False)
            assert initial['updates']==0 and initial['initialization']=='fresh random'
            assert state_hash(initial['model'])==p['initial_state_sha256']
            states.append(initial['model'])
            rows=read_lines(root/arm/'metrics.jsonl');gradient=read_lines(root/arm/'gradient_trajectory.jsonl')
            assert [r['epoch'] for r in rows]==list(range(1,81)) and [r['updates'] for r in rows]==list(range(4,321,4))
            manifest=json.loads((root/'case_station_protocol.json').read_text())
            assert all(r['train']['case_order']==manifest['train_order'] and len(r['train']['steps'])==4 for r in rows)
            assert len(gradient)==96
            for epoch in range(10,81,10):
                subset=[r for r in gradient if r['epoch']==epoch]
                assert [r['case_id'] for r in subset]==p['gradient_cases']
                if arm=='Control':assert all(r['applied_consistency_norm']==0 for r in subset)
            for path in checkpoints(root,arm):
                assert path.exists()
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['arm']==arm and saved['protocol_sha256']==digest(root/'protocol.json')
                if saved['epoch']:
                    assert saved['updates']==4*saved['epoch'] and 'optimizer' in saved and 'rng' in saved
                    assert state_hash(saved['model'])==saved['state_sha256']
                    assert max(float(s['step']) for s in saved['optimizer']['state'].values())==saved['updates']
                ckpts[str(path.relative_to(root))]=dict(sha256=digest(path),size_bytes=path.stat().st_size,
                    epoch=saved['epoch'],updates=saved['updates'],state_sha256=state_hash(saved['model']))
                if path.name.startswith('epoch_'):
                    assert all(r['checkpoint_sha256']==ckpts[str(path.relative_to(root))]['sha256'] for r in gradient if r['epoch']==saved['epoch'])
            assert ckpts[f'checkpoints/{arm}/epoch_080.pt']['sha256']==summary['training'][arm]['final_checkpoint_sha256']
        assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
        assert audit_is_clean(g.summary())
    initial=[json.loads((root/arm/'initial_fixed_evaluation.json').read_text()) for arm in ('Control','Treatment')]
    assert initial[0]['measurement']==initial[1]['measurement']
    assert initial[0]['output']==initial[1]['output'] and initial[0]['teacher_output']==initial[1]['teacher_output']
    assert all(initial[0]['loss'][k]==initial[1]['loss'][k] for k in initial[0]['loss'] if k!='total')
    sessions=read_lines(root/'access_sessions.jsonl');assert all(audit_is_clean(s) and not s['failed'] for s in sessions)
    write_json(root/'checkpoint_manifest.json',ckpts)
    match=dict(registered_before_optimization=True,registration_commit='a2484c3',protocol_sha256=digest(root/'protocol.json'),
        initial_states_tensor_equal=True,initial_state_sha256=p['initial_state_sha256'],only_arm_and_weight_differ=True,
        initial_measurements_outputs_and_loss_parts_equal=True,
        same_train_case_station_order_all_epochs=True,epochs_per_arm=80,updates_per_arm=320,
        final_checkpoint_only=True,own_optimizer_rng_resume_states_verified=True,
        gradient_epochs_verified=list(range(10,81,10)),diagnostic_cases_per_checkpoint=12,
        gradient_rows_per_arm=96,Control_applied_consistency_gradient_zero=True,
        forbidden_access_attempts_zero=True,all_sessions_successful=True,volumetric_GT_opened=False)
    write_json(root/'matching_verification.json',match)
    return match


def figures(root,summary,paired):
    out=root/'figures';out.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for arm,color in [('Control','tab:blue'),('Treatment','tab:orange')]:
        rows=read_lines(root/arm/'metrics.jsonl')
        for ax,field,label in [(axs[0],('measurement','heldout','relative_l1'),'Gate H L1 (within epoch)'),
                              (axs[1],('loss','teacher'),'Dynamic student-teacher D'),
                              (axs[2],('teacher_output','mean'),'Teacher mean output')]:
            values=[]
            for row in rows:
                value=row['train']
                for key in field:value=value[key]
                values.append(value)
            ax.plot(range(1,81),values,label=arm,color=color);ax.set_title(label);ax.set_xlabel('Epoch');ax.set_yscale('log');ax.legend()
    fig.savefig(out/'training_dynamics.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,cohort,domain,label in [(axs[0],'Val-A','H','Unseen cases: H L1'),(axs[1],'Val-A','H','Unseen cases: H L2'),(axs[2],'gate','U','Seen cases: new U L1')]:
        metric='relative_l2' if ax==axs[1] else 'relative_l1'
        rows=[r for r in paired if r['cohort']==cohort and r['domain']==domain]
        delta=np.array([r['Treatment_minus_Control_'+metric] for r in rows])
        ax.scatter(range(1,len(delta)+1),delta,c=np.where(delta<0,'tab:green','tab:red'))
        ax.axhline(0,color='k',lw=1);ax.axhline(delta.mean(),color='tab:blue',ls='--',label='mean')
        ax.set_title(label);ax.set_xlabel('Frozen case order');ax.set_ylabel('Treatment - Control');ax.legend()
    fig.savefig(out/'paired_measurements.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for arm,color in [('Control','tab:blue'),('Treatment','tab:orange')]:
        rows=read_lines(root/arm/'gradient_trajectory.jsonl')
        for cohort,style in [('gate','-'),('Val-A','--')]:
            for ax,key,label in [(axs[0],'applied_consistency_to_measurement_sum_norm','Applied ||gD|| / ||gM||'),
                                 (axs[1],'hypothetical_0p1_cosine_measurement_consistency','cos(gM, hypothetical 0.1 gD)'),
                                 (axs[2],'hypothetical_0p1_consistency_to_measurement_sum_norm','Hypothetical 0.1 ||gD|| / ||gM||')]:
                values=[np.mean([r[key] for r in rows if r['epoch']==e and r['cohort']==cohort]) for e in range(10,81,10)]
                ax.plot(range(10,81,10),values,color=color,ls=style,label=arm+' '+cohort);ax.set_title(label);ax.set_xlabel('Fixed checkpoint epoch');ax.legend(fontsize=8)
    axs[1].axhline(0,color='k',lw=1)
    fig.savefig(out/'gradient_trajectory.png',dpi=150);plt.close(fig)


def report(root,summary,match):
    effects=summary['effects'];primary=summary['primary'];training=summary['training']
    names={'extra_measurement_benefit':'支持本次有限实验的额外测量收益','measurement_harm':'支持本次有限实验的测量性能受损','inconclusive':'结果不确定'}
    decision=names[primary['decision']]
    recommendation='本次结果不支持一致性提供未见案例的额外测量收益；建议暂停原体积一致性蒸馏的推进，保留v1。' if primary['decision']=='measurement_harm' else '保留v1；这一有限结论不自动授权后续训练。'
    effect_rows=[]
    for group,label in [('Val-A_H','Val-A H（未见案例）'),('gate_U','gate U4（已见案例新站位）'),('gate_H','gate H（训练使用过）')]:
        for metric in ('relative_l1','relative_l2'):
            s=effects[group][metric];lo,hi=s['paired_case_95ci']
            effect_rows.append([label,metric,s['case_count'],f"{s['control_mean']:.8f}",f"{s['treatment_mean']:.8f}",f"{s['mean_delta']:+.8f}",f"{s['median_delta']:+.8f}",f"{s['treatment_wins']}/{s['treatment_losses']}/{s['ties']}",f'[{lo:+.8f}, {hi:+.8f}]',names[s['decision']]])
    effect_table=table(['域','指标','n','Control','Treatment','T−C均值','中位数','胜/负/平','配对95%CI','规则结论'],effect_rows)
    auxiliary_notes='；'.join(label+'：'+names[effects[group][metric]['decision']] for group,metric,label in
        [('Val-A_H','relative_l2','Val-A L2'),('gate_U','relative_l1','U L1'),('gate_U','relative_l2','U L2')])
    agree=all(effects[group][metric]['decision']==primary['decision'] for group,metric in
        [('Val-A_H','relative_l2'),('gate_U','relative_l1'),('gate_U','relative_l2')])
    auxiliary_notes+='。辅助判据与主指标'+('一致。' if agree else '并非全部一致，不能合并成统一有效性结论。')
    teacher_rows=[]
    for arm in ('Control','Treatment'):
        for group in ('Val-A_H','gate_U','gate_H'):
            for metric in ('relative_l1','relative_l2'):
                s=summary['teacher_advantage'][arm+'_'+group][metric];lo,hi=s['paired_mean_delta_percentile_95ci']
                teacher_rows.append([arm,group,metric,f"{s['student_mean']:.8f}",f"{s['teacher_mean']:.8f}",f"{s['teacher_minus_student_mean']:+.8f}",f"{s['teacher_win_rate']:.1%}",f'[{lo:+.8f}, {hi:+.8f}]'])
    teacher_table=table(['组','域','指标','稀疏学生','富视角教师','教师−学生','教师胜率','配对95%CI'],teacher_rows)
    gradient_rows=[];final_gradient_rows=[];step_rows=[];trajectory_notes=[]
    for arm in ('Control','Treatment'):
        gr=read_lines(root/arm/'gradient_trajectory.jsonl')
        for epoch in range(10,81,10):
            for cohort in ('gate','Val-A'):
                rows=[r for r in gr if r['epoch']==epoch and r['cohort']==cohort]
                cos=[r['hypothetical_0p1_cosine_measurement_consistency'] for r in rows]
                ratio=[r['hypothetical_0p1_consistency_to_measurement_sum_norm'] for r in rows]
                applied=[r['applied_consistency_to_measurement_sum_norm'] for r in rows]
                gradient_rows.append(dict(arm=arm,epoch=epoch,cohort=cohort,case_count=len(rows),
                    mean_cosine=float(np.mean(cos)),median_cosine=float(np.median(cos)),negative_cases=sum(v<0 for v in cos),
                    mean_hypothetical_ratio=float(np.mean(ratio)),max_hypothetical_ratio=float(np.max(ratio)),mean_applied_ratio=float(np.mean(applied))))
                if epoch==80:final_gradient_rows.append([arm,cohort,len(rows),f'{np.mean(cos):+.5f}',f'{np.median(cos):+.5f}',sum(v<0 for v in cos),f'{np.mean(ratio):.2%}',f'{np.mean(applied):.2%}'])
        for r in read_lines(root/arm/'metrics.jsonl'):
            for step in r['train']['steps']:step_rows.append(dict(arm=arm,epoch=r['epoch'],update=(r['epoch']-1)*4+step['update_in_epoch'],**step))
        for cohort in ('gate','Val-A'):
            subset=[r for r in gradient_rows if r['arm']==arm and r['cohort']==cohort]
            raw=[r for r in gr if r['cohort']==cohort]
            trajectory_notes.append(f"{arm} {cohort}：假设原0.1梯度的逐case平均余弦由epoch10的{subset[0]['mean_cosine']:+.5f}变为epoch80的{subset[-1]['mean_cosine']:+.5f}，负余弦共{sum(r['hypothetical_0p1_cosine_measurement_consistency']<0 for r in raw)}/{len(raw)}条；逐case范数比范围{min(r['hypothetical_0p1_consistency_to_measurement_sum_norm'] for r in raw):.2%}–{max(r['hypothetical_0p1_consistency_to_measurement_sum_norm'] for r in raw):.2%}。")
    write_csv(root/'gradient_trajectory_summary.csv',gradient_rows);write_csv(root/'optimizer_steps.csv',step_rows)
    grad_table=table(['组','域','n','余弦均值','中位数','负余弦数','假设0.1范数比均值','实际范数比均值'],final_gradient_rows)
    trajectory_text='\n\n'.join(trajectory_notes)
    windows=table(['组','前10轮动态D','后10轮动态D','最终固定D','最终学生H L1','最终教师H L1'],[[arm,f"{s['consistency_first10']:.8f}",f"{s['consistency_last10']:.8f}",f"{s['final_fixed_evaluation']['loss']['teacher']:.8f}",f"{s['final_fixed_evaluation']['measurement']['heldout']['relative_l1']:.8f}",f"{s['final_fixed_evaluation']['measurement']['teacher_heldout']['relative_l1']:.8f}"] for arm,s in training.items()])
    scales=table(['组','稀疏mean / std','教师mean / std','非塌缩','集合检查'],[[arm,f"{s['final_fixed_evaluation']['output']['mean']:.8f} / {s['final_fixed_evaluation']['output']['std']:.8f}",f"{s['final_fixed_evaluation']['teacher_output']['mean']:.8f} / {s['final_fixed_evaluation']['teacher_output']['std']:.8f}",str(not s['output_collapsed']),str(all(p['passed'] for p in s['set_properties'].values()))] for arm,s in training.items()])
    lo,hi=primary['paired_case_95ci']
    content=f'''# RVTC 严格匹配 Control / Treatment 有限对照

**按优化前登记规则，主指标结论：{decision}。** Val-A36 稀疏 H relative L1 的 Treatment−Control 均值为 {primary['mean_delta']:+.8f}，95% 配对 case bootstrap 区间 [{lo:+.8f}, {hi:+.8f}]。本结论限定于一次单seed、gate8/80轮训练后的独立测量，不是三维准确性结论。保留 SelfSup v1 和原 v2 NO-GO；本轮结束后停止，无正式300轮或后续自动实验。

## 1. 实际执行与匹配证据（已确认事实）

开始时远端/本地均为 `39a65fd22efb0de0e1c721f6a679dcbda35c5900`，读取原v2报告、失败诊断、草案、原配置/入口/目标和归档；未找到适用 AGENTS.md 或同协议完整/未完成实验。已有无关未跟踪 v1 绘图脚本未修改或提交。用户本次明确授权这一项有限训练，原暂停自动实验边界不扩展。

独立入口 `scripts/run_selfsup_rvtc_matched.py` 和两组新配置，未改原v1/v2数值代码、模型、projector或全局 FROZEN_LOSS。模型为原 GeoLift-OARI-MRGR-Res3D，332,070参数，共享 F(A)/F(B)/F(T)，T=A∪B。复用原 objective 产生各项损失，由独立入口组成 L_measure=hold+.5cross+.1observed+hold_teacher+.1observed_teacher。Control仅优化L_measure，Treatment加原 .1teacher；Control诊断一致性不进入训练图。原教师目标与分母均detach，测量归一化/mask/projector保持原实现。

正式协议 `protocol.json` 在零更新时登记，SHA256 `{match['protocol_sha256']}`；登记代码和协议提前提交为 `a2484c3`，之后才开始实验优化。两组分别fresh random，seed20261011，初始tensor逐项相等；canonical state SHA256 `{match['initial_state_sha256']}`。没有加载监督、v1、原gate或其他实验权重。初始checkpoint由两次独立构造保存，不作为warm start。

两组初始固定测量指标、学生/教师输出统计及全部未加权loss项也逐项完全相等；初始total仅有规定的一致性贡献（FP32求和存在舍入）。

两组均 **80/80 epochs、320/320 optimizer更新**。固定原gate8/epoch0站位与顺序；batch1/accum2，每轮4更新；FP32、AdamW lr2e-4/wd1e-4、clip1，无scheduler、shuffle、early stop或best选择。K96/all rays/view_chunk2。相同初始化、站位、每轮顺序、预算、checkpoint内optimizer step/RNG均核对于 `matching_verification.json`。组间唯一训练差别是0/.1一致性权重。新目录 `{root}` 保留完整权重和恢复状态，Git归档不含它们。

初始state与注册hash、站位、源码、配置已冻结；固定epoch80权重各自哈希在 `checkpoint_manifest.json`。每轮last用于自身恢复，每10轮固定checkpoint保存optimizer/RNG/累计更新数；本轮两组均首次连续完成，未恢复、未重复训练。两组使用同一GPU顺序训练，Treatment是新训练而非复用历史gate。

训练前本轮运行35项独立入口/原RVTC/前次诊断针对性测试，审计日志补充后复跑9项新增测试，全部通过（35个不同测试，44次测试执行）。真实数据显存smoke各组2次microbatch反传、裁剪与AdamW状态缓冲，**零step，权重未变化**；注册最大T10，峰值约8.03GB。mask/padding/permutation/all-invalid与GT/Pool B/外部checkpoint/test阻断有测试证据，真实case集合检查亦通过。历史179项完整测试未冒称本轮重跑。

## 2. 配对测量增量（已确认事实）

学生始终为0.5×[error(C_A)+error(C_B)]，不做体积ensemble。H/U仅投影目标；训练损失使用原gate H，U和Val-A及其梯度没有参与参数更新。所有36例Val-A、8例gate U4完整评价，站位沿用前次诊断冻结清单，不按结果选择。gate H为样本内证据；U为已见案例新站位；Val-A为未见案例。

{effect_table}

{auxiliary_notes}

胜/负/平指Treatment稀疏误差相对于Control逐case更低/更高/相等。完整52条域×case配对结果在 `paired_per_case.csv`，104条组×域×case测量在 `measurement_per_case.csv`，包含A/B/T各自误差、师生绝对差异、教师分母/相对差异和体积尺度。各case、站位和checkpoint哈希均可追溯。输出mean/std等详细字段在原始CSV和固定最终训练summary中。

配对case bootstrap 10,000次，seed2026101202，percentile95%；同一case成对重采样，射线和视角不作为独立样本。主规则优化前登记：均值负且区间全负支持额外测量收益；区间含0不确定；均值正且区间全正支持测量受损。L2/U为辅助，不替代主L1。未做多重比较校正，单seed探索性区间不覆盖跨seed或工况家族相关性。

![配对测量](artifacts/selfsup_rvtc_matched_80/figures/paired_measurements.png)

## 3. 富视角教师优势与一致性增量分开（已确认事实）

{teacher_table}

教师−学生为负只表示同一组内富视角输入的测量优势，不能当作Treatment−Control的训练增量收益。两类差值分别保存，不因教师平均更好就推断蒸馏有效。独立测量改善也不能证明三维更准。

本次Control教师在全部36例Val-A的L1/L2上均优于稀疏学生，但开启一致性并未带来主指标增益；共享网络的教师自身也随训练改变，Treatment的Val-A教师L1为0.31444334，Control为0.24214604。因此富视角输入的可靠测量优势，不能替代对体积蒸馏训练增量的检验。教师测量变差是本次配对结果的一部分，不等于已定位造成它的优化机制。

## 4. 训练统计与固定checkpoint（已确认事实）

{windows}

轮内统计混合每轮4次更新前后的参数状态；固定最终评价保持第80轮参数不动，二者不混用。前/后10轮D仅诊断，没有据此通过/否决新实验或追溯改判原v2。原v2历史日志/门槛/checkpoint均未覆盖。

{scales}

非塌缩采用冻结原output_collapsed定义，集合采用原1e-5容差。工程正确性不等于方法有效性。每轮loss、测量、输出尺度，以及全部640个更新的原始梯度范数、裁剪后范数/系数在metrics.jsonl和optimizer_steps.csv；没有用裁剪后的范数替代原始诊断。

![训练动态](artifacts/selfsup_rvtc_matched_80/figures/training_dynamics.png)

## 5. 梯度轨迹（事实与解释边界）

预登记全部gate8加固定顺序前4例Val-A，共12例，每10轮和最终固定checkpoint计算加权学生measurement、教师measurement及一致性共享参数梯度；总计192条真实梯度记录。Val子集仅用于轨迹诊断，主评价仍为全36例。统计为**逐case余弦/范数比的算术平均**，不与前次报告的总体平均向量余弦混用。全部原始范数、S/T/D两两余弦在 `gradient_trajectory.jsonl`；以下为epoch80摘要：

{grad_table}

{trajectory_text}

**已确认事实：** Control实际一致性训练梯度所有记录均为零；其hypothetical_0p1字段是仅诊断的假设原0.1梯度，未用于更新。Treatment应用0.1后的原始梯度与测量项比较。两组教师detach及学生输出方向检查通过；它们不能替代共享参数冲突检查。诊断逐次验证参数/缓冲区、.grad、RNG与训练模式未改变，不执行额外step，不改变采样或顺序。

**解释假设：** 反向余弦和范数比可描述局部优化竞争及其随训练变化，配对实验可识别这次协议下一致性开关的测量净效果，但不能把净效果唯一归因于某个梯度轨迹。原始欧氏梯度不是AdamW矩估计/预条件后的真实方向；未执行反事实优化扰动或做整个路径的机制证明。

**尚未验证：** 全Val-A梯度轨迹、跨seed复现、训练样本之外的集中视角退化、任何GT体积指标。周期梯度仅12例，不能把局部梯度统计扩展为所有物理场景的因果规律。

![梯度轨迹](artifacts/selfsup_rvtc_matched_80/figures/gradient_trajectory.png)

## 6. 审计、不可变性与结论范围

全部实际实验session均成功，GT64/GT128、Pool B、外部checkpoint、禁止投影/test访问尝试均为0；允许的checkpoint仅本实验各组精确自身谱系。审计包括登记、smoke、训练、评价与产物核验；完整 `access_sessions.jsonl` 与各阶段日志保留。单元测试fixture的预期阻断计数不混入真实实验审计。独立评价打开本实验最终checkpoint用于前向，未加载其他权重。worker=0，原审计继承worker字段仅为兼容字段。

`integrity.json` 确认数据及源GT128仅metadata核对、所有历史实验文件metadata、原模型/projector/训练源码hash未变；没有打开GT或监督checkpoint进行hash。历史文件内容的既有审计hash继续原样保留。所有新权重仅外部保存，Git结果归档附SHA256清单，不提交模型或测量数据。

**结论：{decision}。** 这是相同富视角测量训练下开启原三维一致性项的测量增量结论，辅助指标结论见第2节。不是论文级最终有效性或三维质量确认，也不改变原v2 NO-GO。**三维准确性、集中视角退化、跨seed稳定性均未评价**。本次有限授权工作结束，保留v1，不自动启动正式训练、v3、模块、先验或更多实验。

{recommendation}

## 7. 复现入口

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests \\
  /scratch/djc/envs/gas-torch/bin/python -m unittest test_rvtc_matched test_selfsup_mrgr_v2 test_rvtc_failure_diagnostics -v
# 已登记/完成时核对并退出；未完成时仅允许自身last恢复，不重新初始化。
/scratch/djc/envs/gas-torch/bin/python scripts/run_selfsup_rvtc_matched.py register
/scratch/djc/envs/gas-torch/bin/python scripts/run_selfsup_rvtc_matched.py smoke
/scratch/djc/envs/gas-torch/bin/python scripts/run_selfsup_rvtc_matched.py train --arm Control
/scratch/djc/envs/gas-torch/bin/python scripts/run_selfsup_rvtc_matched.py train --arm Treatment
/scratch/djc/envs/gas-torch/bin/python scripts/run_selfsup_rvtc_matched.py evaluate
/scratch/djc/envs/gas-torch/bin/python scripts/report_selfsup_rvtc_matched.py
```

现有完整结果无需重训；缺失输入/不匹配/非有限/访问违规直接报错停止，不合成结果、不搜索参数重试。协议/source/站位文件已有hash；任何结果后的变更不能覆盖此次登记。
'''
    (REPO/'docs/reports/selfsup_rvtc_matched_80.md').write_text(content)


def archive(root):
    target=REPO/'docs/reports/artifacts/selfsup_rvtc_matched_80'
    names=['protocol.json','protocol.sha256','case_station_protocol.json','registration_audit.json',
           'memory_smoke.json','matching_verification.json','checkpoint_manifest.json','paired_summary.json',
           'measurement_per_case.csv','paired_per_case.csv','optimizer_steps.csv','gradient_trajectory_summary.csv',
           'access_sessions.jsonl','evaluation_audit.json','integrity.json','smoke_console.log',
           'Control_train_console.log','Treatment_train_console.log','evaluation_console.log']
    for name in names:shutil.copy2(root/name,target/name)
    for arm in ('Control','Treatment'):
        (target/arm).mkdir(exist_ok=True)
        for name in ['metrics.jsonl','gradient_trajectory.jsonl','training_summary.json','training_started.json',
                     'initial_fixed_evaluation.json','environment.json','gt_access_audit.json']:
            shutil.copy2(root/arm/name,target/arm/name)
    shutil.copytree(root/'figures',target/'figures',dirs_exist_ok=True)
    # Full inventories remain outside Git; archive their content hashes and verified conclusions.
    write_json(target/'external_inventory_manifest.json',dict(inventory_before_path=str(root/'inventory_before.json'),
        inventory_before_sha256=digest(root/'inventory_before.json'),comparison='exact metadata inventory equality checked during evaluation',
        model_weights_archived=False,measurement_data_archived=False))
    source_paths=['scripts/report_selfsup_rvtc_matched.py','docs/reports/selfsup_rvtc_matched_80.md']
    write_json(target/'reporting_source_manifest.json',dict(sha256={p:digest(REPO/p) for p in source_paths}))
    (target/'README.md').write_text('# RVTC 严格匹配 Control / Treatment 有限对照\n\n[中文报告与图表](../../selfsup_rvtc_matched_80.md)。\n\n协议在任何实验更新前登记并提交于 a2484c3；两组均fresh random、相同初始化/数据/顺序/站位/优化器/320步，只改变一致性0/.1。最终epoch80评价，不挑best。原v2 NO-GO保留。\n\n本目录归档真实104条测量、52条配对指标、192条梯度、640次更新日志、协议/source/state/checkpoint哈希、测试及实际访问审计。Control假设0.1梯度单独标记，实际一致性梯度为零。35个不同测试通过，审计日志补充后复跑9项。\n\n不含权重、测量数据或GT；完整checkpoint/optimizer/RNG与metadata inventory在外部独立目录。SHA256SUMS校验所有归档文件，artifact_manifest.json记录大小。无GT/Pool B/test/foreign checkpoint访问，无正式300轮或后续实验。\n')
    files={p.relative_to(target).as_posix():dict(sha256=digest(p),size_bytes=p.stat().st_size) for p in sorted(target.rglob('*')) if p.is_file() and p.name not in ('artifact_manifest.json','SHA256SUMS')}
    write_json(target/'artifact_manifest.json',dict(source_directory=str(root),payload_bytes=sum(x['size_bytes'] for x in files.values()),files=files))
    (target/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(target).as_posix()}\n' for p in sorted(target.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(f'Archived {len(files)} files; {sum(x["size_bytes"] for x in files.values())} bytes',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    pair,base,root=configs();summary=json.loads((root/'paired_summary.json').read_text())
    assert summary['completed'];match=validate_results(pair,base,root,summary)
    with (root/'paired_per_case.csv').open() as f:paired=list(csv.DictReader(f))
    for row in paired:
        for k in list(row):
            if k.startswith(('Control_relative','Treatment_relative','Treatment_minus_Control')):row[k]=float(row[k])
    figures(root,summary,paired);report(root,summary,match);archive(root)


if __name__=='__main__':main()
