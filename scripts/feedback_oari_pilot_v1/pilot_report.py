"""Status and eventual development-only pilot interpretation."""
from runtime import *

def main():
    d=verified(BASE/'design.json');s=read(BASE/'status.json') if (BASE/'status.json').exists() else {}
    lines=['# MRGR 残差反馈分支加入 OARI：20轮探索性筛查','',f'更新：{now()}。状态：`{s.get("state","registered")}`。',
        '', '仅使用166训练案例与36开发案例；不读取test投影、test GT64或GT128。原始冻结模型与正在进行的C/D实验不修改。',
        '', '## 结构与固定预算','',
        'C0与原始第一阶段不变；计算原P(C0)及有符号Y−P(C0)，残差编码与提升后同context0拼接，经原48→32卷积与SiLU，再经过独立OARI，最后由原32→32零初始化卷积产生Δh，context1=context0+Δh，再用原GCA和解码器生成C1。第二个OARI保留原射线关系、描述符和掩码。',
        '', '三个版本使用同一种子20261011、相同原始参数初始化，各完整运行前20轮、1660次更新，batch2/FP32/AdamW2e-4/裁剪1与全部损失、采样、物理算子不变。保留原cosine T_max300，因此是早期训练前缀，不能与已有300轮模型直接比较。只在预登记20轮内按验证H4最低值选轮。',
        '', '新增OARI从同种子原OARI初始权重复制为独立可训练参数，不消耗新RNG；原有全部同名张量保持精确一致。零残差版本只将残差输入置零，仍计算P(C0)，掩码/路径/原特征与新增参数均保留。',
        '', '| 版本 | 参数 | 进度 | 最佳H4 |','|---|---:|---|---:|']
    for name in d['runs']:
        run=Path(name);p=verified(run/'protocol.json');st=read(run/'status.json') if (run/'status.json').exists() else {}
        score=st.get('best_val_H');epoch=st.get('epochs',st.get('epoch',0))
        lines.append(f'| {p["arm"]} | {p["parameters"]} | {epoch}/20，{st.get("state","等待GPU验收")} | {100*score:.4f}%'+' |' if score is not None else f'| {p["arm"]} | {p["parameters"]} | {epoch}/20，{st.get("state","等待GPU验收")} | — |')
    lines+=['','## 资源和验收','',
        '每15秒查询UUID、进程/用户、显存、利用率和主机资源；使用C/D控制器共同遵守的gpu_locks，在最多一张卡上为本次短测保留优先级。保留锁不占GPU显存，也不影响外部任务；只在无计算进程、已用<100MiB、空闲≥20000MiB、利用率≤5%连续两次并复查后运行，不共享忙卡或发信号。结束后释放锁，原C/D队列继续使用资源。',
        '',f'CPU检查：{"通过" if (BASE/"cpu_checks.json").exists() else "待完成"}；GPU梯度/跨进程恢复：{"通过" if (BASE/"gpu_checks.json").exists() else "等待独立空闲GPU"}。',
        '', 'CPU验收覆盖真实64³、原参数初始化、非零适配器时旁路新增OARI与原实现逐张量完全相同、初始C1=C0、零残差通道和视角重排；GPU验收覆盖完整自监督损失反传和1+1/2轮跨进程恢复。浮点恢复rtol1e-4/atol1e-6，整数与RNG精确。失败记录保留并阻断训练。',
        '',f'[预登记/SHA]({BASE}/design.json)；[实时状态]({BASE}/status.json)；[资源和调度]({BASE}/scheduling/events.jsonl)；[CPU验收]({BASE}/cpu_checks.json)；[GPU验收]({BASE}/gpu_checks.json)。',
        '', '## 结果与解释','']
    if (BASE/'analysis/summary.json').exists():
        result=read(BASE/'analysis/summary.json');lines+=['| 布局 | 版本 | L1 | L2 | 峰比绝对误差 | GT峰处比 | 质心mm | Dice10 | Dice50 | H4 L1 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in result['panels']:
            if not(r['condition']=='Val-A-V4' or r['condition']=='clustered-V4' and len(r['draws'])==5):continue
            mean=lambda k:r['metrics'][k]['mean']
            keys=['volume.relative_l1','volume.relative_l2','morphology.absolute_peak_ratio_error','morphology.gt_peak_location_ratio','morphology.centroid_distance_mm','morphology.dice_0.1','morphology.dice_0.5','measurement.heldout.relative_l1']
            vals=[mean(k) for k in keys];formatted=['—' if v is None else f'{100*v:.4f}%' if i in (0,1,7) else f'{v:.4f}' for i,v in enumerate(vals)]
            lines.append('| '+ ' | '.join([r['condition'],r['arm'],*formatted])+' |')
        lines+=['','| 比较（前−后） | 布局 | L1差百分点 | 探索性95%案例区间 | 胜/平/负 |','|---|---|---:|---|---|']
        for r in result['paired']:
            if r['metric']!='volume.relative_l1' or not(r['condition']=='Val-A-V4' or r['condition']=='clustered-V4' and len(r['draws'])==5):continue
            lo,hi=[100*x for x in r['ci95']];lines.append(f'| {r["comparison"]} | {r["condition"]} | {100*r["delta"]:+.4f} | [{lo:+.4f},{hi:+.4f}] | {r["wins_lower"]}/{r["ties"]}/{r["losses_lower"]} |')
        lines+=['',result['recommendation']+'。', '', '扩大实验的预登记筛查信号要求：真实残差OARI在普通/聚簇两布局均降低平均L1，且不恶化平均峰比绝对误差与质心，并在L1上优于零残差版本。该规则只是下一步资源分配依据，不构成显著性、非劣性或最终性能证明。',
            '', '聚簇五draw先在同案例内平均后进行36案例配对bootstrap20000次；未经多重校正，不覆盖训练种子方差。所有17布局L1/L2、逐案例中位/P90/最大、峰值与定位、输入/H4一致性以及C0/C1原始标量全部保留。未定义形态指标会明确记为缺失，不静默删例。',
            '',f'[完整统计]({BASE}/analysis/summary.json)；[逐例CSV]({BASE}/analysis/per_case_metrics.csv)；[独立重算]({BASE}/independent_verification.json)。',
            '', '| 版本 | 训练秒 | 训练峰MiB | 最佳轮 |','|---|---:|---:|---:|']
        for name in d['runs']:
            r=verified(Path(name)/'pilot_summary.json');lines.append(f'| {r["arm"]} | {r["seconds"]:.2f} | {r["peak_allocated_bytes"]/2**20:.1f} | {r["best_epoch"]} |')
    else:lines+=['任务尚未完成，不能判断是否产生性能改善。不会把形状/梯度检查通过视为性能证据。']
    lines+=['','## 局限与复现','', '预检查首版CPU结构检查通过，但控制器因报告模块同名导入而退出，尚未启动GPU任务；完整源码、状态与日志保存在`mrgr_feedback_oari_pilot_v1_preflight_revision1_20261019_1711`。更名报告模块后重新登记验收，模型、种子及预算未变；另修正未定义形态指标的JSON表示，保留缺失而不删除案例。',
        '', '单种子20轮只检验早期训练趋势；没有参数匹配的非注意力新增模块，故不能把与baseline的差异完全归因于OARI机制。正结果仍需匹配预算、更长训练和多种子确认，负结果也不能否定收敛后可能有效。不能使用已用35例测试集作独立确认。',
        '',f'worktree：`{PROJECT}`；分支`experiment/feedback-oari-pilot-v1`；基线`{d["git_base"]}`。解释器`/scratch/djc/envs/gas-torch/bin/python`，入口`{SCRIPT}/controller.py`；权重/数组/日志位于`{BASE}`，不提交Git。原始主仓库和C/D源码保持不变。']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(lines)+'\n');print(REPORT,flush=True)

if __name__=='__main__':main()
