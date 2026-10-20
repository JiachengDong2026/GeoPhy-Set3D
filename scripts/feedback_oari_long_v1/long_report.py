"""Separate continuation report; completed pilot report remains immutable."""
from runtime import *

def main():
    d=verified(BASE/'design.json');status=read(BASE/'status.json') if (BASE/'status.json').exists() else {}
    lines=['# MRGR 残差反馈 OARI：300轮延长训练', '',f'更新：{now()}；状态：{status.get("state","registered")}。','',
        '根据20轮开发结果追加的探索性实验。三组均从各自第20轮last.pt继续至300轮，每组追加280轮；保留原cosine T_max300、优化器、全部RNG与数据流、历史最佳轮和前20轮指标，不重置学习率。原20轮结果独立冻结。','',
        '仅166训练例与36已用开发例，不读test投影、test GT64或任何GT128。训练和选轮只用投影；选轮为全300轮H4相对L1最小值，无提前结束。此为观察开发结果后追加的单种子实验，不能视作独立确认。','',
        '| 版本 | 进度 | 最佳H4投影相对L1 |','|---|---:|---:|']
    for name in d['runs']:
        p=Path(name);s=read(p/'status.json');lines.append(f'| {verified(p/"protocol.json")["arm"]} | {s.get("epoch",s.get("epochs",20))}/300，{s["state"]} | {100*s["best_val_H"]:.4f}% |')
    lines+=['','每次只使用一张完全空闲GPU。共享既有UUID锁，连续两次15秒检查无计算进程、已用<100MiB、可用≥20000MiB且利用率≤5%，启动前再次核查。正在运行的消融继续执行。', '',
        '三组依次为真实残差OARI、原始MRGR、同参数零残差OARI。模型结构与已通过的20轮CPU/GPU/跨进程恢复检查完全相同；新检查逐字段验证断点迁移，模型/优化器/调度器/RNG/计数/配置完全相同，只有protocol_sha256显式绑定新预算协议。','',
        f'[预登记]({BASE}/design.json)；[迁移验收]({BASE}/migration_verification.json)；[实时状态]({BASE}/status.json)。']
    if (BASE/'analysis/summary.json').exists():
        s=read(BASE/'analysis/summary.json');lines+=['','| 布局 | 版本 | 三维L1 | 三维L2 | 峰比绝对误差 | 质心mm |','|---|---|---:|---:|---:|---:|']
        for r in s['panels']:
            if not (r['condition']=='Val-A-V4' or r['condition']=='clustered-V4' and len(r['draws'])==5):continue
            vals=[r['metrics'][k]['mean'] for k in ('volume.relative_l1','volume.relative_l2','morphology.absolute_peak_ratio_error','morphology.centroid_distance_mm')]
            out=['—' if v is None else f'{100*v:.4f}%' if i<2 else f'{v:.4f}' for i,v in enumerate(vals)]
            lines.append('| '+' | '.join([r['condition'],r['arm'],*out])+' |')
        lines+=['',s['recommendation']+'。', '', '全部17布局先封存预测，再用开发GT64评分；聚簇五draw先按案例平均后配对bootstrap20000次，区间不覆盖训练种子不确定性。', '',
            f'[完整统计]({BASE}/analysis/summary.json)；[独立重算]({BASE}/independent_verification.json)。']
    else:lines+=['','尚未完成全部300轮和统一三维评估，暂不报告长程结构优劣。']
    lines+=['',f'原20轮结果：`{PILOT}`。本轮产物：`{BASE}`；入口：`{SCRIPT}/controller.py`。']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(lines)+'\n');print(REPORT,flush=True)

if __name__=='__main__':main()
