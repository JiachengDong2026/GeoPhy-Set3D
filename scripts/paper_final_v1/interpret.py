"""Write final factual conclusions using only preregistered statistical rules."""
from common import *
import numpy as np


def main():
    base=OUT/'evaluation/test';summary=read(base/'summary.json');assert read(base/'independent_verification.json')['passed']
    lock=verified(OUT/'final_lock.json');access=read(base/'gt_access.json');ablation=read(base/'ablation_paired.json')['results']
    panels=summary['panels'];ordinary=next(p for p in panels if p['method']=='MRGR' and p['stage']=='C1' and p['condition']=='Val-A-V4');m=ordinary['metrics']['volume.relative_l1']
    lines=['# 论文最终实验：按预注册规则生成的结果判读','',f'生成时间：{now()}（北京时间）。仅在{len(lock["methods"])}种方法、35案例、17布局的全部预测封存、独立评分与复核通过后生成；主模型始终为原第278轮MRGR。','',
        f'唯一主要终点——普通固定V4 C1三维相对L1：均值{m["mean"]:.4%}，中位数{m["median"]:.4%}，P90 {m["p90"]:.4%}，最大{m["max"]:.4%}。','',
        '## 主要终点的预登记判断','', '| 对照 | MRGR−对照（百分点） | Bonferroni同时区间（百分点） | 按预注册规则判断 |','|---|---:|---|---|']
    decisions=[]
    for p in summary['paired']:
        if not p['primary']:continue
        lo,hi=p['simultaneous_ci'];decision='MRGR误差更低' if hi<0 else 'MRGR误差更高' if lo>0 else '区间跨零，比较未确定'
        lines.append(f'| {p["reference"]} | {p["delta"]*100:.4f} | [{lo*100:.4f},{hi*100:.4f}] | {decision} |');decisions.append(dict(reference=p['reference'],decision=decision))
    lines+=['',f'判断按锁定的同一主要终点和全部{len(lock["methods"])-1}个对照同时区间执行；无论方向如何都保留原主模型。普通95%区间、逐例差值及胜/平/负见总报告和机器汇总。','',
        '## 结构和损失的独立训练对照','', '| 因素 | 普通V4两种子差值（消融−完整，百分点） | 聚簇五draw两种子差值（百分点） | 观察范围 |','|---|---|---|---|']
    from module_registry_v2 import ablation_arms
    for arm in ablation_arms():
        a=[next(r for r in ablation if r['arm']==arm and r['seed']==s and r['condition']=='Val-A-V4' and r['metric']=='relative_l1') for s in (20261011,20261019)]
        b=[next(r for r in ablation if r['arm']==arm and r['seed']==s and r['condition']=='clustered-V4' and r['metric']=='relative_l1') for s in (20261011,20261019)]
        delta=[r['delta_ablation_minus_full'] for r in a+b]
        observation='两种子、两布局均为完整模型误差较低的方向' if all(d>0 for d in delta) else '两种子、两布局均为消融误差较低的方向' if all(d<0 for d in delta) else '存在种子或布局间的方向差异'
        lines.append(f'| {arm} | '+', '.join(f'{r["delta_ablation_minus_full"]*100:+.4f}' for r in a)+' | '+', '.join(f'{r["delta_ablation_minus_full"]*100:+.4f}' for r in b)+f' | {observation} |')
    lines+=['','这是同种子独立训练比较，不使用联合模型C0/C1替代消融。两个种子的结果不能估计总体训练种子方差；各案例置信区间与探索性限制见总报告。单尺度匹配参数但计算量/感受野改变；无反馈模型同时减少12,512个参数，不能将差异全部归因于纯反馈机制。第一种子参考复用历史冻结主模型，还需保留跨执行非确定性的限制。','',
        'OARI两种输入消融保持名义参数量，但被置零输入对应的权重缺少任务梯度；GCA均匀融合删除学习权重并改变计算量。零残差对照保留完整反馈容量和几何输入，仅移除测量残差信息。各项效应是相对完整模型的条件效应，不表示模块间没有交互，也不支持替换主模型。','',
        '## 峰值、定位和联合模型轨迹','', '| 布局 | C0 L1 | C1 L1 | C1 GT峰处浓度比 | C1质心mm | C1 Dice50 |','|---|---:|---:|---:|---:|---:|']
    for condition,draws in [('Val-A-V4',[0]),('clustered-V4',list(range(5)))]:
        c0=next(p for p in panels if p['method']=='MRGR' and p['stage']=='C0' and p['condition']==condition and p['draws']==draws)['metrics'];c1=next(p for p in panels if p['method']=='MRGR' and p['stage']=='C1' and p['condition']==condition and p['draws']==draws)['metrics']
        lines.append(f'| {condition} | {c0["volume.relative_l1"]["mean"]:.4%} | {c1["volume.relative_l1"]["mean"]:.4%} | {c1["morphology.gt_peak_location_ratio"]["mean"]:.4f} | {c1["morphology.centroid_distance_mm"]["mean"]:.4f} | {c1["morphology.dice_0.5"]["mean"]:.4f} |')
    lines+=['','GT峰处浓度比描述真实峰坐标处的幅度，不是预测最大峰的定位距离；高浓度恢复、质心和全场误差不能相互替代。C0/C1只是联合训练模型的阶段诊断。','',
        '## 经典求解器的有限预算与负结果','', '| 方法 | 数值失败 | 预算耗尽 | 稳定停止 |','|---|---:|---:|---:|']
    for r in summary['failure_rates']:
        if r['method'] in ('NNLS-NL','Tikhonov','TV','SART'):lines.append(f'| {r["method"]} | {r["numerical_failures"]} | {r["budget_exhausted"]} | {r["objective_stable"]} |')
    lines+=['','固定200步、零初始化与空间先验代表本次登记预算；预算耗尽不能当作最优解证明。经典方法和学习方法在分布先验与运行成本上不同，不能仅从最终误差归因于多尺度结构本身。所有方法均保留，最高误差案例完整列入worst_cases.csv。','',
        'SART是节点摩尔分数线性插值的补充基线，离散前向与MRGR不同；其所有投影评价仍统一使用冻结P(C)。SART使用CPU，GPU方法的计时不能直接解释为算法速度倍数；CPU峰RSS另列。','',
        '## 数据与复核边界','',f'全部预测封存：{access["seal_at"]}；独立GT64首次评分读取：{access["first_GT64_at"]}。35份GT64身份在封存后计算哈希并由复核进程再次验证；GT128、测试调参和评分后训练均未使用。', '',
        '固定索引0/17/34的图像、逐例逐布局CSV、全部次要指标、配对区间和任务成本已汇总。次要指标均探索性，案例bootstrap不覆盖训练种子变化，未检验全新几何域、噪声或标定误差。远程实时HEAD无法确认，源码以记录的本地commit、工作树状态及文件哈希为准。','',
        '[论文总报告](mrgr_paper_final_experiments_zh.md)；[经典开发报告](mrgr_paper_final_classical_development_v1_zh.md)；[最终逐例指标](../../outputs/research/mrgr_paper_final_v1/evaluation/test/per_case_metrics.csv)；[最终统计](../../outputs/research/mrgr_paper_final_v1/evaluation/test/summary.json)；[独立复核](../../outputs/research/mrgr_paper_final_v1/evaluation/test/independent_verification.json)。']
    path=ROOT/'docs/reports/mrgr_paper_final_interpretation_zh.md';path.write_text('\n'.join(lines)+'\n');return dict(path=str(path),sha256=sha(path),primary_decisions=decisions)

if __name__=='__main__':print(main())
