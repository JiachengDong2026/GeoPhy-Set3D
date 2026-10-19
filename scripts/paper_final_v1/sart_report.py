"""SART development paper report, preserving the original classical report."""
from common import *
import numpy as np


def main():
    from report import table,secondary_tables
    base=OUT/'evaluation/sart_dev';summary=read(base/'summary.json');verification=read(base/'independent_verification.json')
    assert verification['passed'];choice=verified(OUT/'sart/selected.json');checks=verified(OUT/'sart/numerical_checks.json')
    discrepancy=rows(OUT/'sart/operator_discrepancy_train.jsonl')
    lines=['# GeoPhy-Set3D：SART 补充经典算法开发结果','',f'生成时间：{now()}（北京时间）。以下仅为训练校准和36例开发验证，不是35例最终测试。','',
        '[方法、注册与测试前扩展协议](mrgr_paper_final_sart_supplement_v1_zh.md)。主模型身份与主要终点保持冻结，SART无论开发成绩如何均保留进入正式测试。','',
        '## 参数选择与正确性','',
        f'预登记松弛系数0.5、1.0，各在6个训练案例和2个布局上运行，共24次。所有预测封存后才读取训练GT64。选中候选{choice["candidate"]}，配置：`{json.dumps(choice["config"])}`。','',
        '| 候选 | 松弛系数 | 训练校准平均三维L1 |','|---|---:|---:|']
    for r in choice['scores']:lines.append(f'| {r["candidate"]} | {r["config"]["relaxation"]} | {r["mean_l1"]:.4%} |')
    lines+=['',f'前向/转置、单位、恢复和更新检查通过；phantom线性投影相对L2={checks["phantom_relative_l2"]:.6g}。这不构成真实测量数据或三维重建误差的精度保证。','',
        '## 离散前向差异（训练GT64，仅诊断）','',
        '对同一训练浓度场分别计算冻结P(C)与A·g(C)。这是非线性转换和插值顺序的离散差异，不作为SART调参指标。','',
        '| 布局 | 平均相对L1 | 最大案例相对L1 | 平均相对L2 | 最大绝对差 ppm·m |','|---|---:|---:|---:|---:|']
    for condition in ('Val-A-V4','clustered-V4'):
        rr=[r for r in discrepancy if r['condition']==condition]
        lines.append(f'| {condition} | {np.mean([r["relative_l1"] for r in rr]):.6%} | {max(r["relative_l1"] for r in rr):.6%} | {np.mean([r["relative_l2"] for r in rr]):.6%} | {max(r["max_absolute_ppm_m"] for r in rr):.6f} |')
    lines+=['','SART使用节点摩尔分数表示，不能把它称为P(C)的精确线性解。下列输入及额外H4投影指标统一使用P(C)，不使用求解器自己的线性残差替代。','',
        '[原始MRGR与全部四种经典方法的同切面对比：统一绝对色标及形状归一化](mrgr_paper_final_classical_slices_v2_zh.md)。','',
        '## 36例全布局结果','']+table(summary)+secondary_tables(summary)
    original=read(OUT/'evaluation/classical_dev/summary.json')
    combined=dict(original,panels=[*original['panels'],*[p for p in summary['panels'] if p['method']=='SART']])
    lines+=['','## 与此前三种经典方法统一查看','',
        '以下表格并列此前已封存复核的开发结果与新增SART。方法共享案例、布局和评分公式；SART的离散算子及CPU执行设备不同。全方法配对区间将在完整开发汇总中统一计算，不能把分别开发的同时区间当作同一个比较族。','']+table(combined)
    lines+=['','## 配对统计、有限预算与失败','',
        '| 对照 | 普通V4 MRGR−对照L1（百分点） | 案例95%CI（百分点） |','|---|---:|---|']
    for r in summary['paired']:
        if r['primary']:
            lo,hi=np.array(r['ci95'])*100;lines.append(f'| {r["reference"]} | {r["delta"]*100:.4f} | [{lo:.4f}, {hi:.4f}] |')
    rate=next(r for r in summary['failure_rates'] if r['method']=='SART');manifest=rows(OUT/'classical/val/SART/frozen/manifest.jsonl')
    lines+=['',f'SART共{rate["reconstructions"]}次开发重建：数值失败{rate["numerical_failures"]}，预算耗尽{rate["budget_exhausted"]}，稳定停止{rate["objective_stable"]}。停止判据是目标值稳定，不是全局最优证明。',
        f'CPU求解时间中位数{np.median([r["reconstruct_seconds"] for r in manifest]):.4f}秒；几何准备中位数{np.median([r["setup_seconds"] for r in manifest]):.4f}秒；进程累计峰RSS {max(r["process_peak_rss_bytes"] for r in manifest)/2**20:.1f} MiB。求解时间含周期恢复写盘及摩尔分数到质量分数转换，不含最终预测压缩写盘和几何准备。CPU与GPU计时不作未限定设备的倍数比较。','',
        f'独立复核重算{verification["fields_recomputed"]}个数组，验证{verification["paired_intervals_verified"]}组配对区间；L1/L2最大绝对差{verification["max_L1_L2_absolute_difference"]:.3g}。','',
        '重建误差超过100%是保留的科学结果，不等价数值失败。稀疏欠定问题、零初始化、固定扫描预算、离散表示与先验差异都会影响结果，不能将全部差异归因于神经结构。','',
        '[逐例CSV](../../outputs/research/mrgr_paper_final_v1/evaluation/sart_dev/per_case_metrics.csv)；[完整统计](../../outputs/research/mrgr_paper_final_v1/evaluation/sart_dev/summary.json)；[独立复核](../../outputs/research/mrgr_paper_final_v1/evaluation/sart_dev/independent_verification.json)；[最高误差案例](../../outputs/research/mrgr_paper_final_v1/evaluation/sart_dev/worst_cases.csv)。','']
    for p in sorted(base.glob('fixed_reconstructions_*.png')):lines += [f'![固定案例图](../../outputs/research/mrgr_paper_final_v1/evaluation/sart_dev/{p.name})','']
    path=ROOT/'docs/reports/mrgr_paper_final_sart_development_v1_zh.md';path.write_text('\n'.join(lines)+'\n');return path


if __name__=='__main__':print(main())
