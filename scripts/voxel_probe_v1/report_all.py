"""Combine fixed Adam comparison with the separately registered mechanism probes."""
from probe_context import *
import report as base_report
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    base_report.main()
    bounded=read(ART/'bounded_independent_review.json')
    network=read(ART/'network_independent_review.json')
    teacher=read(ART/'teacher_independent_review.json')
    assert bounded['passed'] and network['passed'] and teacher['passed']
    path=PROJECT/'docs/reports/voxel_network_probe_v1_zh.md'
    text=path.read_text()
    solver=[read(ART/f'case{i}_bounded_solver.json') for i in [0,82]]
    net=read(ART/'case0_unet_lbfgs.json')
    dense=read(ART/'case0_teacher_probe.json')
    lines=['','## 补充：原目标的有效体素拟合路径已得到独立确认','',
        '**这两例的原始物理目标存在明显优于零场的可拟合路径；当前U-Net失败不能统一解释为物理投影不可微、目标根本无有效解或只需更多同样的Adam小步更新。** 有效场由案例专属体素求解得到，尚不构成测量到场的条件网络，也没有三维GT证明其空间解唯一或准确。','',
        '直接Adam尺度臂仍受固定步长和200步上限限制，因此另行预登记一次[0,1]浓度边界L-BFGS-B存在性检查：同fresh平均场、原objective、FP64求解、每例最多60接受迭代/120函数求值。仅训练A/B/H loss进入线搜索，E从不参与。两个求解器均因60迭代上限停止，未声称达到全局最优或数值收敛。最终指标均重新使用原FP32投影器及目标计算，并独立NumPy复核。','',
        '| 案例 | 接受迭代/函数求值 | 原FP32总loss | A/B cross误差 | H误差 | 独立E误差 |','|---|---|---:|---:|---:|---:|']
    for r in solver:
        m=r['final'];lines.append(f"| {r['index']} | {r['accepted_iterations']}/{r['function_evaluations']} | {m['loss']:.7g} | {m['measurement']['cross']:.7g} | {m['measurement']['heldout']:.7g} | {m['extra_E_H']:.7g} |")
    lines+=['','这项结果把证据从“loss存在常量缩幅方向”推进到“原目标有更好的空间解，但当前网络训练没有到达”。它仍不能单独分离网络参数化、一致性约束、数据动态范围和优化器各自的贡献：共享体素场自动满足A/B一致性且无需推理，网络则有这些额外要求。','',
        '## 补充：只改变U-Net优化器还不足以解决','',
        '看到体素存在性结果后，独立登记一个fresh train0原U-Net的优化器单因素检查：结构/输入/初始化/原FP32目标完全不变，使用torch L-BFGS强Wolfe线搜索，最多20外层调用/60原目标梯度求值，不裁剪以免破坏线搜索导数；没有从体素解或失败网络warm-start。E仍仅评价，不能选步或调参。','',
        f"实际接受{net['accepted_iterations']}次参数变化、{net['closure_evaluations']}次目标/梯度求值；末态A/B cross={net['final']['measurement']['cross']:.7g}，H={net['final']['measurement']['heldout']:.7g}，E={net['final']['extra_E_H']:.7g}。预定测量拟合{'通过' if net['fitted_ABH'] else '未通过'}。线搜索改善了部分H，但{'仍没有有效独立射线收益，单换优化器不足以修复' if not net['E_below099'] else 'E也有改善，仍须结合全部拟合门槛'}。接受迭代与Adam更新不是等价计算预算，不作同预算排名。",'',
        '## 补充：自生成可拟合场的稠密目标对照','',
        '最后只登记一个稠密目标诊断：用上述train0体素解作为固定自生成场，原fresh U-Net同时拟合A/B对应同一个场；AdamW2e-4/wd1e-4、clip1、原cosine更新尺度，200步。唯一训练目标为浓度MSE除以该固定场均方值，以移除稀疏物理目标的空间归因困难。这个场不是真值，也不是监督模型权重；其来源仅为合法训练A/B/H和原目标，E未参与生成或选择。此诊断不改变任何正式训练loss。','',
        f"normalized MSE：{dense['initial']['teacher_normalized_MSE']:.7g} → {dense['final']['teacher_normalized_MSE']:.7g}；对自生成场的relative L1：{dense['initial']['teacher_relative_L1']:.7g} → {dense['final']['teacher_relative_L1']:.7g}。末态原物理H={dense['final']['measurement']['heldout']:.7g}、E={dense['final']['extra_E_H']:.7g}。MSE至少减半条件{'通过' if dense['normalized_MSE_below_half_initial'] else '未通过'}，自生成场L1≤0.9条件{'通过' if dense['teacher_relative_L1_below090'] else '未通过'}。",'']
    if dense['normalized_MSE_below_half_initial'] and dense['teacher_relative_L1_below090']:
        lines.append('稠密目标下网络能够学习该空间场，支持物理监督下的优化/空间信用分配是重要障碍；仍不能据单例证明所有架构或跨案例重建都能解决，也不能将自生成场拟合质量等同真实场精度。')
    else:
        lines.append('连稠密自生成目标在固定200步中也未达到预定拟合标准，说明不能把失败唯一归因物理损失：当前初始化、输出动态范围和有限步长下的网络优化本身仍有限制。它不证明U-Net表达不了该场，也不授权继续搜索结构或无限增加预算。')
    lines+=['','## 最终证据边界与后续处理','',
        '已确认：物理/目标在这两个案例可达到有效测量拟合；低浓度直接sigmoid与固定Adam步长确实进展很慢；原relative-L1对弥散输出的缩幅压力已有全166例及实际参数梯度证据；网络并未自动利用这个有效解空间。','',
        '尚未解决：在保持正式公平协议时，如何让经典条件网络稳定学习空间结构与跨案例映射。既有relative/consistency开关失败、当前单次优化器检查与稠密目标结果应共同解释，不能由此宣布“经典算法不行”或“删一个loss即可”。不自动启动300轮，不将当前网络列为已修复成功，不改旧NO_GO。','',
        '本轮停止边界是完成上述各自预登记的有界诊断，而非证明所有可能修复都无效。下一步应围绕已定位的网络优化尺度/条件读出做一项明确假设的独立适配设计；不继续无目的更换架构或扫描学习率、尺度、loss系数。','',
        '累计预算分开记录：原六臂Adam1200更新；独立稠密目标Adam200更新；共享体素L-BFGS-B的120次接受迭代及123次函数求值；网络L-BFGS的'+str(net['accepted_iterations'])+'次接受迭代及'+str(net['closure_evaluations'])+'次梯度求值。它们不混写为“同一种优化器步数”。全部CPU，每进程2线程；补充独立进程与原串行对照可同时运行，总任务线程不超过4。无GPU使用或他人任务干预。','',
        '独立复核合计40个完整A/B场、160组K96积分（包含相同共享场的逐分支复核，并非40个独立案例）。所有科学run的访问审计无GT64/GT128/Pool B/final test/外来权重，原科学源码/数据/冻结结果身份保持。','']
    path.write_text(text+'\n'.join(lines))
    all_rows=[]
    with (ART/'summary.csv').open() as stream:
        for row in csv.DictReader(stream):all_rows.append(dict(row,kind='fixed_Adam',accepted_iterations=row['actual_updates'],function_evaluations=''))
    for r in solver+[net]:
        m=r['final'];all_rows.append(dict(index=r['index'],arm=r['arm'],actual_updates='',initial_loss=r['initial']['loss'],final_loss=m['loss'],
            observed=m['measurement']['observed'],cross=m['measurement']['cross'],H=m['measurement']['heldout'],E=m['extra_E_H'],fit=r['fitted_ABH'],E_below099=r['E_below099'],
            kind='bounded_solver',accepted_iterations=r['accepted_iterations'],function_evaluations=r.get('function_evaluations',r.get('closure_evaluations'))))
    m=dense['final'];all_rows.append(dict(index=0,arm='teacher_probe',actual_updates=200,initial_loss=dense['initial']['loss'],final_loss=m['loss'],
        observed=m['measurement']['observed'],cross=m['measurement']['cross'],H=m['measurement']['heldout'],E=m['extra_E_H'],fit=dense['fitted_ABH'],E_below099=dense['E_below099'],kind='dense_self_generated',accepted_iterations=200,function_evaluations=''))
    with (ART/'all_probe_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(all_rows)
    print('Combined bounded diagnosis report written')


if __name__=='__main__':main()
