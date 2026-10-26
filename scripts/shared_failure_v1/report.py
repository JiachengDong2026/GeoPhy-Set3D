"""Render the read-only diagnosis and its standalone scientific figure."""
from diagnose import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def main():
    measurement = json.loads((ART / 'measurement_decomposition.json').read_text())
    fields = json.loads((ART / 'saved_field_gradients.json').read_text())
    parameter = json.loads((ART / 'parameter_probe.json').read_text())
    review = json.loads((ART / 'independent_review.json').read_text())
    rows = json.loads((ROOT / 'per_case_measurements.json').read_text())
    counts = measurement['measurement_bin_counts']
    signal_total = sum(counts.values())
    stratification = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    constants = measurement['constants']
    x = np.arange(len(constants))
    axes[0].bar(x-.18, [r['huber_gradient'] for r in constants], .36, label='Huber')
    axes[0].bar(x+.18, [r['relative_gradient'] for r in constants], .36, label='Relative L1 (weighted)')
    axes[0].plot(x, [r['joint_constant_gradient'] for r in constants], 'ko-', label='Total')
    axes[0].axhline(0, color='grey', lw=.7)
    axes[0].set_xticks(x, ['0', '1e-10', '1e-8', '1e-6', 'c0'])
    axes[0].set(xlabel='Uniform mass fraction c', ylabel='dL / dc (both branches)',
                title='Mean across 166 train cases')
    axes[0].legend(fontsize=8)
    for c, marker in [(1e-8, '.'), (constants[-1]['c'], 'x')]:
        selected = [r for r in rows if r['c'] == c]
        cases = []
        for index in range(166):
            group = [r for r in selected if r['index'] == index]
            signal = sum((1 if r['branch'] == 'H' else .3)*r['target_sum']/r['rays'] for r in group)
            gradient = sum((1 if r['branch'] == 'H' else .3)*r['total_gradient'] for r in group)
            cases.append((signal, gradient, index))
        cases.sort()
        quartiles = []
        for q in np.array_split(cases, 4):
            quartiles.append(dict(cases=len(q), mean_gradient=float(q[:, 1].mean()),
                                  shrink_direction_cases=int((q[:, 1] > 0).sum())))
        stratification.append(dict(c=c, groups_weakest_to_strongest=quartiles))
        axes[1].scatter([r[0] for r in cases], [r[1] for r in cases], s=18, marker=marker,
                        alpha=.65, label='c=1e-8' if c == 1e-8 else 'c=c0')
    axes[1].set_xscale('log')
    axes[1].set_xticks([.01, .03, .1, .3], ['0.01', '0.03', '0.1', '0.3'])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].axhline(0, color='grey', lw=.7)
    axes[1].set(xlabel='Weighted mean normalized measurement per case', ylabel='dL / dc',
                title='Positive gradient: reducing uniform c lowers loss')
    axes[1].legend(fontsize=8)
    fig.savefig(ART / 'gradient_mechanism.png', dpi=180)
    fig.savefig(ART / 'gradient_mechanism.svg')
    svg = ART / 'gradient_mechanism.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    write('signal_stratification.json', stratification)
    lines = ['# 多架构自监督基线失败机制诊断 v1', '',
             '日期：2026-10-26（Asia/Shanghai）。本轮为只读诊断，不是新修复实验。', '',
             '**判断：目前证据最符合“测量目标对弥散输出的幅度压低倾向，与网络未能有效学习测量条件空间结构相互作用”。relative-L1是已定位的幅度压力来源之一；尺度和sigmoid可进一步放大失败。不能将loss、某一种结构或训练预算认定为唯一根因。**', '',
             '## 本轮实际做了什么', '',
             '- 全部166 train、epoch0固定合法A/B/H，498组测量、五个固定常量浓度，独立NumPy FP64分解原Huber及relative-L1梯度；不读取三维GT。',
             '- 在U-Net已保存的train0/1/2、epoch0/30完整A/B场上，调用原objective做六组场空间梯度与固定±1%幅度有限差分；没有优化这些场。',
             '- CPU只读自身U-Net epoch0/30权重，在固定train0计算原网络参数梯度；参数哈希前后相同、grad buffer均None、无optimizer。',
             f"- 原Torch measurement_discrepancy与NumPy共{review['comparisons']}条件交叉核验通过；loss最大绝对差{review['max_loss_absolute_error']:.3g}，梯度最大差{review['max_gradient_absolute_error']:.3g}。常量投影使用解析物理公式，未冒称本轮再次独立重做完整K96核验。", '',
             '## 1. 幅度下降的来源已分解到测量项', '',
             '令两分支同为常量c，原总目标等于D_H(c)+0.3[D_A(c)+D_B(c)]，一致性项严格为零。下表导数是同时改变两分支c的dL/dc，正数表示降低常量可降loss。', '',
             '| c | 全train平均loss | Huber导数 | 加权relative-L1导数 | 总导数 | 倾向降低c的案例数 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in constants:
        lines.append(f"| {r['c']:.8g} | {r['original_objective_mean']:.9g} | {r['huber_gradient']:.7g} | {r['relative_gradient']:.7g} | {r['joint_constant_gradient']:.7g} | {r['decreasing_output_is_descent_cases']}/166 |")
    weak = (counts['zero']+counts['(0,1e-8]'])/signal_total
    lines += ['', f"共{signal_total:,}条采样有效射线，其中精确零{counts['zero']:,}条，0<Y/Y_SCALE≤1e-8有{counts['(0,1e-8]']:,}条；两者合计{weak:.2%}。这里按测量数值分组，不表示有已知噪声阈值，也不删除任何射线。", '',
              '在c=1e-8时，relative-L1的零组导数+12.0625，(0,1e-8]组+28.9107，其余组合计约−20.3094，relative合计+20.6638；Huber为−14.7582，二者合成+5.90558。因此低测量射线对弥散预测的惩罚足以反转平均常量方向。该relative项按每案例sum|Y|归一化，并非逐射线除以Y；不能误诊为“对每条零射线直接除零”。', '',
              '但平均值不能代表全部案例：c=1e-8只有60/166个案例方向向下。按加权平均测量强度从弱到强分为42/42/41/41例，四组平均导数分别约+73.64、−2.34、−21.04、−28.08；最弱组41/42向下，最强两组均0/41向下。这提示测量强度不均与逐案例相对归约共同影响整体方向，不能将两个前序batch外推为“全部训练案例都偏好零场”。', '',
              'c0是合法train A/B等射线权Huber匹配常量，不是真实浓度均值。该估计目标的导数在c0约为0；原目标的全案例平均Huber导数也仅+0.03468，而加权relative导数+30.5689。尺度初始化没有也不应被解释为原完整目标的驻点。该差异本身是初始化定义和训练目标不同，不是程序bug。', '',
              '![全训练集梯度机制](artifacts/shared_failure_v1/gradient_mechanism.png)', '',
              '精确零点的AD子梯度为负，极小正值的方向可能相反：abs在零残差处不可微，极弱正目标还存在大量细小折点。不能用零点AD替代严格右导数，更不能证明零场是唯一全局最优。仅常量子空间存在更低loss，也不否认局部化羽流能进一步降低测量误差。', '',
              '## 2. 这种幅度压力也出现在实际U-Net输出上', '',
              '保存场的六组固定探针中，整体幅度导数全部为正；将场固定乘0.99的loss均低于乘1.01。这个判断没有改变空间形状或重新训练。末态三个案例的dL/dalpha约0.00320、0.01235、0.03737，说明当前已有空间结构仍不能完全消除降低幅度的下降方向。该结论限于固定train0..2、epoch0采样，不代表全部训练轨迹。', '',
              '进一步在原网络参数空间检查train0，末层bias的梯度如下。梯度下降时正梯度会降低bias、进而降低所有logits；负梯度反之。', '',
              '| checkpoint epoch | Huber | 加权relative-L1 | 加权consistency | 总bias梯度 |',
              '|---|---:|---:|---:|---:|']
    for r in parameter['rows']:
        g = r['readout_bias_gradient']
        lines.append(f"| {r['epoch']} | {g['smooth']:.8g} | {g['relative']:.8g} | {g['consistency']:.8g} | {g['total']:.8g} |")
    lines += ['', '末态Huber希望提高共同bias，relative-L1则希望降低，后者幅度约为前者15.18倍，合成后仍向下。该例末态两项在readout参数空间cosine约−0.9942，确有局部优化冲突；encoder空间cosine却约+0.9849，不能说所有层都发生同样冲突。初态两项整体cosine约+0.9523，也不能把末态关系套用到初态。', '',
              '场空间中末态consistency梯度范数可比测量项大2.15–11.67倍，但同一个train0在实际参数空间，其加权consistency总范数仅0.0003624，relative约0.010869。共享网络Jacobian会改变方向和抵消关系，因此“场梯度大”不能被写成“实际参数更新由consistency主导”。此例直接缩幅bias梯度也主要来自relative而非consistency。AdamW还含动量及预条件，单次原梯度不等于实际更新方向。', '',
              '## 3. 为什么不能把根因简单归为relative-L1或consistency', '',
              '已有Senseiver relative开关对照中，关闭relative后的末态均值约为Control的10.4倍，logit饱和缓解，但两臂都缺少有效测量依赖。随后在relative=0条件下再关闭consistency，末态H L1仍约1.06057，与Control约1.06063几乎相同；仍无有效重建。DeepSets/GNOT的齐次consistency修复也未取得GO。', '',
              '所以这些损失项能够解释一部分幅度压力或梯度分配，却没有被证明是充分失败原因；此前相关开关实验已经做过，不应将它们当作尚未执行的新建议重复搜索。', '',
              '## 4. 测量信息进入模型，与有效条件重建是两回事', '',
              'Senseiver已有固定几何读出实验：冻结latent的岭回归对独立检查测量均值MSE为1.31629e-8，常量基准4.96934e-4；原query结构的标准化标量头检查MSE为2.38860e-5。说明至少测量幅度信息进入并可读；这只验证标量，不证明三维位置/形状信息足够。新增全局latent条件残差仍未解决重建。', '',
              'DeepSets/GNOT在完全相同几何、不同合法合成测量下仍弱响应，说明“只要去掉几何捷径就能成功”也不成立。部分合成案例又能降低测量和三维误差，排除了物理/梯度链完全断开、解码器完全不能表达非均匀场的绝对解释。', '',
              'FNO与U-Net都先用固定回投影汇总测量，再学习网格；Senseiver/GNOT/Ray-DeepSets则依赖全局表示与条件解码。成功的GeoLift-DeepSets-Sum采用先学习每站完整角域局部特征、再几何提升到空间、最后卷积解码的路径。这是一个值得检验的归纳偏置差异，但尚未做严格同预算/同初始化的因果对照，也未证明固定回投影在本数据上丢掉了哪些不可恢复信息。', '',
              '冻结Plain主模型还保留物理残差反馈；正式GeoLift-DeepSets-Sum没有这个反馈也已成功。因此不能宣称只有MRGR才能训练，或把基线失败直接当作MRGR必要性的证明。', '',
              '## 5. 对其他解释的判断', '',
              '| 解释 | 当前证据 | 判断 |',
              '|---|---|---|',
              '| 根本没有梯度或optimizer未更新 | CPU实际更新、逐步日志、参数变化及本轮原网络梯度 | 不是当前主要解释 |',
              '| 单位、轴序、mask或物理投影整体错误 | 既有独立物理核验及成功GeoLift基线 | 未发现支持；不等于所有潜在bug已被穷尽排除 |',
              '| 只是随机输出尺度错误 | v2尺度已修正，DeepONet仍塌缩、U-Net仍弱条件收益 | 是早期因素，不是充分原因 |',
              '| 只是训练不够长 | Senseiver已9379步仍塌缩，真实少样本拟合不足 | 不能统一解释；480步pilot也不能证明其他架构充分训练后必然失败 |',
              '| 所有数据都不可辨识 | 正式GeoLift有成功结果；部分合成可恢复 | 不能成立为普遍解释；稀疏布局仍可能有局部非唯一性 |',
              '| 相同超参数就是所有算法公平充分优化 | 架构对尺度、归一化、lr的反应不同 | 固定协议一致不等于各架构已充分调优，NO_GO仅限当前适配 |', '',
              '历史正式GeoLift-DeepSets-Sum在第3轮/249步Val H约0.8135，第6轮/498步约0.4304。其训练166例，当前pilot32例，样本覆盖/架构/初态并非严格配对；只能说明同一原目标存在可学习路径，不能把数值差唯一归因到某个组件。', '',
              '## 诊断结论与下一步', '',
              '已定位的局部机制是：零及极弱测量在当前relative-L1归约下对弥散浓度产生明显惩罚，部分弱测量案例推动整体幅度降低；网络若尚未学会正确定位羽流，就容易沿全局缩幅路径降低loss。随后部分架构进入sigmoid深负饱和；另一些架构保留空间结构，却仍未形成足够的正常测量收益。', '',
              '尚未唯一定位的是：为什么当前经典适配没能沿“背景降低、真实羽流位置升高”的空间方向学习。候选包括条件信息的空间组织、共享模板、参数化优化、数据与测量动态范围以及有限预算。成功GeoLift反例和已有损失开关失败要求同时保留这些解释。', '',
              '优先下一步应是在固定合法训练案例上，独立登记“直接优化体素场”和“原网络拟合同一测量”的对照，并检查独立射线；若前者可拟合而网络失败，重点查参数化和条件解码；若二者都只能降到零场附近，重点查目标与该案例测量约束。直接体素优化也可能过拟合，因此不作为正式基线或三维正确性的证明。本轮未执行任何此类优化，也不自动修改loss或启动新训练。', '',
              '## 证据与复算', '',
              '- 新脚本：scripts/shared_failure_v1/{diagnose,review,parameter_probe,report}.py。',
              '- 机器证据：docs/reports/artifacts/shared_failure_v1/；大逐案例分解：/scratch/djc/GeoPhy-Set3D/shared_failure_v1/per_case_measurements.json。',
              '- 原始报告：[U-Net](unet_baseline_v1_zh.md)、[DeepSets/GNOT根因审查](deep_baselines_collapse_root_cause_repair_zh.md)、[固定几何条件诊断](deep_baselines_identifiability_v2_zh.md)、[Senseiver relative](ray_senseiver_relative_probe_v1_zh.md)、[consistency](ray_senseiver_consistency_probe_v1_zh.md)、[latent读出与修复](ray_senseiver_quick_repair_v1_zh.md)、[正式GeoLift-DeepSets-Sum](deepsets_selfsup_v1_zh.md)。',
              '- 本轮全部CPU、0次optimizer更新、无GT64/GT128/Pool B/test访问。原147源码、207输入及已封存U-Net证据最终再核验；只新增本诊断目录，不修改历史NO_GO或冻结模型。', '']
    (PROJECT / 'docs/reports/shared_selfsup_failure_diagnosis_v1_zh.md').write_text('\n'.join(lines))
    print('Diagnosis report and standalone figure written')


if __name__ == '__main__':
    main()
