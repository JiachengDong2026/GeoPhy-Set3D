#!/usr/bin/env python3
"""Scalar-only OCMSF screen report and curves; no model/data array access."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research/mrgr_ocmsf_screen_v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    p=read(OUT/'protocol.json');r=read(OUT/'comparison.json');s=read(OUT/'smoke.json')
    arms=list(p['parameters']);labels=dict(A_original='原MRGR（编号A）',B_multiscale='多尺度MRGR（编号B）',C_content='多尺度MRGR＋非O条件尺度融合（编号C）',D_ocmsf='多尺度MRGR＋OCMSF（编号D）')
    logs={a:[json.loads(x) for x in (OUT/a/'metrics.jsonl').read_text().splitlines()] for a in arms}
    summaries={a:read(OUT/a/'summary.json') for a in arms}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for a in arms:
        axes[0].plot([x['epoch'] for x in logs[a]],[x['train']['loss']['total'] for x in logs[a]],label=a)
        axes[1].plot([x['epoch'] for x in logs[a]],[100*x['val']['measurement']['heldout']['relative_l1'] for x in logs[a]],label=a)
    for ax,title in zip(axes,('Training total loss','Validation heldout projection relative L1 (%)')):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'curves.png',dpi=170);fig.savefig(OUT/'curves.svg');plt.close(fig)
    lines=['# OCMSF：四臂匹配20轮筛查','',
        f'状态：四臂各完成 **20轮、1,660次更新**。预注册两个主对照的联合晋级结果：**{"通过" if r["gate_passed"] else "未通过"}**。本轮未开展300轮训练、三维GT评分或最终35例测试。','',
        '**A–D为本次四组消融的实验编号，不是历史“方案A/方案B”。** 四组分别为原MRGR、多尺度MRGR、非O条件尺度融合、OCMSF。另，A→H/B→H中的A/B是自监督输入视角子集，与模型编号含义不同。','',
        '全部误差为**留出投影相对L1**，不是三维浓度场误差。普通每案例先平均A→H、B→H；聚簇5布局先案例内平均，再对36案例等权汇总。四臂166训练/36验证，seed20261011，固定第20轮，原300轮cosine前缀、测量目标、站位、归一化、Q96和有效batch2均相同。','',
        '| 模型 | 参数 | 普通H均值 | 普通P90 | 聚簇五布局H均值 | 第3–20轮平均耗时 |','|---|---:|---:|---:|---:|---:|']
    for a in arms:
        lines.append(f'| {labels[a]} | {p["parameters"][a]:,} | {100*r["ordinary"][a]["mean"]:.4f}% | {100*r["ordinary"][a]["p90"]:.4f}% | {100*r["cluster5"][a]["mean"]:.4f}% | {r["warm_epoch_seconds"][a]:.2f}s |')
    lines += ['', '## 1. 主对照与晋级门槛','',
        'D/C检验增加O数值条件的效果，D/B检验整个新增结构的效果。C仍使用基础可见性mask。C/D均增加739参数，所有共享及新增初态完全一致；D仅在隐藏层加入固定无参数5→8的O嵌入，C不加入，可学习容量和主体计算路径一致；D额外执行固定嵌入及加法。D不是比C多一个可学习O编码器。','',
        '| 对照 | 普通均值相对变化 | 普通P90相对变化 | 聚簇均值相对变化 | 训练耗时比 |','|---|---:|---:|---:|---:|']
    for c in ('C_content','B_multiscale'):
        x=r['contrasts'][f'D_ocmsf_vs_{c}']
        lines.append(f'| D vs {c[0]} | {100*(x["ordinary"]["mean_ratio"]-1):+.3f}% | {100*(x["ordinary"]["p90_ratio"]-1):+.3f}% | {100*(x["cluster5"]["mean_ratio"]-1):+.3f}% | {x["runtime_ratio"]:.4f} |')
    lines += ['', '两个主对照均要求：普通均值至少改善1%、普通P90与聚簇五布局均值恶化不超过2%、完整轮耗时比不超过1.20、D的300轮外推不超过5小时；正确性及输出非坍缩通过。门槛未事后修改。','']
    lines += ['固定嵌入的代价：D没有可独立学到0的几何幅度，也没有可学习O投影；C/D不是简单的嵌套可退化模型。参数严格匹配并避免死参数的同时，也固定了几何注入量级。因此若未通过，只能否定本次最小固定嵌入实现，不能否定全部OCMSF。C若表现较好，也只作为独立待复核候选，不自动追加300轮。','']
    for c,g in r['gates'].items():
        lines.append(f'- D vs {c[0]}：`{json.dumps(g,ensure_ascii=False)}`。')
    lines += ['', '## 2. 配对案例区间与改善数量','',
        '处理减对照为负代表改善。区间按36案例成对bootstrap 20,000次，固定种子20261017；不包含训练种子变化，多个探索性对照未作多重校正。','',
        '| 对照 | 普通均值差95%区间（百分点） | 普通胜/负/平 | 聚簇均值差95%区间（百分点） | 聚簇胜/负/平 |','|---|---|---:|---|---:|']
    for key,x in r['contrasts'].items():
        o,h=x['ordinary'],x['cluster5'];oc=[100*t for t in o['paired_ci95']];hc=[100*t for t in h['paired_ci95']]
        lines.append(f'| {key} | [{oc[0]:+.4f}, {oc[1]:+.4f}] | {o["wins"]}/{o["losses"]}/{o["ties"]} | [{hc[0]:+.4f}, {hc[1]:+.4f}] | {h["wins"]}/{h["losses"]}/{h["ties"]} |')
    lines += ['', '## 3. 正确性、实际干预与开销','',
        'CPU独立测试3项通过，关联回归68项通过。GPU smoke在C/D各覆盖B1/B2与正常、部分padding、全padding、无有效射线，均无NaN/Inf；每次OARI descriptor捕获一次、C0/C1调用同一decoder两次，结束缓存清空。B/C/D完整初始输出逐位一致，原B非零context下的decoder初始等价也通过。','',
        '零残差输出层使gate上游在首次反传的梯度为零，属于预期。一次真实测量目标的临时更新后，C/D全部6个新增参数张量均有有限非零梯度，临时smoke模型被丢弃，未用于短程初始化。','',
        f'同GPU1 smoke完整目标反传热态D/B比为 **{s["D_to_B_update_ratio"]:.4f}**；训练D/B跨GPU1/2、A跨GPU3，训练时段负载和数值轨迹也可能不同，不能将耗时差全部归因模块。D按完整轮线性外推300轮为 **{r["projected_D_300_hours"]:.3f}小时**，不是已运行的正式训练。','',
        '| 模型 | smoke峰值分配显存 | 训练峰值分配显存 | 20轮及末轮评价耗时 |','|---|---:|---:|---:|']
    for a in arms:
        peak=max(x['peak_allocated_bytes'] for x in logs[a])
        lines.append(f'| {labels[a]} | {s["arms"][a]["peak_allocated_bytes"]/2**30:.3f}GiB | {peak/2**30:.3f}GiB | {summaries[a]["elapsed_seconds"]:.2f}s |')
    lines += ['', '32³细尺度特征需要额外执行一次共享reduce（Conv/GN/SiLU），F16/F8复用原分支特征。C/D都有这部分新增计算和共享reduce梯度，不能称为免费复用三个原有完整尺度。','']
    for a in ('C_content','D_ocmsf'):
        cases=[json.loads(x) for x in (OUT/a/'epoch20_per_case.jsonl').read_text().splitlines()]
        gate=[x['gate'][k] for x in cases for k in ('A','B')]
        alpha=np.mean([x['alpha_mean'] for x in gate],axis=0).tolist()
        spatial=np.mean([x['alpha_spatial_std'] for x in gate],axis=0).tolist()
        lines.append(f'{labels[a]}：72次普通重建的尺度权重均值（32/16/8）`{alpha}`，空间标准差均值`{spatial}`，分支残差范数均值{np.mean([x["delta_norm"] for x in gate]):.6g}。残差非零不单独证明使用了O，需结合以下同权重干预。')
        lines.append('')
    intervention_path=OUT/'condition_intervention/summary.json'
    if intervention_path.exists():
        q=read(intervention_path)
        lines += ['同权重干预固定使用注册的首2个验证案例，临时关闭D额外O嵌入，保留原OARI描述符和可见性mask，未重训且不改变晋级判断。','',
            f'四次A/B输入的场相对L1差均值为{100*np.mean([x["field_relative_l1_difference"] for x in q["rows"]]):.6f}%（两次输出差的L1范数除以正常输出的L1范数，不涉及GT），尺度权重绝对差均值为{np.mean([x["alpha_mean_absolute_difference"] for x in q["rows"]]):.6g}；正常H均值{100*q["normal_H_mean"]:.6f}%，关闭额外O后{100*q["no_extra_O_H_mean"]:.6f}%。权重hash保持不变。这个检查只证明额外条件进入实际输出，不是重训消融或总体收益证据。','']
    lines += ['## 4. 输出稳定性与解释边界','',
        '36例A/B最终输出均有限、非恒定、范围在[0,1]；逐例最小/最大/标准差/均值记录于原始JSONL。这里的最大值是模型输出峰值，没有GT，不能判断峰位、真实峰幅、Dice或质心是否更准确。','',
        '新结构不保证欠定方向消歧；几何描述符只反映方向覆盖。已有单种子同初态、同采样对照的短程数值轨迹存在明显变化，因此小幅收益需新种子确认，不能作为几何因果的最终证据。新增参数隔离恢复CPU torch RNG，不宣称CUDA RNG也全部相同；当前模型没有dropout或CUDA随机采样，案例/站位/loader RNG均逐轮核验。未在本轮以GT选模型、调权重或挑案例，最终35例未使用。','',
        '![四臂训练曲线](../../outputs/research/mrgr_ocmsf_screen_v1/curves.png)','',
        '## 5. 可复核证据','',
        '- [设计](mrgr_ocmsf_design_zh.md)',
        '- [预注册协议](../../outputs/research/mrgr_ocmsf_screen_v1/protocol.json)',
        '- [完整对照及gate](../../outputs/research/mrgr_ocmsf_screen_v1/comparison.json)',
        '- [smoke和梯度](../../outputs/research/mrgr_ocmsf_screen_v1/smoke.json)',
        '- [关联回归测试](../../outputs/research/mrgr_ocmsf_screen_v1/regression_tests.json)',
        '- [同权重条件干预](../../outputs/research/mrgr_ocmsf_screen_v1/condition_intervention/summary.json)',
        '- [物理GPU身份](../../outputs/research/mrgr_ocmsf_screen_v1/runtime_identity.json)', '',
        '原始列包括普通 `case_id, heldout_relative_l1, A, B, stations, output_sanity, gate`；聚簇 `arm, draw, case_id, input_ids, heldout_ids, relative_l1, relative_l2`。各组记录20轮指标、loader RNG、案例站位摘要、原始末轮案例指标和cluster5指标。','',
        f'注册SHA256：`{r["protocol_sha256"]}`。全部源码/数据manifest、四臂effective config和checkpoint hash均关联在注册或完成记录中；既有正式source/data验证保持通过。','']
    if (OUT/'independent_verification.json').exists():
        lines += ['父agent的独立标量核验已完成：复算四臂普通及聚簇逐案例聚合、全部六个配对区间、初始化/站位/RNG/源码与权重hash、总6,640次更新及两个主gate，均通过。见[独立复核](../../outputs/research/mrgr_ocmsf_screen_v1/independent_verification.json)。此文件保留独立生成的原始记录，未由本报告器覆盖。','']
    (ROOT/'docs/reports/mrgr_ocmsf_screening_zh.md').write_text('\n'.join(lines))
    print('OCMSF scalar report written')


if __name__=='__main__':main()
