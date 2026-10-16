#!/usr/bin/env python3
"""Read-only scalar report for the completed, preregistered local-detail screen."""
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research/mrgr_local_detail_v1_seed20261011'


def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main():
    protocol=read(OUT/'protocol.json');gate=read(OUT/'gate.json')
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==(OUT/'protocol.sha256').read_text().strip()
    for name,digest in protocol['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    data={a:rows(OUT/a/'epoch20_per_case.jsonl') for a in ('multiscale','local')}
    assert [x['case_id'] for x in data['multiscale']]==[x['case_id'] for x in data['local']]
    arrays={a:np.array([x['heldout_relative_l1'] for x in rr]) for a,rr in data.items()}
    summaries={a:dict(mean=float(x.mean()),p90=float(np.quantile(x,.9)),maximum=float(x.max())) for a,x in arrays.items()}
    assert np.isclose(summaries['local']['p90']/summaries['multiscale']['p90'],gate['ordinary_P90_ratio'])
    worst=np.argsort(arrays['local'])[::-1][:6]
    records=[dict(case_id=data['local'][i]['case_id'],reference=float(arrays['multiscale'][i]),local=float(arrays['local'][i]),delta=float(arrays['local'][i]-arrays['multiscale'][i])) for i in worst]
    hard=read(OUT/'hard_panel.json')['means']
    metrics={a:rows(OUT/a/'metrics.jsonl') for a in data}
    for x,y in zip(metrics['multiscale'],metrics['local']):
        assert x['loader_rng_sha256']==y['loader_rng_sha256']
        for split in ('train','val'):assert x[split]['case_station_digest']==y[split]['case_station_digest']
    audit={a:read(OUT/a/'summary_epoch20.json')['gt_audit'] for a in data}
    assert all(all(x[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts')) for x in audit.values())
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for a,rr in metrics.items():
        axes[0].plot([x['epoch'] for x in rr],[x['train']['loss']['total'] for x in rr],label=a)
        axes[1].plot([x['epoch'] for x in rr],[100*x['val']['measurement']['heldout']['relative_l1'] for x in rr],label=a)
    for ax,label in zip(axes,('Training objective','Validation heldout projection relative L1 (%)')):
        ax.set(xlabel='Epoch (300-epoch cosine prefix)',ylabel=label,yscale='log');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'screen_comparison.png',dpi=160);plt.close(fig)
    result=dict(gate=gate,ordinary=summaries,hard_H=hard,worst_six_by_local_H=records,
        gt_audit=audit,independent_scalar_verification_passed=True,formal300_started=False,volumetric_GT_scoring=False,
        short_screen_case_unit=36,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'screen_report_facts.json').write_text(json.dumps(result,indent=2)+'\n')
    prefix='../../outputs/research/mrgr_local_detail_v1_seed20261011/'
    lines=['# 多尺度上下文＋轻量局部细节分支：20轮匹配筛查结果','',
        '**结论：未通过预先固定的全部晋级条件，未启动300轮训练。** 普通H均值及聚簇五布局均有小幅改善，但普通案例P90恶化约4.78%，超过2%上限。不能根据平均值收益忽略尾部，也不能据此断定局部细节分支长期无效。','',
        '## 1. 实验与资源','',
        '两组均为166训练/36验证案例，seed20261011，全新且共享初始张量一致；同GPU0，各20轮/1660更新。FP32、batch2、原测量自监督目标和300轮cosine前缀不变。局部版345,574参数，较原多尺度344,678新增896（0.26%）。初始完整输出逐位相同，新增输出层梯度有限且非零。','',
        '## 2. 固定第20轮结果','',
        '**下列均为投影误差，不是三维浓度场L1/L2。** 普通H逐案例先对A/B重建平均；聚簇沿用五个冻结布局，先案例内平均再汇总36案例。','',
        '| 指标 | 原多尺度 | 局部细节版 | 变化 |','|---|---:|---:|---|',
        f"| 普通H均值 | {summaries['multiscale']['mean']:.4%} | {summaries['local']['mean']:.4%} | 相对降低{1-gate['ordinary_H_ratio']:.2%} |",
        f"| 普通H逐案例P90 | {summaries['multiscale']['p90']:.4%} | {summaries['local']['p90']:.4%} | 相对升高{gate['ordinary_P90_ratio']-1:.2%} |",
        f"| 普通H最大案例 | {summaries['multiscale']['maximum']:.4%} | {summaries['local']['maximum']:.4%} | 各自最差案例可不同 |",
        f"| 聚簇五布局H均值 | {hard['multiscale']:.4%} | {hard['local']:.4%} | 相对降低{1-gate['clustered_H_ratio']:.2%} |",
        f"| 第3–20轮平均单轮时间（含验证） | {gate['warm_epoch_seconds']['multiscale']:.2f}s | {gate['warm_epoch_seconds']['local']:.2f}s | 比值{gate['runtime_ratio']:.4f} |",'',
        f"普通H逐案例22胜14负；处理减对照的案例bootstrap均值差95%区间为[{gate['paired_case_delta_ci95'][0]*100:.4f}, {gate['paired_case_delta_ci95'][1]*100:.4f}]个百分点，跨零。短程均值的约1.05%相对改善还不足以认定稳定收益。该区间不覆盖训练种子变化。",'',
        f"![短程训练曲线]({prefix}screen_comparison.png)",'',
        '## 3. 预注册门槛与决定','',
        '| 条件 | 门槛 | 结果 |','|---|---|---|']
    names={'ordinary_H':('普通H均值','至少降低1%'),'ordinary_P90':('普通H的P90','恶化不超过2%'),'clustered_H':('聚簇五布局H','恶化不超过2%'),'runtime':('完整轮耗时比','不超过1.20'),'five_hour_budget':('300轮时间外推','不超过5h'),'smoke':('正确性/梯度/零GT读取','通过')}
    for k,passed in gate['checks'].items():
        name,threshold=names[k];lines.append(f"| {name} | {threshold} | {'通过' if passed else '未通过'} |")
    lines += ['',f"局部版300轮线性外推约{gate['projected_local_300_hours']:.2f}小时，但本轮没有执行300轮；因此没有正式训练完成的权重、三维L1/L2或重建形态结果。已准备的冻结三维评价入口未运行。",'',
        '## 4. 尾部案例','',
        '以下固定按局部版普通H误差从大到小取六例，不代表全体均退化。Δ为局部版减对照。','',
        '| case_id | 原多尺度H | 局部版H | Δ（百分点） |','|---|---:|---:|---:|']
    for x in records:lines.append(f"| {x['case_id']} | {x['reference']:.4%} | {x['local']:.4%} | {x['delta']*100:+.4f} |")
    lines += ['', '## 5. 解释与后续','',
        '局部建模对均值可能有帮助，但当前形式没有保证尾部案例收益。零输出初始化仅保证起点相同，不保证训练后不会产生额外的局部错误。P90来自36例的上尾，有限案例、单种子和短程波动都影响结果；本轮不事后放宽门槛，也不将第20轮表现等同于300轮收敛结论。','',
        'P90升高是本次预注册门槛的点估计失败，不是统计显著退化的证明。两组各自最大案例误差反而从19.2990%降至17.6743%，因此不能写成所有困难案例或整个尾部都变差；当前更准确的判断是收益分布不一致。','',
        '建议保留原多尺度正式模型。若继续局部路线，应另行登记一个有界局部残差控制/不同局部路径的匹配试验，同时保留困难面板；若检验损失或困难采样，应作为另一因素单独比较。仅有投影变化无法确定峰值、质心及三维形态是否改善。','',
        '训练读取审计均为零GT/禁止NPY/外部监督权重尝试，未读最终test、未进行本轮GT三维评分。原多尺度/历史/A/B源码、权重和旧评价不变。','',
        f"证据：[冻结协议]({prefix}protocol.json)、[smoke]({prefix}smoke.json)、[晋级判断]({prefix}gate.json)、[逐案例及独立复核]({prefix}screen_report_facts.json)、[原多尺度逐案例]({prefix}multiscale/epoch20_per_case.jsonl)、[局部版逐案例]({prefix}local/epoch20_per_case.jsonl)。",'',
        'GPU3另行完成的[反馈控制探索](mrgr_feedback_gate_v2_screening_zh.md)同样暂未晋级正式训练。']
    diagnostic=OUT/'diagnostics/local_epoch20_intervention_v1/summary.json'
    if diagnostic.exists():
        lines += ['', '## 6. 局部分支实际干预复核','',
            '另在冻结第20轮权重上，对固定前8验证案例的A/B输入做正常与强制局部支路为零的同权重反事实。局部分支输出约为输入特征范数的1.64%–1.66%，关闭后C1场相对L1变化约2.08%，16/16输入变化非零，说明支路确实参与了重建。','',
            '该8例C1的H误差正常为8.9560%，关支路为8.8524%；这项检查没有显示正向局部贡献，但网络是联合训练的，不能据此认定独立重训的无局部模型必然更优。该固定8例子集也不能替代完整36例比较。诊断没有读取GT，不改变原失败门槛。','',
            f"[实际干预记录]({prefix}diagnostics/local_epoch20_intervention_v1/summary.json)、[诊断脚本](../../scripts/diagnose_mrgr_local_detail_v1.py)。"]
    report=ROOT/'docs/reports/mrgr_local_detail_screening_zh.md'
    report.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(ordinary=summaries,worst_cases=records,gate_passed=gate['passed'],report=str(report)),indent=2))


if __name__=='__main__':main()
