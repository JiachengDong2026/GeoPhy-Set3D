#!/usr/bin/env python3
"""Summarize the independent B decoder-capacity probe without reading GT."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/research/b_decoder_capacity_checkpoint_v1'
ARMS = ('reference64', 'wide128')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    reg = read(BASE / 'registration.json')
    summaries = {arm:read(BASE / arm / 'summary.json') for arm in ARMS}
    logs = {arm:rows(BASE / arm / 'metrics.jsonl') for arm in ARMS}
    assert all(s['complete'] and s['epochs']==20 and s['optimizer_updates']==160 for s in summaries.values())
    assert len({s['reference_initialization_sha256'] for s in summaries.values()}) == 1
    assert len({s['final_loader_generator_sha256'] for s in summaries.values()}) == 1
    for arm, summary in summaries.items():
        assert summary['source_preserved']
        assert all(summary['gt_audit'][k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        assert len(logs[arm]) == 20
        assert all(row[k]['complete_epoch'] for row in logs[arm] for k in ('train','val'))
    for a, b in zip(logs[ARMS[0]],logs[ARMS[1]]):
        assert a['epoch'] == b['epoch'] and a['learning_rate'] == b['learning_rate']
    original = summaries[ARMS[0]]['initial_val']['measurement']['heldout']['relative_l1']
    wide = summaries[ARMS[1]]['initial_val']['measurement']['heldout']['relative_l1']
    assert abs(original-wide) < 1e-5
    h = lambda s: s['final']['val']['measurement']['heldout']['relative_l1']
    final = {a:h(s) for a,s in summaries.items()}
    improvement = 1-final[ARMS[1]]/final[ARMS[0]]
    last5 = {a:sum(r['val']['measurement']['heldout']['relative_l1'] for r in logs[a][-5:])/5 for a in ARMS}
    result = dict(complete=True, summaries=summaries,
        matched_common_initialization=True, matched_loader_generator=True,
        matched_lr_prefix=True, initial_val_H_abs_difference=abs(original-wide),
        final_val_H=final, final_relative_improvement=improvement,
        posthoc_last5_val_H=last5,
        gt_evaluated=False, formal_qualification=False,
        sources_preserved=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in reg['code'].items()))
    assert result['sources_preserved']
    (BASE / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for arm in ARMS:
        x=[r['epoch'] for r in logs[arm]]
        axes[0].plot(x,[100*r['train']['measurement']['heldout']['relative_l1'] for r in logs[arm]],label=arm)
        axes[1].plot(x,[100*r['val']['measurement']['heldout']['relative_l1'] for r in logs[arm]],label=arm)
    for ax,title in zip(axes,('Online training measurement error','Independent 8-case validation error')):
        ax.set_title(title);ax.set_xlabel('Epoch (cosine T_max=300 prefix)')
        ax.set_ylabel('Projection relative L1 (%)');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
    fig.savefig(BASE/'curves.png',dpi=170);plt.close(fig)
    lines=[
        '# B 解码器容量：固定小数据测量诊断 v1（匹配激活检查点）','',
        '**执行调整：** 原始width128测速在24GB GPU1上显存不足，失败日志与原reference64完整20轮保留。另注册本实验，两臂均启用相同的activation checkpointing，并从随机初始化重做，不混用原reference指标或测速权重。Q96、射线与query chunk、FP32、网络定义和损失保持不变；此处时间对应checkpoint运行方式，不能当作正式perf_v2默认执行速度。','',
        '**范围：** 固定16个训练案例及其epoch0站位，另8个验证案例；两臂各20轮、160次更新，使用300轮cosine学习率的前缀。所有训练从随机权重开始，采用原B的完整Q96、六路连续投影、64³一致性与损失。未读取体积GT或最终test，也没有启动新的300轮。','',
        '**唯一结构变量：** 四层坐标MLP宽度64→128，encoder/refiner/坐标编码/观测条件均保持不变。将原64个神经元及连接嵌入128维，额外神经元向原子网和最终输出的连接零初始化，确保初始函数保持一致。新神经元保留随机输入，允许新增连接学习。该初始化有利于匹配，并非检验所有宽网络初始化。','',
        '| 项目 | reference64 | wide128 |','|---|---:|---:|',
    ]
    metric_rows=[
        ('参数量',lambda s:f"{s['parameters']:,}"),
        ('初始验证投影L1',lambda s:f"{100*s['initial_val']['measurement']['heldout']['relative_l1']:.4f}%"),
        ('第20轮在线训练投影L1',lambda s:f"{100*s['final']['train']['measurement']['heldout']['relative_l1']:.4f}%"),
        ('第20轮验证投影L1',lambda s:f"{100*h(s):.4f}%"),
        ('前20轮最佳验证投影L1',lambda s:f"{100*s['best_val_H']:.4f}%"),
        ('总运行时间',lambda s:f"{s['elapsed_seconds']/60:.2f}分钟"),
    ]
    for label,fn in metric_rows:
        lines.append('| '+label+' | '+' | '.join(fn(summaries[a]) for a in ARMS)+' |')
    lines += ['',f"宽MLP第20轮独立案例验证误差相对变化为 **{(-improvement)*100:+.2f}%**（负值为改善）。这是当前固定小数据和早期学习率前缀的结果，不能预测完整训练的终局，也不构成三维优劣或跨种子结论。",'',
        f"曲线存在交叉：宽MLP早期学习更快，最佳验证值略低；事后描述的末5轮验证均值为{100*last5['wide128']:.4f}%，对照为{100*last5['reference64']:.4f}%。这些混合信号不能替代事先指定的第20轮终点，也说明不能简单归结为扩宽完全没有作用。",'',
        '![B两臂小规模测量曲线](../../outputs/research/b_decoder_capacity_checkpoint_v1/curves.png)','',
        '**解释边界：** 训练H参与原目标，本来就是训练测量，不是独立的泛化验证；表中训练误差还是一个epoch内随权重更新的在线均值，并非末权重重新遍历训练集的拟合误差。独立验证使用另外8个案例，但样本量小、只看一种固定站位。完整166训练集测量仅用于沿用原B的全训练池标量浓度初始化，两个结构完全相同；不应声称整个过程只读取16例。','',
        '**匹配与审计：** 共同初始化哈希、最终loader generator状态、每轮学习率完全匹配；16训练/8验证及A/B/H站位在注册文件固定。两臂均完成完整epoch，GT/禁止NPY/外来检查点尝试为0，冻结源与本实验注册源哈希未变。全程GPU1上的FP32训练，未复用测速权重。没有对短程检查点做体积评价，也没有导出可正式选模的权重。','',
        '**后续：** 仅凭这一容量探针不启动B长训练。若训练和独立验证同步改善，可将容量/条件特征列为后续受控因素；若无改善，则暂缓扩大同类MLP。无论结果如何，都还需区分条件特征、连续积分与空间消歧的贡献。','',
        '入口：[probe_b_decoder_capacity_checkpoint_v1.py](../../scripts/probe_b_decoder_capacity_checkpoint_v1.py)；注册：[registration.json](../../outputs/research/b_decoder_capacity_checkpoint_v1/registration.json)；结果：[comparison.json](../../outputs/research/b_decoder_capacity_checkpoint_v1/comparison.json)。原失败证据：[OOM日志](../../outputs/research/b_decoder_capacity_v1/benchmark_wide128.log)；额外数据身份核验：[supplement](../../outputs/research/b_decoder_capacity_v1/data_identity_supplement.json)，全部匹配先前正式B数据哈希。',
    ]
    path=ROOT/'docs/reports/b_decoder_capacity_v1_zh.md'
    path.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(path),final_val_H=final,relative_improvement=improvement)))


if __name__=='__main__':
    main()
