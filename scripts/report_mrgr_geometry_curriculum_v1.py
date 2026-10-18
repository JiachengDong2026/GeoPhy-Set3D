#!/usr/bin/env python3
"""Report frozen low-dose geometry curriculum screening scalars only."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research/mrgr_geometry_curriculum_v1'
REPORT=ROOT/'docs/reports/mrgr_geometry_curriculum_v1_zh.md'
def read(p):return json.loads(Path(p).read_text())

def main():
    c=read(OUT/'comparison.json');p=read(OUT/'protocol.json');s=read(OUT/'sampler_audit.json')
    assert c['completed'] and not c['gt_used'] and not c['formal_training_started']
    logs={a:[json.loads(x) for x in (OUT/a/'metrics.jsonl').read_text().splitlines()] for a in ('control','curriculum')}
    for a,rows in logs.items():
        assert len(rows)==20 and [x['epoch'] for x in rows]==list(range(1,21))
        audit=read(OUT/a/'gt_access_audit.json')
        assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    prefix=[]
    for x,y in zip(logs['control'][:5],logs['curriculum'][:5]):
        assert x['train']['case_station_digest']==y['train']['case_station_digest']
        prefix.append(dict(epoch=x['epoch'],control_train_loss=x['train']['loss']['total'],
            curriculum_train_loss=y['train']['loss']['total'],train_loss_delta=y['train']['loss']['total']-x['train']['loss']['total'],
            control_val_H=x['val']['measurement']['heldout']['relative_l1'],curriculum_val_H=y['val']['measurement']['heldout']['relative_l1'],
            val_H_delta=y['val']['measurement']['heldout']['relative_l1']-x['val']['measurement']['heldout']['relative_l1']))
    (OUT/'pre_intervention_numeric_drift.json').write_text(json.dumps(dict(epochs=prefix,
        max_abs_train_loss_delta=max(abs(x['train_loss_delta']) for x in prefix),max_abs_val_H_delta=max(abs(x['val_H_delta']) for x in prefix),
        all_training_stations_identical=True,interpretation='CUDA grid_sample backward is nondeterministic; drift before intervention prevents strong causal attribution of small final effects. No intermediate epoch1-5 weights retained, so parameter drift is not measured.'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for arm,rows in logs.items():
        axes[0].plot([r['epoch'] for r in rows],[r['train']['loss']['total'] for r in rows],label=arm)
        axes[1].plot([r['epoch'] for r in rows],[r['val']['measurement']['heldout']['relative_l1']*100 for r in rows],label=arm)
    for ax,title in zip(axes,('Training objective','Fixed ordinary validation H relative L1 (%)')):
        ax.set_xlabel('Epoch (prefix of 300-epoch cosine schedule)');ax.set_title(title);ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'curves.png',dpi=160);plt.close(fig)
    replacement=sum(x['replacements'] for x in s['epoch_stats']);spans=[x for e in s['epoch_stats'] for x in e['spans_deg']]
    def change(r):return f"{(r-1)*100:+.2f}%"
    table=[]
    entries=[('普通 H 均值','ordinary_H',c['ordinary']['control']['mean'],c['ordinary']['curriculum']['mean']),
        ('普通 H 案例 P90','ordinary_P90',c['ordinary']['control']['p90'],c['ordinary']['curriculum']['p90']),
        ('聚簇五布局 H 均值','clustered_H',c['clustered_five_draw']['control'],c['clustered_five_draw']['curriculum'])]
    for name,key,x,y in entries:table.append(f"| {name} | {x*100:.4f}% | {y*100:.4f}% | {change(c['ratios'][key])} | {'通过' if c['checks'][key] else '未通过'} |")
    failures='、'.join(k for k,v in c['checks'].items() if not v) or '无'
    state='通过所有预注册小规模筛查条件，可以作为后续完整训练候选。尚未启动完整训练。' if c['passed'] else f'未通过所有晋级条件（{failures}），本轮止于20轮，不启动300轮。'
    text=f'''# 低比例困难几何课程：多尺度 MRGR 匹配筛查

{state}

这是单种子、已用于开发的36案例验证集、固定20轮的 **measurement-only** 结果；不是最终35例测试集评价，也没有三维GT评分。不得把投影改善直接称为三维形态改善。

## 预注册干预与匹配

基础网络保持344,678参数多尺度MRGR，两个arm均fresh初始化，共享初始权重、每轮166例案例顺序、A3–5/B3–5/H2视角数量、FP32、原自监督损失、AdamW和300轮cosine前缀。每臂20完整轮、1,660次更新，普通验证36例A4/B4/H4固定不变。唯一干预为训练站位采样。

前5轮完全不替换；第6–20轮按预定线性比例逐渐增加到每轮floor(166×10%)=16例。以epoch/case身份哈希排名确定配额案例；A保留原视角数量，改为20–40°局部方位窗口内的站位，两端跨度真实满足范围，内部站位确定性抽样。B/H从余下站位确定性抽取，三集合严格不交。原池不均匀，严格连续3站没有20–40°候选，因此定义为局部方位窗口内选站；候选3/4/5视角数量为{s['candidate_counts']}。

实际总替换 **{replacement}/3320={replacement/3320*100:.2f}%** epoch-case；每轮替换数为{[e['replacements'] for e in s['epoch_stats']]}。替换A的实际跨度最小/平均/最大 **{min(spans):.3f}/{np.mean(spans):.3f}/{max(spans):.3f}°**。3320条原/新ABH记录保存在station_plan.jsonl；确定性、未替换逐项相等、视角预算相等、三集合不交均通过。训练前5轮站位digest相同，后15轮按计划不同；全部轮次shuffle RNG和验证站位digest一致。

旧50/25/25增强导致普通三维误差恶化11.13%，本次采用独立预固定的低剂量渐进方案，不根据验证GT搜索比例或跨度。站位边际频率也随干预改变，因此不能将结果归因为纯粹角覆盖作用。

## 固定第20轮结果

| 指标（越低越好） | 对照 | 课程采样 | 相对变化 | 晋级条件 |
|---|---:|---:|---:|---|
{chr(10).join(table)}

普通主指标为每案例分别以A、B重建后投影到共同H，两个相对L1平均，再对36案例等权平均。P90按36案例指标计算。聚簇采用既有冻结5套布局，先每案例平均5draw，再平均36案例，不能把180条记录当独立案例。

普通逐案例candidate−control平均差95%配对bootstrap区间：{c['paired_ordinary_ci95']}，胜/负 **{c['wins']}/{c['losses']}**。聚簇差区间：{c['paired_cluster_ci95']}，胜/负 **{c['cluster_wins']}/{c['cluster_losses']}**。bootstrap20,000，seed2026101703；区间只描述固定seed/布局下案例差异，不代表训练种子稳定性。

## 干预前数值漂移

前5轮干预尚未开始，训练案例/站位和学习率完全相同，但两次运行仍产生数值漂移：Val H最大绝对差 **{max(abs(x['val_H_delta']) for x in prefix):.8f}**（原比例），训练total loss最大绝对差 **{max(abs(x['train_loss_delta']) for x in prefix):.8f}**。逐轮记录见pre_intervention_numeric_drift.json。CUDA grid_sample backward存在非确定性，微小数值差在优化中可以累积；本轮没有独立的多次重复训练，不能把最终小幅差异都归因于课程策略。未保存前5轮各轮权重，因此不声称量化了中间参数漂移。预固定gate保持原定义，但其通过仅为工程筛查，不能替代多种子机制验证。

## 成本与完整性

去掉最初2轮后平均含验证epoch耗时：对照 **{c['warm_epoch_seconds']['control']:.2f}s**、课程 **{c['warm_epoch_seconds']['curriculum']:.2f}s**；比值 **{c['ratios']['runtime']:.4f}**。按本次共享GPU3负载外推300轮 **{c['estimated300_hours']:.3f}小时**，不代表独占性能保证。两arm顺序运行，时段负载可影响耗时比较。

预固定晋级要求普通均值至少改善1%、P90退化≤2%、聚簇五布局退化≤2%、暖epoch耗时增幅≤20%、300轮外推≤5小时，并通过数值和输出检查。各项判定：`{c['checks']}`。即使全通过，也仅给出完整训练候选建议，本脚本不自动长训。

smoke初始完整模型输出bitwise一致、反向梯度有限；冻结第20轮全部36例输出有限且非空间常数。GT64/GT128、Pool B、final test和外来checkpoint禁止访问计数均为0。原数值代码、旧协议和历史checkpoint未改；首次smoke数值完成但JSON遇到numpy.bool_序列化异常，优化器更新前单独登记纯序列化适配器并重跑，不改变数学计算。

![训练曲线](../../outputs/research/mrgr_geometry_curriculum_v1/curves.png)

完整登记及标量结果：[protocol](../../outputs/research/mrgr_geometry_curriculum_v1/protocol.json)、[sampler audit](../../outputs/research/mrgr_geometry_curriculum_v1/sampler_audit.json)、[comparison](../../outputs/research/mrgr_geometry_curriculum_v1/comparison.json)。
'''
    # docs/reports is two levels below repository root.
    REPORT.write_text(text)
    print(REPORT)

if __name__=='__main__':main()
