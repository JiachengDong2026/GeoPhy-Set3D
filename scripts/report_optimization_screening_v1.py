#!/usr/bin/env python3
"""Assemble completed screening evidence; no training, selection or GT access."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/optimization_screening_20261016'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    snapshot=read(OUT/'frozen_source_snapshot.json')
    scientific_sources={str(p.relative_to(ROOT)) for folder in ('src','research/plan_b') for p in (ROOT/folder).rglob('*.py')}
    recorded_sources={p for p in snapshot['files'] if p.startswith(('src/','research/plan_b/'))}
    assert scientific_sources == recorded_sources, 'Frozen scientific-source membership changed'
    changed=[p for p,h in snapshot['files'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed, changed
    a_dir=ROOT/'outputs/research/a_dynamic_steps_v1'
    m_dir=ROOT/'outputs/research/mrgr_multiscale_v1'
    b_dir=ROOT/'outputs/research/b_decoder_capacity_checkpoint_v1'
    a={arm:read(a_dir/'runs'/arm/'screening_summary.json') for arm in ('reference','bounded_dynamic')}
    m=read(m_dir/'comparison.json')
    hard=read(m_dir/'hard_panel/summary.json')
    b=read(b_dir/'comparison.json')
    scale=read(ROOT/'outputs/research/a_step_scaling_v1/summary.json')
    draws=read(ROOT/'outputs/research/ab_remaining_draws_v1/summary.json')
    runtime=read(ROOT/'outputs/research/mrgr_multiscale_runtime_v1/summary.json')
    a_logs={arm:rows(a_dir/'runs'/arm/'metrics.jsonl') for arm in a}
    m_logs={arm:rows(m_dir/arm/'metrics.jsonl') for arm in ('reference','multiscale')}
    b_logs={arm:rows(b_dir/arm/'metrics.jsonl') for arm in ('reference64','wide128')}
    b_last5={arm:sum(r['val']['measurement']['heldout']['relative_l1'] for r in log[-5:])/5 for arm,log in b_logs.items()}
    assert all(s['completed_screening'] and s['completed_epochs']==16 and s['optimizer_updates']==1328 for s in a.values())
    assert m['completed'] and b['complete']
    assert hard['completed'] and hard['exploratory'] and not hard['selection_changed'] and not hard['gt_used']
    assert all(s['epochs']==20 and s['updates']==1660 for s in m['summaries'].values())
    a_improvement=1-a['bounded_dynamic']['final_H']/a['reference']['final_H']
    summaries=dict(
        frozen_sources_preserved=True, checked_source_files=len(snapshot['files']),
        A_dynamic=a, A_final_H_relative_improvement=a_improvement,
        MRGR_multiscale=m, MRGR_hard_measurement_panel=hard,
        B_decoder_capacity=b, B_posthoc_last5_val_H=b_last5,
        frozen_AB_remaining_draws=draws, MRGR_full_epoch_runtime=runtime,
        final_test_evaluated=False, new_formal300_started=False,
        interpretation='Exploratory matched short-prefix measurement screens; only frozen-model step scaling and layout evaluation are scored with validation GT after prediction freeze.',
    )
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for condition,label in [('Val-A-V4','Ordinary V4'),('clustered-V4','Narrow clustered V4')]:
        records=sorted([r for r in scale['results'] if r['condition']==condition],key=lambda r:r['scale'])
        original=next(r['relative_l1'] for r in records if r['scale']==1)
        axes[0,0].plot([r['scale'] for r in records],[100*(r['relative_l1']/original-1) for r in records],marker='o',label=label)
    axes[0,0].axhline(0,color='gray',linewidth=.8)
    axes[0,0].set(title='Frozen A: late physical-step scaling',xlabel='Scale of steps 3 and 4',ylabel='Relative change in 3D L1 (%)')
    for arm,log in a_logs.items():
        axes[0,1].plot([r['epoch']+1 for r in log],[100*r['selection_metric'] for r in log],label=arm)
    axes[0,1].set(title='A: matched 16-epoch prefix',xlabel='Epoch',ylabel='Validation projection L1 (%)')
    for arm,log in m_logs.items():
        axes[1,0].plot([r['epoch'] for r in log],[100*r['val']['measurement']['heldout']['relative_l1'] for r in log],label=arm)
    axes[1,0].set(title='MRGR: matched 20-epoch prefix',xlabel='Epoch',ylabel='Validation projection L1 (%)')
    axes[1,0].set_yscale('log')
    for arm,log in b_logs.items():
        axes[1,1].plot([r['epoch'] for r in log],[100*r['val']['measurement']['heldout']['relative_l1'] for r in log],label=arm)
    axes[1,1].set(title='B: fixed 16-train / 8-val fitting probe',xlabel='Epoch',ylabel='Validation projection L1 (%)')
    axes[1,1].set_yscale('log')
    for ax in axes.flat:
        ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'screening_curves.png',dpi=170);plt.close(fig)
    tests=[
        ('A：有界动态步长',16,1328,a['reference']['final_H'],a['bounded_dynamic']['final_H'],a_improvement,'+2'),
        ('MRGR：多尺度空间分支',20,1660,m['final_reference'],m['final_multiscale'],m['relative_improvement'],'+12,608（3.80%）'),
        ('B：坐标MLP宽64→128',20,160,b['final_val_H']['reference64'],b['final_val_H']['wide128'],b['final_relative_improvement'],f"+{b['summaries']['wide128']['parameters']-b['summaries']['reference64']['parameters']:,}"),
    ]
    lines=[
        '# 三模型后续优化：并行小预算筛查结果','',
        '**本轮完成的范围：** A冻结后期步长诊断、A动态步长匹配短程训练、MRGR轻量多尺度匹配短程训练，以及B固定小数据解码容量诊断。没有启动新的300轮训练，没有访问最终test。','',
        '**读表方式：** 下表是各自同协议Control/Treatment的验证留出投影相对L1，均以固定终点为主；不是三维误差。A/MRGR全166训练与36验证案例，B为固定16训练与8验证案例。B不能与另外两行横向排名。所有短程采用300轮cosine前缀，不能当作完整收敛结果。','',
        '| 实验 | 每臂轮数 / 更新数 | 对照终点投影L1 | 改动终点投影L1 | 相对改善 | 新增参数 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for label,ep,updates,x,y,gain,params in tests:
        lines.append(f'| {label} | {ep} / {updates} | {100*x:.4f}% | {100*y:.4f}% | {100*gain:+.2f}% | {params} |')
    lines += ['',
        '![各线路开发诊断及短程投影曲线](../../outputs/research/optimization_screening_20261016/screening_curves.png)','',
        '**冻结A的已确认现象：** 将后两步物理位移缩为0，普通三维L1从9.0734%升至9.1313%，极窄聚簇从56.7503%降至55.7250%；困难组相对只改善1.81%，仍明显差于历史43.9439%。因此全局关步不是主要解决方案。此诊断复用冻结C2、保留反馈与先验，预测先全部封存，再独立读取验证GT；不是重新训练的模型。','',
        '**A动态控制的检验范围：** 只为后两步增加两个零初始化标量，按输入物理位移相对场范数调制步长，gamma在(0,2)，初始为1。不是完整几何控制器，也不保证关闭更新。共同初始化、数据站位及shuffle、损失、更新数和学习率匹配；零初始化近似继承原前向。诊断统计detached，gate不读取H或GT。','',
        '**A实际干预很弱：** 末轮后两步gamma均值仅1.0001024、1.0000538，偏离1约万分之一。相同末checkpoint正常gate/强制gamma1的验证H为8.3772968%/8.3773851%，只差0.0000883个百分点；正常gate再跑一次为8.3772960%。该版本无积极收益、不晋级，但两臂约1.1%的末轮差不能泛化为所有动态控制无效，当前主要局限是控制器几乎没有实际调制。','',
        '**MRGR多尺度的检验范围：** 原OARI、GCA、残差反馈不变，在共享decoder32³特征上加入12通道16³/8³上下文残差。参数332070→344678，零输出初始化保证初始完整前向相同。两臂分别占GPU0和GPU3共享运行，精度协议一致，但耗时比较受负载干扰。','',
        f"**MRGR是本轮优先候选：** 普通验证H相对降低{100*m['relative_improvement']:.2f}%，{m['wins']}/36案例改善；配对案例差的95%区间为[{100*m['paired_case_bootstrap_95ci'][0]:.4f}, {100*m['paired_case_bootstrap_95ci'][1]:.4f}]个百分点，普通P90从15.6033%降至14.2190%。两臂最佳均在第20轮，但中间有波动，不能说全程稳定领先。该区间不覆盖训练种子不确定性。",'',
        f"**MRGR困难测量补充：** 固定上述末轮权重，36案例各用5个聚簇布局，先在案例内平均，再汇总。留出H从{100*hard['summary']['reference']['heldout']['mean']:.4f}%降至{100*hard['summary']['multiscale']['heldout']['mean']:.4f}%，相对降低{100*hard['paired']['heldout']['relative_improvement']:.2f}%，{hard['paired']['heldout']['wins']}/36改善；P90从{100*hard['summary']['reference']['heldout']['p90']:.4f}%降至{100*hard['summary']['multiscale']['heldout']['p90']:.4f}%。这是事后探索面板，未改变选模，未读取三维GT；困难投影改善仍不能证明三维消歧改善。",'',
        '**B容量的检验范围：** 共同64维子网络嵌入128维，初始函数匹配，固定训练案例与站位，原连续积分和无标签损失不变。width128原测速显存不足后，独立注册并重做两臂相同activation checkpointing，失败日志及旧reference结果保留，不混入匹配表。训练H参与目标，在线训练误差不等于末权重的完整训练集评价。该探针不能确定所有连续表示是否有效，也不能给出容量误差下限。','',
        'B的“16训练案例”仅指梯度训练集合；两臂都沿用正式B的全166训练案例、全部Pool A投影统计初始化最终sigmoid bias，没有读取验证或GT作初始化。该共同标量初始化不受case limit限制。','',
        f"**B曲线有混合信号：** 宽MLP早期学习更快，前20轮最佳验证H为39.3823%，接近对照39.7021%；事后描述的末5轮均值为{100*b_last5['wide128']:.4f}%（对照{100*b_last5['reference64']:.4f}%）。这与第20轮较差的结论应同时保留，不能另选有利统计替换事先固定终点。当前证据不足以支持扩大MLP后直接长训，也不足以断言容量完全无贡献。",'',
        '**证据强度：** 三项训练均是单种子、不完整训练的开发筛查，没有体积GT评估；测量改善不能直接兑换成三维改善。MRGR的案例bootstrap只反映这36个案例变化，不含训练种子变化。不能因短程无收益就否定整类机制，也不能因短程有收益就宣称终局超过历史。','',
        f"**保全检查：** {len(snapshot['files'])}个历史src、B模型源及冻结运行适配器哈希均未变；旧检查点与旧结果未覆盖。各训练审计GT/禁止NPY/外来检查点尝试为0。新增代码均位于独立scripts，输出位于独立开发目录。",'',
        '另对全166训练/36验证投影及source_index、statistics、train geometry共205文件做[独立内容身份核验](../../outputs/research/optimization_screening_20261016/data_identity_postflight.json)，全部精确匹配旧正式A/B记录。这是筛查期间的独立读取核验，不称作补做的事前注册。','',
        '**详细报告：** [冻结A缩放](a_step_scaling_v1_zh.md)、[A动态步长](a_dynamic_steps_v1_zh.md)、[MRGR多尺度](mrgr_multiscale_v1_zh.md)、[B容量](b_decoder_capacity_v1_zh.md)、[A/B剩余draw](ab_remaining_draws_v1_zh.md)。完整数值见[汇总JSON](../../outputs/research/optimization_screening_20261016/summary.json)。',
    ]
    lines += ['', '**补齐历史五draw后的三维结果：** 新增864场，全部先保存封存后独立评分。先在每个案例内平均5个draw，再对36案例汇总和配对统计，不能把180份布局当作独立案例。','',
        '| 条件（五draw） | 历史三维L1 | A三维L1 | B三维L1 | A对历史相对改善 |',
        '|---|---:|---:|---:|---:|']
    for row in draws['results']:
        if row['draws_per_case']==5:
            v=row['relative_l1']
            lines.append(f"| {row['condition']} | {100*v['history']:.4f}% | {100*v['A']:.4f}% | {100*v['B']:.4f}% | {100*row['A_versus_history']['relative_error_reduction']:+.2f}% |")
    lines += ['', 'A在V3/V6每个draw均宏平均改善，在聚簇每个draw均恶化；B三条件的案例内五draw平均均为0胜36负，但个别单draw有少量案例胜利。困难退化不是draw0偶然，不应把某个单draw的绝对误差与五draw均值混用。','',
        '**空闲GPU2完整轮次测速：** 两臂均从共有随机初始化开始，分别跑2个完整训练+验证epoch；测速权重不保存、不用于模型筛选，包含83次实际更新及36案例验证。下表采用第二轮热态，排除启动、建模和checkpoint写盘。','',
        '| 模型 | 训练秒/更新 | 完整训练+验证秒/轮 | 300轮线性估算 |',
        '|---|---:|---:|---:|']
    for arm in ('reference','multiscale'):
        rr=runtime['arms'][arm];warm=rr['records'][-1]
        lines.append(f"| {arm} | {warm['seconds_per_update']:.4f} | {warm['epoch_seconds']:.2f} | {rr['estimated_300_hours']:.2f}小时 |")
    lines += ['', '这是当时GPU2无其他计算进程时的短测外推；仅两轮，几何采样随epoch变化，未计完整300轮checkpoint成本，未来共享负载也可能改变，因此不承诺墙钟时间。原GPU0/3并行训练记录不能作为独占GPU速度排名。原始测速见[mrgr_multiscale_runtime_v1/summary.json](../../outputs/research/mrgr_multiscale_runtime_v1/summary.json)。']
    lines += ['', '**下一步优先级：** 先将MRGR轻量多尺度版作为唯一优先完整训练候选，保留当前对照协议、300轮日程与固定验证选择规则；空闲卡早期外推约2.89小时，新增成本约3.74%，符合5小时预算的可行性要求。完整训练后先封存预测，再在相同普通/五draw聚簇验证面板评价三维L1、P90及逐案例变化，同时报告测量误差；只有此后才能判断是否值得多种子重复与最终test。A当前近恒等动态gate暂不晋级，后续控制器须先证明有实际调制；B暂缓单纯扩宽，优先查条件特征与空间约束。上述完整训练尚未启动。']
    path=ROOT/'docs/reports/optimization_screening_20261016_zh.md'
    path.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(path),A_improvement=a_improvement,MRGR_improvement=m['relative_improvement'],B_improvement=b['final_relative_improvement'])))


if __name__=='__main__':
    main()
