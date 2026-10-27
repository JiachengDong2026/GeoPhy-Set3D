"""Read-only progress snapshot; never changes models, gates, or run state."""
from feasible_context import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo


def journal(path):
    if not path.exists():return []
    out=[]
    for line in path.read_text().splitlines():
        try:out.append(json.loads(line))
        except json.JSONDecodeError:break
    return out


def main():
    teacher=journal(ROOT/'teachers/ledger.jsonl')
    lines=['# 经典基线正式训练与调试：进度快照','',datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'',
        f'仅train测量自生成场：{len(teacher)}/166，GT未用于训练。此文件为进度快照，最终状态以各阶段summary及测试封存记录为准。','',
        '## U-Net完整训练','']
    rows=journal(ROOT/'unet/metrics.jsonl')
    if rows:
        last=rows[-1];lines.append(f"当前阶段：{last['stage']}；epoch {last['epoch']}；该阶段更新 {last['updates']}；完整36Val H={selection_metric(last['val']):.7g}。")
        figure,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for stage in ['preheat','physical']:
            points=[v for v in rows if v['stage']==stage]
            if points:
                axes[0].plot([v['epoch'] for v in points],[v['train'].get('normalized_MSE',v['train'].get('loss',{}).get('total')) for v in points],label=stage)
                axes[1].plot([v['epoch'] for v in points],[selection_metric(v['val']) for v in points],label=stage)
        axes[0].set(title='Training losses (different stage objectives)',xlabel='Stage epoch',ylabel='Loss')
        axes[1].set(title='Full36Val physical H relative L1',xlabel='Stage epoch',ylabel='Relative L1')
        for ax in axes:ax.legend()
        figure.savefig(ART/'unet_progress.png',dpi=160);plt.close(figure)
        lines+=['','![U-Net训练曲线](artifacts/classical_feasible_v1/unet_progress.png)']
    else:lines.append('等待166个训练案例自生成场准备完成，尚未开始完整训练。')
    if (ART/'unet_summary.json').exists():
        r=read(ART/'unet_summary.json');lines+=['',f"当前训练结束状态：{r['decision']}；物理epoch={r['epochs']}；bestVal H={r['best_val_H']:.7g}。"]
    corrected=journal(ROOT/'sparse_guard_v2/unet/metrics.jsonl')
    if corrected:
        r=corrected[-1]
        lines+=['','## U-Net：独立稀疏场判据修订 v2','',
            '上图和原始STOP记录属于v1。v2保留原175轮停止结论，恢复同一完整边界继续训练；只修订饱和区域的统计权重，不改变网络或原目标。',
            '',f"v2当前物理epoch {r['epoch']}/300，累计更新 {r['updates']}，完整36Val H={selection_metric(r['val']):.7g}。",
            '', '[完整对比、判据变化和测试结果](classical_feasible_v1_training_test_zh.md)。']
    lines+=['','## 其他算法：单案例固定预算诊断','', '| 模型 | 稠密预热后H | 原目标末态H | E（未参与优化） | 状态 |','|---|---:|---:|---:|---|']
    summary=[]
    for arm in ['deeponet','fno','senseiver','gnot','deepsets']:
        path=ART/f'debug_{arm}.json'
        if path.exists():
            r=read(path)
            if r['status']=='bounded_complete':
                d,f=r['dense_final'],r['final'];passed=r['dense_MSE_half'] and r['fitted_ABH'] and r['E_below099']
                lines.append(f"| {arm} | {d['measurement']['heldout']:.7g} | {f['measurement']['heldout']:.7g} | {f['extra_E_H']:.7g} | {'局部通过，需多案例验证' if passed else '本预算未通过'} |")
                summary.append(dict(arm=arm,dense_updates=200,physical_updates=200,dense_H=d['measurement']['heldout'],physical_H=f['measurement']['heldout'],E=f['extra_E_H'],local_pass=passed))
            else:lines.append(f"| {arm} | — | — | — | {r['status']} |")
        else:
            dense=len(journal(ROOT/'debug'/arm/'dense_updates.jsonl'));physics=len(journal(ROOT/'debug'/arm/'physical_updates.jsonl'))
            lines.append(f'| {arm} | — | — | — | dense {dense}/200，physical {physics}/200 |')
    lines+=['','## 其他算法：多案例检查','']
    for arm in ['deeponet','fno','senseiver','gnot','deepsets']:
        pilot=journal(ROOT/('pilot_'+arm)/'metrics.jsonl')
        if pilot:
            r=pilot[-1];lines.append(f"- {arm}：{r['stage']} epoch{r['epoch']}，完整Val H={selection_metric(r['val']):.7g}。")
    if summary:
        with (ART/'debug_progress.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
    lines+=['','正式测试仅在U-Net完成300轮、模型选择与独立复核冻结后运行。单案例误差不构成三维GT精度或跨案例成功证明。','']
    (PROJECT/'docs/reports/classical_feasible_v1_progress_zh.md').write_text('\n'.join(lines))
    print('Progress snapshot written; teachers',len(teacher),'training records',len(rows))


if __name__=='__main__':main()
