"""Summarize monitored continuation and preserve exact journals and provenance."""
from continuation_context import *
import csv,gzip,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    r=read(ART/'result.json');review=read(ART/'independent_review.json');assert review['passed']
    rows=r['metrics'];first,last=r['initial'],r['final']
    figure,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    axes[0].plot([v['step'] for v in rows],[v['loss'] for v in rows],'o-',label='Total')
    for key in ['hold','cross','observed','consistency']:
        axes[0].plot([v['step'] for v in rows],[v['parts'][key] for v in rows],label=key,alpha=.7)
    axes[0].set(title='Original objective and unweighted parts',ylabel='Loss')
    for key in ['heldout','cross','observed']:
        axes[1].plot([v['step'] for v in rows],[v['measurement'][key] for v in rows],label=key)
    axes[1].plot([v['step'] for v in rows],[v['extra_E_H'] for v in rows],'o-',label='E (used to monitor stopping)')
    axes[1].set(title='Measurement relative L1',ylabel='Relative L1')
    for label in ['A','B']:
        axes[2].plot([v['step'] for v in rows],[next(f for f in v['fields'] if f['branch']==label)['mean'] for v in rows],label=label+' mean')
    axes[2].set(title='Field concentration mean',ylabel='Concentration')
    for ax in axes:ax.set_xlabel('Additional AdamW updates');ax.legend(fontsize=6)
    figure.savefig(ART/'training_curves.png',dpi=180);plt.close(figure)
    compact=[]
    for v in rows:
        compact.append(dict(additional_updates=v['step'],original_objective_updates=100+v['step'],loss=v['loss'],H=v['measurement']['heldout'],
            observed=v['measurement']['observed'],cross=v['measurement']['cross'],E_monitor=v['extra_E_H'],
            mean_A=v['fields'][0]['mean'],mean_B=v['fields'][1]['mean'],std_A=v['fields'][0]['std'],std_B=v['fields'][1]['std']))
    with (ART/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(compact[0]),lineterminator='\n');writer.writeheader();writer.writerows(compact)
    for p in (ROOT/'case0').glob('*.jsonl'):
        with p.open('rb') as source,gzip.open(ART/(p.name+'.gz'),'wb') as target:shutil.copyfileobj(source,target)
    for name in ['launch.log','launch_attempt2.log','preflight.log','preflight_attempt2.log']:
        p=ROOT/name
        if p.exists():shutil.copyfile(p,ART/name)
    task=read(ROOT/'scheduling/case0_continuation.json')
    write(ART/'resources.json',dict(task=task,snapshots=str(ROOT/'scheduling/resources.jsonl'),double_idle_snapshots=read_snapshots(),
        global_lock_directory='/scratch/djc/GeoPhy-Set3D/gpu_locks',peak_gpu_bytes=r['peak_gpu_bytes'],foreign_processes_terminated=False))
    shutil.copyfile(Path(task['log']),ART/'train.log')
    reason=r['stop_reason']
    reason_zh={'budget_complete':'达到预定400步追加上限','unseen_E_persistently_worse_25percent':'E连续三次比起点恶化超过25%，按规则早停',
        'loss_and_H_plateau_two_100step_windows':'loss及H的两个100步窗口改善均不足1%，按规则早停',
        'loss_and_H_divergence':'loss和H持续发散，按规则早停','output_collapse':'输出塌缩，按规则早停'}.get(reason,reason)
    lines=['# U-Net 可行场预热后的原目标续训 v1','',
        '日期：2026-10-27（北京时间）。用户已授权继续训练、监测loss/误差并在异常或不收敛时中断。','',
        f"实际追加{r['actual_additional_updates']}次更新，原目标累计{r['total_original_objective_updates']}步，另有此前200步自生成场预热。停止原因：**{reason_zh}**。本轮沿用固定train0，并未扩大为跨案例正式训练。",'',
        '| 指标 | 续训起点 | 末态 | 相对起点改善 |','|---|---:|---:|---:|']
    for name,key in [('原总loss','loss'),('observed','observed'),('cross','cross'),('H（参与目标）','heldout'),('E（仅监测，参与早停）','E')]:
        a=first['loss'] if key=='loss' else first['extra_E_H'] if key=='E' else first['measurement'][key]
        b=last['loss'] if key=='loss' else last['extra_E_H'] if key=='E' else last['measurement'][key]
        lines.append(f'| {name} | {a:.8g} | {b:.8g} | {(1-b/a)*100:.2f}% |')
    lines+=['','![续训曲线](artifacts/unet_continuation_v1/training_curves.png)','',
        '## 结果解释','',
        '原目标的持续改善应同时看H和E，H始终参与训练。E来自同案例未输入的四个站位，本次用于早停监测，不能再称为未使用的最终测试集。结果不等同三维真值精度或跨案例泛化，未读取GT64/GT128、Pool B、final test或外来权重。','']
    if last['loss']<first['loss'] and last['measurement']['heldout']<first['measurement']['heldout'] and last['extra_E_H']<first['extra_E_H']:
        lines+=['原loss、H与E末态均较续训起点改善，支持自生成场预热后原目标可继续学习的局部可行路径。该结果针对单案例记忆，下一步关键问题是该路径能否在训练集多个案例上共享并泛化；不能据此宣布经典基线已经修复或与主模型公平比较。','']
    else:
        lines+=['原loss、H与E未同时改善，应结合下述监测节点分析；训练目标的下降不能单独证明重建改善。此时不自动延长预算或修改损失。','']
    if reason!='budget_complete':
        lines+=['本轮触发了预先固定的停止条件，已保留末态而没有自动改动超参数重试。E恶化指向同案例未见射线收益退化；loss/H平台指向当前路径下的优化停滞，二者不能混为物理目标无解。','']
    lines+=['## 监测节点','', '| 追加步数 | loss | H | cross | E监测 |','|---|---:|---:|---:|---:|']
    for v in rows:lines.append(f"| {v['step']} | {v['loss']:.7g} | {v['measurement']['heldout']:.7g} | {v['measurement']['cross']:.7g} | {v['extra_E_H']:.7g} |")
    lines+=['','## 恢复、审计与证据','',
        f"从原目标100步checkpoint恢复模型/AdamW/scheduler/RNG，AdamW累计step={r['total_original_objective_updates']}，scheduler={review['scheduler_last_epoch']}。保持原FP32目标、K96/64³和全部有效射线；未重置动量。CPU只读恢复逐值匹配原末态，GPU启动基线也通过容差核验。",'',
        f"独立NumPy重算初末4个完整场/16组K96积分，最大投影差相对目标L1={review['max_projection_relative_to_target']:.4g}；原目标及误差一致。原147源码/207输入、冻结checkpoint、历史证据与本轮源权重再次哈希核验不变。",'',
        '首次启动发生本地context模块名和既有调度器context的导入冲突，在GPU申请/optimizer更新前失败（0更新），随后仅重命名本地模块并重新检查。原登记protocol_initial.json、失败日志和新登记保留，未重放训练更新。','',
        '资源使用既有UUID全局锁与15秒双空闲快照，运行中监测外来进程冲突，仅占用空闲GPU。完整字段、checkpoint、日志：`/scratch/djc/GeoPhy-Set3D/unet_continuation_v1/`；Git保留小型证据和哈希，不合并main。','',
        f"末态checkpoint：`{r['checkpoint']}`；SHA256：`{r['checkpoint_sha256']}`。",'']
    (PROJECT/'docs/reports/unet_feasible_continuation_v1_zh.md').write_text('\n'.join(lines))
    print('Continuation report and training curves written')


def read_snapshots():
    return [json.loads(line) for line in (ROOT/'gpu_snapshots.jsonl').read_text().splitlines()]


if __name__=='__main__':main()
