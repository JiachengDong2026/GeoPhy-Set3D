#!/usr/bin/env python3
"""Report saved probe scalars only: never opens data, GT, or any checkpoint."""
import csv
import gzip
import json
from pathlib import Path
import sys
import shutil
REPO=Path(__file__).resolve().parents[1]
NAME='ray_senseiver_init_probe_v1'
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons')/NAME
ART=REPO/'docs/reports/artifacts'/NAME
REPORT=REPO/'docs/reports/ray_senseiver_init_probe_v1_zh.md'


def read(p):return json.loads(p.read_text())
def number(x):return f'{x:.8g}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms={arm:dict(history=read(OUT/arm/'history.json'),health=read(OUT/arm/'health.json'),events=read(OUT/arm/'events.json'),complete=read(OUT/arm/'complete.json')) for arm in ['control','treatment']}
    scale=read(OUT/'scale.json');registration=read(OUT/'registration.json');initial=read(OUT/'initialization.json')
    last=arms['treatment']['health'][-1];last3=arms['treatment']['health'][-3:]
    checks=dict(last3_H_below_090=len(last3)==3 and all(h['full_fixed_subset']['measurement']['heldout']['relative_l1']<.90 for h in last3),
                no_engineering_or_collapse=not (OUT/'treatment/engineering_error.json').exists() and arms['treatment']['complete']['reason']!='NEAR_ZERO_COLLAPSE',
                identifiable_case_changes=any(r['relative_MA']>=1e-3 for r in last['case_differences']),
                dependence_at_least_two_cases=sum(r['measurement_dependence']['relative_MA']>=1e-3 for r in last['diagnostic_cases'])>=2,
                valid_gradients_and_updates=all(all(v>0 for v in r['gradient_l2'].values()) for h in last3 for r in h['diagnostic_cases']) and all(all(v>0 for v in o['update_l2'].values()) for r in arms['treatment']['history'] for o in r['update_observations']))
    decision='GO' if all(checks.values()) else 'NO-GO'
    common=min(len(a['history']) for a in arms.values())
    matched=[]
    for epoch in [0,*range(1,common+1)]:
        row=dict(epoch=epoch,updates=4*epoch)
        for arm,a in arms.items():
            stat=a['history'][epoch-1]['eval'] if epoch else a['health'][0]['full_fixed_subset']
            row[arm]=dict(H_L1=stat['measurement']['heldout']['relative_l1'],H_L2=stat['measurement']['heldout']['relative_l2'],loss=stat['loss'],output=stat['output'])
        row['Treatment_minus_Control_H_L1']=row['treatment']['H_L1']-row['control']['H_L1'];matched.append(row)
    result=dict(decision=decision,checks=checks,matched_last_epoch=common,matched_last_updates=common*4,
                treatment_last_three_health_H=[dict(epoch=h['epoch'],H=h['full_fixed_subset']['measurement']['heldout']['relative_l1']) for h in last3],
                budgets={arm:a['complete'] for arm,a in arms.items()},boundary='Fixed train subset, H included in loss; no independent Val/generalization or voxel accuracy claim; no GT/PoolB/test; no automatic formal training')
    (ART/'decision.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (ART/'matched_comparison.json').write_text(json.dumps(matched,indent=2,allow_nan=False)+'\n')
    with (ART/'epoch_scalars.csv').open('w') as f:
        fields=['arm','epoch','updates',*[f'{g}_{m}' for g in ['heldout','cross','observed'] for m in ['relative_l1','relative_l2']],*[f'loss_{k}' for k in ['total','hold','cross','observed','consistency']],*[f'output_{k}' for k in ['min','max','mean','std','case_to_case_mean_absolute_difference']], 'seconds','gpu_event_elapsed_ms','peak_gpu_allocated_bytes']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for arm,a in arms.items():
            for r in [dict(epoch=0,updates=0,eval=a['health'][0]['full_fixed_subset']),*a['history']]:
                stat=r['eval'];writer.writerow(dict(arm=arm,epoch=r['epoch'],updates=r['updates'],**{f'{g}_{m}':stat['measurement'][g][m] for g in ['heldout','cross','observed'] for m in ['relative_l1','relative_l2']},**{f'loss_{k}':v for k,v in stat['loss'].items()},**{f'output_{k}':stat['output'][k] for k in ['min','max','mean','std','case_to_case_mean_absolute_difference']},**{k:r.get(k,0) for k in ['seconds','gpu_event_elapsed_ms','peak_gpu_allocated_bytes']}))
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for arm,a in arms.items():
        values=[a['health'][0]['full_fixed_subset'],*[r['eval'] for r in a['history']]];epochs=list(range(len(values)))
        for i,g in enumerate(['heldout','cross','observed']):
            for j,m in enumerate(['relative_l1','relative_l2']):
                ax=axes[i,j];ax.plot(epochs,[r['measurement'][g][m] for r in values],label=arm);ax.set_title(f'{g} {m} (end epoch)');ax.axhline(1,color='grey',ls='--');ax.set_yscale('log')
                ax.plot(epochs[1:],[r['train']['measurement'][g][m] for r in a['history']],ls=':',alpha=.6,label=arm+' train aggregate')
        for i,k in enumerate(['mean','std','case_to_case_mean_absolute_difference']):
            ax=axes[i,2];ax.plot(epochs,[r['output'][k] for r in values],label=arm);ax.set_title('Output '+k);ax.set_yscale('log')
    for ax in axes.flat:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.savefig(ART/'measurement_output_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for arm,a in arms.items():
        epochs=list(range(len(a['history'])+1));values=[a['health'][0]['full_fixed_subset'],*[r['eval'] for r in a['history']]]
        for i,k in enumerate(['total','hold','cross','observed','consistency']):
            ax=axes.flat[i];ax.plot(epochs,[r['loss'][k] for r in values],label=arm);ax.set_title('Loss '+k);ax.set_yscale('log')
            ax.plot(epochs[1:],[r['train']['loss'][k] for r in a['history']],ls=':',alpha=.6,label=arm+' train aggregate')
        for i,g in enumerate(['encoder','latents','decoder']):
            obs=[o for r in a['history'] for o in r['update_observations']];ax=axes.flat[i+5]
            ax.plot([r['step'] for r in obs],[r['update_l2'][g] for r in obs],label=arm+' update');ax.plot([r['step'] for r in obs],[r['post_clip_gradient_l2'][g] for r in obs],label=arm+' grad',ls=':');ax.set_yscale('log');ax.set_title(g+' update / clipped grad');ax.set_xlabel('Optimizer step')
        axes.flat[8].plot([r['epoch'] for r in a['history']],[r['seconds'] for r in a['history']],label=arm);axes.flat[8].set_title('Wall seconds including health')
    for ax in axes.flat:ax.legend();ax.grid(alpha=.2)
    fig.savefig(ART/'loss_update_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for arm,a in arms.items():
        h=a['health'];epochs=[r['epoch'] for r in h]
        for q,style in [('0.01',':'),('0.5','-'),('0.99','--')]:axes[0,0].plot(epochs,[r['logits']['quantiles'][q] for r in h],style,label=arm+' q'+q)
        axes[0,1].plot(epochs,[r['logits']['fraction_below_minus12'] for r in h],label=arm)
        axes[0,2].plot(epochs,[r['logits']['sigmoid_derivative']['mean'] for r in h],label=arm)
        for i in range(3):
            axes[1,0].plot(epochs,[r['diagnostic_cases'][i]['measurement_dependence']['relative_MA'] for r in h],label=arm+f' case{i}')
        for g in ['encoder','latents','decoder']:
            axes[1,1].plot(epochs,[sum(c['gradient_l2'][g] for c in r['diagnostic_cases'])/3 for r in h],label=arm+' '+g)
        axes[1,2].plot(epochs,[max(c['relative_MA'] for c in r['case_differences']) for r in h],label=arm)
    for ax,title in zip(axes.flat,['Actual postproc logit quantiles','Fraction logits < -12','Mean sigmoid derivative','Normal vs zero relative MA','Mean un-clipped diagnosis gradient L2','Max diagnostic case pair relative MA']):ax.set_title(title);ax.set_xlabel('Epoch');ax.legend(fontsize=7);ax.grid(alpha=.2)
    for ax in [axes[0,2],*axes[1]]:ax.set_yscale('log')
    axes[1,0].axhline(1e-3,color='grey',ls='--')
    fig.savefig(ART/'health_curves.png',dpi=150);plt.close(fig)
    for arm in arms:
        dest=ART/arm;dest.mkdir(exist_ok=True)
        for name in ['history.json','health.json','events.json','complete.json','access.json','environment.json']:
            with gzip.open(dest/(name+'.gz'),'wb') as f:f.write((OUT/arm/name).read_bytes())
        shutil.copyfile(OUT/f'{arm}_train.log',dest/'train.log')
    archive=read(REPO/'docs/reports/artifacts/ray_senseiver_selfsup_v1/early_termination_archive.json')
    lines=[f'# Ray-Senseiver 输出初始化两臂探针：{decision}',
           f'**结论：{decision}。** 本轮实际训练与预登记退出规则已执行。没有读取三维GT、Pool B、test或外部权重，未启动新的300轮。判定只针对固定8个训练case的任务适配可学习性，不代表原Senseiver能力上限，不证明三维重建正确。',
           '## 原失败 run 安全停止与历史边界',
           '原run在第113轮完整保存边界停止，9379 updates；FAILED_LEARNING / NEAR_ZERO_COLLAPSE / EARLY_TERMINATED。last/best保留，末轮Val H L1=1.000000950284751，后续900条GT评价取消。终止仅涉及该训练PID和loader子进程，未终止其他实验。第68轮独立诊断及原协议保留为历史记录；其当时“继续300轮”说明由本轮授权覆盖。',
           '[完整停止归档](artifacts/ray_senseiver_selfsup_v1/early_termination_archive.json)；[原独立诊断](ray_senseiver_learning_diagnostic_zh.md)。',
           '## 预登记、尺度与初态',
           '[中文协议](../ray_senseiver_init_probe_v1_protocol_zh.md)，[配置](../../configs/ray_senseiver_init_probe_v1.yaml)，[固定case/station/ray清单](artifacts/ray_senseiver_init_probe_v1/fixed_inputs.json.gz)。登记commit 8b74618，smoke commit 550ab56，均早于科学训练。两臂顺序前8例、相同A/B/H和batch，sampler始终epoch0，H仍用于原loss，不称泛化验证。',
           f'有效A/B射线{scale["valid_rays"]}条（包括有效零）；float64分子Σlength*y={number(scale["numerator"])} ppm·m²、分母1e6Σlength²={number(scale["denominator"])} 1e6·m²。χ_init={number(scale["chi_init"])}，调用原mole_to_mass后C_init={number(scale["C_init"])}，bias={number(scale["bias_logit"])}；clamp={scale["clamp_applied"]}。原projector全8例A/B常量场最大绝对偏差={number(max(r["max_abs_ppm_m"] for r in scale["projector_checks"]))} ppm·m。',
           'Treatment仅postproc weight乘0.01及bias设置；其余参数逐元素相等，64个输出权重全部非零。Control fresh SHA='+initial['control']+'；Treatment SHA='+initial['treatment']+'。工程smoke各独立1步，不计科学预算，正式初态重新构建且hash完全吻合，无smoke/失败权重warm start。',
           '## 实际预算与退出',
           table(['臂','完成epochs','updates','终止原因','epoch墙钟秒','峰值GiB'],[[arm,a['complete']['completed_epochs'],a['complete']['updates'],a['complete']['reason'],number(a['complete']['total_epoch_seconds']),number(a['complete']['peak_gpu_allocated_bytes']/2**30)] for arm,a in arms.items()]),
           '每轮4次真实更新，FP32/AdamW2e−4/wd1e−4/clip1/cosine T_max300，完整64³、all valid rays、K96和原loss。CUDA event与墙钟完整记录；GPU共享，含健康检查/host间隙的event时间不是独占kernel性能。epoch0健康检查不计epoch墙钟，完整进程墙钟另存。',
           '严格“第5/20轮后”从10/25检查开始，连续两次才退出；[逐事件与窗口证据](artifacts/ray_senseiver_init_probe_v1/control/events.json.gz)及[Treatment事件](artifacts/ray_senseiver_init_probe_v1/treatment/events.json.gz)保留。',
           '## 同预算比较与终点',
           f'两臂共同完成到epoch{common}/update{common*4}；下表只比较共同位置，各自最终状态另列。轮末评价是同一权重，训练中指标跨4次更新另存，不能相混。',
           table(['epoch/updates','Control H L1','Treatment H L1','T−C'],[[f'{r["epoch"]}/{r["updates"]}',number(r['control']['H_L1']),number(r['treatment']['H_L1']),number(r['Treatment_minus_Control_H_L1'])] for r in matched if r['epoch'] in {0,5,10,15,20,25,30,35,40,common}]),
           table(['臂/终点','H L1/L2','cross L1/L2','obs L1/L2','mean/std','case MA'],[[arm+'/'+str(a['history'][-1]['epoch']),*[number(s['measurement'][g]['relative_l1'])+'/'+number(s['measurement'][g]['relative_l2']) for g in ['heldout','cross','observed']],number(s['output']['mean'])+'/'+number(s['output']['std']),number(s['output']['case_to_case_mean_absolute_difference'])] for arm,a in arms.items() for s in [a['history'][-1]['eval']]]),
           table(['臂/轮','total','hold','cross','observed','consistency'],[[arm+'/'+str(epoch),*[number(s['loss'][k]) for k in ['total','hold','cross','observed','consistency']]] for arm,a in arms.items() for epoch,s in [(0,a['health'][0]['full_fixed_subset']),(a['history'][-1]['epoch'],a['history'][-1]['eval'])]]),
           '完整各loss值及L1/L2/输出曲线见[标量CSV](artifacts/ray_senseiver_init_probe_v1/epoch_scalars.csv)、[共同位置JSON](artifacts/ray_senseiver_init_probe_v1/matched_comparison.json)。',
           '![测量及输出](artifacts/ray_senseiver_init_probe_v1/measurement_output_curves.png)',
           '![loss和真实更新](artifacts/ray_senseiver_init_probe_v1/loss_update_curves.png)',
           '## 健康诊断与GO条件',
           table(['臂/轮','logit P01/P50/P99','<-12比例','sigmoid导数均值','诊断case最大relative MA'],[[arm+'/'+str(h['epoch']),'/'.join(number(h['logits']['quantiles'][q]) for q in ['0.01','0.5','0.99']),number(h['logits']['fraction_below_minus12']),number(h['logits']['sigmoid_derivative']['mean']),number(max(r['relative_MA'] for r in h['case_differences']))] for arm,a in arms.items() for h in [a['health'][0],a['health'][-1]]]),
           table(['臂/终点case','encoder/latents/decoder未clip梯度L2','normal−zero绝对MA','normal meanabs','relative MA'],[[arm+'/'+str(r['index']),'/'.join(number(r['gradient_l2'][g]) for g in ['encoder','latents','decoder']),number(r['measurement_dependence']['absolute_MA']),number(r['measurement_dependence']['normal_mean_abs']),number(r['measurement_dependence']['relative_MA'])] for arm,a in arms.items() for r in a['health'][-1]['diagnostic_cases']]),
           'normal/zero只置零测量，几何/path/mask逐tensor一致；relative MA=绝对MA/(normal meanabs+1e−12)。空间XYZ邻点变化、min/max/mean/std、case间绝对及相对差均存health。每step真实参数更新三组L2均非零；query259/37分块检查与全参数梯度检查全部通过，实际输出仍262144 query。原纯相对检查返回值原样保留，工程判断采用预登记的绝对+相对容差。',
           '![健康诊断](artifacts/ray_senseiver_init_probe_v1/health_curves.png)',
           'Treatment最后三次健康H：'+', '.join(f'epoch{h["epoch"]}={number(h["full_fixed_subset"]["measurement"]["heldout"]["relative_l1"])}' for h in last3)+'。',
           table(['最低GO条件','通过'],[[k,str(v)] for k,v in checks.items()]),
           '## 证据解释与下一阶段',
           '初始化尺度被纠正仅是初态事实；是否避免饱和、产生空间/case变化与测量依赖，必须同时看终点与全部轨迹。不同预算的终点不作为matched胜负。训练H含在loss里，下降只说明这8例的测量拟合；没有三维结构/跨观测泛化证据，也没有GT评价。consistency损失小不能解释为梯度小，本轮保持权重不变，不能宣称其唯一导致失败。',
           '## 审计与复现',
           '6项新增测试及9项Ray-Senseiver相关回归通过，覆盖物理单位/含零拟合、仅输出层变化、严格协议、退出边界/窗口、完整last恢复后下一步逐元素一致、访问保护；smoke覆盖原projector、有效ray计数/XYZ/完整输出、首步三组梯度非零、实际更新、自身状态恢复及query分块。边界拒绝测试触发的禁止尝试仅在测试进程，实际科学训练全部为0。',
           table(['阶段','unique Pool A projections','GT尝试','禁止NPY尝试','foreign权重尝试'],[['scale',scale['access']['unique_pool_a_projection_files'],scale['access']['gt_open_attempts'],scale['access']['forbidden_npy_attempts'],scale['access']['supervised_checkpoint_open_attempts']],*[[arm,a['complete']['access']['unique_pool_a_projection_files'],a['complete']['access']['gt_open_attempts'],a['complete']['access']['forbidden_npy_attempts'],a['complete']['access']['supervised_checkpoint_open_attempts']] for arm,a in arms.items()]]),
           'TrainEightGuard在原guard上进一步限制为注册8个train NPY；继承summary里的train/val文字描述基础guard，实际allowed只有8例。没有Val/test/Pool B访问。元数据和8例projection/source hashes封存且每臂启动核验；原数值源码未改。冻结Ours/DeepSets未调用、未训练、未改写。大checkpoint外部存放，Git只含清单、标量、图和报告。报告入口只读取已存JSON，不打开checkpoint或数据。',
           '[最终独立完整性核验](artifacts/ray_senseiver_init_probe_v1/final_integrity.json)确认各臂last全部AdamW state steps=140、scheduler epoch=35、checkpoint hash一致、原v1和探针冻结源码未变；全部8例A/B valid-ray数=实际packed-ray数，输入tensor hash与注册清单完全一致，几何有限且entry/exit在域内，禁止访问尝试0。该核验只读各臂自身last与固定8例train测量，零optimizer step。',
           table(['臂','last checkpoint SHA256','外部目录'],[[arm,a['complete']['last_sha256'],a['complete']['last_checkpoint']] for arm,a in arms.items()]),
           '复现入口：`python scripts/run_senseiver_init_probe.py --phase ...`；已有注册/科学完成目录拒绝覆盖，恢复仅`--phase train --arm ... --resume`读取本臂last。生成报告：`python scripts/report_senseiver_init_probe.py`。本轮到此停止，无额外超参数搜索或正式300轮。']
    idx=lines.index('## 审计与复现')
    if decision=='GO':
        lines.insert(idx,'本轮达到可学习性最低GO条件。唯一下一阶段建议：单独登记同一初始化的完整正式训练与独立Val测量选模；本轮只建议，不启动。')
    elif last['full_fixed_subset']['output']['mean']<.01*scale['C_init']:
        lines.insert(idx,'Treatment初始mean从Control的0.482742纠正到0.00155799，但末轮mean仅7.39725e−6（约0.004747×C_init）；H误差趋近零场基准，case差与测量依赖不足。末轮仅3.03% logits<−12，中位数−11.8144，**尚未触发>95%的严格塌缩规则**；分类是FAILED_TO_LEARN并有趋零/弱依赖迹象，不改写为NEAR_ZERO_COLLAPSE，也不是已观察到有效学习但缓慢。原run第68轮的严重负饱和证据不移植为本探针终点状态。结合原近零常量方向测量分支推低背景的证据，唯一下一阶段建议：单独登记原measurement discrepancy中relative-L1项开/关的有限归约对照，保持初始化与其余loss不变，检验极小/零射线的背景压低机制；本轮不执行，不认定因果已证实。')
    else:
        lines.insert(idx,'Treatment未达到全部最低GO条件，不能按已成功强基线启动正式训练。唯一下一阶段建议：单独登记相同初始化下consistency权重0与原0.05的有限对照，检验测量依赖/结构学习受一致性梯度约束的假设；本轮不执行，不把该分支认定为唯一原因。')
    REPORT.write_text('\n\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
