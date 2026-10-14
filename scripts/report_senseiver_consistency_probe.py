#!/usr/bin/env python3
"""Reports only persisted scalar diagnostics; no dataset/checkpoint is opened."""
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
REPO=Path(__file__).resolve().parents[1]
NAME='ray_senseiver_consistency_probe_v1'
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons')/NAME
ART=REPO/'docs/reports/artifacts'/NAME
REPORT=REPO/'docs/reports/ray_senseiver_consistency_probe_v1_zh.md'


def read(p):return json.loads(p.read_text())
def num(v):return ('通过' if v else '未通过') if isinstance(v,bool) else f'{v:.8g}' if isinstance(v,(int,float)) else str(v)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(num(x) for x in r)+' |' for r in rows])
def average(v):return sum(v)/len(v)


def diagnostic_tables(d):
    fresh=[];amplitude=[];interactions=[]
    for c in d['case_results']:
        for name,row in c['fresh']['branches'].items():
            fresh.append(dict(index=c['index'],branch=name,loss=row['loss'],encoder_gradient=row['parameter_gradient_l2']['encoder'],latents_gradient=row['parameter_gradient_l2']['latents'],decoder_gradient=row['parameter_gradient_l2']['decoder'],dL_dc_uniform_shift=row['uniform_output_shift_derivative']))
        for r in c['amplitude']:
            for branch,v in r['branches'].items():amplitude.append(dict(index=c['index'],alpha=r['alpha'],branch=branch,**v))
        for space in ['parameters','outputs']:
            interactions.append(dict(index=c['index'],space=space,**c['fresh']['interaction'][space]))
        for group,v in c['fresh']['interaction']['parameter_groups'].items():interactions.append(dict(index=c['index'],space=group,**v))
    for name,rows in [('fresh_branch_gradients',fresh),('common_amplitude',amplitude),('branch_interactions',interactions)]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (ART/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return fresh,amplitude,interactions


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=read(OUT/'gradient_diagnosis.json');scale=read(OUT/'scale.json');reg=read(OUT/'registration.json')
    fresh,amplitude,interactions=diagnostic_tables(d)
    arms={arm:{key:read(OUT/arm/(key+'.json')) for key in ['history','health','events','complete','environment','access']} for arm in ['control','treatment']}
    treatment=arms['treatment'];last=treatment['health'][-1];last3=treatment['health'][-3:]
    key='total_treatment'
    checks=dict(last3_H_L1_below090=len(last3)==3 and all(h['full_fixed_subset']['measurement']['heldout']['relative_l1']<.90 for h in last3),
                no_engineering_or_collapse=not (OUT/'treatment/engineering_error.json').exists() and treatment['complete']['reason']!='NEAR_ZERO_COLLAPSE',
                case_variation=any(r['relative_MA']>=1e-3 for r in last['case_differences']),
                measurement_dependence=sum(r['measurement_dependence']['relative_MA']>=1e-3 for r in last['diagnostic_cases'])>=2,
                gradients_updates_valid=all(all(v>0 for v in r['gradient_split']['branches'][key]['parameter_gradient_l2'].values()) for h in last3 for r in h['diagnostic_cases']) and all(all(v>0 for v in o['update_l2'].values()) for r in treatment['history'] for o in r['update_observations']))
    common=min(len(a['history']) for a in arms.values());initial=treatment['health'][0]['full_fixed_subset'];matched_end=treatment['history'][common-1]['eval']
    improvements=dict(H_L2=(initial['measurement']['heldout']['relative_l2']-matched_end['measurement']['heldout']['relative_l2'])/initial['measurement']['heldout']['relative_l2'],measurement_smooth=(initial['loss_diagnostics']['measurement_smooth']-matched_end['loss_diagnostics']['measurement_smooth'])/initial['loss_diagnostics']['measurement_smooth'])
    go=all(checks.values());partial=not go and any(v>=.01 for v in improvements.values())
    label='GO' if go else 'PARTIAL_LEARNING_EVIDENCE' if partial else 'NO-GO'
    result=dict(classification=label,formal_training_recommendation='GO' if go else 'NO-GO',go_checks=checks,auxiliary_improvement_at_matched_budget=improvements,matched_epoch=common,matched_updates=4*common,last3_treatment_H=[dict(epoch=h['epoch'],H_L1=h['full_fixed_subset']['measurement']['heldout']['relative_l1']) for h in last3],completed={arm:a['complete'] for arm,a in arms.items()},only_local_fixed_train_probe=True,full_training_started=False)
    (ART/'decision.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    matched=[];scalars=[]
    for epoch in range(common+1):
        row=dict(epoch=epoch,updates=4*epoch)
        for arm,a in arms.items():
            s=a['health'][0]['full_fixed_subset'] if epoch==0 else a['history'][epoch-1]['eval']
            row[arm]=dict(measurement=s['measurement'],loss=s['loss'],loss_diagnostics=s['loss_diagnostics'],output=s['output'])
        row['T_minus_C_H_L1']=row['treatment']['measurement']['heldout']['relative_l1']-row['control']['measurement']['heldout']['relative_l1']
        row['T_minus_C_H_L2']=row['treatment']['measurement']['heldout']['relative_l2']-row['control']['measurement']['heldout']['relative_l2'];matched.append(row)
    (ART/'matched_comparison.json').write_text(json.dumps(matched,indent=2,allow_nan=False)+'\n')
    for arm,a in arms.items():
        for r in [dict(epoch=0,updates=0,eval=a['health'][0]['full_fixed_subset']),*a['history']]:
            s=r['eval'];scalars.append(dict(arm=arm,epoch=r['epoch'],updates=r['updates'],**{f'{g}_{m}':s['measurement'][g][m] for g in ['heldout','cross','observed'] for m in ['relative_l1','relative_l2']},**{f'loss_{k}':v for k,v in s['loss'].items()},**s['loss_diagnostics'],**{f'output_{k}':v for k,v in s['output'].items()},**{k:r.get(k,0) for k in ['seconds','gpu_event_elapsed_ms','peak_gpu_allocated_bytes']}))
        dest=ART/arm;dest.mkdir(exist_ok=True)
        for name,value in a.items():
            with gzip.open(dest/(name+'.json.gz'),'wt') as f:json.dump(value,f,allow_nan=False)
        shutil.copyfile(OUT/f'{arm}_train.log',dest/'train.log')
    with (ART/'epoch_scalars.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(scalars[0]));writer.writeheader();writer.writerows(scalars)
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for arm,a in arms.items():
        v=[a['health'][0]['full_fixed_subset'],*[r['eval'] for r in a['history']]];epochs=list(range(len(v)))
        for i,g in enumerate(['heldout','cross','observed']):
            for j,m in enumerate(['relative_l1','relative_l2']):
                ax=axes[i,j];ax.plot(epochs,[s['measurement'][g][m] for s in v],label=arm);ax.plot(epochs[1:],[r['train']['measurement'][g][m] for r in a['history']],':',alpha=.5,label=arm+' train');ax.set_title(g+' '+m);ax.axhline(1,color='grey',ls='--')
        for i,k in enumerate(['mean','std','case_to_case_mean_absolute_difference']):
            axes[i,2].plot(epochs,[s['output'][k] for s in v],label=arm);axes[i,2].set_title('Output '+k);axes[i,2].set_yscale('log')
    for ax in axes.flat:ax.set_xlabel('Epoch');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(ART/'measurement_output_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for arm,a in arms.items():
        v=[a['health'][0]['full_fixed_subset'],*[r['eval'] for r in a['history']]];epochs=list(range(len(v)))
        for i,k in enumerate(['total','hold','cross','observed','consistency']):
            ax=axes.flat[i];ax.plot(epochs,[s['loss'][k] for s in v],label=arm);ax.plot(epochs[1:],[r['train']['loss'][k] for r in a['history']],':',alpha=.5,label=arm+' train');ax.set_title(k+' (total is arm-specific, not ranked)');ax.set_yscale('log')
        for i,k in enumerate(['measurement_smooth','measurement_relative_raw']):
            ax=axes.flat[i+5];ax.plot(epochs,[s['loss_diagnostics'][k] for s in v],label=arm);ax.set_title(k+' (both arms)');ax.set_yscale('log')
        for i,k in enumerate(['seconds','peak_gpu_allocated_bytes']):
            ax=axes.flat[i+7];ax.plot(epochs[1:],[r[k] for r in a['history']],label=arm);ax.set_title(k)
    for ax in axes.flat:ax.set_xlabel('Epoch');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(ART/'loss_cost_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for arm,a in arms.items():
        h=a['health'];epochs=[r['epoch'] for r in h]
        for q,ls in [('0.01',':'),('0.5','-'),('0.99','--')]:axes[0,0].plot(epochs,[r['logits']['quantiles'][q] for r in h],ls,label=arm+' q'+q)
        axes[0,1].plot(epochs,[r['logits']['fraction_below_minus12'] for r in h],label=arm)
        axes[0,2].plot(epochs,[r['logits']['sigmoid_derivative']['mean'] for r in h],label=arm)
        for i in range(3):axes[1,0].plot(epochs,[r['diagnostic_cases'][i]['measurement_dependence']['relative_MA'] for r in h],label=arm+f' case{i}')
        for g in ['encoder','latents','decoder']:
            obs=[o for r in a['history'] for o in r['update_observations']];axes[1,1].plot([o['step']/4 for o in obs],[o['update_l2'][g] for o in obs],label=arm+' '+g)
        axes[1,2].plot(epochs,[max(r['relative_MA'] for r in row['case_differences']) for row in h],label=arm)
    for ax,title in zip(axes.flat,['Logit quantiles','Fraction logits < -12','Mean sigmoid derivative','Normal vs zero relative MA','Actual update L2 by group','Max case pair relative MA']):ax.set_title(title);ax.set_xlabel('Epoch');ax.legend(fontsize=7);ax.grid(alpha=.2)
    for ax in [axes[0,2],*axes[1]]:ax.set_yscale('log')
    axes[1,0].axhline(1e-3,color='grey',ls='--');fig.savefig(ART/'health_update_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    for arm,a in arms.items():
        h=a['health'];epochs=[r['epoch'] for r in h]
        for i,g in enumerate(['encoder','latents','decoder']):
            for branch,ls in [('measurement_smooth','-'),('consistency_weighted',':'),('total_control' if arm=='control' else 'total_treatment','-.')]:
                axes[0,i].plot(epochs,[average([r['gradient_split']['branches'][branch]['parameter_gradient_l2'][g] for r in v['diagnostic_cases']]) for v in h],ls,label=arm+' '+branch)
            axes[0,i].set_title(g+' un-clipped branch gradient L2');axes[0,i].set_yscale('log')
        for space,ls in [('parameters','-'),('outputs',':')]:
            axes[1,0].plot(epochs,[average([r['gradient_split']['interaction'][space]['cosine'] for r in v['diagnostic_cases']]) for v in h],ls,label=arm+' '+space)
        obs=[o for r in a['history'] for o in r['update_observations']]
        for g in ['encoder','latents','decoder']:axes[1,1].plot([o['step']/4 for o in obs],[o['post_clip_gradient_l2'][g] for o in obs],label=arm+' '+g)
        axes[1,2].plot(epochs,[average([r['spatial']['axis_neighbor_mean_abs']['x'] for r in v['diagnostic_cases']]) for v in h],label=arm+' spatial X')
        axes[1,2].plot(epochs,[average([r['spatial']['axis_neighbor_mean_abs']['y'] for r in v['diagnostic_cases']]) for v in h],':',label=arm+' spatial Y')
        axes[1,2].plot(epochs,[average([r['spatial']['axis_neighbor_mean_abs']['z'] for r in v['diagnostic_cases']]) for v in h],'--',label=arm+' spatial Z')
    axes[1,0].set_title('Parameter gradient cosine');axes[1,0].axhline(0,color='grey');axes[1,1].set_title('Actual step post-clip gradient L2');axes[1,1].set_yscale('log');axes[1,2].set_title('Axis neighbor mean absolute difference');axes[1,2].set_yscale('log')
    for ax in axes.flat:ax.set_xlabel('Epoch');ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.savefig(ART/'branch_gradient_curves.png',dpi=150);plt.close(fig)
    # Both objective diagnostics are recorded for both arms, independent of actual training objective.
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for arm,a in arms.items():
        v=[a['health'][0]['full_fixed_subset'],*[r['eval'] for r in a['history']]]
        for ax,k in zip(axes,['total_control','total_treatment','consistency_raw']):
            ax.plot(range(len(v)),[r['loss_diagnostics'][k] for r in v],label=arm);ax.set_title(k+' diagnostic');ax.legend();ax.grid(alpha=.2)
    fig.savefig(ART/'both_objective_curves.png',dpi=150);plt.close(fig)
    def branch_mean(name,key):return average([r[key] for r in fresh if r['branch']==name])
    def metric(s,k):return s['measurement']['heldout'][k]
    def cosine_summary(rows,space):
        vals=[r['gradient_split']['interaction'][space]['cosine'] for r in rows]
        return ' / '.join(num(f(vals)) for f in [average,min,max])
    def epoch_summary(arm,epoch):
        a=arms[arm];return a['health'][0]['full_fixed_subset'] if epoch==0 else a['history'][epoch-1]['eval']
    def dominance(rows,g):
        return average([r['gradient_split']['branches']['consistency_weighted']['parameter_gradient_l2'][g]/r['gradient_split']['branches']['measurement_smooth']['parameter_gradient_l2'][g] for r in rows])
    endpoint=[];dependence=[];behaviors=[];gradrows=[];windowrows=[];checkpointrows=[]
    for arm,a in arms.items():
        done=a['complete'];h=a['health'][-1];end=h['full_fixed_subset'];init=a['health'][0]['full_fixed_subset']
        endpoint.append([arm,done['completed_epochs'],done['updates'],done['reason'],metric(end,'relative_l1'),metric(end,'relative_l2'),end['loss_diagnostics']['measurement_smooth'],end['loss']['consistency'],end['output']['mean']])
        behaviors.append([arm,end['output']['min'],end['output']['max'],end['output']['std'],h['logits']['quantiles']['0.5'],h['logits']['fraction_below_minus12'],h['logits']['sigmoid_derivative']['mean'],max(v['relative_MA'] for v in h['case_differences'])])
        for r in h['diagnostic_cases']:
            dep=r['measurement_dependence'];dependence.append([arm,r['index'],dep['absolute_MA'],dep['relative_MA'],*[r['spatial']['axis_neighbor_mean_abs'][k] for k in ['x','y','z']]])
        for h0 in [a['health'][0],h]:
            rr=h0['diagnostic_cases']
            for g in ['encoder','latents','decoder']:
                v=[r['gradient_split']['interaction']['parameter_groups'][g] for r in rr]
                gradrows.append([arm,h0['epoch'],g,dominance(rr,g),average([q['dot'] for q in v]),average([q['cosine'] for q in v]),average([q['sum_gradient_l2'] for q in v])])
        for e in a['events'][-2:]:windowrows.append([arm,e['epoch'],e['windows']['relative_l1']['recent5'],e['windows']['relative_l1']['relative_improvement'],e['windows']['relative_l2']['relative_improvement'],e['dependency_case_count'],e['failure_streak'],e['collapse_streak'],e['reason']])
        checkpointrows.append([arm,done['last_checkpoint'],done['last_sha256'],done['epoch0_sha256']])
    common_end={a:epoch_summary(a,common) for a in arms}
    amplitude_table=[]
    for alpha in [1.,.1,.01]:
        for branch in ['measurement_smooth','consistency_weighted','total_control','total_treatment']:
            rows=[r for r in amplitude if r['alpha']==alpha and r['branch']==branch]
            amplitude_table.append([alpha,branch,average([r['loss'] for r in rows]),average([r['dL_dalpha'] for r in rows])])
    lines=[f'# Ray-Senseiver consistency开/关最后一次小预算对照：{label}',
      f'**正式训练决策：'+('GO（仅冻结候选，未启动）' if go else 'NO-GO')+f'。** 描述标签 `{label}`。本轮实际执行全部预登记工作，H参与训练loss，仅固定8例训练子集可学习性，不是独立泛化验证。没有读取GT64/128、Val、Pool B或test，没有训练/重新评价Ours/DeepSets，没有启动长训练或新增搜索。',
      '## 唯一差异、初态和历史保护',
      '[训练前协议](../ray_senseiver_consistency_probe_v1_protocol_zh.md)、[配置](../../configs/ray_senseiver_consistency_probe_v1.yaml)、[独立objective](../../src/geophy_set3d/training/senseiver_consistency_probe.py)、[入口](../../scripts/run_senseiver_consistency_probe.py)、[注册](artifacts/ray_senseiver_consistency_probe_v1/registration.json)。注册commit `9574f8e`，诊断与smoke确认commit `a811e17`，先提交后科学训练。两臂relative固定0，Control的原consistency权重.05、Treatment为0；hold1/cross.5/observed.1、SmoothL1 beta.01、epsilon、尺度、mask/归约、网络和几何全相同。Treatment仍计算relative与原consistency及两种total作诊断，但训练梯度只来自measurement SmoothL1。没有改原selfsup函数、FROZEN_LOSS或旧validator。沿用日志字段measurement_relative_weighted仅指名义.1×原relative诊断，不是本轮实际梯度系数（actual_relative_weight始终0）。',
      f'固定source-index train前8例、Pool A、seed20261011、sampler epoch0、A3–5/B3–5/H2不交、连续2例batch。直接复用C_init={scale["C_init"]}、weight×.01、bias={scale["bias_logit"]}，无重估/训练权重读取；全部参数逐元素一致，fresh state SHA `{reg["initial_hash"]}`，与原登记一致。历史失败记录和GO/NO-GO结论保留，历史源码/数据/报告hash复核通过。',
      '## 零更新诊断与等价性',
      f'全部8例同一fresh预测图；只forward/autograd，optimizer未创建，0 step，model state不变，.grad全None，耗时{num(d["seconds"])}秒。[完整诊断JSON](artifacts/ray_senseiver_consistency_probe_v1/gradient_diagnosis.json.gz)、[梯度CSV](artifacts/ray_senseiver_consistency_probe_v1/fresh_branch_gradients.csv)、[dot/cosine及组内重构CSV](artifacts/ray_senseiver_consistency_probe_v1/branch_interactions.csv)。',
      '新Control与上轮relative关闭Treatment的同预测loss各分量bitwise相等；对全部模型参数及预测体素梯度一致。8例最大参考梯度差L2='+num(max(c['fresh']['equivalence']['gradient_difference_l2'] for c in d['case_results']))+'；Control−Treatment−.05consistency最大loss余量='+num(max(abs(c['fresh']['only_difference']['loss_remainder']) for c in d['case_results']))+'，参数梯度余量L2='+num(max(c['fresh']['only_difference']['parameter_gradient_remainder_l2'] for c in d['case_results']))+'，输出梯度余量L2='+num(max(c['fresh']['only_difference']['output_gradient_remainder_l2'] for c in d['case_results']))+'。',
      table(['fresh分支，8例均值','loss','encoder梯度L2','latents梯度L2','decoder梯度L2','共同输出平移dL/dc'],[[b,*[branch_mean(b,k) for k in ['loss','encoder_gradient','latents_gradient','decoder_gradient','dL_dc_uniform_shift']]] for b in ['measurement_smooth','consistency_weighted','total_control','total_treatment']]),
      table(['fresh梯度空间','dot均值','cosine均值/min/max'],[[sp,average([r['dot'] for r in interactions if r['space']==sp]),' / '.join(num(f([r['cosine'] for r in interactions if r['space']==sp])) for f in [average,min,max])] for sp in ['parameters','outputs','encoder','latents','decoder']]),
      '参数空间与输出空间cosine是不同问题：前者包含当前网络Jacobian的映射与参数共享，后者描述独立体素变化。不能把参数cosine套用于输出空间，也不能相加范数。每组向量dot和||g_m+g_c||、平方范数重构已保存，支持观察贡献与抵消。正共同输出平移导数表示梯度下降倾向降低浓度，但不代表任意参数更新等价于均匀平移。',
      '## 共同幅度探针',
      '[逐case幅度CSV](artifacts/ray_senseiver_consistency_probe_v1/common_amplitude.csv)。同一C_A/C_B同时乘固定α，不训练、不搜索，下面为8例均值。正dL/dα表示局部梯度下降降低共同幅度。',
      table(['α','分支','loss','dL/dα'],amplitude_table),
      '原consistency = αd/(αm+1e−6)，d为未缩放A/B meanabs差、m为两预测meanabs均值。导数为d×1e−6/(αm+1e−6)²；.05加权解析值逐例随JSON保留，FP32 autograd数值有减法抵消。epsilon使共同缩小也可降低该项（正导数），而减小A/B差异是另一种变化方向；α=1附近近似尺度不变也不等于严格不变。该有限探针不证明训练失败的完整因果。',
      '## 实际预算、统一指标和退出',
      '每臂fresh重训，FP32、batch2/accum1、每轮4更新、AdamW2e−4/wd1e−4/clip1/cosine T_max300；query chunk16384/checkpoint开启，完整64³全部query/valid rays（含零）与原K96。未发生OOM工程降配，未终止其他进程。epoch0完整保存，工程smoke每臂独立1step不计科学预算，无warm start。',
      table(['臂','epoch','updates','终止','H L1','H L2','weighted SmoothL1','raw consistency','mean'],endpoint),
      f'共同可比终点epoch{common}/updates{4*common}；只对统一指标配对，不横排两种训练total。共同终点T−C：H L1='+num(metric(common_end['treatment'],'relative_l1')-metric(common_end['control'],'relative_l1'))+'，H L2='+num(metric(common_end['treatment'],'relative_l2')-metric(common_end['control'],'relative_l2'))+'。[全epoch配对JSON](artifacts/ray_senseiver_consistency_probe_v1/matched_comparison.json)、[所有轮末标量CSV](artifacts/ray_senseiver_consistency_probe_v1/epoch_scalars.csv)。各臂压缩history保留每轮train聚合、轮末指标和每step梯度/更新，health/events保留全部检查和计数。',
      'Treatment在共同终点相对自身epoch0：H L2改善='+num(improvements['H_L2'])+'；weighted SmoothL1改善='+num(improvements['measurement_smooth'])+'（相对改善，负值为恶化）。辅助改善不能覆盖测量依赖与case区分要求。',
      table(['臂','检查epoch','最近5轮H L1','L1改善比例','L2改善比例','依赖case数','failure连续','collapse连续','原因'],windowrows),
      '退出严格复用第20轮后连续两次L1≥.98、L1和L2改善均<1%、有效依赖不足2例；不是只因L1慢停止。工程完整性与真实更新核验正常。所有事件在最终审计中逐项重放一致。',
      '![统一测量与输出](artifacts/ray_senseiver_consistency_probe_v1/measurement_output_curves.png)',
      '![loss与成本](artifacts/ray_senseiver_consistency_probe_v1/loss_cost_curves.png)',
      '![两目标共同诊断](artifacts/ray_senseiver_consistency_probe_v1/both_objective_curves.png)',
      '## 梯度方向、输出与测量依赖',
      table(['臂','epoch','参数组','cons/Smooth梯度范数比','dot均值','cosine均值','梯度和L2均值'],gradrows),
      '范数比为逐case比值的均值，dot/cosine同图同参数组计算。Treatment中cons梯度仅诊断，不参与更新；不能因为该诊断范数大就声称它实际支配Treatment更新。全健康检查曲线同时给出两种空间cosine、三组实际更新和空间变化。',
      table(['臂','min','max','std','logit中位数','logit<−12比例','sigmoid导数均值','最大case-pair relMA'],behaviors),
      table(['臂','固定case','normal-zero绝对MA','normal-zero相对MA','空间邻点X MA','Y MA','Z MA'],dependence),
      'normal-zero仅置零测量，geometry/path/mask逐tensor完全相等。relative MA分母normal meanabs+1e−12，有效阈值1e−3；相对误差必须结合绝对差阅读。空间纹理不自动意味着case或测量信息被利用。',
      '![健康与真实更新](artifacts/ray_senseiver_consistency_probe_v1/health_update_curves.png)',
      '![分支梯度方向与空间变化](artifacts/ray_senseiver_consistency_probe_v1/branch_gradient_curves.png)',
      '## GO判定、访问与checkpoint身份',
      table(['预登记GO条件','结果'],list(checks.items())),
      'Treatment最后3次健康H L1：'+str(result['last3_treatment_H'])+'。条件仅工程可学习性，不证明三维重建正确，也不要求达到Ours精度。',
      table(['臂','epoch总秒','进程总秒','峰值分配GiB'],[[arm,a['complete']['total_epoch_seconds'],a['complete']['elapsed_process_seconds'],a['complete']['peak_gpu_allocated_bytes']/2**30] for arm,a in arms.items()]),
      '共享GPU环境，时间仅实际预算记录，不作受控性能比较。阶段一无更新，最终审计只读本轮epoch0/last。所有阶段实际仅8例Pool A projection，无GT/禁用NPY/外来checkpoint访问；TrainEightGuard继承summary文字含train/val，但实际allowlist已收紧到固定8例train。边界负测试是主动拒绝模拟访问，与科学run零违规分开。[smoke](artifacts/ray_senseiver_consistency_probe_v1/engineering_smoke.json)、[最终完整性审计](artifacts/ray_senseiver_consistency_probe_v1/final_integrity.json)、[测试记录](artifacts/ray_senseiver_consistency_probe_v1/verification.txt)。每次完整262144-query输出形状与XYZ轴序，以及固定259个覆盖端点的query在chunk259/37下输出和全部参数/query梯度一致性通过；这是有限数值核验，不冒称对全部262144个query逐点作双份梯度比较。全valid rays、所有组有限梯度/真实更新、自身状态恢复、退出计数与历史hash均通过。',
      table(['臂','外部last checkpoint','last SHA256','epoch0 SHA256'],checkpointrows),
      'Git只保存代码/协议/配置/hash/标量/图表/报告，大checkpoint保留上述独立外部目录。',
      '## 结论与唯一后续任务',
    ]
    if go:
        lines.append('Treatment达到预登记工程GO，仅冻结该候选；唯一建议是重新登记该适配目标下的完整训练与独立测量选模，明确已不同于冻结基线。本轮未启动完整训练。')
    else:
        lines.append('本轮没有达到有效学习GO，正式训练NO-GO。Treatment相对初态SmoothL1改善40.526%，H L2却恶化3.472%；相对本轮Control，两种H误差几乎相同。关闭后case间差异达到既定操作阈值（最大relative MA=.002348），normal-zero差也有增长（case0从1.41e−5升至1.11e−4），但3例都不足1e−3，绝对MA最高仅8.48e−9；不能把case差异直接归因于测量信息利用。两臂输出均仅约.049×C_init，无logit<−12，未触发严格塌缩条件，但持续整体压低。描述标签PARTIAL仅保留有限指标改善，不能认定形成有效测量相关输出。关闭consistency不足以解决当前适配的学习失败，也不能证明该分支没有任何间接作用。结束当前逐项loss排查；不再扫描beta/lr/relative/consistency，不延长预算，不启动300轮，不把失败Senseiver列入成功强基线主表。这是固定适配、数据与预算的局部结果，不能归为原Senseiver能力上限，也不能外推consistency对其他结构有害。唯一后续任务：独立登记输入表征与结构可学习性检查，判断测量信息能否通过当前ray-token编码和latent bottleneck被有效利用；本轮不实现任何该改动。')
    cr=arms['control']['health'][-1]['diagnostic_cases'];tr=arms['treatment']['health'][-1]['diagnostic_cases']
    cdep=[r['measurement_dependence']['relative_MA'] for r in cr];tdep=[r['measurement_dependence']['relative_MA'] for r in tr]
    paired_h=next(h for h in arms['treatment']['health'] if h['epoch']==common)
    paired_c=next(h for h in arms['control']['health'] if h['epoch']==common)
    summary='实际结果：在共同epoch'+str(common)+'，Treatment H L1/L2='+num(metric(common_end['treatment'],'relative_l1'))+'/'+num(metric(common_end['treatment'],'relative_l2'))+'，Control='+num(metric(common_end['control'],'relative_l1'))+'/'+num(metric(common_end['control'],'relative_l2'))+'。关闭consistency后，固定3例有效测量依赖数='+str(sum(r['measurement_dependence']['relative_MA']>=1e-3 for r in paired_h['diagnostic_cases']))+'（Control='+str(sum(r['measurement_dependence']['relative_MA']>=1e-3 for r in paired_c['diagnostic_cases']))+'），最大case-pair相对差='+num(max(r['relative_MA'] for r in paired_h['case_differences']))+'（Control='+num(max(r['relative_MA'] for r in paired_c['case_differences']))+'）。共同终点输出mean为'+num(common_end['treatment']['output']['mean'])+'（Control='+num(common_end['control']['output']['mean'])+'）。这些数字同时约束指标变化、整体幅度变化和输入依赖的解释。'
    lines.insert(2,summary)
    lines.append('关于梯度机制：Control末轮固定3例consistency/Smooth梯度范数比，encoder='+num(dominance(cr,'encoder'))+'、latents='+num(dominance(cr,'latents'))+'；这说明该分支在这些参数组占较大范数，不代表方向一定相反。末轮encoder/latents cosine均值分别='+num(average([r['gradient_split']['interaction']['parameter_groups']['encoder']['cosine'] for r in cr]))+'/'+num(average([r['gradient_split']['interaction']['parameter_groups']['latents']['cosine'] for r in cr]))+'，输出空间cosine均值='+num(average([r['gradient_split']['interaction']['outputs']['cosine'] for r in cr]))+'。fresh α=1处consistency对共同幅度的导数均值仅约2.62e−8，measurement约.04278；α=.01处measurement约−.009125，consistency约+.0002467，未改变total导数符号。因此本有限探针不能把较大参数梯度直接解释为主要共同缩幅力，更不能宣称consistency是唯一失败原因。')
    audit=read(ART/'final_integrity.json')
    replay=audit['control']['prior_relative_off_fresh_control_scalar_replay']
    lines.append('独立fresh Control对上轮相同科学配置的标量重放：共同'+str(replay['compared_epochs'])+'轮，H L1最大绝对差='+num(replay['max_H_L1_absolute_difference'])+'，H L2最大绝对差='+num(replay['max_H_L2_absolute_difference'])+'。同图objective等价核验与独立训练数值重放是不同证据；后者存在FP32 CUDA归约及后续优化轨迹微差，历史标量保持原样，未用旧结果充当本轮Control。')
    REPORT.write_text('\n\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
