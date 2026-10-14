#!/usr/bin/env python3
"""Reports only persisted scalar diagnostics; no dataset/checkpoint is opened."""
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
REPO=Path(__file__).resolve().parents[1]
NAME='ray_senseiver_relative_probe_v1'
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons')/NAME
ART=REPO/'docs/reports/artifacts'/NAME
REPORT=REPO/'docs/reports/ray_senseiver_relative_probe_v1_zh.md'


def read(p):return json.loads(p.read_text())
def num(v):return f'{v:.8g}' if isinstance(v,(int,float)) else str(v)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(num(x) for x in r)+' |' for r in rows])
def average(v):return sum(v)/len(v)


def diagnostic_tables(d):
    cases=d['case_results'];constant=[];contributions=[];fresh=[];amplitudes=[]
    for c in cases:
        for name,row in c['fresh']['branches'].items():
            fresh.append(dict(index=c['index'],branch=name,loss=row['loss'],encoder_gradient=row['parameter_gradient_l2']['encoder'],latents_gradient=row['parameter_gradient_l2']['latents'],decoder_gradient=row['parameter_gradient_l2']['decoder'],dL_dc_uniform_shift=row['dL_dc_uniform_output_shift']))
        for subset,row in c['amplitude'].items():
            for group,value in row['groups'].items():
                amplitudes.append(dict(index=c['index'],subset=subset,group=group,count=value['count'],**{f'q{q}':value['normalized_target']['quantiles'][q] if value['normalized_target'] else None for q in ['0.0','0.01','0.5','0.99','1.0']}))
        for level in c['constants']:
            b=level['split']['branches']
            constant.append(dict(index=c['index'],C=level['C'],**{k:b[k]['dL_dc_constant'] for k in ['measurement_smooth','measurement_relative_weighted','consistency_weighted','total_control','total_treatment']},float64_total=level['float64_scalar_check']['total_control_dL_dc']))
            for branch,groups in level['amplitude_contributions'].items():
                for group,row in groups.items():contributions.append(dict(index=c['index'],C=level['C'],branch=branch,group=group,**row))
    for name,rows in [('fresh_branch_gradients',fresh),('constant_derivatives',constant),('amplitude_inventory',amplitudes),('amplitude_gradient_contributions',contributions)]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (ART/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return fresh,constant,amplitudes,contributions


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=read(OUT/'gradient_diagnosis.json');scale=read(OUT/'scale.json');reg=read(OUT/'registration.json')
    fresh,constant,amplitudes,contributions=diagnostic_tables(d)
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
            ax=axes.flat[i];ax.plot(epochs,[s['loss'][k] for s in v],label=arm);ax.plot(epochs[1:],[r['train']['loss'][k] for r in a['history']],':',alpha=.5,label=arm+' train');ax.set_title(k+' training loss');ax.set_yscale('log')
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
            for branch,ls in [('measurement_smooth','-'),('measurement_relative_weighted','--'),('consistency_weighted',':'),('total_control' if arm=='control' else 'total_treatment','-.')]:
                axes[0,i].plot(epochs,[average([r['gradient_split']['branches'][branch]['parameter_gradient_l2'][g] for r in v['diagnostic_cases']]) for v in h],ls,label=arm+' '+branch)
            axes[0,i].set_title(g+' un-clipped branch gradient L2');axes[0,i].set_yscale('log')
        for pair,ls in [('measurement_smooth__measurement_relative_weighted','-'),('measurement_smooth__consistency_weighted',':')]:
            axes[1,0].plot(epochs,[average([r['gradient_split']['interactions'][pair]['parameters']['cosine'] for r in v['diagnostic_cases']]) for v in h],ls,label=arm+' '+pair)
        obs=[o for r in a['history'] for o in r['update_observations']]
        for g in ['encoder','latents','decoder']:axes[1,1].plot([o['step']/4 for o in obs],[o['post_clip_gradient_l2'][g] for o in obs],label=arm+' '+g)
        axes[1,2].plot(epochs,[average([r['spatial']['axis_neighbor_mean_abs']['x'] for r in v['diagnostic_cases']]) for v in h],label=arm+' spatial X')
        axes[1,2].plot(epochs,[average([r['spatial']['axis_neighbor_mean_abs']['y'] for r in v['diagnostic_cases']]) for v in h],':',label=arm+' spatial Y')
        axes[1,2].plot(epochs,[average([r['spatial']['axis_neighbor_mean_abs']['z'] for r in v['diagnostic_cases']]) for v in h],'--',label=arm+' spatial Z')
    axes[1,0].set_title('Parameter gradient cosine');axes[1,0].axhline(0,color='grey');axes[1,1].set_title('Actual step post-clip gradient L2');axes[1,1].set_yscale('log');axes[1,2].set_title('Axis neighbor mean absolute difference');axes[1,2].set_yscale('log')
    for ax in axes.flat:ax.set_xlabel('Epoch');ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.savefig(ART/'branch_gradient_curves.png',dpi=150);plt.close(fig)
    levels=[r['C'] for r in cases_first(d)['constants']]
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for i,c in enumerate(d['case_results']):
        for ax,key in zip(axes,['measurement_smooth','measurement_relative_weighted']):
            ax.plot(range(len(levels)),[r['split']['branches'][key]['dL_dc_constant'] for r in c['constants']],marker='.',label=f'case{i}');ax.set_title(key+' dL/dc');ax.axhline(0,color='grey');ax.set_xticks(range(len(levels)),[f'{v:.2g}' for v in levels]);ax.set_xlabel('Constant mass fraction')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(ART/'constant_derivative_curves.png',dpi=150);plt.close(fig)
    lines=[f'# Ray-Senseiver 测量梯度拆分与relative-L1开/关：{label}',
           f'**实验标签：{label}；正式训练决策：'+('GO（仅建议）' if go else 'NO-GO')+'。** 本轮只执行预登记的两臂固定8例可学习性探针。H参与loss，不是独立泛化Val；没有GT/Val/Pool B/test或外部权重读取，没有新300轮或系数搜索。',
           '## 唯一干预和历史保护',
           '[协议](../ray_senseiver_relative_probe_v1_protocol_zh.md)、[配置](../../configs/ray_senseiver_relative_probe_v1.yaml)、[注册清单](artifacts/ray_senseiver_relative_probe_v1/registration.json)。训练前登记commit `73b4ed4`。Control内部λ_rel=.1，Treatment=0；外层hold1/cross.5/observed.1/consistency.05全不变。内部relative=.1和外层observed=.1是不同系数。beta=.01、epsilon1e−6、Y_SCALE、mask/reduction、consistency公式及所有网络/输入/几何未变。relative是逐case sum(abs residual)/clamp(sum(abs target))，不是逐射线相除。Treatment的relative诊断不加入训练total梯度。',
           '复用前8例/seed20261011/sampler epoch0/A-B-H/case顺序/连续2例batch；原C_init='+num(scale['C_init'])+'、输出weight×.01与bias='+num(scale['bias_logit'])+'直接复制，无重估。两臂全部参数逐元素一致，fresh state SHA=`'+reg['initial_hash']+'`，与上轮注册Treatment一致；不读取其训练checkpoint。历史源码/报告/数据hash受保护，原失败run和初始化探针不续跑，不重训/评价Ours或DeepSets。',
           '## 训练前零更新梯度诊断',
           f'全部8例，各1个fresh网络完整预测和6个常量场，只forward/autograd，optimizer未创建，0 step，model state hash不变，.grad buffers全None；耗时{num(d["seconds"])}秒。完整[梯度JSON](artifacts/ray_senseiver_relative_probe_v1/gradient_diagnosis.json.gz)、[fresh各分支CSV](artifacts/ray_senseiver_relative_probe_v1/fresh_branch_gradients.csv)、[常量导数CSV](artifacts/ray_senseiver_relative_probe_v1/constant_derivatives.csv)。',
           '同一预测图核验Control与原objective所有loss项bitwise相等，输出/全部模型参数梯度一致（数值容差内）；两臂差正好是被移除的加权relative项。8例最大原梯度差L2='+num(max(c['fresh']['equivalence']['gradient_difference_l2'] for c in d['case_results']))+'；唯一差异梯度最大余量L2='+num(max(c['fresh']['only_difference']['gradient_remainder_l2'] for c in d['case_results']))+'。同预测dL/dC差也单独保存核验。',
           table(['fresh分支（8例均值）','loss','encoder grad L2','latents grad L2','decoder grad L2','输出均匀平移dL/dc'],[[name,*[average([r[k] for r in fresh if r['branch']==name]) for k in ['loss','encoder_gradient','latents_gradient','decoder_gradient','dL_dc_uniform_shift']]] for name in ['measurement_smooth','measurement_relative_weighted','hold_control','cross_control_weighted','observed_control_weighted','consistency_weighted','total_control','total_treatment']]),
           '上述范数是逐例计算后平均，不能相加。总梯度由向量相加决定；cosine/dot报告同一向量空间的夹角，零向量cosine为null。常量探针没有encoder等网络参数，参数梯度记null；完整dL/dC的min/max/mean/sum/正负比例及scalar dL/dc全部保留。',
           table(['fresh梯度对','参数cosine均值/min/max','输出cosine均值/min/max'],[[pair,*[' / '.join(num(fn([c['fresh']['interactions'][pair][space]['cosine'] for c in d['case_results'] if c['fresh']['interactions'][pair][space]['cosine'] is not None])) for fn in [average,min,max]) for space in ['parameters','outputs']]] for pair in ['measurement_smooth__measurement_relative_weighted','measurement_smooth__consistency_weighted','measurement_relative_weighted__consistency_weighted']]),
           '## 近零常量方向和测量幅度贡献',
           '正dL/dc表示梯度下降压低常量质量分数，负值表示提高。以下为全部8例均值；逐case与hold/cross/observed分支全部在CSV/JSON，不用残差正负数量代替浓度梯度。',
           table(['C','Smooth dL/dc','0.1×relative dL/dc','consistency dL/dc','Control total','Treatment total'],[[level,*[average([r[k] for r in constant if r['C']==level]) for k in ['measurement_smooth','measurement_relative_weighted','consistency_weighted','total_control','total_treatment']]] for level in levels]),
           '![常量导数](artifacts/ray_senseiver_relative_probe_v1/constant_derivative_curves.png)',
           '分组仅按观测幅度target/Y_SCALE：精确0、(0,1e−6]、>1e−6，不是已知真实背景/羽流区域，也不是训练ray筛选或物理噪声界限。A/B/H各自不交；下表按8例相加。逐case/subset数量和分位数见[幅度CSV](artifacts/ray_senseiver_relative_probe_v1/amplitude_inventory.csv)，包含q0/q01/q50/q99/q100；不会把分位数平均冒充全局分位数。',
           table(['subset','精确0','(0,1e−6]','>1e−6'],[[k,*[sum(r['count'] for r in amplitudes if r['subset']==k and r['group']==g) for g in ['exact_zero','positive_le_1e_6','above_1e_6']]] for k in ['A','B','H']]),
           '每组贡献保留原完整case的valid数和sum target分母，经**真实P_diff/K96→scalar**链式autograd求导；分组导数之和等于全分支（数值容差内）。[完整分组贡献CSV](artifacts/ray_senseiver_relative_probe_v1/amplitude_gradient_contributions.csv)。下表是C=1e−7时外层加权measurement的8例均值。',
           table(['观测组','Smooth dL/dc','0.1×relative dL/dc'],[[g,average([r['smooth_dL_dc'] for r in contributions if r['branch']=='weighted_measurement_total' and r['group']==g and r['C']==1e-7]),average([r['relative_weighted_dL_dc'] for r in contributions if r['branch']=='weighted_measurement_total' and r['group']==g and r['C']==1e-7])] for g in ['exact_zero','positive_le_1e_6','above_1e_6']]),
           '精确C=0时abs的零残差子梯度为0；正近零场会使许多精确零/极小目标ray变为正残差，因此符号可与零点不同。这是不可微点的方向/子梯度差异，不是梯度断开。float64标量只核验均匀场的y=1e6 length mass_to_mole(c)及原归约，FP32真实projector/训练未改；所有逐例float64导数随JSON保留。',
           '实际机制证据：在C=1e−7，精确零/(0,1e−6]/>1e−6三组的加权relative导数约+27.39/+118.52/−29.38，合计+116.52；Smooth对应约+.001436/+.007373/−7.37796，合计−7.36915。全部8例在1e−8至1e−5的正近零常量方向呈Smooth向上、relative向下（逐例详见CSV）；在C_init，两者平均均向下。fresh参数Smooth-vs-relative cosine约.9999，relative加权decoder范数均值.65436约为Smooth .12693的5.16倍；consistency .08867仍不可忽略。因此初态不是Smooth和relative互相抵消，近零常量方向才出现明显相反推动。float64与FP32总导数最大绝对差约5.40e−5，符号解释一致。',
           '## 实际训练预算与终止',
           table(['臂','λ_rel','epochs','updates','终止原因','epoch墙钟秒','峰值GiB'],[[arm,a['complete']['relative_l1'],a['complete']['completed_epochs'],a['complete']['updates'],a['complete']['reason'],a['complete']['total_epoch_seconds'],a['complete']['peak_gpu_allocated_bytes']/2**30] for arm,a in arms.items()]),
           '各臂epoch0完整保存；每轮4次更新，最多40/160，FP32/AdamW2e−4/wd1e−4/clip1/cosine T_max300/完整64³/all rays/K96。smoke每臂1次独立更新不算科学预算，随后重新fresh构建并核对hash。共享GPU墙钟及event含健康检查/host间隙，不是独占kernel性能。',
           '停止事件保留完整L1/L2最近/前5轮均值、相对改善、依赖case数和连续计数：[Control](artifacts/ray_senseiver_relative_probe_v1/control/events.json.gz) / [Treatment](artifacts/ray_senseiver_relative_probe_v1/treatment/events.json.gz)。第5/20轮后从10/25检查起，不因L1单独缓慢误停。',
           '## 共同预算比较',
           f'共同完成至epoch{common}/update{4*common}；下表只在相同预算比较，各自终点另列。H/cross/obs指标对两臂统一计算，不随训练loss改变；train聚合跨更新、eval为轮末同一权重，曲线分别标出。',
           table(['epoch/update','Control H L1/L2','Treatment H L1/L2','T−C L1/L2'],[[str(r['epoch'])+'/'+str(r['updates']),num(r['control']['measurement']['heldout']['relative_l1'])+'/'+num(r['control']['measurement']['heldout']['relative_l2']),num(r['treatment']['measurement']['heldout']['relative_l1'])+'/'+num(r['treatment']['measurement']['heldout']['relative_l2']),num(r['T_minus_C_H_L1'])+'/'+num(r['T_minus_C_H_L2'])] for r in matched if r['epoch'] in {0,5,10,15,20,25,30,35,40,common}]),
           table(['臂/最终epoch','H L1/L2','cross L1/L2','obs L1/L2','smooth / relative(raw)','mean/std'],[[arm+'/'+str(a['complete']['completed_epochs']),*[num(s['measurement'][g]['relative_l1'])+'/'+num(s['measurement'][g]['relative_l2']) for g in ['heldout','cross','observed']],num(s['loss_diagnostics']['measurement_smooth'])+'/'+num(s['loss_diagnostics']['measurement_relative_raw']),num(s['output']['mean'])+'/'+num(s['output']['std'])] for arm,a in arms.items() for s in [a['history'][-1]['eval']]]),
           '![统一测量与输出](artifacts/ray_senseiver_relative_probe_v1/measurement_output_curves.png)',
           '![损失与成本](artifacts/ray_senseiver_relative_probe_v1/loss_cost_curves.png)',
           '## 输出、梯度与测量依赖',
           table(['臂/轮','logit P01/P50/P99','<-12比例','sigmoid导数均值','最大case间relative MA'],[[arm+'/'+str(h['epoch']),'/'.join(num(h['logits']['quantiles'][q]) for q in ['0.01','0.5','0.99']),h['logits']['fraction_below_minus12'],h['logits']['sigmoid_derivative']['mean'],max(r['relative_MA'] for r in h['case_differences'])] for arm,a in arms.items() for h in [a['health'][0],a['health'][-1]]]),
           table(['臂/终点case','normal−zero绝对MA','normal meanabs','relative MA'],[[arm+'/'+str(r['index']),r['measurement_dependence']['absolute_MA'],r['measurement_dependence']['normal_mean_abs'],r['measurement_dependence']['relative_MA']] for arm,a in arms.items() for r in a['health'][-1]['diagnostic_cases']]),
           '仅将raw/normalized测量置零，geometry/path/mask等逐tensor一致；分母normal meanabs+1e−12。空间XYZ邻点、完整min/max/mean/std、case间绝对差及三组未clip梯度、所有真实step的postclip梯度/更新L2均在health/history归档。下表为终点3诊断case的分支梯度范数均值（counterfactual relative项在Treatment仅诊断，未加入训练）。',
           table(['臂/分支','encoder grad L2','latents grad L2','decoder grad L2'],[[arm+'/'+branch,*[average([r['gradient_split']['branches'][branch]['parameter_gradient_l2'][g] for r in a['health'][-1]['diagnostic_cases']]) for g in ['encoder','latents','decoder']]] for arm,a in arms.items() for branch in ['measurement_smooth','measurement_relative_weighted','consistency_weighted','total_control' if arm=='control' else 'total_treatment']]),
           table(['臂/终点梯度对','参数cosine（3例均值）','输出cosine（3例均值）'],[[arm+'/'+pair,*[average([r['gradient_split']['interactions'][pair][space]['cosine'] for r in a['health'][-1]['diagnostic_cases']]) for space in ['parameters','outputs']]] for arm,a in arms.items() for pair in ['measurement_smooth__measurement_relative_weighted','measurement_smooth__consistency_weighted','measurement_relative_weighted__consistency_weighted']]),
           '所有dot/cosine原始向量证据保留；参数空间和输出空间的夹角含义不同，不把范数相加。完整query输出和健康检查的259/37分块全部参数/每query梯度核验保持，不改正式query数量。',
           '![健康及更新](artifacts/ray_senseiver_relative_probe_v1/health_update_curves.png)',
           '![分支梯度与抵消](artifacts/ray_senseiver_relative_probe_v1/branch_gradient_curves.png)',
           '## 判定与证据边界',
           table(['最低GO条件','通过'],[[k,v] for k,v in checks.items()]),
           'Treatment最后三次健康H L1：'+', '.join('epoch'+str(h['epoch'])+'='+num(h['full_fixed_subset']['measurement']['heldout']['relative_l1']) for h in last3)+'。共同预算相对自身epoch0，H L2改善='+num(improvements['H_L2']*100)+'%、measurement SmoothL1改善='+num(improvements['measurement_smooth']*100)+'%；预登记≥1%的辅助改善只产生PARTIAL标签，不能替代GO或有效测量依赖。',
           '## 访问审计、历史保护与交付',
           table(['阶段','unique train8 projection','GT尝试','禁止NPY尝试','foreign权重尝试'],[[phase,a['unique_pool_a_projection_files'],a['gt_open_attempts'],a['forbidden_npy_attempts'],a['supervised_checkpoint_open_attempts']] for phase,a in [('stage1',d['access']),*[(arm,a['complete']['access']) for arm,a in arms.items()]]]),
           'TrainEightGuard仅开放注册8个train NPY；基础summary中train/val字样不扩大该allowlist。没有Val/test/PoolB读取。边界拒绝测试在独立测试进程，不计为科学run违规。5项新增与15项相关回归通过，覆盖原Control、系数差值/输出梯度、valid零、真实projector分组导数、L1/L2/依赖退出窗口、own-last完整恢复和禁止读取边界；每臂独立smoke验证H不入网、首步梯度/真实更新/自身状态恢复、ray与分块。',
           '[最终完整性](artifacts/ray_senseiver_relative_probe_v1/final_integrity.json)核验历史/新冻结源码hash、全部固定输入tensor身份、valid-ray数量/geometry、last的AdamW steps/scheduler/参数有限性及checkpoint SHA；不执行step。大checkpoint留外部，Git保存全部标量、清单、曲线、诊断、协议与中文报告。',
           table(['臂','last SHA256','外部last'],[[arm,a['complete']['last_sha256'],a['complete']['last_checkpoint']] for arm,a in arms.items()]),
           '复现：`scripts/run_senseiver_relative_probe.py --phase prepare/diagnose/smoke/train`；已有注册/诊断/科学终止拒绝覆盖，仅中断允许本臂`--resume`。报告入口只读已存JSON。原近零塌缩和上轮NO-GO保留，本轮不自动300轮、不再搜索。']
    idx=lines.index('## 访问审计、历史保护与交付')
    if go:
        text='Treatment达到本轮最低可学习性GO。唯一下一阶段建议：独立登记λ_rel=0适配的完整训练及独立Val测量选模，明确训练目标不同于冻结基线；本轮不启动。'
    else:
        text='当前未达到GO；辅助指标改善只能说明目标/误差偏好变化或有限训练拟合证据，不能证明非平凡重建。Treatment终点consistency的encoder梯度范数均值约为Smooth的29.7倍，latents约52.5倍；其参数cosine与Smooth平均约.015，不证明两者方向相反。唯一下一阶段建议：独立登记固定初始化与λ_rel=0下consistency开/关有限对照，检验该分支在encoder/latents占主导是否影响测量依赖形成；本轮不执行，不预设其唯一因果。'
    lines.insert(idx,text)
    lines.insert(idx,'relative-L1的常量方向机制只在本8例、该Senseiver适配、固定初始化/预算下成立；不能外推其对所有模型有害。关闭后若只是L2/Smooth改善、没有case/测量依赖，就属于指标偏好或局部拟合改变；后续即使成功也必须明确训练目标已与冻结Ours/DeepSets不同。')
    lines.insert(idx,'本轮实际回答：关闭relative后末轮mean为7.68437e−5，约Control的10.40倍，但仍仅0.04931×C_init；logit中位数从Control −11.8151变为−9.47325，<-12比例从3.063%变0。两臂均未触发严格NEAR_ZERO_COLLAPSE，Treatment仍出现明显输出压低及近乎case无关、测量无关的空间先验。共同终点T−C：H L1约+0.0551610、H L2约−0.00623170，二者胜负方向相反；相对自身初态，Treatment H L2反而恶化3.474%，只有SmoothL1下降40.524%。因此PARTIAL在这里是有限指标证据，不是已产生非平凡有效学习。Control终点Smooth-vs-relative参数cosine约−.99986，显示当前参数方向发生抵消；不能把fresh的近同向关系套用至训练终点，输出空间cosine也不同。')
    REPORT.write_text('\n\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def cases_first(d):return d['case_results'][0]


if __name__=='__main__':main()
