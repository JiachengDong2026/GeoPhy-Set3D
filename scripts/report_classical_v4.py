#!/usr/bin/env python3
"""Reporting from actual frozen reconstructions and historical network metrics only."""
import sys,json,csv,hashlib,gzip,shutil,itertools,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,yaml
from run_classical_v4 import verify,verify_selection,write,digest
METRICS=['relative_l1','relative_l2','mae','rmse','integrated_concentration_relative_error','peak_value_absolute_error','heldout_relative_l1','heldout_relative_l2','observed_relative_l1','observed_relative_l2']
CONDITIONS=['Original','Targeted','Random']


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def case_means(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['model'],r['condition'],r['case_id']].append(r)
    means=[]
    for (model,condition,case),items in sorted(groups.items()):
        assert len(items)==5 and sorted(r['draw'] for r in items)==list(range(5))
        means.append(dict(model=model,condition=condition,case_id=case,**{m:float(np.mean([r[m] for r in items])) for m in METRICS}))
    return means


def paired(means,model_left,cond_left,model_right,cond_right,metric,index):
    a={r['case_id']:r[metric] for r in means if r['model']==model_left and r['condition']==cond_left};b={r['case_id']:r[metric] for r in means if r['model']==model_right and r['condition']==cond_right}
    assert len(a)==len(b)==36 and set(a)==set(b)
    delta=np.array([a[k]-b[k] for k in sorted(a)]);ci=np.quantile(delta[index].mean(1),[.025,.975])
    return dict(left_model=model_left,left_condition=cond_left,right_model=model_right,right_condition=cond_right,metric=metric,
        left_mean=float(np.mean(list(a.values()))),right_mean=float(np.mean(list(b.values()))),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),CI_low=float(ci[0]),CI_high=float(ci[1]),
        wins=int((delta<0).sum()),losses=int((delta>0).sum()),ties=int((delta==0).sum()),case_count=36,draws_per_case=5)


def markdown_table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    cfg=yaml.safe_load((ROOT/'configs/classical_v4_v1.yaml').read_text());out=Path(cfg['output']);verify(cfg,out);selection=verify_selection(out)
    complete=json.loads((out/'evaluation_complete.json').read_text());assert digest(out/'classical_rows.json')==complete['rows_sha256']
    classic=json.loads((out/'classical_rows.json').read_text());network=json.loads((out/'network_historical_rows.json').read_text());times=json.loads((out/'network_forward_times.json').read_text())
    rows=list(network)
    for r in classic:
        for identity in r['selection_identities']:rows.append(dict(r,model=r['method']+'_'+identity))
    models=sorted({r['model'] for r in rows});means=case_means(rows);index=np.random.default_rng(cfg['bootstrap']['seed']).integers(0,36,(10000,36));summary=[]
    for model in models:
        for condition in CONDITIONS:
            sub=[r for r in means if r['model']==model and r['condition']==condition];assert len(sub)==36
            result=dict(model=model,condition=condition)
            for metric in METRICS:
                values=np.array([r[metric] for r in sub]);ci=np.quantile(values[index].mean(1),[.025,.975]);result[metric]=float(values.mean());result[metric+'_CI']=[float(x) for x in ci]
            summary.append(result)
    comparisons=[]
    for model in models:
        for left,right in [('Targeted','Original'),('Targeted','Random'),('Random','Original')]:
            for metric in METRICS:comparisons.append(paired(means,model,left,model,right,metric,index))
    for condition in CONDITIONS:
        for a,b in itertools.combinations(models,2):
            for metric in METRICS:comparisons.append(paired(means,a,condition,b,condition,metric,index))
    runtime=[]
    for method in ['SART','Tikhonov','TV']:
        for candidate in sorted({r['candidate'] for r in classic if r['method']==method}):
            group=[r for r in classic if r['method']==method and r['candidate']==candidate]
            runtime.append(dict(method=method,candidate=candidate,actual_reconstructions=len(group),selection_identities=group[0]['selection_identities'],
                **{field+'_mean':float(np.mean([r[field] for r in group])) for field in ['operator_setup_seconds','iteration_seconds','conversion_seconds','evaluation_seconds','equivalent_cold_geometry_seconds']},
                iteration_seconds_total=sum(r['iteration_seconds'] for r in group),iteration_seconds_median=float(np.median([r['iteration_seconds'] for r in group])),
                actual_iterations_mean=float(np.mean([r['solver_stats']['iterations'] for r in group])),statuses=dict(collections.Counter(r['solver_stats']['status'] for r in group)),
                GPU_peak_allocated_MiB=max(r['GPU_peak_allocated_bytes'] for r in group)/2**20,GPU_peak_reserved_MiB=max(r['GPU_peak_reserved_bytes'] for r in group)/2**20,
                RSS_peak_MiB=max(r['process_peak_rss_kib'] for r in group)/1024,compute_device=group[0].get('compute_device',cfg['device'])))
    for model in ['SelfSup','Supervised']:
        group=[r for r in times if r['model']==model]
        runtime.append(dict(method=model,actual_reconstructions=0,actual_timed_forwards=len(group),forward_seconds_mean=float(np.mean([r['forward_seconds'] for r in group])),forward_seconds_total=sum(r['forward_seconds'] for r in group),GPU_peak_allocated_MiB=max(r['GPU_peak_allocated_bytes'] for r in group)/2**20,GPU_peak_reserved_MiB=max(r['GPU_peak_reserved_bytes'] for r in group)/2**20))
    case_runtime=[]
    groups=collections.defaultdict(list)
    for r in classic:groups[r['method'],r['candidate'],r['condition'],r['case_id']].append(r)
    for (method,candidate,condition,case),items in sorted(groups.items()):
        assert len(items)==5
        case_runtime.append(dict(method=method,candidate=candidate,condition=condition,case_id=case,draws=5,
            **{field+'_draw_mean':float(np.mean([r[field] for r in items])) for field in ['operator_setup_seconds','iteration_seconds','conversion_seconds','evaluation_seconds','equivalent_cold_geometry_seconds']},
            actual_iterations_draw_mean=float(np.mean([r['solver_stats']['iterations'] for r in items])),statuses=dict(collections.Counter(r['solver_stats']['status'] for r in items))))
    groups=collections.defaultdict(list)
    for r in times:groups[r['model'],r['condition'],r['case_id']].append(r)
    for (model,condition,case),items in sorted(groups.items()):
        assert len(items)==5
        case_runtime.append(dict(method=model,condition=condition,case_id=case,draws=5,forward_seconds_draw_mean=float(np.mean([r['forward_seconds'] for r in items]))))
    csv_write(out/'runtime_case_means.csv',case_runtime)
    write(out/'summary.json',dict(summary=summary,runtime=runtime,bootstrap=cfg['bootstrap'],unique_classical_reconstructions=len(classic),reused_network_metric_rows=len(network),actual_timed_network_forwards=len(times)))
    write(out/'paired_statistics.json',comparisons);csv_write(out/'paired_statistics.csv',comparisons);csv_write(out/'paired_case_means.csv',means);csv_write(out/'per_case_draw.csv',rows);csv_write(out/'runtime_summary.csv',runtime);csv_write(out/'summary.csv',summary)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    figure=out/'figures';figure.mkdir(exist_ok=True);x=np.arange(3);width=.8/len(models)
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for i,model in enumerate(models):
        for ax,metric in zip(axes,['relative_l1','heldout_relative_l1']):
            group=[next(r for r in summary if r['model']==model and r['condition']==condition) for condition in CONDITIONS]
            mean=np.array([r[metric] for r in group]);ci=np.array([r[metric+'_CI'] for r in group]);err=np.maximum(np.stack([mean-ci[:,0],ci[:,1]-mean]),0)
            ax.bar(x-.4+width*(i+.5),mean,width,label=model,yerr=err,capsize=2)
            ax.set_xticks(x,CONDITIONS);ax.set_ylabel(metric);ax.set_ylim(bottom=0)
    axes[1].legend(fontsize=7,bbox_to_anchor=(1.02,1));fig.suptitle('36 paired Val cases; each case averaged over 5 draws; 95% case bootstrap CI');fig.tight_layout();fig.savefig(figure/'paired_errors.png',dpi=170);fig.savefig(figure/'paired_errors.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,method in zip(axes,['SART','Tikhonov','TV']):
        for candidate in sorted({r['candidate'] for r in classic if r['method']==method}):
            for condition in CONDITIONS:
                r=next(r for r in classic if r['method']==method and r['candidate']==candidate and r['case_index']==0 and r['draw']==0 and r['condition']==condition)
                curve=json.loads(Path(r['curve_file']).read_text());values=np.array([v['objective'] for v in curve]);ax.semilogy([v['iteration'] for v in curve],values/max(values[0],1e-20),label=f'c{candidate} {condition}')
        ax.set_title(method+' input objective / first objective');ax.set_xlabel('Sweep' if method=='SART' else 'Iteration');ax.legend(fontsize=7)
    fig.suptitle('Preregistered first Val case / draw0; no GT or H trajectory selection');fig.tight_layout();fig.savefig(figure/'convergence_examples.png',dpi=170);plt.close(fig)
    # Export light scalar artifacts; never copy GT, measurements, weights, volumes or operators.
    archive=ROOT/'docs/reports/artifacts/classical_v4_v1';archive.mkdir(parents=True,exist_ok=True)
    names=['selected_configs.json','selected_configs.sha256','calibration_scores.json','calibration_rows.json','classical_rows.json','network_historical_rows.json','network_forward_times.json','network_timing_audit.json','summary.json','summary.csv','paired_statistics.json','paired_statistics.csv','paired_case_means.csv','per_case_draw.csv','runtime_summary.csv','runtime_case_means.csv','evaluation_complete.json','execution_supplement.json','integrity.json','calibrate.log','full_tests.log','boundary_tests.log','access_sessions.jsonl','operator_build_calibrate.json','numerical_validation.json','gt_discretization.json','numerical_validation.log','network_timing.log','network_timing_attempt1_failed.log','render_audit.json','render.log','boundary_tests_final.log','path_and_mask_validation.json','path_validation.log','path_validation_attempt1_failed.log']
    for name in names:
        if (out/name).exists():shutil.copy2(out/name,archive/name)
    (archive/'worker_audits').mkdir(exist_ok=True)
    for method in ['SART','Tikhonov','TV']:
        for name in ['access_sessions.jsonl','operator_build_evaluate.json','complete.json','run.log']:
            p=out/'workers'/method/name
            if p.exists():shutil.copy2(p,archive/'worker_audits'/(method+'_'+name))
    for p in figure.iterdir():
        if p.is_file():destination=archive/'figures'/p.name;destination.parent.mkdir(exist_ok=True);shutil.copy2(p,destination)
    with gzip.open(archive/'convergence_curves.jsonl.gz','wt') as stream:
        for p in sorted(out.glob('curves/calibrate/*.json'))+sorted(out.glob('workers/*/curves/evaluate/*.json')):
            stream.write(json.dumps(dict(file=str(p.relative_to(out)),curve=json.loads(p.read_text())),separators=(',',':'))+'\n')
    solver_summary=[]
    for method in ['SART','Tikhonov','TV']:
        for candidate in sorted({r['candidate'] for r in classic if r['method']==method}):
            for condition in CONDITIONS:
                sub=[r for r in classic if r['method']==method and r['candidate']==candidate and r['condition']==condition]
                solver_summary.append(dict(method=method,candidate=candidate,condition=condition,
                    solver_relative_l2=float(np.mean([r['solver_stats']['solver_relative_l2'] for r in sub])),
                    solver_objective=float(np.mean([r['solver_stats'].get('objective',.5*r['solver_stats']['solver_relative_l2']**2) for r in sub])),
                    observed_relative_l1=float(np.mean([r['observed_relative_l1'] for r in sub])),
                    observed_relative_l2=float(np.mean([r['observed_relative_l2'] for r in sub])),
                    heldout_relative_l1=float(np.mean([r['heldout_relative_l1'] for r in sub]))))
    csv_write(out/'solver_objectives.csv',solver_summary);shutil.copy2(out/'solver_objectives.csv',archive/'solver_objectives.csv')
    # Chinese report: actual final values only.
    lines=['# 三维经典算法与冻结 MRGR：V4 配对比较 v1','',f'本轮实际完成 **{len(classic)} 次 Val 经典重建**，复用两个冻结网络的 **1080 条完整历史指标**，另实际完成1080次网络前向计时。范围固定为原36 Val case×5draw×Original/Targeted/Random，使用原 station IDs 和共同 H4；没有新网络训练、Pool B、最终 test 或 GT128。','',
        '## 1. 方案与方法边界','',
        '自由稀疏视角定义及物理范围见 [方案修订](../research_revision_geometry_classical_v1_zh.md)，算法来源、公式、目标与更新见 [算法登记](../classical_methods_v1_zh.md)。SART 为有界非负 station-block 三维线性 chi 重建，Tikhonov 为直接使用冻结 P_diff(C) 的有界非线性三维平滑重建，TV 为有界平滑 isotropic 三维 TV。均从零场开始，输出 XYZ64³ CH4质量分数[0,1]。TV epsilon=1e-4/m；三轴物理步长0.6/63m、末端 Neumann。',
        '', 'SART 使用真实 origin/direction/near/path length/valid mask 与 K96 三线性稀疏权重及真正转置；不是普通梯度下降或二维 Radon。一次 sweep 为输入顺序下4个站位 block 各更新一次。ppm·m经摩尔分数线积分解释，最后精确转换C。原投影器插值C后转换chi，SART插值chi；同一路径不代表同一离散化。两种变分算法采用投影BB梯度与单调Armijo，非闭式线性解或精确非平滑TV解。','',
        '## 2. 验证、校准与冻结','',
        f'优化前协议 SHA256 `{digest(out/"protocol.json")}`；校准配置冻结 SHA256 `{digest(out/"selected_configs.json")}`。登记提交 a3201a1；数值源码/参数/停止规则在读取新Val结果前冻结。后续执行补充仅将不同方法分配到相同型号4090的cuda:0/1及独立目录，未改变数值规则。',
        '', '校准采用冻结 source-index train 顺序的12个等间隔 case（rint(linspace(0,165,12))），每例自然随机 V4 与 clustered draw0 两种布局，各有独立不重叠H4。共216次实际校准，每方法3候选；SART150 sweep，变分200步。GT辅助选择按全部12case×2布局平均三维relative L1；测量选择按同一批输出的独立train H4 relative L1，不使用GT选择。身份严格分开。',
        '',markdown_table(['方法','候选','train三维L1','train H4 L1','GT选择','测量选择'],[
            [r['method'],r['candidate'],f"{r['train_relative_l1']:.8f}",f"{r['train_H4_relative_l1']:.8f}",'是' if selection['selected'][r['method']]['GT_assisted']['candidate']==r['candidate'] else '', '是' if selection['selected'][r['method']]['measurement_only']['candidate']==r['candidate'] else ''] for r in json.loads((out/'calibration_scores.json').read_text())]),
        '', '最终全局配置：','']
    for method,identities in selection['selected'].items():
        for identity,s in identities.items():lines.append(f'- {method} / {identity}：candidate{s["candidate"]}，`{json.dumps(s["config"],ensure_ascii=False)}`。')
    lines+=['','两种身份若同候选只实际重建一次，同时进入两类身份表。case重建只使用输入测量与几何；输出先落盘，再读取GT64和H4作评价。不得按GT/H选择停止点；停止只看输入目标相对变化1e-5连续8次，SART至少30/变分至少40步，否则预算耗尽。所有最终输出为实际停止的最后一个iterate，无轨迹后验最优筛选。','',
        '测试含零/常量场、ppm·m、XYZ、路径、边界转换往返、伴随内积、三维phantom、有效零测量/invalid排除、物理梯度和autograd差分。完整仓库212项测试通过，另4项独立读取阻断测试通过，证据在归档日志；真实运行访问审计与测试故意阻断分开。数值验证另存 numerical_validation.json；原projectors.py和自监督训练guard未改。','',
        '## 3. Original / Targeted / Random 完整主表','',
        '各数值先平均同case五draw，再平均36case。relative L1/L2与积分proxy无量纲；MAE/RMSE/峰值绝对误差为质量分数。积分proxy沿用冻结 abs(sum(C)-sum(GT))/sum(abs(GT))，不是kg质量。H4误差通过同一冻结P_diff/K96，单位ppm·m后取相对误差。','',
        markdown_table(['方法身份','条件','三维L1 [95%CI]','三维L2','MAE','RMSE','H4 L1','H4 L2','积分proxy误差','峰值绝对误差'],[
            [r['model'],r['condition'],f"{r['relative_l1']:.6f} [{r['relative_l1_CI'][0]:.6f},{r['relative_l1_CI'][1]:.6f}]",*[f'{r[k]:.6g}' for k in ['relative_l2','mae','rmse','heldout_relative_l1','heldout_relative_l2','integrated_concentration_relative_error','peak_value_absolute_error']]] for r in summary]),
        '',f'bootstrap固定seed{cfg["bootstrap"]["seed"]}、10,000次、36case成对percentile95%CI；射线/180draw不作独立样本。全部10指标的均值及区间见summary.csv；同布局各方法全部配对与同方法布局配对见paired_statistics.csv，逐case均值及逐case/draw来源完整归档。','',
        '## 4. 配对结果：算法能力与布局收益分别报告','',
        '以下算法间差值统一为经典算法−网络，同一输入布局；负值表示经典误差更低。GT_assisted带train GT调参，measurement_only不以三维GT选配置；SelfSup训练无体积GT且用原测量Val选checkpoint，Supervised训练使用GT且历史checkpoint由同Val监督损失选择，存在Val模型选择偏倚。','']
    network_pairs=[]
    for condition in CONDITIONS:
        for metric in ['relative_l1','relative_l2','heldout_relative_l1','heldout_relative_l2']:
            result=paired(means,'SelfSup',condition,'Supervised',condition,metric,index)
            network_pairs.append([condition,metric,f'{result["mean_delta"]:+.6f}',f'[{result["CI_low"]:+.6f},{result["CI_high"]:+.6f}]',f'{result["wins"]}/{result["losses"]}/{result["ties"]}'])
    lines+=['两个网络也按同输入比较，差值SelfSup−Supervised；差异同时包含训练目标/标签/模型选择条件，不能解释成标签使用的单一因果效果：','',markdown_table(['条件','指标','SelfSup−Supervised','95%CI','胜/负/平'],network_pairs),'']
    pairs=[]
    for condition in CONDITIONS:
        for model in [m for m in models if m not in ['SelfSup','Supervised']]:
            for net in ['SelfSup','Supervised']:
                result=paired(means,model,condition,net,condition,'relative_l1',index)
                pairs.append([model,net,condition,f'{result["mean_delta"]:+.6f}',f'[{result["CI_low"]:+.6f},{result["CI_high"]:+.6f}]',f'{result["wins"]}/{result["losses"]}/{result["ties"]}'])
    lines.append(markdown_table(['经典身份','网络','相同条件','三维L1差','配对95%CI','胜/负/平'],pairs))
    lines+=['','自身求解残差与统一P_diff重投影分别如下；各方法objective形式/归一化不同，不能直接把objective数值横向排名：','',markdown_table(['方法','条件','自身relative L2','自身objective','P_diff输入L1','P_diff输入L2','共同H4 L1'],[[r['method']+'/c'+str(r['candidate']),r['condition'],*[f'{r[k]:.6g}' for k in ['solver_relative_l2','solver_objective','observed_relative_l1','observed_relative_l2','heldout_relative_l1']]] for r in solver_summary])]
    cold=json.loads((out/'workers/SART/operator_build_evaluate.json').read_text())
    lines+=['',f'SART本次评价进程实际冷构建{len(cold["stations"])}个站位稀疏算子，共{cold["total_build_seconds"]:.3f}s；其后按station缓存复用。每case/draw未缓存等价构建时间也保存，平均值为{np.mean([r["equivalent_cold_geometry_seconds"] for r in classic if r["method"]=="SART"]):.3f}s。这个准备成本不算作网络前向开销，亦不隐藏为SART速度优势。']
    layout=[]
    for model in models:
        for left,right in [('Targeted','Original'),('Targeted','Random'),('Random','Original')]:
            r=paired(means,model,left,model,right,'relative_l1',index);layout.append([model,left+'−'+right,f'{r["mean_delta"]:+.6f}',f'[{r["CI_low"]:+.6f},{r["CI_high"]:+.6f}]',f'{r["wins"]}/{r["losses"]}/{r["ties"]}'])
    lines+=['','固定方法与V4预算的布局对照如下，表示输入几何改变的收益，不解释为网络结构更新：','',markdown_table(['方法身份','布局配对','三维L1差','95%CI','胜/负/平'],layout),'','![三维及共同H误差](artifacts/classical_v4_v1/figures/paired_errors.png)','',
        '## 5. 时间、内存与收敛','',markdown_table(['方法/candidate','实际重建/前向数','构建/查找均值s','迭代/前向均值s','转换均值s','评价均值s','平均实际迭代','停止状态','GPU peak MiB'],[
            [r['method']+('/c'+str(r['candidate']) if 'candidate' in r else ''),r.get('actual_timed_forwards',r['actual_reconstructions']),f"{r.get('operator_setup_seconds_mean',0):.5f}",f"{r.get('iteration_seconds_mean',r.get('forward_seconds_mean',0)):.5f}",f"{r.get('conversion_seconds_mean',0):.5f}",f"{r.get('evaluation_seconds_mean',0):.5f}",f"{r.get('actual_iterations_mean',0):.1f}",json.dumps(r.get('statuses',{})),f"{r['GPU_peak_allocated_MiB']:.1f}"] for r in runtime]),
        '', '计时用GPU同步的wall clock；网络仅前向（输入准备、投影评价另计），5次warmup/model。经典算法GPU峰值为求解阶段，包含已有驻留稀疏cache；评价阶段峰值未单独记录，不能视为端到端峰值。RSS是进程高水位，reserved memory亦完整保存。两张4090为共享服务器，结果不等于专用硬件速度基准。SART geometry cache按站位复用，冷构建总成本与每case未缓存等价成本在operator_build和runtime表中保存；不能把缓存准备藏为速度优势。变分直接调用原P_diff，每次仍有坐标生成开销。停止状态仅表示统一有限规则，objective_stable不是最优性证明。','',
        '![输入目标收敛示例](artifacts/classical_v4_v1/figures/convergence_examples.png)','', '![固定首个case重建](artifacts/classical_v4_v1/figures/reconstruction_example.png)','', '重建图固定原case0/draw0，显示沿Y的最大值投影，仅用于形态检查；所有图共享GT确定的显示色标，超过色标的预测峰值另标在标题，不用于参数或停止选择。','',
        '## 6. 结果解释、尚未解决问题与下一阶段','']
    for condition in CONDITIONS:
        sub=[r for r in summary if r['condition']==condition];rank=sorted(sub,key=lambda r:r['relative_l1'])
        lines.append(f'- {condition}：本预算三维L1最低为 {rank[0]["model"]}={rank[0]["relative_l1"]:.6f}；SelfSup={next(r for r in sub if r["model"]=="SelfSup")["relative_l1"]:.6f}，Supervised={next(r for r in sub if r["model"]=="Supervised")["relative_l1"]:.6f}。方法能力比较严格限于本行同输入。')
    paths=json.loads((out/'path_and_mask_validation.json').read_text())
    lines+=['',f'冻结Pool A实际射线：单位方向最大偏差{paths["ray_unit_norm_max_error"]:.3e}；near/far端点超出物理域最大量{paths["near_far_domain_bound_max_error_m"]:.3e}m；path length与far−near最大差{paths["path_length_vs_far_minus_near_max_error_m"]:.3e}m。首个校准case的精确valid零测量数={paths["valid_zero_measurements"]}，invalid非零数={paths["invalid_nonzero_measurements"]}；有效零值保留另由合成零场测试验证，不要求真实case必须包含精确零值。']
    numerical=json.loads((out/'numerical_validation.json').read_text())
    lines+=['',f'实际Pool A算子伴随内积相对差={numerical["actual_PoolA_adjoint_relative_discrepancy"]:.3e}；常量C=0.2时稀疏算子与冻结P_diff的相对L1差={numerical["constant_C02_sparse_vs_frozen_Pdiff_relative_l1"]:.3e}；mass/mole往返最大绝对差={numerical["mass_mole_roundtrip_max_absolute_error"]:.3e}。小型三维phantom的SART/Tikhonov/TV三维L1为'+ '/'.join(f'{numerical["phantom"][m]["relative_l1"]:.6f}' for m in ['SART','Tikhonov','TV'])+'；它只验证小规模实现，不代表稀疏Val可唯一恢复。']
    discretization=[r['discretization_relative_l1'] for r in classic if r['method']=='SART']
    gt_disc=json.loads((out/'gt_discretization.json').read_text())['rows']
    lines+=['',f'SART最终C上，插值chi与插值C后转换chi的差异relative L1均值={np.mean(discretization):.6g}、最大={max(discretization):.6g}；独立train GT64诊断均值={np.mean([r["relative_l1"] for r in gt_disc]):.6g}。这些是实际输出/GT64诊断，不作为Val调参。差异还包含稀疏权重float64构建与P_diff FP32坐标的浮点差异；常量场检查给出其数值量级参考。']
    all_bad=[]
    for model in models:
        a=next(r for r in summary if r['model']==model and r['condition']=='Original')['relative_l1'];b=next(r for r in summary if r['model']==model and r['condition']=='Targeted')['relative_l1']
        all_bad.append((model,a>b))
    lines+=['', 'Original clustered的平均三维L1高于Targeted的身份：'+ '、'.join(m for m,bad in all_bad if bad)+'。这是布局退化的实测范围；若某方法不符合该方向也保留，不声称所有指标/所有case一致。']
    lines+=['','优势并不覆盖全部指标：SART在Original/Targeted/Random的积分proxy误差均值为0.0255455/0.0130534/0.0127580，低于监督MRGR的0.107936/0.0238415/0.0271228；对应配对差的95%CI均全负。Original中SART还优于SelfSup的积分proxy（差−0.0155082，CI[−0.0286878,−0.00402573]），但Targeted/Random相对SelfSup的积分proxy区间包含0。Original峰值绝对误差SART也低于监督MRGR（差−0.0897569，CI[−0.124412,−0.0535724]），同时劣于SelfSup。网络三维分布/共同H优势不能替代这些独立量化指标，也不能被概括为所有指标更优。']
    failure=sum(sum(v for k,v in r.get('statuses',{}).items() if k in ['line_search_failed','nonfinite']) for r in runtime)
    exhausted=sum(r.get('statuses',{}).get('budget_exhausted',0) for r in runtime)
    lines+=['',f'经典评价数值失败/线搜索失败计数={failure}；达到预算上限={exhausted}。未按Val追加参数、延长迭代或调参重试。若经典预算耗尽，其结果只能说明这个有限协议的性能，不能将未收敛差距包装为网络理论优势或经典理论下限。',
        '', '网络的摊销推理与训练形成的先验、经典的零初始化/稀疏测量歧义，以及GT调参身份是不同条件。监督网络标签优势也不能被视为无标签方法能力。几何覆盖和方向不足时，误差增加可能混合信息不足、表示误差和方法优化能力，不能由一次比较唯一归因。',
        '', '所有输入均来自128³/K256冻结参考，而输出64³/K96；表示、积分和mass/mole插值非交换误差仍在。SART最终输出的离散化误差单独记录；没有把它全算成SART算法不足。几何分数不是条件数，Targeted只是方向启发式，非全局最优设计。单网络seed、同Val选择的监督checkpoint、case家族相关性、探索性多重比较、有限经典预算均限制结论。',
        '', '下一阶段判断：值得优先登记一次匹配的几何分层采样训练受控验证。依据是冻结网络在相同V4预算下仅改变一个输入站位，Original→Targeted的误差已有明确下降；这条同网络证据比把未收敛经典差距当作训练收益更可靠。固定结构、标签条件、训练budget与自然随机控制，仅改变预登记Pool A采样分布，并在相同自然随机/分层/压力布局上配对。训练能否帮助原退化布局尚未验证，覆盖不足与歧义不能靠改变采样保证消除。经典TV/SART等预算耗尽限制仍需承认，不能宣称已解决几何歧义。本轮只提出建议，停止，不训练或追加蒸馏/模块/先验。',
        '', '## 7. 完整性、归档与未运行扩展','',
        '完整性审计核对冻结数据/原GT128 metadata（不打开GT128）、历史checkpoint/日志/比较/诊断metadata、原tracked文件与数值源码hash。两网络checkpoint hash及1080条输入/H IDs逐条核对，原Original复现2240项通过证据保留。原自监督读取保护、投影器、历史结论与未跟踪绘图脚本均保留。没有重训、覆盖或恢复其他实验。',
        '', '归档含协议、station清单、1323条几何与180条V4描述、参数身份与候选、216次校准、Val逐case/draw、配对统计、全部scalar收敛曲线、关键图、时间/内存、源码及来源hash、测试/访问/完整性审计。体积/测量/GT/checkpoint/大型算子不提交Git。外部输出目录为 `/data/djc/GeoPhy-Set3D/outputs/evaluations/classical_v4_v1`。',
        '', 'V3/V6及完整几何分层扩展仅完成协议与几何清单，未运行模型矩阵；普通固定V4参照本轮未追加。Pool B、最终test、GT128、新网络训练及下一轮实验均未执行。',
        '', '补充路径诊断首次误设首个真实case必须含精确valid零测量的检查而停止；该case实际为0条精确零值，后续如实记录，invalid非零检查保持。有效零值逻辑已通过合成测试。原失败日志与审计保留，未修改求解器、数据、参数或任何Val输出。',
        '', '网络计时首次因漏设原关闭TF32开关在首条输出一致性核对时停止，失败日志保留（首条失败前向和5次warmup共6次未纳入成功计时）；恢复原seed_everything设置后才完成计时，没有采用失败计时或替换历史指标。经典数值源码/参数及全部Val重建没有因此修改。',
        '', '复现先运行 register_geometry_catalog.py / run_classical_v4.py register；完整校准或evaluation已有结果会核验并退出，缺项只按自身保存的重建/评价边界恢复，不重复已完成重建。worker入口与执行补充登记互相核对。报告入口只读实际标量，不重建、不填估计数字。']
    (ROOT/'docs/reports/classical_v4_comparison_v1_zh.md').write_text('\n'.join(lines)+'\n')
    manifest={p.relative_to(archive).as_posix():digest(p) for p in sorted(archive.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
    write(archive/'artifact_manifest.json',manifest);print('Chinese report and actual artifacts exported',len(classic))
if __name__=='__main__':main()
