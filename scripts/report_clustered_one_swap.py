#!/usr/bin/env python3
"""Render the already completed registered experiment; no model inference."""
import csv
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from geophy_set3d.evaluation.clustered_one_swap import ClusteredEvaluationReadGuard, paired_station_sets

ROOT=Path('/data/djc/GeoPhy-Set3D/outputs/diagnostics/clustered_v4_one_swap')
ARCHIVE=REPO/'docs/reports/artifacts/clustered_v4_one_swap'
REPORT=REPO/'docs/reports/clustered_v4_one_swap.md'
CONDITIONS=['Original','Targeted','Random']
COLORS=['#677582','#167b64','#db8738']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def decision(s):return dict(improvement='支持改善',inconclusive='不确定',worsening='支持恶化')[s['decision']]
def statrow(s):return [s['model'],s['metric'],s['comparison'],f"{s['left_mean']:.8f}",f"{s['right_mean']:.8f}",f"{s['mean_delta']:+.8f}",f"{s['median_delta']:+.8f}",f"{100*s['relative_improvement']:+.2f}%",f"{s['wins']}/{s['losses']}/{s['ties']}",f"[{s['case_paired_95ci'][0]:+.8f}, {s['case_paired_95ci'][1]:+.8f}]",decision(s)]


def complete_saved_summary():
    """Repair export only, preserving registered sources and the failure record."""
    if (ROOT/'summary.json').exists():return
    failures=list(ROOT.glob('failure_evaluate_*.json'))
    assert len(failures)==1 and read(failures[0])['error']=="TypeError('Object of type bool is not JSON serializable')"
    runner=runpy.run_path(str(REPO/'scripts/diagnose_clustered_one_swap.py'))
    cfg,root=runner['configuration']();runner['verify'](cfg,root)
    assert read(ROOT/'inventory_before.json')==runner['inventory'](cfg)
    integrity=read(ROOT/'integrity.json')
    assert integrity['parameters_and_buffers_unchanged'] and integrity['checkpoint_hashes_unchanged'] and integrity['dataset_metadata_unchanged']
    assert read(ROOT/'evaluation_audit.json')['forbidden_attempts']==0
    rows=read(ROOT/'per_case_draw.json');stats=read(ROOT/'paired_statistics.json');records=read(ROOT/'station_protocol.json')['records']
    independent_verification(rows,records,stats)
    metric_names=['relative_l1','relative_l2','mae','rmse','heldout_relative_l1','heldout_relative_l2']
    draws=[dict(model=m,condition=c,draw=d,**{k:float(np.mean([r[k] for r in rows if (r['model'],r['condition'],r['draw'])==(m,c,d)])) for k in metric_names}) for m in cfg['models'] for c in CONDITIONS for d in range(5)]
    correlations=[];worst={}
    for model in cfg['models']:
        for comparison in ('Targeted-Original','Random-Original','Targeted-Random'):
            effect=next(s for s in stats if (s['model'],s['metric'],s['comparison'])==(model,'relative_l1',comparison))
            left,right=comparison.split('-');gx=[];dy=[]
            for case in effect['per_case']:
                relevant=[r for r in records if r['case_id']==case['case_id']]
                gx.append(float(np.mean([r['geometry'][left]['score']-r['geometry'][right]['score'] for r in relevant])));dy.append(case['delta'])
            constant=bool(np.std(gx)<=1e-12)
            corr=float(np.corrcoef(gx,dy)[0,1]) if not constant and np.std(dy)>1e-12 else None
            correlations.append(dict(model=model,comparison=comparison,case_mean_score_delta=gx,case_mean_error_delta=dy,pearson=corr,constant_geometry_across_cases=constant,interpretation='descriptive only; case/family dependence, no p-value'))
        s=next(s for s in stats if (s['model'],s['metric'],s['comparison'])==(model,'relative_l1','Targeted-Original'))
        worst[model]=sorted(s['per_case'],key=lambda r:r['left_mean'],reverse=True)[:5]
    primary=next(s for s in stats if (s['model'],s['metric'],s['comparison'])==('SelfSup','relative_l1','Targeted-Original'))
    extra=next(s for s in stats if (s['model'],s['metric'],s['comparison'])==('SelfSup','relative_l1','Targeted-Random'))
    recovery=dict(reason='NumPy bool in descriptive summary was not JSON serializable; all actual forwards, raw metrics, statistics and integrity were saved before failure',
        numerical_source_changed=False,protocol_changed=False,model_forward_repeated=False,original_failure_record=failures[0].name,
        verified_raw_rows=1080,statistics_rechecked=36,canonical_raw_result_sha256=sha(ROOT/'per_case_draw.json'))
    (ROOT/'export_recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
    summary=dict(completed=True,protocol_sha256=sha(ROOT/'protocol.json'),row_count=len(rows),primary=primary,targeted_vs_random=extra,
        paired_statistics=stats,per_draw=draws,geometry_correlations=correlations,worst_targeted_cases=worst,
        random_targeted_coincidences=sum(r['random_equals_targeted'] for r in records),integrity=integrity,audit=read(ROOT/'evaluation_audit.json'),
        original_reproduction=read(ROOT/'original_reproduction.json'),export_recovery=recovery)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


def independent_verification(rows,records,stats):
    assert len(rows)==1080 and len(records)==180
    assert len({(r['model'],r['condition'],r['case_id'],r['draw']) for r in rows})==1080
    ids={(r['case_id'],r['draw']):r for r in records}
    grouped={}
    for r in rows:
        p=ids[r['case_id'],r['draw']]
        assert r['input_station_ids']==p[r['condition']] and r['H_station_ids']==p['H']
        assert len(set(r['input_station_ids']))==4 and len(set(p['H']))==4
        assert not set(p['H']) & set(r['input_station_ids'])
        grouped.setdefault((r['model'],r['case_id'],r['draw']),[]).append(r)
    for pair in grouped.values():
        assert {r['condition'] for r in pair}==set(CONDITIONS)
        assert len({r['heldout_valid_rays'] for r in pair})==1
    errors=[]
    for s in stats:
        left,right=s['comparison'].split('-');cases=sorted({r['case_id'] for r in rows})
        delta=[]
        for case in cases:
            pairs={c:{r['draw']:r[s['metric']] for r in rows if r['model']==s['model'] and r['case_id']==case and r['condition']==c} for c in (left,right)}
            assert all(set(p)==set(range(5)) for p in pairs.values())
            delta.append(np.mean([pairs[left][d]-pairs[right][d] for d in range(5)]))
        delta=np.array(delta)
        indices=np.random.default_rng(s['bootstrap_seed']).integers(0,36,size=(10000,36))
        ci=np.percentile(delta[indices].mean(axis=1),[2.5,97.5])
        np.testing.assert_allclose(ci,s['case_paired_95ci'],atol=1e-14,rtol=1e-12)
        np.testing.assert_allclose(delta.mean(),s['mean_delta'],atol=1e-14,rtol=1e-12)
        errors.append(float(np.max(np.abs(ci-s['case_paired_95ci']))))
    result=dict(passed=True,scalar_rows=1080,case_draw_station_rows=180,statistics_checked=36,
        bootstrap_unit='36 cases, five paired differences averaged first',max_ci_absolute_recheck_error=max(errors),
        same_H_ids_and_valid_ray_counts=True,all_V4_and_H_disjoint=True,no_model_forward_or_optimizer=True)
    (ROOT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')


def plots(summary,records):
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for i,model in enumerate(('SelfSup','Supervised')):
        for j,comparison in enumerate(('Targeted-Original','Targeted-Random')):
            s=next(s for s in summary['paired_statistics'] if s['model']==model and s['metric']=='relative_l1' and s['comparison']==comparison)
            ax=axes[i,j];x=[r['right_mean'] for r in s['per_case']];y=[r['left_mean'] for r in s['per_case']]
            hi=max(x+y)*1.05;ax.plot([0,hi],[0,hi],color='gray',linestyle='--');ax.scatter(x,y,s=22,color=COLORS[1])
            ax.set(xlabel=comparison.split('-')[1]+' volume relative L1',ylabel='Targeted volume relative L1',title=model+' / 5 draws averaged per case',xlim=(0,hi),ylim=(0,hi))
            ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'paired_volume_errors.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,field,title in zip(axes,['score','crossing_angle_mean_deg','direction_tensor_eigenvalues'],['Direction diversity score','Mean acute crossing angle (deg)','Smallest direction tensor eigenvalue']):
        for condition,color,offset in zip(CONDITIONS,COLORS,[-.22,0,.22]):
            vals=[[r['geometry'][condition][field][0] if field=='direction_tensor_eigenvalues' else r['geometry'][condition][field] for r in records if r['draw']==d] for d in range(5)]
            ax.bar(np.arange(5)+offset,[np.mean(v) for v in vals],width=.22,color=color,label=condition,
                yerr=[[max(0.,np.mean(v)-min(v)) for v in vals],[max(0.,max(v)-np.mean(v)) for v in vals]],capsize=2)
        ax.set(xlabel='Draw',title=title);ax.set_xticks(range(5));ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Geometry only; bars average 36 cases, whiskers show ranges');fig.tight_layout()
    fig.savefig(out/'geometry_changes.png',dpi=160);plt.close(fig)
    fig=plt.figure(figsize=(15,4))
    for draw in range(5):
        r=next(r for r in records if r['case_index']==0 and r['draw']==draw)
        ax=fig.add_subplot(1,5,draw+1,projection='3d')
        for cond,color in zip(('Original','Targeted'),COLORS):
            origins=np.array(r['geometry'][cond]['origins_m']);d=.3-origins;d/=np.linalg.norm(d,axis=1)[:,None]
            ax.quiver(np.zeros(4),np.zeros(4),np.zeros(4),*d.T,color=color,alpha=.65,length=1,arrow_length_ratio=.1)
        ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),title=f'Draw {draw}: {r["deleted"]} -> {r["targeted_added"]}',xlabel='dx',ylabel='dy',zlabel='dz')
        ax.view_init(elev=25,azim=-60)
    fig.suptitle('True 3D station-to-center directions: gray Original, green Targeted');fig.tight_layout()
    fig.savefig(out/'direction_changes.png',dpi=160);plt.close(fig)
    cfg=read(ROOT/'protocol.json')['config'];spec=cfg['figures'];assert spec['block_mean_voxel_size']==2
    for model in ('SelfSup','Supervised'):
        for index in spec['case_indices']:
            file=ROOT/'case_images'/f'{model}_val{index}_Original.npz'
            with np.load(file) as z:volumes=[z['GT'],z['Original']]
            for cond in ('Targeted','Random'):
                with np.load(ROOT/'case_images'/f'{model}_val{index}_{cond}.npz') as z:volumes.append(z[cond])
            small=[v.reshape(32,2,32,2,32,2).mean((1,3,5)) for v in volumes]
            norm=Normalize(vmin=spec['mass_fraction_threshold'],vmax=max(.021,max(float(v.max()) for v in small)))
            fig=plt.figure(figsize=(13,4));axes=[]
            coords=np.indices((33,33,33))*.6/32
            for j,(name,v) in enumerate(zip(['GT64']+CONDITIONS,small)):
                ax=fig.add_subplot(1,4,j+1,projection='3d');axes.append(ax)
                ax.voxels(*coords,v>=spec['mass_fraction_threshold'],facecolors=plt.cm.viridis(norm(v)),edgecolor=None,shade=False)
                ax.set(xlim=(0,.6),ylim=(0,.6),zlim=(0,.6),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=name)
                ax.set_box_aspect((1,1,1));ax.view_init(elev=spec['elevation'],azim=spec['azimuth'])
            case=next(r['case_id'] for r in records if r['case_index']==index)
            fig.suptitle(f'{model} / Val index {index} / {case} / draw0\n2x2x2 block means; fixed CH4 mass-fraction threshold 0.02')
            fig.subplots_adjust(left=.01,right=.91,top=.79,bottom=.03,wspace=.04)
            fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=axes,shrink=.65,pad=.03,label='CH4 mass fraction')
            fig.savefig(out/f'volume_{model}_val{index}.png',dpi=130);plt.close(fig)


def report(summary,records):
    primary=summary['primary'];extra=summary['targeted_vs_random'];stats=summary['paired_statistics'];protocol=read(ROOT/'protocol.json')
    header=['模型','指标','配对','左均值','右均值','均值差','中位数差','相对改善','胜/负/平','case配对95%CI','规则结论']
    main=table(header,[statrow(s) for s in stats if s['metric']=='relative_l1'])
    aux=table(header,[statrow(s) for s in stats if s['metric']!='relative_l1'])
    geom=[]
    for draw in range(5):
        r=next(r for r in records if r['case_index']==0 and r['draw']==draw)
        random=[x['geometry']['Random']['score'] for x in records if x['draw']==draw]
        geom.append([draw,str(r['Original']),f"{r['deleted']} → {r['targeted_added']}",f"{r['geometry']['Original']['score']:.8f}",f"{r['geometry']['Targeted']['score']:.8f}",f"{np.mean(random):.8f}",f"{r['geometry']['Original']['crossing_angle_mean_deg']:.2f} → {r['geometry']['Targeted']['crossing_angle_mean_deg']:.2f}",f"{r['geometry']['Original']['direction_tensor_eigenvalues'][0]:.3g} → {r['geometry']['Targeted']['direction_tensor_eigenvalues'][0]:.3g}"])
    worst=[]
    for model,items in summary['worst_targeted_cases'].items():
        for r in items:
            rr=next(s for s in stats if s['model']==model and s['metric']=='relative_l1' and s['comparison']=='Targeted-Random')
            rand=next(c['right_mean'] for c in rr['per_case'] if c['case_id']==r['case_id'])
            worst.append([model,r['case_id'],f"{r['right_mean']:.6f}",f"{r['left_mean']:.6f}",f"{rand:.6f}",f"{r['delta']:+.6f}"])
    draws=table(['模型','draw','Original L1','Targeted L1','Random L1'],[[m,d]+[f"{next(r['relative_l1'] for r in summary['per_draw'] if r['model']==m and r['draw']==d and r['condition']==c):.8f}" for c in CONDITIONS] for m in ('SelfSup','Supervised') for d in range(5)])
    ci=primary['case_paired_95ci'];ec=extra['case_paired_95ci']
    conclusions='Targeted 相对 Original 的主指标'+decision(primary)+'；Targeted 相对 Random '+decision(extra)+'。'
    if extra['decision']!='improvement':conclusions+=' 因未明确优于随机替换，不能声称互补选择策略具有额外优势。'
    corr=table(['模型','配对','case均值几何变化与L1变化 Pearson（描述性）'],[[r['model'],r['comparison'],'不可识别：几何增量在case间恒定' if r['pearson'] is None else f"{r['pearson']:+.5f}"] for r in summary['geometry_correlations']])
    text=f'''# 冻结模型 clustered V4：单站位互补替换诊断

**{conclusions}** SelfSup v1 C1 的 Targeted−Original 三维 relative L1 均值差为 **{primary['mean_delta']:+.8f}**，95% case 配对区间 **[{ci[0]:+.8f}, {ci[1]:+.8f}]**；Targeted−Random 为 **{extra['mean_delta']:+.8f}**，区间 **[{ec[0]:+.8f}, {ec[1]:+.8f}]**。这是固定 V=4、更换一次输入几何的单checkpoint诊断，未更新模型。

## 1. 实际执行与原结果复现（已确认事实）

开始时本地/远端均为 `0a010bb9a2163ebec2c322f972e512795c8128e5`；未发现适用的祖先/仓库 AGENTS.md、同协议已完成或未完成产物。读取 SelfSup v1、监督MRGR、原clustered协议与逐case/draw记录、最新RVTC匹配报告。最新RVTC报告的 Val-A36 测量 Treatment−Control L1 为 +0.03139504，区间 [+0.01683308,+0.04647491]；这是历史引用，本轮未重跑对照、训练或梯度诊断。保留 SelfSup v1 与原v2 NO-GO，不调整历史判据。

独立入口 `scripts/diagnose_clustered_one_swap.py`，配置 `configs/diagnose_clustered_one_swap.yaml`，独立外部目录 `{ROOT}`。协议在零前向推理时登记，SHA256 `{summary['protocol_sha256']}`；登记代码、180条站位清单与测试先提交为 `ac6c4e9`，然后才运行 Original 和新条件。登记阶段没有读取投影或GT。站位采样器与两模型历史ID逐条一致；全部36 Val / Pool A案例、原draw0–4保留。

两模型均为 GeoLift-OARI-MRGR-Res3D，332,070参数，C1、eval、FP32，推理时 requires_grad=False / inference_mode。SelfSup v1 checkpoint SHA256 `{protocol['models']['SelfSup']['sha256']}`，源提交 `{protocol['models']['SelfSup']['training_source_commit']}`，第278轮，原300轮完全结束后按无GT held-out测量选定。监督MRGR SHA256 `{protocol['models']['Supervised']['sha256']}`，源提交 `{protocol['models']['Supervised']['training_source_commit']}`，第295轮，按同一Val的监督损失选定；辅助结果因此带有Val模型选择偏倚。checkpoint未改写，未加载其他权重。

Original两模型共360次真实前向；四项三维指标、两项输入测量指标逐case复现，另核对draw mean/pooled，总计 **2240项比较全部通过**。预登记容差 abs1e-8 + rel1e-5×|历史值|，沿用冻结参考容差，未按结果调整。最大绝对差 {summary['original_reproduction']['max_absolute_difference']:.3g}。三维L1 SelfSup=0.39340238915257336，监督=0.4952099720969504，与历史一致。

新条件720次真实前向，共1080条模型×条件×case×draw结果。没有optimizer、反向传播、更新、GT128、Pool B或最终test读取。模型参数和buffer前后canonical hash一致，数据/GT128及历史产物的size/mtime/inode清单前后相同；原数值源码、历史指标、checkpoint SHA256逐项相同。数据完整性证据是访问写阻断和全文件元数据清单对比，没有冒称重读全部训练数据做逐字节哈希。

导出时曾出现一次summary内NumPy bool无法JSON序列化的实现错误。错误发生在全部实际前向、逐条指标、配对统计和完整性记录成功保存之后；保留failure_evaluate与原日志，通过独立报告入口仅从已存标量补写摘要并重新验证1080条/36项区间，**没有重复推理，没有修改已登记源码、选择或数值规则**。几何图首次绘制还因常量的浮点舍入使误差棒出现微小负值而停止；仅将绘图范围长度截到0后完成，保留绘图错误日志与failed审计session，不影响任何数值结果。

## 2. 冻结选择、共同H与统计（已确认事实）

几何中心(0.3,0.3,0.3)m，真实三维 d=normalize(center−station_origin)，score=mean(1−|d_i·d_j|)，不使用测量、GT、误差或输出。遍历4种删除与60个Original外Pool A候选，最大分数；精确float64平分按删除ID、新增ID字典顺序。原三个站位及其输入位置保留，新站位占删除位置。绝对内积使近乎反向的平行方向仍计为不互补。这是方向多样性启发式，不是完整可观测性、条件数或数学最优站位。

Random从排序的Original外站位均匀选1个，删除与Targeted相同ID，seed={protocol['config']['random_seed']}；180次中随机恰好等于Targeted **{summary['random_targeted_coincidences']}次**，全部保留。H4从三条件输入并集之外均匀无放回抽样，独立seed={protocol['config']['heldout_seed']}，各模型/条件同case/draw使用同一H4与完全相同有效射线。identity seed采用 purpose+NUL+case_id 的 blake2b16/little-endian u32，经SeedSequence(base,draw,words)与PCG64重放。完整ID、origin、azimuth、删除/新增、H、方向描述与checkpoint哈希已归档。

测量、全部geometry/mask、station IDs沿同一索引同步，H不输入网络。保留linear_p99_5尺度32284.894580078173 ppm·m，冻结投影64³/K96/all rays/view_chunk2；参考测量128³/K256。分辨率/积分误差仍存在，共同H只保证配对公平，不消除建模误差。

所有均值先对每case的5draw平均，再对36case平均；差值为左−右，负为误差降低，相对改善=−均值差/右均值。胜/负/平由case平均差值的<0/>0/==0确定。95% percentile配对bootstrap共10,000次，seed={protocol['bootstrap']['seed']}，只重采样36个case；不将180记录、视角或射线视为独立样本。所有比较用同一重采样seed。区间全负且均值负支持改善，全正且均值正支持恶化，含0不确定；主指标之外均为辅助探索，未多重校正。独立汇总代码重新按draw差值聚合并核算36项区间，通过。

## 3. 三维主指标与随机替换比较（已确认事实）

{main}

SelfSup主L1改善约{100*primary['relative_improvement']:.2f}%，相对随机替换进一步改善约{100*extra['relative_improvement']:.2f}%；对应L2、MAE/RMSE及共同H的L1/L2方向一致且区间全负。监督MRGR的三维四指标都支持Targeted优于Original，L1/L2/RMSE还支持优于Random，说明该几何干预的体积收益在这两个不同训练方式的冻结checkpoint上均出现；但监督MRGR的Targeted−Random **MAE及共同H L1/L2区间包含0**，因此辅助优势并非全部得到确认。Targeted对Random的case L1并非全胜：SelfSup有9例更差，监督有13例更差，全部保留。

![case配对体积误差](artifacts/clustered_v4_one_swap/figures/paired_volume_errors.png)

## 4. 三维与共同H辅助指标（已确认事实）

MAE/RMSE单位为CH4质量分数，relative L1/L2无量纲。各行是独立指标，不以测量改善替代体积结论。

{aux}

## 5. 每draw与几何描述（已确认事实）

{draws}

draw0中两模型Targeted的平均L1都高于Random，另外4个draw都低于Random；5draw预登记聚合结果支持额外优势，不能包装为每draw都更好，也没有因此剔除draw0或改变规则。

Original/Targeted几何跨case相同，Random依case变化；下表Random分数取36case均值，角度/特征值仅描述，不用于调整选择。

{table(['draw','Original IDs','删除 → 新增','Original分数','Targeted分数','Random分数均值','平均锐crossing angle °','Q最小特征值'],geom)}

![几何变化](artifacts/clustered_v4_one_swap/figures/geometry_changes.png)
![真实三维方向](artifacts/clustered_v4_one_swap/figures/direction_changes.png)

crossing angle=acos(|dot|)，Q=mean(d dᵀ)，特征值升序，全部6个pair角度与3个特征值存于原始记录。Q仅描述中心方向的空间分布，不是重建系统条件数。Targeted分数显著增大并不保证所有方向都得到改善：最小特征值仍约1e-6至1e-5，draw2/3甚至下降，中心方向仍接近一个平面；这阻止将分数改善解释为三维完整可观测性。

{corr}

## 6. 仍存在的大误差案例与固定图例（事实／解释边界）

按预登记规则列出各模型Targeted case平均L1最大的5例，没有结果后设定“严重失败”阈值或删去不利案例。其完整5draw、H误差和尺度在逐条文件。

{table(['模型','case','Original L1','Targeted L1','Random L1','Targeted−Original'],worst)}

原始与Targeted方向分数在各case完全相同，却对应不同误差，因此该分数不能解释所有困难案例，也不能在case层面估计Targeted−Original的分数/误差相关性。随机条件的描述性相关不构成因果或策略最优性证据。

图例在评价前固定为Val顺序索引0、18、35，draw0，每模型各3例，未按结果挑成功。图为2×2×2体素块均值≥0.02的可视化，固定elev25/azim−60、0–0.6m域；低于阈值部分不显示，所有指标仍在原64³全体素计算。每case四面板共用颜色尺度，case之间颜色尺度可不同，不能将阈值图当完整误差图。

'''
    for model in ('SelfSup','Supervised'):
        for index in (0,18,35):text+=f'![{model}固定Val索引{index}](artifacts/clustered_v4_one_swap/figures/volume_{model}_val{index}.png)\n\n'
    text+='''## 7. 证据解释与尚未验证

**已确认事实：** 本轮配对结果检验同一冻结模型、固定四站位预算下的一次输入替换；两种训练方式的指标分别完整报告。若两者同方向，仅支持几何影响在这两个checkpoint上跨训练方式出现，不支持普遍规律或优于所有随机策略。原clustered输入上的能力没有因此提升。

**解释假设：** 两模型在替换后均改善、随机替换也明显改善，与集中输入的方向冗余限制重建这一解释一致；Targeted相对Random的增量说明预设方向多样性选择在本轮有帮助。集中输入方向的信息不足、训练分布适配及网络利用多样几何的能力，均可能影响结果。替换带来几何与站位对应测量一起变化，是合理输入干预；测量与体积改善是否一致应按表逐项判断，不能仅凭几何分数因果归因。仅4个方向到中心的描述也不包含全部射线覆盖和体积结构。

**尚未验证：** 唯一恢复、真实完整可观测性、跨checkpoint/seed稳定性、独立数据来源/工况、最终test及其他几何池均未评价。同来源案例可能有工况相关性，case bootstrap不能消除此相关性；两个模型各仅一个checkpoint，监督checkpoint的Val选择偏倚进一步限制外推。本轮没有训练新模型或评估模型可改进空间的优化收益，不调整评分，不扩池、不增视角，不自动启动后续实验。

## 8. 测试、审计、归档与重放

本轮实际执行27项测试全部通过（新增8项与原视角/测量测试19项），并独立重核1080条实际结果、36项配对区间，以及只用冻结几何重放全部180组站位。几何穷举/平分、反向平行、保留三站位、Random相同删除与巧合保留、H隔离、同步mask/geometry/测量索引、case5draw聚合、参数bufferhash及允许/禁止路径均有针对性测试。合成数据仅作为单元测试fixture，没有用于实际评价或补齐结果。

`access_sessions.jsonl`记录register/original/evaluate/render各阶段；运行审计中禁止访问尝试均0，允许GT64/Pool A均仅36个Val文件，登记0测量/GT读取；optimizer创建/更新均0。测试日志里的预期拒绝属于测试fixture，与实际数据运行审计分开。原训练访问保护完全不变。

仓库归档包含协议、180条完整站位描述、1080条CSV/JSON、逐case配对、逐draw、Original2240项复现比较、完整访问和不可变性日志、测试、关键图以及源码/产物SHA256。`SHA256SUMS`校验归档和新脚本/配置/测试/报告；`artifact_manifest.json`记录外部完整清单的hash和排除项。不提交checkpoint、投影、GT或用于作图的体积NPZ。完整模型/数据仍在原位置，外部缓存保留不覆盖历史。

```bash
PYTHONPATH=src:tests /scratch/djc/envs/gas-torch/bin/python -m unittest test_clustered_one_swap test_evaluation_view_selection test_measurement_consistency -v
/scratch/djc/envs/gas-torch/bin/python scripts/diagnose_clustered_one_swap.py register
/scratch/djc/envs/gas-torch/bin/python scripts/diagnose_clustered_one_swap.py original
/scratch/djc/envs/gas-torch/bin/python scripts/diagnose_clustered_one_swap.py evaluate
/scratch/djc/envs/gas-torch/bin/python scripts/report_clustered_one_swap.py
sha256sum -c docs/reports/artifacts/clustered_v4_one_swap/SHA256SUMS
```

已有完整同协议文件时诊断入口只校验并跳过，不重复前向。首次register要求27项测试日志在独立输出目录的父目录，命名clustered_one_swap_tests.log且以OK结束；已有未完成标记时拒绝覆盖并报错。报告入口只读取已有结果、冻结几何与固定缓存，不运行模型。完成本轮后停止；SelfSup v1及原v2 NO-GO均保留。
'''
    REPORT.write_text(text)


def main():
    complete_saved_summary()
    runner=runpy.run_path(str(REPO/'scripts/diagnose_clustered_one_swap.py'))
    cfg,root=runner['configuration']();runner['verify'](cfg,root)
    assert read(ROOT/'inventory_before.json')==runner['inventory'](cfg)
    summary=read(ROOT/'summary.json');assert summary['completed']
    rows=read(ROOT/'per_case_draw.json');records=read(ROOT/'station_protocol.json')['records']
    independent_verification(rows,records,summary['paired_statistics'])
    protocol=read(ROOT/'protocol.json');cfg=protocol['config']
    with ClusteredEvaluationReadGuard(cfg['dataset'],[m['checkpoint'] for m in cfg['models'].values()],ROOT) as guard:
        ds=runner['dataset'](cfg)
        for r in records:
            replay=paired_station_sets(r['Original'],ds.geometry['station_origin_m'],r['case_id'],r['draw'],cfg['random_seed'],cfg['heldout_seed'],cfg['center_m'])
            assert all(r[k]==v for k,v in replay.items())
            for c in ('Targeted','Random'):
                assert set(r[c]) & set(r['Original'])==set(r['Original'])-{r['deleted']}
        (ROOT/'station_replay_verification.json').write_text(json.dumps(dict(passed=True,rows=180,geometry_only=True,
            deleted_same=True,three_original_preserved=True,deterministic_random_and_H_match=True,selection_rule_unchanged=True),indent=2)+'\n')
        plots(summary,records);report(summary,records)
        assert guard.counts['forbidden_attempts']==0
        (ROOT/'render_audit.json').write_text(json.dumps(guard.summary(),indent=2)+'\n')
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    exclude={'case_images'}
    for p in ROOT.iterdir():
        if p.name in exclude:continue
        if p.is_file():shutil.copy2(p,ARCHIVE/p.name)
        elif p.name=='figures':shutil.copytree(p,ARCHIVE/'figures',dirs_exist_ok=True)
    manifest=dict(external_output=str(ROOT),excluded_from_git=['case_images/*.npz (GT/prediction volumes)','checkpoints and measurement data'],
        external_inventory_before_sha256=sha(ROOT/'inventory_before.json'),external_inventory_before_bytes=(ROOT/'inventory_before.json').stat().st_size,
        registration_commit='ac6c4e9',no_model_inference_in_report=True)
    (ARCHIVE/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files=[p for p in ARCHIVE.rglob('*') if p.is_file() and p.name!='SHA256SUMS']
    files += [REPORT,Path(__file__),REPO/'scripts/diagnose_clustered_one_swap.py',REPO/'configs/diagnose_clustered_one_swap.yaml',REPO/'tests/test_clustered_one_swap.py',REPO/'src/geophy_set3d/evaluation/clustered_one_swap.py']
    (ARCHIVE/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(REPO).as_posix()+'\n' for p in sorted(files)))
    print(REPORT)


if __name__=='__main__':main()
