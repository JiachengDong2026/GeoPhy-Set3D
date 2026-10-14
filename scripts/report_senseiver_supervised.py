#!/usr/bin/env python3
"""Render only registered scalar artifacts, without opening measurements or GT arrays."""
import gzip,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
A=REPO/'docs/reports/artifacts/ray_senseiver_supervised_stability_v1';B=REPO/'docs/reports/artifacts/ray_senseiver_supervised_v1'

def read(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt') as f:return json.load(f)
    return json.loads(p.read_text())

def main():
    a=read(A/'complete.json');ae=read(A/'evaluations.json.gz');old=read(REPO/'docs/reports/artifacts/ray_senseiver_supervised_probe_v1/B/evaluations.json.gz')
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for key,ax in zip(['nmse','relative_l1'],axes):
        for rows,label in [(old,'Previous fixed LR (historical)'),(ae,'Fresh cosine stability gate')]:ax.plot([r['updates'] for r in rows],[r['mean'][key] for r in rows],label=label)
        ax.set_xlabel('Optimizer updates');ax.set_ylabel(key);ax.axvline(2000,color='gray',ls=':',lw=1);ax.grid(alpha=.2)
    axes[0].axhline(.2,color='gray',ls='--',lw=1);axes[1].axhline(.4,color='gray',ls='--',lw=1);axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(B/'stability_curves.png',dpi=150);plt.close(fig)
    rows=read(B/'history.json.gz') if (B/'history.json.gz').exists() else []
    complete=read(B/'complete.json') if (B/'complete.json').exists() else None
    frozen=read(B/'frozen_best.json') if (B/'frozen_best.json').exists() else None
    if rows:
        fig,axes=plt.subplots(1,3,figsize=(13,3.4))
        for key,ax in zip(['nmse','relative_l1','relative_l2'],axes):
            for split in ['train','val']:ax.plot([r['epoch'] for r in rows],[r[split]['mean'][key] for r in rows],label=split)
            ax.set_xlabel('Epoch');ax.set_ylabel(key);ax.grid(alpha=.2)
        axes[0].legend();fig.tight_layout();fig.savefig(B/'formal_curves.png',dpi=150);plt.close(fig)
    state=('正式训练进行中，尚不能宣称独立基线已完成。' if complete is None else f"正式训练已停止：{complete['reason']}，最佳Val NMSE={complete['best_nmse']:.6g}（epoch {complete['best_epoch']}）。")
    evaldone=(B/'evaluation_complete.json').exists()
    summary='';comparison_intro=''
    if evaldone:
        import pandas as pd
        s=pd.read_csv(B/'group_summary.csv');g1=s[(s.group=='G1')&(s.metric=='relative_l1')].set_index('model');g5=s[(s.group=='G5')&(s.metric=='relative_l1')].set_index('model');comparison_intro=f' 900条冻结评价G1/G5 relative L1：监督Senseiver={g1.loc["Senseiver","mean"]:.4f}/{g5.loc["Senseiver","mean"]:.4f}，监督MRGR={g1.loc["SupervisedMRGR","mean"]:.4f}/{g5.loc["SupervisedMRGR","mean"]:.4f}，自监督Ours={g1.loc["Ours","mean"]:.4f}/{g5.loc["Ours","mean"]:.4f}，DeepSets={g1.loc["DeepSets","mean"]:.4f}/{g5.loc["DeepSets","mean"]:.4f}。标签、目标与预算不同，不能解释为纯架构比较。';summary='\n五档900条评价已完成。下表先对每case的5layout平均，再对36case取均值。\n\n|方法|G1 L1|G2 L1|G3 L1|G4 L1|G5 L1|\n|---|---:|---:|---:|---:|---:|\n'
        for name in ['Senseiver','SupervisedMRGR','Ours','DeepSets']:
            v=s[(s.model==name)&(s.metric=='relative_l1')].set_index('group');summary+='|'+name+'|'+'|'.join(f'{v.loc[g,"mean"]:.6f}' for g in ['G1','G2','G3','G4','G5'])+'|\n'
    if evaldone:
        summary+='\n辅助指标、峰值、总量及观测误差（同一历史定义；完整五档/逐case/区间见CSV）。\n\n|组|方法|3D L2|峰值绝对误差|总量relative error|observed L1|held-out L1|\n|---|---|---:|---:|---:|---:|---:|\n'
        for g in ['G1','G2','G3','G4','G5']:
            for name in ['Senseiver','SupervisedMRGR','Ours','DeepSets']:
                v=s[(s.model==name)&(s.group==g)].set_index('metric')
                metrics=['relative_l2','peak_value_absolute_error','integrated_concentration_relative_error','observed_relative_l1','heldout_relative_l1']
                summary+='|'+g+'|'+name+'|'+'|'.join(f'{v.loc[m,"mean"]:.6g}' for m in metrics)+'|\n'
        summary+='\n[900配对明细](artifacts/ray_senseiver_supervised_v1/paired_900.csv)、[case×group均值](artifacts/ray_senseiver_supervised_v1/case_group_180.csv)、[配对差与95%区间](artifacts/ray_senseiver_supervised_v1/paired_statistics.csv)、[全部十指标汇总](artifacts/ray_senseiver_supervised_v1/group_summary.csv)。峰值和积分浓度proxy沿用质量分数指标，积分项不是kg甲烷质量。\n'
    if (B/'timing.json').exists():
        timing=read(B/'timing.json');historical=read(REPO/'docs/reports/artifacts/deepsets_selfsup_v1/timing.json')
        summary+='\n成本：9个固定输入、每个10次warmup/30次同步FP32 batch1前向。\n\n|方法|参数量|前向median(s)|来源|\n|---|---:|---:|---|\n'
        for name in ['Senseiver','SupervisedMRGR']:
            values=[x['seconds'] for x in timing['rows'] if x['model']==name]
            summary+=f'|{name}|{timing["parameters"][name]}|{np.median(values):.6g}|本轮共享GPU实测|\n'
        for name in ['Ours','DeepSets']:
            values=[v for x in historical['rows'] if x['model']==name for v in x['seconds']]
            summary+=f'|{name}|{historical["identities"][name]["parameter_count"]}|{np.median(values):.6g}|历史冻结实测，硬件负载不匹配|\n'
        summary+='\n同一计时方法不等于同一资源竞争条件，不据跨日期计时比值归因架构；本轮显存及每次原始时间见[timing.json](artifacts/ray_senseiver_supervised_v1/timing.json)。\n'
    reg=read(B/'registration.json');t=a['final'];checks=[r for r in ae if r['updates'] in [2000,2200,2400]]
    report=f'''# 有监督Ray-Senseiver稳定性、正式训练与冻结评价

八样本监督稳定性已经验证，原始254913参数结构在2000/2200/2400步连续通过联合门槛。{state} {'已冻结有效监督候选并完成独立900条评价；比较结果见下文。' if evaldone else '尚未完成冻结900条评价，不报告成功基线排名。'}{comparison_intro} {'下一步仅冻结本次结论与候选，不自动追加配置或训练。' if complete else '下一步按已登记配置继续正式训练，完成后条件冻结评价。'}历史纯自监督REPAIR_FAILED_NO_GO保持，不将本轮监督结果纳入纯自监督榜单。

## 实际执行与阶段A

起始HEAD 33c5785；协议先提交c952ef9；正式训练源码及完整train/Val登记ce5e6e2。协议见[训练前协议](../ray_senseiver_supervised_v1_protocol_zh.md)，原始记录见[阶段A](artifacts/ray_senseiver_supervised_stability_v1/)与[正式阶段](artifacts/ray_senseiver_supervised_v1/)。

阶段A使用过滤train序号0–7，原manifest行0/1/2/3/4/5/6/8，同一四站[26,27,45,42]，seed20261011，fresh初态hash `{a['initial_hash']}`。沿用历史GT均值.0013875673876921429及weight×.01，原sigmoid、NMSE、AdamW2e-4/wd1e-4/clip1/FP32/有效batch2不变，仅预设cosine和预算相对历史变化。实际{a['updates']}更新、{a['elapsed_seconds']:.2f}秒，停止{a['reason']}；最大4000更新/1200秒未用尽。平均模板NMSE={a['references']['mean_template']['mean']['nmse']:.9f}。

|更新|NMSE|3D relative L1|3D relative L2|测量依赖达标例数|错配误差增大例数|
|---:|---:|---:|---:|---:|---:|
'''
    for e in checks:report+=f"|{e['updates']}|{e['mean']['nmse']:.8f}|{e['mean']['relative_l1']:.8f}|{e['mean']['relative_l2']:.8f}|{sum(x['relative_MA']>=.001 for x in e['interventions'])}/3|{sum(x['wrong_minus_normal_nmse']>0 for x in e['interventions'])}/3|\n"
    report+='\n'+'''![阶段A曲线](artifacts/ray_senseiver_supervised_v1/stability_curves.png)

新旧仅共同前2000步可作历史对照；后续点预算不同，不能将差异单独归因学习率调度。固定8例拟合不是独立泛化证据。

## 正式监督实验

使用官方166train/36Val、Pool A，原ViewSampler训练3–6站并逐epoch变化，Val固定4站/epoch0；GT64质量分数、XYZ/y-height、[0,.6]³及原预处理不变。全valid rays含精确零、完整262144 query，单进程可审计加载；train曲线为逐批更新前的在线统计，Val为每轮结束后完整固定输入评价。GT仅用于监督loss与评价，不进入网络。原始RaySenseiverSS，没有全局残差/lifting/OARI/GCA/MRGR/3D卷积。与监督MRGR的输入采样一致，但本轮逐样本归一化voxel MSE不同于其SmoothL1+.1relative-L1，不能称全部训练条件相同或纯架构因果。

'''
    report+=f"初始化只用完整train GT均值{reg['scale']['gt_all_voxel_mean']:.17g}，bias={reg['scale']['bias_logit']:.12g}，weight×.01；无截断，初态hash `{reg['initial_hash']}`。从fresh开始，不使用稳定性、smoke、历史失败或其他模型权重。\n\n"
    report+='预算固定300epoch/7200秒；FP32 AdamW lr2e-4/wd1e-4/clip1，有效batch2，epoch cosine T_max300/eta_min2e-5。唯一selector=Val平均逐case NMSE，实际最低值保存best；独立0.5%显著改善计数至少50轮/30轮patience。每20轮固定train0/1/2同几何normal/zero/wrong干预；不使用900条结果选模。完整optimizer/scheduler/RNG/游标/退出计数和last保存在外部。\n\n'
    if rows:
        last=rows[-1];report+=f"当前完整{last['epoch']}轮／{last['updates']}次更新，末轮train/Val NMSE={last['train']['mean']['nmse']:.6g}/{last['val']['mean']['nmse']:.6g}。\n\n![正式曲线](artifacts/ray_senseiver_supervised_v1/formal_curves.png)\n\n"
    if (B/'healths.json.gz').exists():
        healths=read(B/'healths.json.gz')
        report+='每20轮固定训练三例的同几何干预（不用于选择best；最低依赖要求至少2例≥.001）。\n\n|epoch|case|normal NMSE|zero relative MA|absolute MA|wrong-normal NMSE|\n|---:|---:|---:|---:|---:|---:|\n'
        for h in healths:
            if h['epoch']==0:continue
            for d in h['interventions']:report+=f'|{h["epoch"]}|{d["index"]}|{d["normal_nmse"]:.6g}|{d["relative_MA"]:.6g}|{d["absolute_MA"]:.6g}|{d["wrong_minus_normal_nmse"]:.6g}|\n'
        report+='\n'
    if complete:report+=f"实际停止{complete['reason']}，{complete['epochs']}轮/{complete['updates']}更新/{complete['elapsed_seconds']:.2f}秒。best epoch={complete['best_epoch']}，SHA256 `{complete['best_sha256']}`，路径 `{complete['best_checkpoint']}`。\n\n"
    if frozen:
        report+=f"冻结门槛effective_learning={frozen['effective_learning']}，Val selector重放NMSE={frozen['selector_replay']['mean']['nmse']:.8f}；三组梯度和真实更新有效={frozen['valid_gradient_updates']}。\n\n"
        report+='|固定训练case|normal-zero absolute MA|relative MA|normal NMSE|wrong NMSE|\n|---:|---:|---:|---:|---:|\n'
        for d in frozen['health']['interventions']:report+=f"|{d['index']}|{d['absolute_MA']:.6g}|{d['relative_MA']:.6g}|{d['normal_nmse']:.6g}|{d['wrong_measurement_nmse']:.6g}|\n"
    report+='\n## 冻结比较、访问与局限\n'+summary
    report+='\n|方法|训练GT|选模GT／selector|训练目标|训练输入|预算与停止|\n|---|---|---|---|---|---|\n|监督Ray-Senseiver|166 train GT64|36 Val GT64／平均逐case NMSE|归一化voxel MSE|Pool A自然3–6站、epoch变化|本轮实际预算见上，预登记早停／300轮／2h|\n|冻结监督MRGR|166 train GT64|36 Val GT64／C1监督loss|SmoothL1(beta1)+.1relative L1|Pool A自然3–6站、epoch变化|300轮／24900更新，预算结束；best295|\n|冻结自监督Ours|无训练GT|无选模GT／Val held-out relative L1|原hold/cross/obs/consistency|A3–5/B3–5/H2，H不输入|300轮／24900更新，预算结束|\n|冻结自监督DeepSets|无训练GT|无选模GT／Val held-out relative L1|同冻结自监督Ours|A3–5/B3–5/H2，H不输入|300轮／24900更新，预算结束；best292|\n\n监督Senseiver和监督MRGR使用训练GT及Val GT选模；Ours/DeepSets为冻结自监督参考，不能混为同标签排名。本轮900预测全部先落盘并hash封存，再独立GT64步骤沿用原VolumeMetrics及P_diff/K96观测指标；五档×五layout×36Val不变，case内先平均5draw，36case配对bootstrap10000次/seed2026101801、沿用tie epsilon=1e-8。900条仍来自同一官方Val split，不是额外test集。新训练不读取test、Pool B、GT128，不改划分及冻结历史结果。GT128来源hash仅引用元数据。已知GT64与历史GT128投影离散化差异保留，不新增缩放、插值或重投影标签。\n\n'
    report+='监督相关13项检查、原Ray-Senseiver4项、数据12项、采样3项、独立评价边界2项，共34项适用检查通过；原始测试日志保留。历史FP32 consistency回归仍有1项失败，本轮未修改其函数/容差，不能称历史全部测试通过。共享GPU且仅单seed，计时不是独占设备基准；监督可学不证明原纯测量loss失败的唯一根因，也不代表原论文Senseiver实验的完全复现。\n\n'
    report+='下一步：'+('冻结本轮有监督候选和独立比较结果，研究是否需要更广的监督泛化验证；本轮不追加训练、结构或loss搜索。' if evaldone else '按当前已登记配置完成正式训练及条件冻结评价；未完成前保留运行状态，不据小样本拟合宣称成功独立基线。')+'\n'
    (REPO/'docs/reports/ray_senseiver_supervised_v1_zh.md').write_text(report)
if __name__=='__main__':main()
