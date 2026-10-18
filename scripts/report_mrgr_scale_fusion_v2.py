#!/usr/bin/env python3
"""Scalar-only integrity checks and final scale-fusion screening report."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARMS=('B_multiscale','C_content','E_global')
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def paired(a,b):
    d=b-a;rng=np.random.default_rng(20261017)
    return dict(mean_ratio=float(b.mean()/a.mean()),p90_ratio=float(np.quantile(b,.9)/np.quantile(a,.9)),
      mean_delta=float(d.mean()),paired_ci95=np.quantile(d[rng.integers(0,36,size=(20000,36))].mean(1),[.025,.975]).tolist(),
      wins=int((d < -1e-8).sum()),losses=int((d > 1e-8).sum()),ties=int((abs(d)<=1e-8).sum()))
def verify(out):
    p=read(out/'protocol.json');r=read(out/'comparison.json');smoke=read(out/'smoke.json')
    assert digest(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()==r['protocol_sha256']
    for name,h in p['sources'].items():assert digest(ROOT/name)==h
    assert smoke['passed'] and smoke['initial_B_C_E_bitwise_equal']
    ordinary={};hard={};logs={};times={}
    for a in ARMS:
        assert digest(out/a/'effective_config.json')==p['effective_config_sha256'][a]
        log=logs[a]=lines(out/a/'metrics.jsonl');cases=lines(out/a/'epoch20_per_case.jsonl');cluster=lines(out/a/'cluster5_per_case.jsonl')
        s=read(out/a/'summary.json');h=read(out/a/'cluster5_summary.json')
        assert len(log)==20 and [x['epoch'] for x in log]==list(range(1,21))
        assert s['completed'] and s['epochs']==20 and s['updates']==1660 and s['output_sanity_passed']
        assert digest(out/'checkpoints'/a/'last.pt')==s['checkpoint_sha256']
        for audit in (s['gt_audit'],h['gt_audit']):
            assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        assert len(cases)==36 and [x['case_id'] for x in cases]==p['validation_case_ids']
        assert len(cluster)==180 and len({(x['case_id'],x['draw']) for x in cluster})==180
        for x in cases:
            assert x['stations']==p['station_map'][x['case_id']]
            for sanity in x['output_sanity'].values():assert sanity['std']>1e-8 and sanity['min']>=0 and sanity['max']<=1
        for row in cluster:
            setting=next(x for x in p['cluster_settings'] if x['draw']==row['draw'])
            assert row['input_ids']==setting['input_ids'][row['case_id']] and row['heldout_ids']==setting['heldout_ids'][row['case_id']]
        ordinary[a]=np.array([x['heldout_relative_l1'] for x in cases])
        hard[a]=np.array([np.mean([x['relative_l1'] for x in cluster if x['case_id']==case]) for case in p['validation_case_ids']])
        times[a]=np.mean([x['seconds'] for x in log[2:]])
        for mode,values in (('ordinary',ordinary[a]),('cluster5',hard[a])):
            assert np.isclose(values.mean(),r[mode][a]['mean']) and np.isclose(np.quantile(values,.9),r[mode][a]['p90'])
    for a in ARMS:
        for x,y in zip(logs['B_multiscale'],logs[a]):
            assert x['learning_rate']==y['learning_rate'] and x['loader_rng_sha256']==y['loader_rng_sha256']
            for split in ('train','val'):
                assert x[split]['case_station_digest']==y[split]['case_station_digest']
                assert x[split]['sample_count']==(166 if split=='train' else 36)
    for i,c in enumerate(ARMS):
        for t in ARMS[i+1:]:
            x=r['contrasts'][f'{t}_vs_{c}']
            for mode,values in (('ordinary',ordinary),('cluster5',hard)):
                recalculated=paired(values[c],values[t])
                for k,v in recalculated.items():assert np.allclose(v,x[mode][k])
            assert np.isclose(x['runtime_ratio'],times[t]/times[c])
    thresholds=p['promotion_gate'];gates={}
    for a in ('C_content','E_global'):
        x=r['contrasts'][f'{a}_vs_B_multiscale']
        g=dict(ordinary_mean=x['ordinary']['mean_ratio']<=thresholds['ordinary_mean_ratio_max'],ordinary_p90=x['ordinary']['p90_ratio']<=thresholds['ordinary_p90_ratio_max'],
           cluster5_mean=x['cluster5']['mean_ratio']<=thresholds['cluster5_mean_ratio_max'],runtime=x['runtime_ratio']<=thresholds['runtime_ratio_max'],
           five_hour=times[a]*300/3600<=thresholds['projected300_hours_max'],correctness=True,output_sanity=True)
        assert g==r['gates'][a]['checks'] and bool(all(g.values()))==r['gates'][a]['gate_passed'];gates[a]=bool(all(g.values()))
    verification=dict(passed=True,seed=p['seed'],sources=len(p['sources']),epochs=60,updates=4980,ordinary_rows=108,cluster_rows=540,
       matched_stations_loader_lr=True,no_gt=True,no_final_test=True,recomputed_case_aggregates_CIs_gates=True,gates=gates,
       protocol_sha256=digest(out/'protocol.json'))
    (out/'scalar_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return p,r,logs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',default=[20261018]);args=ap.parse_args()
    all_results=[]
    for seed in args.seeds:
        out=ROOT/f'outputs/research/mrgr_scale_fusion_v2_seed{seed}'
        p,r,logs=verify(out);all_results.append((out,p,r))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for a in ARMS:
            axes[0].plot(range(1,21),[x['train']['loss']['total'] for x in logs[a]],label=a)
            axes[1].plot(range(1,21),[100*x['val']['measurement']['heldout']['relative_l1'] for x in logs[a]],label=a)
        for ax,title in zip(axes,('Training total loss','Heldout projection relative L1 (%)')):
            ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend();ax.set_yscale('log')
        fig.savefig(out/'curves.png',dpi=160);fig.savefig(out/'curves.svg');plt.close(fig)
    text=['# 尺度融合 v2：全局权重候选与 content 独立重复','',
       '本实验检验多尺度分支是否需要逐体素的内容门控，增加一个仅学习全局尺度权重的更简单候选。全部结果为投影指标；未读取三维 GT，也未使用最终 35 例测试。正式长训由主 agent 统一决定。','',
       '每种子三个模型各独立新训 20 轮：166 训练 / 36 验证，batch2，每轮 83 更新。固定末轮，不按最优 epoch 选结果。沿用 300 轮 cosine 前缀、AdamW2e-4/wd1e-4、Q96、FP32、原自监督目标和归一化。每种子各模型初态共享参数逐位相同、完整初始输出逐位相同，逐轮案例站位摘要、loader RNG 与 LR 一致；GPU3 上顺序运行，未混用其他卡。','',
       '| 模型 | 参数 | 新增融合参数 | 设计 |','|---|---:|---:|---|',
       '| B_multiscale | 344678 | 0 | 冻结多尺度基线 |','| C_content | 345417 | 739 | 内容空间 softmax gate，独立新种子复核 |','| E_global | 345097 | 419 | 3 个全局 softmax logits + 零初始化 12→32 输出层 |','',
       'C/E 使用相同的 reduce(x32)、local16、context8 对齐特征、相同可见性 mask 和共享 C0/C1 解码器。E 的权重不依赖案例或空间位置，C/E 新增融合分支均不输入 O 数值，只读取 f>0 作为有效性；原 OARI 保持使用几何描述符。无效位置使用均匀权重且输出零残差。新增初始化使用 seed+2 并恢复 CPU RNG；base 与 context 使用 seed 与 seed+1。32³ reduce 仍增加真实计算。','',
       'CPU 单元检查覆盖 mask、无 O 数值依赖和有效学习；GPU smoke 覆盖 B1/B2 × 正常/部分 padding/全 padding/无有效射线，每模型 descriptor 捕获一次，C0/C1 解码两次，缓存均清理。零输出初始化使 gate 首次梯度为零，第二次反传已非零；两步临时更新后 global alpha 离开均匀、分支残差非零，临时模型均丢弃。','',
       '预登记门槛：候选相对本种子 B 普通均值至少改善 1%，普通 P90 与聚簇五布局均值恶化不超过 2%，完整轮耗时增加不超过 20%，300 轮线性外推不超过 5 小时，正确性和输出非坍缩通过。通过一次仅进入第二种子复核，不自动开始 300 轮。','']
    for out,p,r in all_results:
        seed=p['seed'];rel='../../outputs/research/'+out.name
        text += [f'## 种子 {seed}','', '| 模型 | 普通相对 L1 | 普通 P90 | 聚簇 5 布局相对 L1 | 热态完整轮 | 300 轮外推 |','|---|---:|---:|---:|---:|---:|']
        for a in ARMS:text.append(f'| {a} | {100*r["ordinary"][a]["mean"]:.4f}% | {100*r["ordinary"][a]["p90"]:.4f}% | {100*r["cluster5"][a]["mean"]:.4f}% | {r["warm_epoch_seconds"][a]:.2f}s | {r["projected300_hours"][a]:.3f}h |')
        text += ['', '| 对照 | 普通变化 | 普通差 95% CI（百分点） | 普通胜/负/平 | P90 变化 | 聚簇变化 |','|---|---:|---|---:|---:|---:|']
        for k,x in r['contrasts'].items():
            o=x['ordinary'];ci=[100*v for v in o['paired_ci95']]
            text.append(f'| {k} | {100*(o["mean_ratio"]-1):+.3f}% | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {o["wins"]}/{o["losses"]}/{o["ties"]} | {100*(o["p90_ratio"]-1):+.3f}% | {100*(x["cluster5"]["mean_ratio"]-1):+.3f}% |')
        for a in ('C_content','E_global'):
            gs=[x['gate'][k] for x in lines(out/a/'epoch20_per_case.jsonl') for k in ('A','B')]
            alpha=np.mean([x['alpha_mean'] for x in gs],axis=0)
            spatial=np.mean([x['alpha_spatial_std'] for x in gs],axis=0)
            text += ['',f'{a}：32/16/8 尺度权重均值 `{alpha.tolist()}`，空间标准差均值 `{spatial.tolist()}`，分支残差范数均值 {np.mean([x["delta_norm"] for x in gs]):.6g}。']
        text += ['',f'门槛结果：C_content **{"通过" if r["gates"]["C_content"]["gate_passed"] else "未通过"}**；E_global **{"通过" if r["gates"]["E_global"]["gate_passed"] else "未通过"}**。', '',
         f'![训练和留出投影曲线]({rel}/curves.png)','',f'[协议]({rel}/protocol.json) · [逐对照与门槛]({rel}/comparison.json) · [独立标量复算]({rel}/scalar_verification.json) · [smoke]({rel}/smoke.json)','',f'协议 SHA256：`{r["protocol_sha256"]}`。','']
    passed={a:all(r['gates'][a]['gate_passed'] for _,_,r in all_results) for a in ('C_content','E_global')}
    if len(all_results)==1 and not any(passed.values()):
        text += ['## 路径决定','', '本种子两个候选均未达到登记门槛，不开展本路径的第二种子训练或300轮正式训练。20261019协议保留为提前登记、未执行状态。既往seed20261011的content小幅收益未在本次复核得到支持；不能根据旧单次结果直接晋级。GPU3释放给主agent的其他已授权工作。','']
    else:
        text += ['## 路径决定','',f'所有已执行种子共同通过的候选：{[a for a,v in passed.items() if v]}。单种子通过仅取得复核资格，最终正式晋级由主agent结合其余路线统一决定。','']
    text += ['## 解释边界','',
       '普通指标先按案例平均两个输入子集的 H 误差；困难指标先将 5 个聚簇布局在案例内平均，再对 36 案例等权。CI 为 36 案例成对 bootstrap 20000 次，不包含训练种子方差。全部对照列出，未搜索末轮窗口。普通验证站位随种子改变，每个种子内部严格匹配；聚簇面板跨种子固定。跨种子仅比较对照的相对改变量，不直接比较原始均值。小幅改善可能受 GPU 非确定性轨迹和验证集复用影响。投影误差改善不能替代三维浓度场精度，需要通过复核的候选在正式长训结束后再做冻结的三维测试。','']
    (ROOT/'docs/reports/mrgr_scale_fusion_v2_zh.md').write_text('\n'.join(text))
    print(json.dumps(dict(reports_written=True,seeds=args.seeds)))
if __name__=='__main__':main()
