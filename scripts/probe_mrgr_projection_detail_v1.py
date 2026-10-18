#!/usr/bin/env python3
"""Registered paired measurement-only 20-epoch projection-gradient screen; never formal train."""
import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
import probe_mrgr_multiscale_v1 as base
import train_mrgr_multiscale_formal_v1 as formal
import run_mrgr_local_detail_v1 as runner

ROOT=base.ROOT
OUT=ROOT/'outputs/research/mrgr_projection_detail_v1'
ARMS=('multiscale','detail')
LAMBDA=0.05
write=formal.write
read=runner.read
ORIGINAL_OBJECTIVE=base.selfsup.objective
ORIGINAL_EPOCH=base.selfsup.run_epoch


def gradient_discrepancy(projected,target):
    """Per-case mean over all valid horizontal+vertical neighbor pairs, normalized by frozen Y_SCALE."""
    valid=target['input_valid_ray'].detach().bool() & target['view_mask'][:,:,None,None,None]
    reference=target['input_projection_raw'].detach()
    # Mask BEFORE differences: even invalid NaN inputs cannot leak through invalid pairs.
    p=torch.where(valid,projected,0)/base.selfsup.Y_SCALE
    y=torch.where(valid,reference,0)/base.selfsup.Y_SCALE
    sums=torch.zeros(p.shape[0],device=p.device,dtype=p.dtype)
    counts=torch.zeros_like(sums)
    for dim in (-2,-1):
        one=[slice(None)]*p.ndim;two=one.copy();one[dim]=slice(1,None);two[dim]=slice(None,-1)
        one,two=tuple(one),tuple(two)
        mask=valid[one]&valid[two]
        difference=(p[one]-p[two])-(y[one]-y[two])
        error=F.smooth_l1_loss(difference,torch.zeros_like(difference),beta=.01,reduction='none')
        sums+=(error*mask).flatten(1).sum(1)
        counts+=mask.flatten(1).sum(1)
    return (sums/counts.clamp_min(1)).mean()


@contextlib.contextmanager
def augmented(weight):
    stats=[]
    def objective(model,batch):
        total,parts,volumes,projections=ORIGINAL_OBJECTIVE(model,batch)
        with torch.set_grad_enabled(torch.is_grad_enabled() and weight!=0):
            gradient=.5*(gradient_discrepancy(projections['C_A']['H'],batch['H'])+
                          gradient_discrepancy(projections['C_B']['H'],batch['H']))
        stats.append((float(total.detach()),float(gradient.detach()),len(volumes[0])))
        result=total if weight==0 else total+weight*gradient
        if not torch.isfinite(result):raise FloatingPointError('Nonfinite augmented loss')
        return result,parts,volumes,projections
    def epoch(*args,**kwargs):
        stats.clear();result=ORIGINAL_EPOCH(*args,**kwargs)
        n=sum(r[2] for r in stats)
        result['loss']['base_total']=sum(r[0]*r[2] for r in stats)/n
        result['loss']['projection_gradient']=sum(r[1]*r[2] for r in stats)/n
        result['loss']['weighted_projection_gradient']=weight*result['loss']['projection_gradient']
        return result
    assert base.selfsup.objective is ORIGINAL_OBJECTIVE and base.selfsup.run_epoch is ORIGINAL_EPOCH
    base.selfsup.objective=objective;base.selfsup.run_epoch=epoch
    try:yield
    finally:
        base.selfsup.objective=ORIGINAL_OBJECTIVE;base.selfsup.run_epoch=ORIGINAL_EPOCH


def sources():
    result=formal.sources()
    for name in ('scripts/run_mrgr_local_detail_v1.py','scripts/mrgr_local_detail_v1.py',str(Path(__file__).relative_to(ROOT))):
        result[name]=base.digest(ROOT/name)
    return result


class ModelProvider:
    source_hashes=staticmethod(sources)
    @staticmethod
    def model_for(config,arm,device):
        assert arm in ARMS
        return base.model_for(config,'multiscale',device)


def verify(out):
    p=read(out/'protocol.json')
    assert base.digest(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    assert sources()==p['sources']
    base.verify();formal.verify_data(p)
    return p


def register(out,device):
    assert not out.exists()
    p=copy.deepcopy(base.verify());config=p['config'];config['training']['device']=device
    config['experiment']=dict(name=out.name,output_root=str(out))
    out.mkdir(parents=True)
    for arm in ARMS:(out/arm/'checkpoints').mkdir(parents=True)
    with base.GTReadGuard(config['dataset']['root'],out) as guard:
        models={a:ModelProvider.model_for(config,a,'cpu') for a in ARMS}
        hashes={a:base.state_hash(m.state_dict()) for a,m in models.items()}
        assert len(set(hashes.values()))==1
        train=base.selfsup.make_dataset(config);val=base.selfsup.make_dataset(config,validation=True)
        assert (len(train),len(val))==(166,36)
        settings_path=ROOT/'outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/registration.json'
        p=dict(config=config,sources=sources(),registered_at=base.now(),parameters=344678,seed=20261011,initial_sha256=hashes,
            data_files=read(ROOT/'outputs/research/optimization_screening_20261016/data_identity_postflight.json')['files'],
            train_case_ids=[x['case_id'] for x in train.records],validation_case_ids=[x['case_id'] for x in val.records],
            epochs=20,updates_per_epoch=83,updates_per_arm=1660,precision='FP32; TF32 off; cudnn benchmark off',
            optimizer='AdamW lr2e-4 wd1e-4 batch2 clip1',schedule='20 epoch prefix of original cosine T_max300',
            single_factor='original total + 0.05 * mean(C_A_H_gradient,C_B_H_gradient)',lambda_gradient=LAMBDA,
            loss_definition='Horizontal and vertical neighboring valid ray pairs, both endpoints valid; differences of raw projections divided by frozen Y_SCALE; SmoothL1 beta0.01; combined pair mean per case then case mean; empty pair case contributes0. No relative-gradient renormalization, edge weighting, GT or extra forward.',
            scale_explanation='Both base smooth projection term and gradient term are dimensionless in frozen Y_SCALE units; finite difference is per detector-pixel index (no physical spacing division). Fixed lambda0.05 is not tuned.',
            ordinary_selection='lowest ordinary heldout projection relative L1 for checkpoint recording; gate uses fixed epoch20 only',
            gt_allowed=False,final_test_allowed=False,automatic_formal_training=False,
            promotion_gate=dict(ordinary_H_min_relative_improvement=.01,ordinary_case_P90_max_ratio=1.02,clustered_five_draw_H_max_ratio=1.02,mean_warm_epoch_runtime_max_ratio=1.20,projected_300_hour_ceiling=5.),
            hard_panel_source=str(settings_path),hard_panel_source_sha256=base.digest(settings_path),
            registration_audit=guard.summary(),gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv'],text=True))
        formal.verify_data(p);write(out/'protocol.json',p)
        (out/'protocol.sha256').write_text(base.digest(out/'protocol.json')+'\n')
    write(out/'status.json',dict(state='registered',updated_at=base.now()))
    print('Registered fixed lambda .05 matching 20-epoch pair',flush=True)


def unit_checks():
    t={'input_projection_raw':torch.zeros(2,1,1,4,4),'input_valid_ray':torch.ones(2,1,1,4,4,dtype=torch.bool),'view_mask':torch.ones(2,1,dtype=torch.bool)}
    x=torch.ones_like(t['input_projection_raw'],requires_grad=True)
    assert float(gradient_discrepancy(x,t))==0 # constant offset has zero gradient discrepancy
    q=x.detach().clone();q[0,0,0,2,2]=4;q.requires_grad_()
    loss=gradient_discrepancy(q,t);assert float(loss)>0;loss.backward();assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0
    t['input_valid_ray'][:]=False
    assert float(gradient_discrepancy(q,t))==0
    t['input_valid_ray'][0,0,0,0,0]=True # isolated valid ray produces no pairs
    assert float(gradient_discrepancy(q,t))==0
    t['input_valid_ray'][:]=True;t['input_valid_ray'][0,0,0,2,2]=False
    q2=torch.ones_like(q);q2[0,0,0,2,2]=float('nan')
    assert float(gradient_discrepancy(q2,t))==0
    return dict(constant=True,finite_backward=True,empty_and_isolated_pairs=True,invalid_endpoint_nan=True)


def smoke(out):
    p=verify(out);config=p['config'];device=config['training']['device'];unit=unit_checks()
    with base.GTReadGuard(config['dataset']['root'],out) as guard:
        ds=base.selfsup.make_dataset(config)
        batch=base.selfsup.move_dual_batch(base.dual_measurement_collate([ds[0],ds[1]]),device)
        model=ModelProvider.model_for(config,'multiscale',device);model.train()
        original_state=base.state_hash(model.state_dict())
        # One shared forward, inspect gradient contribution against original objective.
        total,parts,volumes,projections=ORIGINAL_OBJECTIVE(model,batch)
        grad=.5*(gradient_discrepancy(projections['C_A']['H'],batch['H'])+gradient_discrepancy(projections['C_B']['H'],batch['H']))
        parameter=model.decoder.context.output.weight
        g0=torch.autograd.grad(total,parameter,retain_graph=True)[0]
        gd=torch.autograd.grad(LAMBDA*grad,parameter,retain_graph=True)[0]
        assert torch.isfinite(gd).all() and gd.abs().sum()>0
        (total+LAMBDA*grad).backward();assert torch.allclose(parameter.grad,g0+gd,rtol=1e-4,atol=1e-6)
        record=dict(base_loss=float(total.detach()),gradient_loss=float(grad.detach()),weighted_gradient_loss=float((LAMBDA*grad).detach()),
            base_parameter_grad_norm=float(g0.norm()),added_parameter_grad_norm=float(gd.norm()),added_to_base_grad_norm=float(gd.norm()/g0.norm()))
        del total,parts,volumes,projections,grad,g0,gd;model.zero_grad(set_to_none=True)
        times={}
        for arm,weight in (('multiscale',0.),('detail',LAMBDA)):
            with augmented(weight):
                values=[]
                for i in range(3):
                    torch.cuda.synchronize(device);start=time.perf_counter();loss,parts,volumes,projections=base.selfsup.objective(model,batch)
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    torch.cuda.synchronize(device);values.append(time.perf_counter()-start)
                    model.zero_grad(set_to_none=True);del loss,parts,volumes,projections
                times[arm]=values
        assert base.selfsup.objective is ORIGINAL_OBJECTIVE and base.selfsup.run_epoch is ORIGINAL_EPOCH
        assert base.state_hash(model.state_dict())==original_state
        estimate=sum(np.mean(v[-2:]) for v in times.values())*83*20*1.3/60
        result=dict(passed=True,unit=unit,gradient_contribution=record,backward_seconds=times,estimated_two_arm_minutes=float(estimate),
            global_bindings_restored=True,weights_unchanged=True,gt_audit=guard.summary())
        write(out/'smoke.json',result);print(json.dumps(dict(stage='smoke',**result)),flush=True)
        del model,batch;gc.collect();torch.cuda.empty_cache()
        assert estimate<=45,'Estimated screen exceeds preauthorized 45min; notify parent'


def gate(out):
    p=verify(out);logs={a:[json.loads(x) for x in (out/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    for a,b in zip(logs[ARMS[0]],logs[ARMS[1]]):
        assert a['epoch']==b['epoch'] and a['learning_rate']==b['learning_rate'] and a['loader_rng_sha256']==b['loader_rng_sha256']
        for split in ('train','val'):assert a[split]['case_station_digest']==b[split]['case_station_digest']
    cases={a:[json.loads(x) for x in (out/a/'epoch20_per_case.jsonl').read_text().splitlines()] for a in ARMS}
    for a in ARMS:assert [r['case_id'] for r in cases[a]]==p['validation_case_ids']
    values={a:np.array([r['heldout_relative_l1'] for r in cases[a]]) for a in ARMS}
    mean={a:float(v.mean()) for a,v in values.items()};p90={a:float(np.quantile(v,.9)) for a,v in values.items()}
    times={a:float(np.mean([r['seconds'] for r in logs[a][2:]])) for a in ARMS}
    hard=read(out/'hard_panel.json')['means']
    ratios=dict(ordinary=mean['detail']/mean['multiscale'],p90=p90['detail']/p90['multiscale'],cluster=hard['detail']/hard['multiscale'],runtime=times['detail']/times['multiscale'])
    checks=dict(ordinary_H=ratios['ordinary']<=.99,ordinary_P90=ratios['p90']<=1.02,clustered_H=ratios['cluster']<=1.02,runtime=ratios['runtime']<=1.2,five_hour_budget=times['detail']*300/3600<=5,smoke=read(out/'smoke.json')['passed'])
    delta=values['detail']-values['multiscale'];rng=np.random.default_rng(20261017)
    ci=np.quantile(delta[rng.integers(0,36,(10000,36))].mean(1),[.025,.975]).tolist()
    result=dict(passed=all(checks.values()),checks=checks,ratios=ratios,ordinary_H=mean,ordinary_P90=p90,clustered_H=hard,warm_epoch_seconds=times,
        projected_detail_300_hours=times['detail']*300/3600,paired_mean_delta_ci95=ci,wins=int((delta<0).sum()),losses=int((delta>0).sum()),
        matched_sampling_and_shuffle=True,gt_used=False,final_test_evaluated=False,formal_training_started=False,finished_at=base.now())
    write(out/'gate.json',result);write(out/'status.json',dict(state='screen_complete',**result));print(json.dumps(dict(stage='gate',**result)),flush=True)
    report(out,result)


def report(out,r):
    def row(label,key):
        a,b=(r[key][x]*100 for x in ARMS)
        return f'| {label} | {a:.4f}% | {b:.4f}% | {(b/a-1)*100:+.2f}% |'
    text='# 多尺度 MRGR：投影梯度细节损失匹配筛查\n\n'
    text+='本轮仅改变训练目标：原损失加 `0.05 × 共同 H 投影梯度 SmoothL1`。模型结构、参数量344,678、初始化、166训练/36验证、站位采样、20轮（各1,660次更新）及300轮余弦调度前缀均匹配。H 从未输入重建网络。\n\n'
    text+='梯度项使用水平/垂直相邻射线差，预测和测量均除以原始固定 Y_SCALE，SmoothL1 beta=0.01。仅统计两端均有效的相邻对；每案例先合并两方向所有有效对平均，再跨案例及 A/B 重建平均；无有效对的案例贡献0。不除以像素物理间距，不额外归一化小梯度，不新增前向计算。\n\n'
    text+='| 留出投影相对 L1 | 原多尺度 | 加投影梯度损失 | 相对变化 |\n|---|---:|---:|---:|\n'
    text+='\n'.join((row('普通均值','ordinary_H'),row('普通P90','ordinary_P90'),row('聚簇五布局均值','clustered_H')))+'\n\n'
    text+=f"完整晋级检查：`{json.dumps(r['checks'],ensure_ascii=False)}`；总判定 **{'通过' if r['passed'] else '未通过'}**。\n\n"
    text+=f"预热后每轮：原多尺度 {r['warm_epoch_seconds']['multiscale']:.2f}s，处理组 {r['warm_epoch_seconds']['detail']:.2f}s；300轮外推 {r['projected_detail_300_hours']:.2f}h。该外推取决于当前GPU负载。\n\n"
    text+=f"普通案例配对均值差95% bootstrap区间（处理−对照，相对L1原始比例）：{r['paired_mean_delta_ci95']}，胜/负 {r['wins']}/{r['losses']}。此为单种子短程测量筛查，不代表收敛后的三维精度或统计显著性证明。\n\n"
    text+='本轮未读取三维GT、未评价最终35例测试集、未启动300轮训练。源代码、数据、协议、权重和全轮采样/打乱证据见输出目录；所有旧文件保持原样。\n\n'
    text+='证据：[协议](../../outputs/research/mrgr_projection_detail_v1/protocol.json)、[smoke](../../outputs/research/mrgr_projection_detail_v1/smoke.json)、[结果](../../outputs/research/mrgr_projection_detail_v1/gate.json)。\n'
    dest=ROOT/'docs/reports/mrgr_projection_detail_v1_zh.md';assert not dest.exists();dest.write_text(text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','smoke','pipeline','verify'));parser.add_argument('--device',default='cuda:2');args=parser.parse_args()
    torch.set_num_threads(2)
    runner.verify=verify;runner.candidate=ModelProvider;runner.ARMS=ARMS
    if args.action=='register':register(OUT,args.device);return
    if args.action=='verify':verify(OUT);print('verified');return
    if not OUT.exists():register(OUT,args.device)
    if not (OUT/'smoke.json').exists():smoke(OUT)
    if args.action=='smoke':return
    for arm in ARMS:
        if not (OUT/arm/'summary_epoch20.json').exists():
            with augmented(0. if arm=='multiscale' else LAMBDA):runner.train_arm(OUT,arm,20)
    if not (OUT/'hard_panel.json').exists():runner.hard_panel(OUT)
    gate(OUT)

if __name__=='__main__':main()
