#!/usr/bin/env python3
"""Registered single-factor parameter screens; no GT or final-test access."""
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
import probe_mrgr_multiscale_v1 as base
import train_mrgr_multiscale_formal_v1 as formal
import probe_mrgr_ocmsf_screen_v1 as evaluation
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.training.utils import move_batch
ROOT=base.ROOT
OUT=ROOT/'outputs/research/mrgr_parameter_optimization_v2'
ARMS=('control','lr4e4','consistency001')
SETTINGS={'control':dict(lr=.0002,consistency=.05),'lr4e4':dict(lr=.0004,consistency=.05),'consistency001':dict(lr=.0002,consistency=.01)}
ORIGINAL=base.selfsup.objective
ORIGINAL_EPOCH=base.selfsup.run_epoch
write=formal.write
read=evaluation.read
PANEL=formal.DEFAULT_OUT/'evaluation/frozen_model_test_v1/registration.json'

def sources():
    r=formal.sources()
    for name in ('scripts/probe_mrgr_ocmsf_screen_v1.py','scripts/mrgr_ocmsf_v1.py',str(Path(__file__).relative_to(ROOT))):r[name]=base.digest(ROOT/name)
    return r

@contextlib.contextmanager
def objective(weight):
    records=[]
    def wrapped(model,batch):
        total,parts,volumes,projections=ORIGINAL(model,batch)
        records.append((float(total.detach()),float(parts['consistency'].detach()),len(volumes[0])))
        if weight!=.05:total=total+(weight-.05)*parts['consistency']
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite adjusted objective')
        return total,parts,volumes,projections
    def epoch(*args,**kwargs):
        records.clear();r=ORIGINAL_EPOCH(*args,**kwargs);n=sum(x[2] for x in records)
        r['loss']['standard_total']=sum(x[0]*x[2] for x in records)/n
        r['loss']['weighted_consistency_actual']=weight*sum(x[1]*x[2] for x in records)/n
        return r
    assert base.selfsup.objective is ORIGINAL and base.selfsup.run_epoch is ORIGINAL_EPOCH
    base.selfsup.objective=wrapped;base.selfsup.run_epoch=epoch
    try:yield
    finally:base.selfsup.objective=ORIGINAL;base.selfsup.run_epoch=ORIGINAL_EPOCH

def config_for(p,arm):
    c=copy.deepcopy(p['config']);c['optimizer']['lr']=SETTINGS[arm]['lr'];c['loss']['consistency']=SETTINGS[arm]['consistency']
    c['experiment']=dict(name=f"mrgr_parameter_optimization_v2_s{p['seed']}_{arm}",output_root=str(OUT/f"seed{p['seed']}"/arm))
    return c

def verify(out):
    p=read(out/'protocol.json')
    assert base.digest(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    assert sources()==p['sources']
    formal.verify(formal.DEFAULT_OUT);formal.verify_data(p)
    assert base.digest(PANEL)==p['panel_sha256']
    for a in p['arms']:assert base.digest(out/a/'effective_config.json')==p['config_sha256'][a]
    return p

def register(seed,arms):
    out=OUT/f'seed{seed}';assert not out.exists(),'Refuse protocol overwrite'
    old=formal.verify(formal.DEFAULT_OUT);formal.verify_data(old)
    c=copy.deepcopy(old['config']);c['training']['device']='cuda:0';c['training']['seed']=seed
    c['experiment']=dict(name=out.name,output_root=str(out));out.mkdir(parents=True)
    with base.GTReadGuard(c['dataset']['root'],out) as guard:
        model=base.model_for(c,'multiscale','cpu');initial=base.state_hash(model.state_dict())
        train=base.selfsup.make_dataset(c);val=base.selfsup.make_dataset(c,validation=True)
        assert (len(train),len(val))==(166,36)
        p=dict(registered_at=base.now(),config=c,seed=seed,arms=list(arms),sources=sources(),data_files=old['data_files'],
            initial_sha256=initial,parameters=sum(x.numel() for x in model.parameters()),epochs=20,updates=1660,
            settings={a:SETTINGS[a] for a in arms},train_case_ids=[r['case_id'] for r in train.records],validation_case_ids=[r['case_id'] for r in val.records],
            station_map={r['case_id']:{k:val[i][k]['input_station_indices'].tolist() for k in ('A','B','H')} for i,r in enumerate(val.records)},
            cluster_settings=[s for s in read(PANEL)['settings'] if s['condition']=='clustered-V4'],panel_sha256=base.digest(PANEL),
            mechanism=dict(lr4e4='One-factor initial AdamW LR doubled to test optimization-limited short prefix. Same cosineTmax300; faster early convergence does not imply final accuracy.',
                consistency001='One-factor cross-input normalized 3D L1 consistency reduced .05 to .01. Tests competition with measurement fit and suppression of input-dependent local detail. Does not impose PDE/GT.',
                control='Unmodified multiscale MRGR, same GPU and same-seed initialization/sample/shuffle.'),
            endpoint='Fixed last epoch20, not best; second fresh seed20261019 matched repeat for all passing arms; no automatic300',
            precision='FP32 TF32off cudnnbenchmarkoff',schedule='First20 of cosineTmax300, originalbatch2 wd1e-4 clip1',
            gate=dict(ordinary_mean_ratio_max=.99,p90_ratio_max=1.02,cluster5_ratio_max=1.02,runtime_ratio_max=1.2,hours300_max=5.),
            no_gt=True,no_final_test=True,second_seed_rule='If any arm passes all gates against same-seed control, register seed20261019 control plus all passing arms before their results.',
            gpu_environment=dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),status=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv'],text=True)),
            limitations='Single factor short screen cannot establish final3D. Numeric training drift exists; paired-case bootstrap excludes training repeat uncertainty.',registration_audit=guard.summary())
        assert p['parameters']==344678
        p['config_sha256']={}
        for a in arms:
            (out/a/'checkpoints').mkdir(parents=True);write(out/a/'effective_config.json',config_for(p,a));p['config_sha256'][a]=base.digest(out/a/'effective_config.json')
        write(out/'protocol.json',p);(out/'protocol.sha256').write_text(base.digest(out/'protocol.json')+'\n')
    print(json.dumps(dict(stage='registered',seed=seed,arms=arms)),flush=True)
    return out

def gradient_probe(model,batch,weight):
    model.train();model.zero_grad(set_to_none=True)
    total,parts,volumes,projections=ORIGINAL(model,batch)
    pars=tuple(model.parameters())
    gt=torch.autograd.grad(total,pars,retain_graph=True,allow_unused=True)
    gcns=torch.autograd.grad(.05*parts['consistency'],pars,retain_graph=True,allow_unused=True)
    dot=sum((g*h).sum() for g,h in zip(gt,gcns) if g is not None and h is not None)
    norm=lambda gs:torch.sqrt(sum(g.square().sum() for g in gs if g is not None))
    gn,cn=norm(gt),norm(gcns)
    result=dict(original_total=float(total.detach()),parts={k:float(v.detach()) for k,v in parts.items()},
        adjusted_total=float((total+(weight-.05)*parts['consistency']).detach()),
        weighted_consistency_original=float((.05*parts['consistency']).detach()),weighted_consistency_candidate=float((weight*parts['consistency']).detach()),
        all_parameter_total_gradient_norm=float(gn),all_parameter_original_consistency_gradient_norm=float(cn),
        original_consistency_to_total_gradient_ratio=float(cn/gn.clamp_min(1e-12)),cosine_consistency_with_total=float(dot/(cn*gn).clamp_min(1e-12)),
        actual_weight=weight)
    del total,parts,volumes,projections,gt,gcns
    return result

def smoke(out):
    p=verify(out);c=p['config'];device=c['training']['device']
    with base.GTReadGuard(c['dataset']['root'],out) as guard:
        ds=base.selfsup.make_dataset(c);b=base.selfsup.move_dual_batch(base.dual_measurement_collate([ds[0],ds[1]]),device)
        model=base.model_for(c,'multiscale',device);before=base.state_hash(model.state_dict());model.eval()
        with torch.no_grad():
            original=ORIGINAL(model,b)
            with objective(.05):wrapped=base.selfsup.objective(model,b)
            assert torch.equal(original[0],wrapped[0]);assert all(torch.equal(x,y) for x,y in zip(original[2],wrapped[2]))
            del original,wrapped
        probes={};times={}
        for arm in p['arms']:
            setting=SETTINGS[arm];probes[arm]=gradient_probe(model,b,setting['consistency'])
            values=[]
            with objective(setting['consistency']):
                for _ in range(2):
                    model.zero_grad(set_to_none=True);torch.cuda.synchronize(device);start=time.perf_counter()
                    total,parts,volumes,projections=base.selfsup.objective(model,b);total.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    torch.cuda.synchronize(device);values.append(time.perf_counter()-start)
                    del total,parts,volumes,projections
            times[arm]=dict(seconds=values,gradient_norm=float(norm))
        assert base.state_hash(model.state_dict())==before
        assert base.selfsup.objective is ORIGINAL
        write(out/'smoke.json',dict(passed=True,control_objective_bitwise_equal=True,weights_unchanged=True,global_binding_restored=True,gradient_probes=probes,timing=times,gt_audit=guard.summary()))
    del model,b;gc.collect();torch.cuda.empty_cache();print('Smoke passed',flush=True)

def train(out,arm):
    p=verify(out);assert read(out/'smoke.json')['passed'];c=config_for(p,arm);run=out/arm;device=c['training']['device']
    assert not (run/'metrics.jsonl').exists(),'Refuse silent restart'
    with base.GTReadGuard(c['dataset']['root'],out) as guard:
        model=base.model_for(c,'multiscale',device);assert base.state_hash(model.state_dict())==p['initial_sha256']
        opt=torch.optim.AdamW(model.parameters(),lr=SETTINGS[arm]['lr'],weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
        generator=torch.Generator().manual_seed(p['seed']);ds=base.selfsup.make_dataset(c);vd=base.selfsup.make_dataset(c,validation=True)
        loader=base.selfsup.make_loader(ds,c,True,generator);vl=base.selfsup.make_loader(vd,c)
        write(run/'environment.json',dict(base.environment(c),pid=os.getpid(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        rows=[];began=time.perf_counter()
        with objective(SETTINGS[arm]['consistency']):
            for epoch in range(20):
                ds.set_epoch(epoch);vd.set_epoch(0);torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);start=time.perf_counter();lr=opt.param_groups[0]['lr']
                tr=base.selfsup.run_epoch(model,loader,c,opt);va=base.selfsup.run_epoch(model,vl,c);torch.cuda.synchronize(device);sch.step()
                assert (tr['sample_count'],va['sample_count'])==(166,36)
                row=dict(epoch=epoch+1,train=tr,val=va,learning_rate=lr,seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),loader_rng_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
                rows.append(row);base.append(run/'metrics.jsonl',row)
                base.atomic_checkpoint(run/'checkpoints/last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=base.rng_state(generator),epoch=epoch+1,updates=(epoch+1)*83,config=c,arm=arm,protocol_sha256=base.digest(out/'protocol.json'),volumetric_gt_used=False))
                write(run/'gt_access_audit.json',guard.summary());write(run/'status.json',dict(state='training',epoch=epoch+1,seed=p['seed'],val_H=base.selfsup.selection_metric(va),updated_at=base.now()))
                print(f"seed{p['seed']} {arm} epoch={epoch+1}/20 valH={base.selfsup.selection_metric(va):.8f} seconds={row['seconds']:.2f}",flush=True)
        cases=evaluation.ordinary_cases(model,vl,c)
        assert [r['case_id'] for r in cases]==p['validation_case_ids']
        for r in cases:assert r['stations']==p['station_map'][r['case_id']]
        (run/'epoch20_per_case.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases))
        ds.set_epoch(0);batch=base.selfsup.move_dual_batch(base.dual_measurement_collate([ds[0],ds[1]]),device)
        probe=gradient_probe(model,batch,SETTINGS[arm]['consistency']);write(run/'late_gradient_probe.json',probe)
        assert base.digest(run/'checkpoints/last.pt')
        result=dict(completed=True,epochs=20,updates=1660,warm_epoch_seconds=float(np.mean([r['seconds'] for r in rows[2:]])),elapsed_seconds=time.perf_counter()-began,
            checkpoint_sha256=base.digest(run/'checkpoints/last.pt'),gt_audit=guard.summary(),output_sanity_passed=True,finished_at=base.now())
        write(run/'summary.json',result);write(run/'status.json',dict(state='complete',**result));formal.plot(run,rows)
    del model,opt,loader,vl,batch;gc.collect();torch.cuda.empty_cache()
    verify(out)

def panel(out,arm):
    p=verify(out);c=config_for(p,arm);device=c['training']['device'];run=out/arm
    assert not (run/'cluster5_per_case.jsonl').exists();assert base.digest(run/'checkpoints/last.pt')==read(run/'summary.json')['checkpoint_sha256']
    records=[]
    with base.GTReadGuard(c['dataset']['root'],out) as guard:
        ckpt=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False);assert (ckpt['epoch'],ckpt['updates'])==(20,1660)
        model=base.model_for(c,'multiscale',device);model.load_state_dict(ckpt['model']);model.eval()
        with torch.no_grad():
            for setting in p['cluster_settings']:
                ds=MeasurementOnlyPoolDataset(c['dataset']['root'],split='val',geometry_pool='train',evaluation_only=True,min_input_views=4,max_input_views=4,hold_views=4,base_seed=p['seed'],normalization='linear_p99_5',statistics_file='train_statistics.json');ds.sampler=evaluation.Replay(setting)
                for i in range(0,len(ds),2):
                    samples=[ds[j] for j in range(i,min(i+2,len(ds)))];b=move_batch(projection_pool_collate(samples),device);inputs={k:v for k,v in b.items() if not k.startswith('hold_')};h=evaluation.heldout(b)
                    field=model(inputs);prediction=base.project_input(field,h);valid=h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]
                    assert torch.isfinite(field).all() and torch.isfinite(prediction).all()
                    for j,case in enumerate(inputs['case_id']):
                        assert samples[j]['input_station_indices'].tolist()==setting['input_ids'][case]
                        records.append(dict(case_id=case,draw=setting['draw'],input_ids=setting['input_ids'][case],heldout_ids=setting['heldout_ids'][case],**base.relative_projection_metrics(prediction[j],h['input_projection_raw'][j],valid[j])))
                print(f'{arm} cluster{setting["draw"]} complete',flush=True)
        assert len(records)==180
        (run/'cluster5_per_case.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));write(run/'cluster5_audit.json',guard.summary())
    del model,ckpt;gc.collect();torch.cuda.empty_cache()

def compare(out):
    p=verify(out);arms=p['arms'];logs={a:[json.loads(x) for x in (out/a/'metrics.jsonl').read_text().splitlines()] for a in arms};ordinary={};hard={};times={}
    for a in arms:
        assert len(logs[a])==20
        for x,y in zip(logs['control'],logs[a]):
            assert x['epoch']==y['epoch'] and x['loader_rng_sha256']==y['loader_rng_sha256']
            assert math.isclose(y['learning_rate']/x['learning_rate'],SETTINGS[a]['lr']/.0002,rel_tol=1e-12)
            for split in ('train','val'):assert x[split]['case_station_digest']==y[split]['case_station_digest']
        cases=[json.loads(x) for x in (out/a/'epoch20_per_case.jsonl').read_text().splitlines()];rows=[json.loads(x) for x in (out/a/'cluster5_per_case.jsonl').read_text().splitlines()]
        assert [r['case_id'] for r in cases]==p['validation_case_ids'];assert len({(r['case_id'],r['draw']) for r in rows})==180
        ordinary[a]=np.array([r['heldout_relative_l1'] for r in cases]);hard[a]=np.array([np.mean([r['relative_l1'] for r in rows if r['case_id']==case]) for case in p['validation_case_ids']]);times[a]=float(np.mean([x['seconds'] for x in logs[a][2:]]))
        for name in ('gt_access_audit.json','cluster5_audit.json'):
            audit=read(out/a/name);assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    contrasts={}
    for a in arms[1:]:
        o=evaluation.paired(ordinary['control'],ordinary[a]);h=evaluation.paired(hard['control'],hard[a]);t=p['gate']
        checks=dict(ordinary=o['mean_ratio']<=t['ordinary_mean_ratio_max'],p90=o['p90_ratio']<=t['p90_ratio_max'],cluster=h['mean_ratio']<=t['cluster5_ratio_max'],runtime=times[a]/times['control']<=t['runtime_ratio_max'],five_hour=times[a]*300/3600<=t['hours300_max'],smoke=read(out/'smoke.json')['passed'])
        contrasts[a]=dict(ordinary=o,cluster5=h,checks=checks,gate_passed=all(checks.values()),runtime_ratio=times[a]/times['control'],projected300_hours=times[a]*300/3600)
    result=dict(completed=True,seed=p['seed'],contrasts=contrasts,ordinary={a:evaluation.describe(x) for a,x in ordinary.items()},cluster5={a:evaluation.describe(x) for a,x in hard.items()},warm_epoch_seconds=times,
        no_gt=True,no_final_test=True,no_formal300=True,matched_sampling_shuffle=True,source_verified=True,protocol_sha256=base.digest(out/'protocol.json'),finished_at=base.now())
    write(out/'comparison.json',result);print(json.dumps(result),flush=True);return result

def report():
    results=[read(path) for path in sorted(OUT.glob('seed*/comparison.json'))]
    text='# 多尺度 MRGR 参数优化 v2\n\n预注册两个独立候选：学习率 2e-4→4e-4；consistency 权重 0.05→0.01。每种子新控制及候选均为同一 GPU2、相同初始权重、站位/打乱、166训练/36验证、固定20轮/1,660更新、300轮余弦前缀。每候选只改变一个变量。\n\n'
    text+='原 consistency 是 A/B 三维场归一化 L1，可能与细节/测量拟合竞争；降低权重测试该竞争。双倍学习率测试短前缀是否受优化速度限制，但早期优势不代表300轮最终精度。没有PDE或三维GT监督。\n\n'
    for r in results:
        text+=f"## 种子 {r['seed']}\n\n| 模型 | 普通 H相对L1 | P90 | 聚簇5均值 | 每轮s |\n|---|---:|---:|---:|---:|\n"
        for a,x in r['ordinary'].items():text+=f"| {a} | {x['mean']*100:.4f}% | {x['p90']*100:.4f}% | {r['cluster5'][a]['mean']*100:.4f}% | {r['warm_epoch_seconds'][a]:.2f} |\n"
        text+='\n'
        for a,x in r['contrasts'].items():
            text+=f"- {a}: 普通均值变化 {(x['ordinary']['mean_ratio']-1)*100:+.2f}%，P90 {(x['ordinary']['p90_ratio']-1)*100:+.2f}%，聚簇 {(x['cluster5']['mean_ratio']-1)*100:+.2f}%；门槛 {'通过' if x['gate_passed'] else '失败'}；检查 `{json.dumps(x['checks'])}`。配对均值差95%区间 `{x['ordinary']['paired_ci95']}`（比例单位），胜/负 {x['ordinary']['wins']}/{x['ordinary']['losses']}。\n"
        text+=f"\n[协议](../../outputs/research/mrgr_parameter_optimization_v2/seed{r['seed']}/protocol.json)；[结果](../../outputs/research/mrgr_parameter_optimization_v2/seed{r['seed']}/comparison.json)。\n\n"
    text+='## 解释边界\n\n所有数字仅是已反复使用开发验证集的投影误差，不是三维重建误差。没有读取三维GT或最终35例测试。病例bootstrap不包含训练数值轨迹与重复种子不确定性。正式训练由主agent综合决定，本脚本不启动300轮。真实分量loss和最大梯度范数逐轮记录；固定训练前2例的初始/末期全参数 consistency 梯度贡献记录在 smoke 和 late_gradient_probe。运行时含训练和验证，不含每轮checkpoint/曲线导出，外推可能受GPU负载影响。\n'
    (ROOT/'docs/reports/mrgr_parameter_optimization_v2_zh.md').write_text(text)

def pipeline():
    first=OUT/'seed20261018'
    if not first.exists():register(20261018,ARMS)
    run_seed(first)
    result=read(first/'comparison.json');passed=[a for a,x in result['contrasts'].items() if x['gate_passed']]
    if passed:
        second=OUT/'seed20261019'
        if not second.exists():register(20261019,('control',*passed))
        run_seed(second)
    report();write(OUT/'status.json',dict(state='complete',second_seed_candidates=passed,finished_at=base.now()))

def run_seed(out):
    p=verify(out)
    if not (out/'smoke.json').exists():smoke(out)
    for arm in p['arms']:
        if not (out/arm/'summary.json').exists():train(out,arm)
        if not (out/arm/'cluster5_per_case.jsonl').exists():panel(out,arm)
    if not (out/'comparison.json').exists():compare(out)
    report()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','smoke','pipeline','verify','report'));parser.add_argument('--seed',type=int,default=20261018);args=parser.parse_args();torch.set_num_threads(2)
    out=OUT/f'seed{args.seed}'
    if args.action=='register':register(args.seed,ARMS)
    elif args.action=='verify':verify(out);print('Verified sources/config/data')
    elif args.action=='smoke':smoke(out)
    elif args.action=='report':report()
    else:pipeline()
if __name__=='__main__':main()
