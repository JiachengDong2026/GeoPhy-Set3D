#!/usr/bin/env python3
"""Matched local-detail screen, conditional 300 epochs and frozen validation test."""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import mrgr_local_detail_v1 as candidate
import probe_mrgr_multiscale_v1 as base
import train_mrgr_multiscale_formal_v1 as old_formal

ROOT = base.ROOT
OUT = ROOT / 'outputs/research/mrgr_local_detail_v1_seed20261011'
ARMS = ('multiscale', 'local')
write = old_formal.write
rows_write = old_formal.write_metrics


def read(p): return json.loads(Path(p).read_text())


def verify(out):
    p = read(out / 'protocol.json')
    assert base.digest(out / 'protocol.json') == (out / 'protocol.sha256').read_text().strip()
    assert candidate.source_hashes() == p['sources'], 'Numerical sources changed'
    base.verify()
    old_formal.verify_data(p)
    return p


def register(out, device):
    original = base.verify()
    assert not out.exists(), f'Refuse existing output {out}'
    config = copy.deepcopy(original['config'])
    config['training']['device'] = device
    config['experiment'] = dict(name=out.name, output_root=str(out))
    base.selfsup.validate_selfsup_config(config)
    out.mkdir(parents=True)
    for arm in ARMS:
        (out / arm / 'checkpoints').mkdir(parents=True)
    data = read(ROOT / 'outputs/research/optimization_screening_20261016/data_identity_postflight.json')['files']
    with base.GTReadGuard(config['dataset']['root'], out) as guard:
        models = {a:candidate.model_for(config, a, 'cpu') for a in ARMS}
        ref, local = (models[a].state_dict() for a in ARMS)
        assert all(torch.equal(v, local[k]) for k,v in ref.items())
        assert all(k.startswith('decoder.local.') for k in local.keys()-ref.keys())
        rng = torch.get_rng_state()
        x = torch.randn(1, 32, 32, 32, 32)
        with torch.no_grad():
            assert torch.equal(models['multiscale'].decoder(x), models['local'].decoder(x))
        torch.set_rng_state(rng)
        counts = {a:sum(x.numel() for x in m.parameters()) for a,m in models.items()}
        assert counts['multiscale'] == 344678
        assert counts['local']/counts['multiscale'] < 1.02
        train = base.selfsup.make_dataset(config)
        val = base.selfsup.make_dataset(config, validation=True)
        assert (len(train),len(val)) == (166,36)
        p = dict(registered_at=base.now(), config=config, sources=candidate.source_hashes(), data_files=data,
            parameters=counts['local'], parameter_counts=counts, seed=20261011,
            train_case_ids=[x['case_id'] for x in train.records], validation_case_ids=[x['case_id'] for x in val.records],
            architecture='Frozen multiscale MRGR + parallel width8 depthwise3 local residual at32³ before upsampling; zero output initialization; shared C0/C1 decoder',
            initial_sha256={a:base.state_hash(m.state_dict()) for a,m in models.items()},
            initial_shared_tensors_exact=True, initial_decoder_bitwise_equal=True,
            epochs=300, updates_per_epoch=83, planned_updates=24900, screening_epochs=20,
            schedule='Unchanged cosine T_max300, fresh initialization per arm; selected local prefix continues exactly from epoch20 to300',
            selector='Lowest mean fixed Val-A H projection relative L1, strict improvement; no volumetric GT selection',
            optimizer='AdamW lr2e-4 wd1e-4 clip1 batch2', precision='FP32, TF32 off, cudnn benchmark off',
            gt_allowed=False, final_test_allowed=False,
            promotion_gate=dict(ordinary_H_min_relative_improvement=0.01, ordinary_case_P90_max_ratio=1.02,
                clustered_five_draw_H_max_ratio=1.02, mean_warm_epoch_runtime_max_ratio=1.20,
                projected_300_hour_ceiling=5.0, finite_gradients=True, initial_full_forward_bitwise_equal=True),
            screening_primary='Fixed epoch20 mean Val-A H; paired case aggregation',
            hard_panel_source=str(ROOT / 'outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/registration.json'),
            hard_panel_source_sha256=base.digest(ROOT / 'outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/registration.json'),
            evaluation_plan='Seal best after300; 36 validation cases,17layouts,C0/C1; independent volumetric scoring; final35 test untouched',
            gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv'],text=True), registration_audit=guard.summary())
        old_formal.verify_data(p)
        write(out/'protocol.json',p)
        (out/'protocol.sha256').write_text(base.digest(out/'protocol.json')+'\n')
    write(out/'status.json',dict(state='registered',updated_at=base.now(),epoch=0))
    print(json.dumps(dict(stage='registered',parameters=counts)),flush=True)
    return p


def smoke(out):
    p=verify(out); config=p['config']; device=config['training']['device']
    assert not (out/'smoke.json').exists()
    with base.GTReadGuard(config['dataset']['root'],out) as guard:
        ds=base.selfsup.make_dataset(config)
        batch=base.selfsup.move_dual_batch(base.dual_measurement_collate([ds[0],ds[1]]),device)
        outputs={}; records={}
        for arm in ARMS:
            model=candidate.model_for(config,arm,device)
            model.eval()
            with torch.no_grad(): outputs[arm]=model(batch['A']).cpu()
            before=base.state_hash(model.state_dict()); model.train()
            torch.cuda.reset_peak_memory_stats(device); times=[]
            for iteration in range(3):
                model.zero_grad(set_to_none=True); torch.cuda.synchronize(device); start=time.perf_counter()
                loss,parts,volumes,projections=base.selfsup.objective(model,batch)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                torch.cuda.synchronize(device); times.append(time.perf_counter()-start)
                del loss,parts,volumes,projections
            gradient=None
            if arm=='local':
                g=model.decoder.local.output.weight.grad
                assert torch.isfinite(g).all() and g.abs().sum()>0
                gradient=float(g.norm())
            assert base.state_hash(model.state_dict())==before
            records[arm]=dict(backward_seconds=times,gradient_norm=float(norm),local_output_gradient_norm=gradient,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,weights_unchanged=True)
            del model; gc.collect(); torch.cuda.empty_cache()
        assert torch.equal(outputs['multiscale'],outputs['local'])
        audit=guard.summary()
    assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    result=dict(passed=True,full_forward_bitwise_equal=True,arms=records,gt_audit=audit,finished_at=base.now())
    write(out/'smoke.json',result); print(json.dumps(dict(stage='smoke',**result)),flush=True)


def train_arm(out,arm,target):
    p=verify(out); config=p['config']; device=config['training']['device']; run=out/arm
    assert read(out/'smoke.json')['passed']
    if target>20:
        assert arm=='local' and read(out/'gate.json')['passed']
    with base.GTReadGuard(config['dataset']['root'],out) as guard:
        model=candidate.model_for(config,arm,device)
        assert base.state_hash(model.state_dict())==p['initial_sha256'][arm]
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
        generator=torch.Generator().manual_seed(20261011)
        ds=base.selfsup.make_dataset(config); vd=base.selfsup.make_dataset(config,validation=True)
        loader=base.selfsup.make_loader(ds,config,True,generator); vl=base.selfsup.make_loader(vd,config)
        metrics=[]; best=float('inf'); best_epoch=0; best_model=None; previous=0.
        last=run/'checkpoints/last.pt'
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False)
            assert saved['protocol_sha256']==base.digest(out/'protocol.json') and saved['arm']==arm
            model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); sch.load_state_dict(saved['scheduler'])
            metrics=saved['metrics']; best=saved['best_score']; best_epoch=saved['best_epoch']; best_model=saved['best_model']; previous=saved['elapsed_seconds']
            old_formal.prime_for_resume(loader,vl); base.restore_rng(saved['rng'],generator)
            del saved
        assert len(metrics)<target
        began=time.perf_counter()
        write(run/'environment.json',dict(base.environment(config),pid=os.getpid(),arm=arm))
        for epoch in range(len(metrics),target):
            ds.set_epoch(epoch);vd.set_epoch(0);torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device); start=time.perf_counter(); lr=opt.param_groups[0]['lr']
            tr=base.selfsup.run_epoch(model,loader,config,opt); va=base.selfsup.run_epoch(model,vl,config)
            torch.cuda.synchronize(device); sch.step(); score=base.selfsup.selection_metric(va)
            assert math.isfinite(score) and (tr['sample_count'],va['sample_count'])==(166,36)
            improved=score<best
            if improved:
                best=score;best_epoch=epoch+1;best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            row=dict(epoch=epoch+1,train=tr,val=va,learning_rate=lr,seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),loader_rng_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
            metrics.append(row); elapsed=previous+time.perf_counter()-began
            payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=base.rng_state(generator),
                epoch=epoch+1,updates=(epoch+1)*83,config=config,arm=arm,protocol_sha256=base.digest(out/'protocol.json'),
                volumetric_gt_used=False,best_model=best_model,best_epoch=best_epoch,best_score=best,metrics=metrics,elapsed_seconds=elapsed)
            base.atomic_checkpoint(last,payload)
            if improved:
                base.atomic_checkpoint(run/'checkpoints/best.pt',dict(model=best_model,config=config,epoch=best_epoch,arm=arm,
                    selection_metric=best,protocol_sha256=base.digest(out/'protocol.json'),volumetric_gt_used=False))
            rows_write(run,metrics);write(run/'gt_access_audit.json',guard.summary())
            status=dict(state='screening' if target==20 else 'formal_training',arm=arm,epoch=epoch+1,max_epochs=target,
                updates=(epoch+1)*83,val_H=score,best_val_H=best,best_epoch=best_epoch,elapsed_seconds=elapsed,
                estimated_remaining_hours=np.mean([x['seconds'] for x in metrics[-10:]])*(target-epoch-1)/3600,pid=os.getpid(),updated_at=base.now())
            write(out/'status.json',status);write(run/'status.json',status)
            print(f'{arm} epoch={epoch+1}/{target} valH={score:.8f} bestH={best:.8f}@{best_epoch} seconds={row["seconds"]:.2f}',flush=True)
            if epoch==0 or (epoch+1)%10==0: old_formal.plot(run,metrics)
            del payload
        cases=base.final_cases(model,vl,config)
        (run/f'epoch{target}_per_case.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in cases))
        summary=dict(completed=True,epochs=target,updates=target*83,best_epoch=best_epoch,best_val_H=best,
            final_val_H=metrics[-1]['val']['measurement']['heldout']['relative_l1'],elapsed_seconds=previous+time.perf_counter()-began,
            finished_at=base.now(),best_checkpoint_sha256=base.digest(run/'checkpoints/best.pt'),last_checkpoint_sha256=base.digest(last),
            gt_audit=guard.summary(),sources_preserved=candidate.source_hashes()==p['sources'],final_test_evaluated=False,
            volumetric_evaluation_completed=False)
        write(run/f'summary_epoch{target}.json',summary)
        write(run/'summary.json',summary)
    del model,opt,loader,vl;gc.collect();torch.cuda.empty_cache()
    return summary


def heldout(batch):
    result={k:v for k,v in batch.items() if not k.startswith(('input_','hold_')) and k!='view_mask'}
    result.update({'input_'+k[5:]:v for k,v in batch.items() if k.startswith('hold_') and k!='hold_view_mask'})
    result['view_mask']=batch['hold_view_mask'];return result


def hard_panel(out):
    from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
    from geophy_set3d.data.collate import projection_pool_collate
    from geophy_set3d.training.utils import move_batch
    p=verify(out);config=p['config'];device=config['training']['device']
    assert base.digest(p['hard_panel_source'])==p['hard_panel_source_sha256']
    settings=[s for s in read(p['hard_panel_source'])['settings'] if s['condition']=='clustered-V4']
    assert len(settings)==5
    class Replay:
        def __init__(self,s):self.s=s
        def sample(self,station_count,*,epoch,sample_index,case_id):
            return np.array(self.s['input_ids'][case_id]),np.array(self.s['heldout_ids'][case_id])
    records=[]
    with base.GTReadGuard(config['dataset']['root'],out) as guard:
        for arm in ARMS:
            ckpt=torch.load(out/arm/'checkpoints/last.pt',map_location='cpu',weights_only=False)
            assert ckpt['epoch']==20
            model=candidate.model_for(config,arm,device);model.load_state_dict(ckpt['model']);model.eval()
            with torch.no_grad():
                for s in settings:
                    ds=MeasurementOnlyPoolDataset(config['dataset']['root'],split='val',geometry_pool='train',evaluation_only=True,
                        min_input_views=4,max_input_views=4,hold_views=4,base_seed=20261011,normalization='linear_p99_5',statistics_file='train_statistics.json')
                    ds.sampler=Replay(s)
                    for i in range(0,len(ds),2):
                        b=move_batch(projection_pool_collate([ds[j] for j in range(i,min(i+2,len(ds)))]),device)
                        inputs={k:v for k,v in b.items() if not k.startswith('hold_')};h=heldout(b)
                        field=model(inputs);pred=base.project_input(field,h)
                        valid=h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]
                        for j,case in enumerate(inputs['case_id']):
                            metric=base.relative_projection_metrics(pred[j],h['input_projection_raw'][j],valid[j])
                            records.append(dict(arm=arm,case_id=case,draw=s['draw'],relative_l1=metric['relative_l1']))
            del model,ckpt;gc.collect();torch.cuda.empty_cache()
        audit=guard.summary()
    assert len(records)==360
    (out/'hard_panel_per_case.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    means={a:float(np.mean([np.mean([x['relative_l1'] for x in records if x['arm']==a and x['case_id']==case]) for case in p['validation_case_ids']])) for a in ARMS}
    write(out/'hard_panel.json',dict(means=means,gt_audit=audit,layouts=5,cases=36))
    print(json.dumps(dict(stage='hard_panel',means=means)),flush=True)


def gate(out):
    p=verify(out);metrics={a:[json.loads(x) for x in (out/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    for x,y in zip(metrics['multiscale'],metrics['local']):
        assert x['epoch']==y['epoch'] and x['learning_rate']==y['learning_rate'] and x['loader_rng_sha256']==y['loader_rng_sha256']
        for split in ('train','val'):assert x[split]['case_station_digest']==y[split]['case_station_digest']
    cases={a:[json.loads(x) for x in (out/a/'epoch20_per_case.jsonl').read_text().splitlines()] for a in ARMS}
    assert all([x['case_id'] for x in cases[a]]==p['validation_case_ids'] for a in ARMS)
    arrays={a:np.array([x['heldout_relative_l1'] for x in cases[a]]) for a in ARMS}
    times={a:float(np.mean([x['seconds'] for x in metrics[a][2:]])) for a in ARMS}
    ratio=float(arrays['local'].mean()/arrays['multiscale'].mean());p90=float(np.quantile(arrays['local'],.9)/np.quantile(arrays['multiscale'],.9))
    hard=read(out/'hard_panel.json')['means'];hard_ratio=hard['local']/hard['multiscale'];runtime=times['local']/times['multiscale']
    delta=arrays['local']-arrays['multiscale'];rng=np.random.default_rng(20261016)
    ci=np.quantile(delta[rng.integers(0,36,(10000,36))].mean(1),[.025,.975]).tolist()
    thresholds=p['promotion_gate']
    checks=dict(ordinary_H=ratio<=1-thresholds['ordinary_H_min_relative_improvement'],ordinary_P90=p90<=thresholds['ordinary_case_P90_max_ratio'],
        clustered_H=hard_ratio<=thresholds['clustered_five_draw_H_max_ratio'],runtime=runtime<=thresholds['mean_warm_epoch_runtime_max_ratio'],
        five_hour_budget=times['local']*300/3600<=thresholds['projected_300_hour_ceiling'],smoke=read(out/'smoke.json')['passed'])
    result=dict(passed=all(checks.values()),checks=checks,ordinary_H_ratio=ratio,ordinary_P90_ratio=p90,clustered_H_ratio=hard_ratio,runtime_ratio=runtime,
        ordinary_H_means={a:float(x.mean()) for a,x in arrays.items()},warm_epoch_seconds=times,projected_local_300_hours=times['local']*300/3600,
        paired_case_delta_ci95=ci,wins=int((delta<0).sum()),losses=int((delta>0).sum()),matched_sampling=True,
        gate_is_short_measurement_screen_not_3D_guarantee=True,finished_at=base.now())
    write(out/'gate.json',result);print(json.dumps(dict(stage='gate',**result)),flush=True)
    return result


def publish_for_eval(out):
    # Evaluation uses root/checkpoints and the already frozen root protocol.
    run=out/'local'; summary=read(run/'summary.json')
    assert summary['epochs']==300
    (out/'checkpoints').mkdir(exist_ok=True)
    for name in ('best.pt','last.pt'):
        link=out/'checkpoints'/name
        if not link.exists():link.symlink_to(Path('../local/checkpoints')/name)
    write(out/'summary.json',summary)


def pipeline(out,device):
    if not out.exists():register(out,device)
    verify(out)
    if not (out/'smoke.json').exists():smoke(out)
    for arm in ARMS:
        if not (out/arm/'summary_epoch20.json').exists():train_arm(out,arm,20)
    if not (out/'hard_panel.json').exists():hard_panel(out)
    result=read(out/'gate.json') if (out/'gate.json').exists() else gate(out)
    if not result['passed']:
        write(out/'status.json',dict(state='screen_failed_no_formal_training',gate=result,updated_at=base.now()))
        return
    if not (out/'local/summary_epoch300.json').exists():train_arm(out,'local',300)
    publish_for_eval(out)
    test=ROOT/'scripts/test_mrgr_local_detail_v1.py';report=ROOT/'scripts/report_mrgr_local_detail_v1.py'
    assert test.exists() and report.exists(), 'Training completed; evaluation scripts missing'
    for action in ('register','predict','score'):
        subprocess.run([sys.executable,str(test),action,'--run-dir',str(out),'--device',device],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(report),'--run-dir',str(out)],cwd=ROOT,check=True)
    write(out/'status.json',dict(state='complete_training_and_validation_test',epoch=300,updated_at=base.now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','smoke','pipeline','verify'))
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.set_num_threads(2);out=args.out.resolve()
    if args.action=='register':register(out,args.device)
    elif args.action=='verify':verify(out);print('Verified sources/data',flush=True)
    else:
        out.mkdir(parents=True,exist_ok=True) if args.action=='smoke' else None
        lock_path=out.parent/(out.name+'.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                if args.action=='smoke':smoke(out)
                else:pipeline(out,args.device)
            except BaseException as error:
                if out.exists():write(out/'failure.json',dict(error=repr(error),timestamp=base.now(),pid=os.getpid()))
                raise


if __name__=='__main__':main()
