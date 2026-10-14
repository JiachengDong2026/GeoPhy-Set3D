#!/usr/bin/env python3
"""One fresh eight-case stability gate. Historical scientific runs are immutable."""
import argparse,gzip,json,signal,subprocess,sys,time,traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
import yaml
from run_senseiver_supervised_probe import load_samples,evaluate,references
from geophy_set3d.training.senseiver_supervised_probe import SupervisedProbeGuard,init_scale,fresh,input_identity
from geophy_set3d.training.senseiver_supervised_stability import stability_exit
from geophy_set3d.training.senseiver_init_probe import ObservedAdamW,group_norm
from geophy_set3d.training.senseiver_supervised_probe import normalized_mse
from geophy_set3d.training.ray_senseiver import sha256
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything,move_batch,write_json,atomic_checkpoint,rng_state,restore_rng
from geophy_set3d.data.collate import projection_pool_collate
NAME='ray_senseiver_supervised_stability_v1';CONFIG=REPO/'configs'/f'{NAME}.yaml'
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons')/NAME;ART=REPO/'docs/reports/artifacts'/NAME
STOP=False


def settings():return yaml.safe_load(CONFIG.read_text())


def prepare(c):
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'registration.json').exists()
    old=json.loads((REPO/'docs/reports/artifacts/ray_senseiver_supervised_probe_v1/registration.json').read_text())
    with SupervisedProbeGuard(c['dataset_root'],OUT,list(range(8))) as guard:
        for path,h in old['data_sha256'].items():assert sha256(path)==h,path
        ds,x,g=load_samples(c,list(range(8)))
        for i,s in enumerate(x):assert input_identity(projection_pool_collate([s]))==old['samples'][i]['input_identity']
        scale=init_scale(g);assert scale==old['scales']['B']
        seed_everything(c['seed']);m=fresh(scale);initial=state_hash(m.state_dict());assert initial==old['initial_hashes']['B']
        files=subprocess.check_output(['git','ls-files','src/**/*.py','scripts/run_senseiver_supervised_probe.py'],cwd=REPO,text=True).splitlines()
        files += ['scripts/run_senseiver_supervised_stability.py','src/geophy_set3d/training/senseiver_supervised_stability.py',
            'configs/'+NAME+'.yaml','configs/train_ray_senseiver_supervised_v1.yaml','docs/ray_senseiver_supervised_v1_protocol_zh.md','tests/test_senseiver_supervised_stability.py']
        reg=dict(config=c,base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),scale=scale,
            initial_hash=initial,samples=old['samples'],data_sha256=old['data_sha256'],source_sha256={p:sha256(REPO/p) for p in sorted(set(files))},
            historical_reports_sha256=old['historical_reports_sha256']|{'docs/reports/ray_senseiver_supervised_probe_v1_zh.md':sha256(REPO/'docs/reports/ray_senseiver_supervised_probe_v1_zh.md')},access=guard.summary())
        write_json(OUT/'registration.json',reg);write_json(ART/'registration.json',reg)
    write_json(ART/'prepare_access.json',guard.summary());print(json.dumps(dict(initial_hash=initial,scale=scale)))


def verify(c):
    p=OUT/'registration.json';r=json.loads(p.read_text());assert r['config']==c
    assert subprocess.check_output(['git','show',f'HEAD:docs/reports/artifacts/{NAME}/registration.json'],cwd=REPO)==p.read_bytes()
    for path,h in r['source_sha256'].items():assert sha256(REPO/path)==h,path
    for path,h in r['historical_reports_sha256'].items():assert sha256(REPO/path)==h,path
    for path,h in r['data_sha256'].items():assert sha256(path)==h,path
    return r


def run(c,resume=False,smoke=False):
    d=OUT/('smoke' if smoke else 'science');d.mkdir(exist_ok=True);ck=d/'checkpoints';ck.mkdir(exist_ok=True)
    assert not (d/'complete.json').exists(),'Scientific stop cannot repeat'
    assert resume or not (ck/'last.pt').exists(),'Explicit own last resume required'
    history=[];evaluations=[];events=[];state={};elapsed_before=0.;gen=torch.Generator().manual_seed(c['seed']);began=time.perf_counter()
    with SupervisedProbeGuard(c['dataset_root'],OUT,list(range(8))) as guard:
        try:
            reg=verify(c);ds,samples,targets=load_samples(c,list(range(8)));seed_everything(c['seed'])
            model=fresh(reg['scale']);assert state_hash(model.state_dict())==reg['initial_hash'];model.to(c['device'])
            opt=ObservedAdamW(model,lr=2e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=4000,eta_min=2e-5)
            ref=references(targets);template=ref['mean_template']['mean']['nmse'];micro=2;oom_reduced=False
            if resume:
                saved=torch.load(ck/'last.pt',map_location='cpu',weights_only=False)
                assert saved['config']==c and saved['registration_sha256']==sha256(OUT/'registration.json')
                assert not saved['exit_state'].get('reason'),'Scientific stop cannot resume'
                model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],gen)
                history=saved['history'];evaluations=saved['evaluations'];events=saved['events'];state=saved['exit_state'];elapsed_before=saved['elapsed_seconds'];micro=saved['micro'];oom_reduced=saved['oom_reduced'];opt.updates=len(history)
            def elapsed():return elapsed_before+time.perf_counter()-began
            def checkpoint(name):
                atomic_checkpoint(ck/name,dict(config=c,registration_sha256=sha256(OUT/'registration.json'),initial_hash=reg['initial_hash'],
                    model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(gen),updates=len(history),
                    history=history,evaluations=evaluations,events=events,exit_state=state,elapsed_seconds=elapsed(),micro=micro,oom_reduced=oom_reduced,GT64_supervised=True))
            def records():
                for name,v in [('history',history),('evaluations',evaluations),('events',events)]:
                    write_json(d/(name+'.json'),v)
                    if not smoke:
                        with gzip.open(ART/(name+'.json.gz'),'wt') as f:json.dump(v,f,allow_nan=False)
            if not resume:
                e,_=evaluate(model,samples,targets,c,'B',0,c['device'],True);evaluations.append(e);checkpoint('epoch0.pt');checkpoint('last.pt')
            while len(history)<(1 if smoke else 4000):
                if STOP:
                    checkpoint('last.pt');records();write_json(d/'pause.json',dict(reason='ENGINEERING_INTERRUPT',updates=len(history)));return
                if elapsed()>=1200 and not smoke:break
                step=len(history)+1;ids=list(range(((step-1)*2)%8,((step-1)*2)%8+2));lr=opt.param_groups[0]['lr'];opt.zero_grad(set_to_none=True);model.train();lossvalue=0.;before_rng=rng_state(gen);t=time.perf_counter()
                try:
                    for off in range(0,2,micro):
                        ii=ids[off:off+micro];b=move_batch(projection_pool_collate([samples[i] for i in ii]),c['device']);g=torch.stack([targets[i] for i in ii]).to(c['device'])
                        p=model(b);assert p.shape==g.shape==(len(ii),1,64,64,64)
                        loss=normalized_mse(p,g);assert torch.isfinite(loss);(loss*len(ii)/2).backward();lossvalue+=float(loss.detach())*len(ii)/2
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True);restore_rng(before_rng,gen)
                    if oom_reduced:
                        checkpoint('last.pt');records();write_json(d/'pause.json',dict(reason='GPU_RESOURCE_PAUSE',updates=len(history)));return
                    oom_reduced=True;micro=1;torch.cuda.empty_cache();events.append(dict(engineering='OOM_BATCH1_ACCUM2',updates=len(history)));continue
                named=list(model.named_parameters());norms=group_norm(named,[p.grad for n,p in named]);assert all(v>0 for v in norms.values())
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();scheduler.step()
                history.append(dict(updates=step,loss_nmse=lossvalue,indices=ids,station_ids=c['station_ids'],lr=lr,next_lr=opt.param_groups[0]['lr'],gradient_l2=norms,preclip_norm=norm,**opt.observations[-1],seconds=time.perf_counter()-t,elapsed_seconds=elapsed(),peak_gpu_bytes=torch.cuda.max_memory_allocated(c['device']),queries_per_case=64**3))
                if step%100==0 and not smoke:
                    e,_=evaluate(model,samples,targets,c,'B',step,c['device'],step%200==0);evaluations.append(e)
                    if step%200==0:
                        state=stability_exit(step,[r['loss_nmse'] for r in history],e,template,state,elapsed());events.append(state)
                    checkpoint('last.pt');records();print(json.dumps(dict(updates=step,lr=lr,mean=e['mean'],state=state,elapsed=elapsed())),flush=True)
                    if state.get('reason'):break
            if smoke:
                checkpoint('last.pt');saved=torch.load(ck/'last.pt',map_location='cpu',weights_only=False);assert saved['updates']==1 and saved['scheduler']['last_epoch']==1
                result=dict(smoke=True,updates=1,state_roundtrip=True,initial_hash=reg['initial_hash']);write_json(d/'complete.json',result);write_json(ART/'smoke.json',result);return
            if not state.get('reason'):
                e,_=evaluate(model,samples,targets,c,'B',len(history),c['device'],True);evaluations.append(e)
                state=stability_exit(len(history),[r['loss_nmse'] for r in history],e,template,state,elapsed());events.append(state)
            checkpoint('last.pt');records()
            result=dict(status=state['status'],reason=state['reason'],updates=len(history),elapsed_seconds=elapsed(),initial_hash=reg['initial_hash'],
                final=evaluations[-1],references=ref,last_checkpoint=str(ck/'last.pt'),last_sha256=sha256(ck/'last.pt'),epoch0_sha256=sha256(ck/'epoch0.pt'),
                training_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),peak_gpu_bytes=torch.cuda.max_memory_allocated(c['device']),access=guard.summary())
            write_json(d/'complete.json',result);write_json(ART/'complete.json',result);print(json.dumps(dict(status=result['status'],updates=len(history))),flush=True)
        except BaseException as exc:
            write_json(d/'engineering_error.json',dict(updates=len(history),exception=repr(exc),traceback=traceback.format_exc()));raise
        finally:write_json(d/'access.json',guard.summary());write_json(ART/('smoke_access.json' if smoke else 'science_access.json'),guard.summary())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','smoke','run'],required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();torch.set_num_threads(2)
    def stop(sig,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    c=settings()
    if a.phase=='prepare':prepare(c)
    else:run(c,a.resume,a.phase=='smoke')
if __name__=='__main__':main()
