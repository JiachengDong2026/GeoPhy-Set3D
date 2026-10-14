#!/usr/bin/env python3
"""Registered fresh full-train supervised baseline; never reuses stability weights."""
import argparse,gzip,json,signal,subprocess,sys,time,traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import torch,yaml
from run_senseiver_supervised_probe import load_samples,evaluate
from geophy_set3d.training.senseiver_supervised_formal import FormalGT64Guard,datasets,batch,metric_rows,summarize,validate_last
from geophy_set3d.training.senseiver_supervised_probe import init_scale,fresh,normalized_mse,input_identity
from geophy_set3d.training.senseiver_supervised_stability import formal_exit
from geophy_set3d.training.senseiver_init_probe import ObservedAdamW,group_norm
from geophy_set3d.training.ray_senseiver import sha256
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything,write_json,atomic_checkpoint,rng_state,restore_rng
NAME='ray_senseiver_supervised_v1';CONFIG=REPO/'configs'/('train_'+NAME+'.yaml')
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons')/NAME;ART=REPO/'docs/reports/artifacts'/NAME
GATE=OUT.parent/'ray_senseiver_supervised_stability_v1/science/complete.json'
STOP=False

def settings():return yaml.safe_load(CONFIG.read_text())

def prepare(c):
    gate=json.loads(GATE.read_text());assert gate['status']=='PASSED','Gate must first pass scientifically'
    assert not (OUT/'registration.json').exists();OUT.mkdir(exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    with FormalGT64Guard(c['dataset_root'],OUT) as guard:
        train,val=datasets(c);assert (len(train),len(val))==(166,36)
        root=Path(c['dataset_root']);registered={};hashes={};targets=[]
        for ds in [train,val]:
            rows=[]
            for i,r in enumerate(ds.records):
                p=root/r['gt_volume_64_path'];meta=json.loads((p.parent/'sample_meta.json').read_text());source=json.loads(Path(r['source_metadata_path']).read_text())
                assert meta['case_id']==meta['sample_id']==r['case_id']==r['sample_id']==source['sample_id']
                assert meta['frame_index']==source['frame_index']==60 and meta['physical_time_s']==source['physical_time_s']==2.
                assert meta['gt_axes']==source['target_axes']==['x','y','z'] and source['target_quantity']=='methane_mass_fraction' and source['target_unit']=='1'
                assert meta['gt_path']==r['source_target_path'] and meta['source_gt_sha256']==r['source_gt_sha256']
                assert all(source['domain_bounds_m'][a]==[0.,.6] for a in ['x','y','z'])
                for key in ['train_projection_path','gt_volume_64_path']:
                    q=root/r[key];h=sha256(q);assert h==meta['hashes'][q.name];hashes[str(q)]=h
                x,g=batch(ds,[i],'cpu');assert torch.isfinite(g).all() and float(g.max())>0 and g.shape==(1,1,64,64,64)
                if ds is train:targets.append(g[0])
                rows.append(dict(index=i,case_id=r['case_id'],sample_id=r['sample_id'],GT64=r['gt_volume_64_path'],stations_epoch0=x['input_station_indices'][0].tolist(),input_identity_epoch0=input_identity(x)))
            registered[ds.split]=rows
        scale=init_scale(targets);seed_everything(c['seed']);model=fresh(scale);assert sum(p.numel() for p in model.parameters())==254913
        files=subprocess.check_output(['git','ls-files','src/**/*.py'],cwd=REPO,text=True).splitlines()
        files += ['src/geophy_set3d/training/senseiver_supervised_formal.py','scripts/train_ray_senseiver_supervised.py','tests/test_senseiver_supervised_formal.py',str(CONFIG.relative_to(REPO)),'docs/ray_senseiver_supervised_v1_protocol_zh.md']
        for n in ['source_index.jsonl','train_statistics.json','geometry/train_geometry.npz','generation_config.yaml']:hashes[str(root/n)]=sha256(root/n)
        reg=dict(config=c,base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),gate_sha256=sha256(GATE),gate_updates=gate['updates'],scale=scale,initial_hash=state_hash(model.state_dict()),samples=registered,data_sha256=hashes,source_sha256={f:sha256(REPO/f) for f in sorted(set(files))},comparison_manifest_sha256=sha256(REPO/c['comparison']['manifest']),access=guard.summary(),initialization_GT_split='train only',model_parameters=254913,sampler='original ViewSampler; epoch zero-based train, fixed zero Val; single-process audited loader',supervised=True)
        write_json(OUT/'registration.json',reg);write_json(ART/'registration.json',reg)
    write_json(ART/'prepare_access.json',guard.summary());print(json.dumps(dict(scale=scale,initial_hash=reg['initial_hash'],gate_updates=gate['updates'])))

def verify(c):
    p=OUT/'registration.json';r=json.loads(p.read_text());assert r['config']==c and sha256(GATE)==r['gate_sha256']
    assert subprocess.check_output(['git','show',f'HEAD:docs/reports/artifacts/{NAME}/registration.json'],cwd=REPO)==p.read_bytes()
    for path,h in r['source_sha256'].items():assert sha256(REPO/path)==h,path
    for path,h in r['data_sha256'].items():assert sha256(path)==h,path
    return r

def validate(model,ds,device):
    model.eval();rows=[]
    with torch.no_grad():
        for i in range(len(ds)):
            b,g=batch(ds,[i],device);p=model(b);assert p.shape==g.shape==(1,1,64,64,64)
            rows+=metric_rows(p,g,b['case_id'],[i])
    return summarize(rows)

def run(c,resume=False,smoke=False):
    d=OUT/('smoke' if smoke else 'science');d.mkdir(exist_ok=True);ck=d/'checkpoints';ck.mkdir(exist_ok=True)
    assert not (d/'complete.json').exists();assert resume or not (ck/'last.pt').exists()
    history=[];steps=[];healths=[];events=[];state={};train_rows=[];order=[];epoch=position=updates=0;best=float('inf');best_epoch=None;elapsed_before=0.;micro=2;oom_reduced=False;started=time.perf_counter();gen=torch.Generator().manual_seed(c['seed'])
    with FormalGT64Guard(c['dataset_root'],OUT) as guard:
        try:
            reg=verify(c);train,val=datasets(c);_,fixed,gt=load_samples(c,[0,1,2]);seed_everything(c['seed']);model=fresh(reg['scale']);assert state_hash(model.state_dict())==reg['initial_hash'];model.to(c['device'])
            opt=ObservedAdamW(model,lr=2e-4,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300,eta_min=2e-5)
            if resume:
                s=torch.load(ck/'last.pt',map_location='cpu',weights_only=False);validate_last(s,c,sha256(OUT/'registration.json'))
                model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sched.load_state_dict(s['scheduler']);restore_rng(s['rng'],gen)
                epoch=s['epoch'];position=s['position'];order=s['order'];train_rows=s['train_rows'];history=s['history'];steps=s['steps'];updates=s['updates'];state=s['exit_state'];healths=s['healths'];events=s['events'];best=s['best'];best_epoch=s['best_epoch'];elapsed_before=s['elapsed_seconds'];micro=s['micro'];oom_reduced=s['oom_reduced'];opt.updates=updates
            def elapsed():return elapsed_before+time.perf_counter()-started
            def save(name):
                atomic_checkpoint(ck/name,dict(config=c,registration_sha256=sha256(OUT/'registration.json'),model=model.state_dict(),initial_hash=reg['initial_hash'],optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state(gen),epoch=epoch,position=position,order=order,train_rows=train_rows,history=history,steps=steps,updates=updates,exit_state=state,healths=healths,events=events,best=best,best_epoch=best_epoch,elapsed_seconds=elapsed(),micro=micro,oom_reduced=oom_reduced,GT64_supervised=True))
            def records():
                for name,value in [('history',history),('steps',steps),('healths',healths),('events',events)]:
                    write_json(d/(name+'.json'),value)
                    if not smoke:
                        with gzip.open(ART/(name+'.json.gz'),'wt') as f:json.dump(value,f,allow_nan=False)
            def health():
                h,_=evaluate(model,fixed,gt,c,'B',updates,c['device'],True);h['epoch']=epoch;healths.append(h);return h
            if not resume:
                h=health();save('epoch0.pt');save('last.pt')
            while epoch<300:
                if STOP or elapsed()>=7200:
                    if STOP:
                        save('last.pt');records();write_json(d/'pause.json',dict(reason='ENGINEERING_INTERRUPT',epoch=epoch,updates=updates));return
                    state=dict(state,reason='WALLTIME');break
                train.set_epoch(epoch);val.set_epoch(0)
                if not order:order=torch.randperm(len(train),generator=gen).tolist()
                while position<83:
                    if STOP or elapsed()>=7200:break
                    ids=order[position*2:position*2+2];opt.zero_grad(set_to_none=True);model.train();t=time.perf_counter();before_rng=rng_state(gen);rows=[];identities=[];lossvalue=0.
                    try:
                        for off in range(0,2,micro):
                            ii=ids[off:off+micro];b,g=batch(train,ii,c['device']);p=model(b);assert p.shape==g.shape==(len(ii),1,64,64,64)
                            loss=normalized_mse(p,g);assert torch.isfinite(loss);(loss*len(ii)/2).backward();lossvalue+=float(loss.detach())*len(ii)/2
                            rows+=metric_rows(p,g,b['case_id'],ii)
                            identities += [dict(index=i,case_id=b['case_id'][j],stations=b['input_station_indices'][j][b['input_station_indices'][j]>=0].tolist(),valid_rays=int(b['input_valid_ray'][j].sum())) for j,i in enumerate(ii)]
                    except torch.cuda.OutOfMemoryError:
                        opt.zero_grad(set_to_none=True);restore_rng(before_rng,gen)
                        if oom_reduced:
                            save('last.pt');records();write_json(d/'pause.json',dict(reason='GPU_RESOURCE_PAUSE',epoch=epoch,updates=updates));return
                        micro=1;oom_reduced=True;torch.cuda.empty_cache();events.append(dict(engineering='OOM_BATCH1_ACCUM2',updates=updates));continue
                    norms=group_norm(list(model.named_parameters()),[p.grad for p in model.parameters()]);assert all(v>0 for v in norms.values())
                    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();updates+=1;position+=1;train_rows+=rows
                    steps.append(dict(updates=updates,epoch_zero_based=epoch,position=position,cases=identities,loss_nmse=lossvalue,gradient_l2=norms,preclip_norm=norm,lr=opt.param_groups[0]['lr'],**opt.observations[-1],seconds=time.perf_counter()-t,elapsed_seconds=elapsed(),peak_gpu_bytes=torch.cuda.max_memory_allocated(c['device']),queries_per_case=64**3))
                    if updates%10==0:save('last.pt')
                    if smoke:
                        save('last.pt');s=torch.load(ck/'last.pt',map_location='cpu',weights_only=False);validate_last(s,c,sha256(OUT/'registration.json'));assert s['updates']==1
                        result=dict(updates=1,fresh_initial_hash=reg['initial_hash'],roundtrip=True,full_queries=64**3,optimizer_update=opt.observations[-1]);write_json(d/'complete.json',result);write_json(ART/'smoke.json',result);return
                if position<83:continue
                v=validate(model,val,c['device']);tr=summarize(train_rows);epoch+=1;position=0;order=[];train_rows=[];sched.step()
                row=dict(epoch=epoch,updates=updates,train=tr,val=v,lr_used=steps[-1]['lr'],next_lr=opt.param_groups[0]['lr'],elapsed_seconds=elapsed());history.append(row)
                h=health() if epoch%20==0 else (healths[-1] if healths else None)
                new_state=formal_exit(epoch,history,h,state)
                if new_state['reason']=='FAILED_TO_LEARN' and (h is None or h['epoch']!=epoch):h=health();new_state=formal_exit(epoch,history,h,state)
                state=new_state;events.append(state)
                if v['mean']['nmse']<best:best=v['mean']['nmse'];best_epoch=epoch;save('best.pt')
                save('last.pt');records();print(json.dumps(dict(epoch=epoch,updates=updates,train=tr['mean'],val=v['mean'],best=best,best_epoch=best_epoch,exit_state=state,elapsed_seconds=elapsed())),flush=True)
                if state.get('reason'):break
            save('last.pt');records()
            result=dict(reason=state['reason'],epochs=epoch,updates=updates,elapsed_seconds=elapsed(),best_nmse=best,best_epoch=best_epoch,best_checkpoint=str(ck/'best.pt'),best_sha256=sha256(ck/'best.pt') if (ck/'best.pt').exists() else None,last_sha256=sha256(ck/'last.pt'),epoch0_sha256=sha256(ck/'epoch0.pt'),initial_hash=reg['initial_hash'],training_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),peak_gpu_bytes=torch.cuda.max_memory_allocated(c['device']),access=guard.summary())
            write_json(d/'complete.json',result);write_json(ART/'complete.json',result);print(json.dumps(result),flush=True)
        except BaseException as exc:
            # Save last *valid* checkpoint; never overwrite with a partially invalid update.
            write_json(d/'engineering_error.json',dict(epoch=epoch,updates=updates,exception=repr(exc),traceback=traceback.format_exc()));raise
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
