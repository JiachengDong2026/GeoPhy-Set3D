#!/usr/bin/env python3
"""Register/run the single supervised GT64 learnability check; no loss search."""
import argparse
import copy
import gzip
import json
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import torch
import yaml
from geophy_set3d.training.senseiver_supervised_probe import (NAME, STATIONS, SupervisedProbeGuard,
    dataset, separate, normalized_mse, volume_metrics, init_scale, fresh, metrics_mean,
    exit_event, verify_resume, input_identity)
from geophy_set3d.training.senseiver_init_probe import ObservedAdamW,group_norm
from geophy_set3d.training.ray_senseiver import sha256
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import (seed_everything,write_json,atomic_checkpoint,move_batch,
    rng_state,restore_rng,save_slices)
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models.ray_senseiver_ss import ray_tokens

CONFIG=REPO/'configs'/f'{NAME}.yaml'
ART=REPO/'docs/reports/artifacts'/NAME
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons')/NAME
CK=OUT/'checkpoints'
STOP=False


def config():
    c=yaml.safe_load(CONFIG.read_text())
    assert c['name']==NAME and c['case_indices']==list(range(8)) and c['station_ids']==STATIONS
    assert c['seed']==20261011 and c['model']=='original_ray_senseiver_ss'
    assert c['optimizer']==dict(name='adamw',lr=.0002,weight_decay=.0001,clip=1.,precision='fp32',scheduler='none')
    assert c['loss']==dict(name='per_sample_normalized_voxel_mse',denominator_floor=1e-12)
    assert (c['stages']['A']['max_updates'],c['stages']['B']['max_updates'])==(1000,2000)
    return c


def load_samples(c,indices):
    ds=dataset(c['dataset_root']);samples=[];targets=[]
    for i in indices:
        x,g=separate(ds[i]);samples.append(x);targets.append(g)
    return ds,samples,targets


def source_manifest():
    files=subprocess.check_output(['git','ls-files','src/**/*.py'],cwd=REPO,text=True).splitlines()
    files += ['src/geophy_set3d/training/senseiver_supervised_probe.py',
        'scripts/run_senseiver_supervised_probe.py','configs/'+NAME+'.yaml',
        'tests/test_senseiver_supervised_probe.py','docs/'+NAME+'_protocol_zh.md']
    return {f:sha256(REPO/f) for f in sorted(set(files))}


def prepare(c):
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True);CK.mkdir(exist_ok=True)
    assert not (OUT/'registration.json').exists(),'Registration already exists'
    guard=SupervisedProbeGuard(c['dataset_root'],OUT,list(range(8)))
    with guard:
        ds,samples,targets=load_samples(c,list(range(8)));root=Path(c['dataset_root']);rows=[];hashes={}
        original=[json.loads(x) for x in (root/'source_index.jsonl').read_text().splitlines() if x.strip()]
        geometry_id=None
        for i,(s,g,r) in enumerate(zip(samples,targets,ds.records)):
            meta_path=root/r['gt_volume_64_path'];meta=json.loads((meta_path.parent/'sample_meta.json').read_text())
            frames=[json.loads(x) for x in (meta_path.parent/'frame_manifest.jsonl').read_text().splitlines() if x.strip()]
            source_meta=json.loads(Path(r['source_metadata_path']).read_text())
            assert meta['case_id']==meta['sample_id']==r['case_id']==r['sample_id']==source_meta['sample_id']
            assert meta['split']=='train' and meta['generation_status']=='complete'
            assert meta['frame_index']==source_meta['frame_index']==frames[0]['frame_index']==60
            assert meta['physical_time_s']==source_meta['physical_time_s']==frames[0]['physical_time_s']==2.
            assert meta['gt_axes']==source_meta['target_axes']==['x','y','z']
            assert meta['gt_path']==r['source_target_path']==frames[0]['gt_path']
            assert meta['source_gt_sha256']==r['source_gt_sha256']
            assert source_meta['target_quantity']=='methane_mass_fraction' and source_meta['target_unit']=='1'
            assert all(source_meta['domain_bounds_m'][a]==[0.,.6] for a in ['x','y','z'])
            assert meta['downsample_method']=='Trilinear resampling at linspace(0,0.6,64); endpoint-aligned point grid'
            assert frames[0]['gt_volume_64_path']=='gt_volume_64.npy' and frames[0]['train_projection_path']=='projections_train.npy'
            assert g.shape==(1,64,64,64) and g.dtype==torch.float32 and float(g.max())>0 and float(g.std())>0
            for key in ['train_projection_path','gt_volume_64_path']:
                p=root/r[key];h=sha256(p);assert h==meta['hashes'][p.name];hashes[str(p)]=h
            for p in [meta_path.parent/'sample_meta.json',meta_path.parent/'frame_manifest.jsonl',Path(r['source_metadata_path'])]:hashes[str(p)]=sha256(p)
            b=projection_pool_collate([s]);geo={k:v for k,v in b.items() if k.startswith('input_') and k not in ['input_projection','input_projection_raw']}
            identity=input_identity(geo);geometry_id=geometry_id or identity;assert identity==geometry_id
            assert b['input_station_indices'].tolist()==[STATIONS]
            valid=b['input_valid_ray'];raw=b['input_projection_raw'];assert torch.isfinite(raw).all()
            _,excluded,count=ray_tokens(b);assert int(count[0])==int(valid.sum())==int((~excluded).sum())
            rows.append(dict(index=i,source_index_row=next(j for j,r0 in enumerate(original) if r0['case_id']==r['case_id']),
                case_id=r['case_id'],sample_id=r['sample_id'],frame=60,time_s=2.,stations=STATIONS,
                geometry_identity=identity,input_identity=input_identity(b),GT64_path=r['gt_volume_64_path'],
                valid_rays=int(valid.sum()),valid_zero_rays=int((raw[valid]==0).sum()),GT=volume_metrics(g,g),
                source_gt128_hash_from_metadata_only=r['source_gt_sha256']))
        scales={k:init_scale([targets[i] for i in v['indices']]) for k,v in c['stages'].items()}
        initials={}
        for stage,scale in scales.items():
            seed_everything(c['seed']);model=fresh(scale);initials[stage]=state_hash(model.state_dict())
            atomic_checkpoint(CK/f'{stage}_registered_initial.pt',dict(model=model.state_dict(),scale=scale,not_warm_start=True))
        for name in ['source_index.jsonl','train_statistics.json','geometry/train_geometry.npz','generation_config.yaml']:
            p=root/name;hashes[str(p)]=sha256(p)
        history={str(p.relative_to(REPO)):sha256(p) for p in (REPO/'docs/reports').glob('ray_senseiver*_zh.md')}
        reg=dict(config=c,base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            samples=rows,scales=scales,initial_hashes=initials,source_sha256=source_manifest(),
            data_sha256=hashes,historical_reports_sha256=history,access=guard.summary(),GT128_opened=False,
            supervised_diagnostic=True,not_for_selfsup_leaderboard_or_warm_start=True)
        write_json(OUT/'registration.json',reg);write_json(ART/'registration.json',reg)
    write_json(ART/'prepare_access.json',guard.summary());print(json.dumps(dict(scales=scales,initials=initials,samples=rows),indent=2))


def verify(c,indices):
    p=OUT/'registration.json';reg=json.loads(p.read_text());assert reg['config']==c
    assert subprocess.check_output(['git','show',f'HEAD:docs/reports/artifacts/{NAME}/registration.json'],cwd=REPO)==p.read_bytes()
    for path,h in {**reg['source_sha256'],**reg['historical_reports_sha256']}.items():assert sha256(REPO/path)==h,path
    # Only hash the current stage's GT/projection/metadata; do not open other GTs in A.
    root=Path(c['dataset_root']);allowed_ids={reg['samples'][i]['case_id'] for i in indices}
    for path,h in reg['data_sha256'].items():
        p=Path(path)
        if any(s['case_id'] in path for s in reg['samples'] if s['case_id'] not in allowed_ids):continue
        assert sha256(p)==h,path
    return reg


def evaluate(model,samples,targets,c,stage,step,device,intervene=False):
    started=time.perf_counter();model.eval();rows=[];preds=[];logits=[]
    hook=model.decoder.register_forward_hook(lambda mod,args,out:logits.append(out.detach().flatten().cpu()))
    try:
        with torch.no_grad():
            for idx,(sample,g) in enumerate(zip(samples,targets)):
                b=move_batch(projection_pool_collate([sample]),device);pred=model(b)
                assert pred.shape==(1,1,64,64,64) and torch.isfinite(pred).all()
                rows.append(dict(index=idx,case_id=sample['case_id'],**volume_metrics(pred,g[None].to(device))));preds.append(pred.cpu())
    finally:hook.remove()
    log=torch.cat(logits).double();p=torch.cat(preds)
    result=dict(updates=step,mean=metrics_mean(rows),cases=rows,
        logit_quantiles={str(q):float(torch.quantile(log,q)) for q in [0.,.01,.5,.99,1.]},
        fraction_logit_below_minus12=float((log < -12).double().mean()),
        mean_sigmoid_derivative=float((p.double()*(1-p.double())).mean()),
        query_count_per_case=64**3,complete_query_finite=True)
    if intervene:
        dep=[]
        with torch.no_grad():
            for idx,donor in zip([0,1,2],[1,2,0]):
                normal=preds[idx].to(device);b=move_batch(projection_pool_collate([samples[idx]]),device)
                zero=dict(b,input_projection_raw=torch.zeros_like(b['input_projection_raw']),input_projection=torch.zeros_like(b['input_projection']))
                db=move_batch(projection_pool_collate([samples[donor]]),device)
                wrong=dict(b,input_projection_raw=db['input_projection_raw'],input_projection=db['input_projection'])
                for x in [zero,wrong]:
                    assert all(x[k] is b[k] for k in b if k not in ['input_projection_raw','input_projection'])
                zp=model(zero);wp=model(wrong);difference=float((normal-zp).double().abs().mean())
                target=targets[idx][None].to(device);wrongmetric=volume_metrics(wp,target)
                dep.append(dict(index=idx,donor_index=donor,absolute_MA=difference,
                    normal_meanabs=float(normal.double().abs().mean()),relative_MA=difference/(float(normal.double().abs().mean())+1e-12),
                    normal_nmse=rows[idx]['nmse'],wrong_measurement_nmse=wrongmetric['nmse'],
                    wrong_minus_normal_nmse=wrongmetric['nmse']-rows[idx]['nmse'],
                    wrong_measurement_relative_l1=wrongmetric['relative_l1'],
                    wrong_output_absolute_MA=float((normal-wp).double().abs().mean())))
        result['interventions']=dep
    result['seconds']=time.perf_counter()-started
    return result,preds


def references(targets):
    gs=torch.stack(targets);template=gs.mean(0)
    values={name:[dict(index=i,**volume_metrics(p[i],g)) for i,g in enumerate(targets)] for name,p in dict(
        zero=torch.zeros_like(gs),uniform=gs.mean(dim=(1,2,3,4),keepdim=True).expand_as(gs),
        mean_template=template[None].expand_as(gs)).items()}
    return {k:dict(mean=metrics_mean(rows),cases=rows) for k,rows in values.items()}


def save_records(stage,history,evals,events,access):
    out=OUT/stage;art=ART/stage;out.mkdir(exist_ok=True);art.mkdir(exist_ok=True)
    for name,value in [('history',history),('evaluations',evals),('events',events)]:
        write_json(out/(name+'.json'),value)
        with gzip.open(art/(name+'.json.gz'),'wt') as f:json.dump(value,f,allow_nan=False)
    write_json(art/'access.json',access)


def train(c,stage,resume=False,smoke=False):
    global STOP
    if stage=='B' and not smoke:
        assert json.loads((OUT/'A'/'complete.json').read_text())['reason']=='PASSED','B requires A passed'
    ids=c['stages'][stage]['indices'];guard=SupervisedProbeGuard(c['dataset_root'],OUT,ids)
    d=OUT/('smoke_'+stage if smoke else stage);d.mkdir(exist_ok=True);path=CK/('smoke_'+stage if smoke else stage);path.mkdir(exist_ok=True)
    assert not (d/'complete.json').exists(),'Scientifically completed experiment cannot repeat'
    assert resume or not (path/'last.pt').exists(),'Use explicit own-stage resume'
    history=[];evals=[];events=[];state={};elapsed_before=0.;initial=None;opt=None;model=None
    generator=torch.Generator().manual_seed(c['seed']);started=time.perf_counter()
    with guard:
        try:
            reg=verify(c,ids);rh=sha256(OUT/'registration.json');ds,samples,targets=load_samples(c,ids)
            for i,s in zip(ids,samples):assert input_identity(projection_pool_collate([s]))==reg['samples'][i]['input_identity']
            seed_everything(c['seed']);model=fresh(reg['scales'][stage],**dict(chunk=c['runtime']['query_chunk_size'],checkpoint=c['runtime']['decoder_checkpoint']))
            initial=state_hash(model.state_dict());assert initial==reg['initial_hashes'][stage]
            # Endpoint-aligned query XYZ, y is axis1 (height), matching frozen GT64.
            coords=model.query_coordinates_m.reshape(64,64,64,3)
            for axis in range(3):
                q=[0,0,0];q[axis]=63;assert abs(float(coords[tuple(q)][axis])-.6)<1e-6
            assert len(model.query_coordinates_m)==64**3 and sum(p.numel() for p in model.parameters())==254913
            model.to(c['device']);opt=ObservedAdamW(model,lr=2e-4,weight_decay=1e-4)
            refs=references(targets);write_json(d/'references.json',refs)
            if not smoke:write_json(ART/stage/'references.json',refs) if (ART/stage).exists() else None
            micro=c['stages'][stage]['effective_batch'];oom_reduced=False
            if resume:
                saved=torch.load(path/'last.pt',map_location='cpu',weights_only=False);verify_resume(saved,stage,c,rh)
                model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],generator)
                history=saved['history'];evals=saved['evaluations'];events=saved['events'];state=saved['exit_state'];elapsed_before=saved['elapsed_seconds']
                opt.updates=len(history);initial=saved['initial_hash'];micro=saved['micro_batch'];oom_reduced=saved['oom_reduced']
            def elapsed():return elapsed_before+time.perf_counter()-started
            def checkpoint(name):
                atomic_checkpoint(path/name,dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(generator),
                    config=c,stage=stage,updates=len(history),history=history,evaluations=evals,events=events,exit_state=state,
                    elapsed_seconds=elapsed(),initial_hash=initial,registration_sha256=rh,micro_batch=micro,oom_reduced=oom_reduced,
                    supervised_diagnostic_GT64=True,never_use_as_no_GT_warm_start=True))
            if not resume:
                e,_=evaluate(model,samples,targets,c,stage,0,c['device'],intervene=stage=='B');evals.append(e)
                checkpoint('epoch0.pt');checkpoint('last.pt')
            max_steps=1 if smoke else c['stages'][stage]['max_updates']
            while len(history)<max_steps:
                if STOP:
                    checkpoint('last.pt');write_json(d/'pause.json',dict(reason='ENGINEERING_INTERRUPT',updates=len(history)));return
                if not smoke and elapsed()>=c['stages'][stage]['max_seconds']:break
                step=len(history)+1;effective=c['stages'][stage]['effective_batch']
                start=((step-1)*effective)%len(samples);batch_ids=list(range(start,start+effective))
                opt.zero_grad(set_to_none=True);model.train();loss_value=0.;step_start=time.perf_counter();before_rng=rng_state(generator)
                try:
                    for off in range(0,effective,micro):
                        selection=batch_ids[off:off+micro]
                        b=move_batch(projection_pool_collate([samples[i] for i in selection]),c['device'])
                        g=torch.stack([targets[i] for i in selection]).to(c['device']);pred=model(b)
                        assert pred.shape==g.shape==(len(selection),1,64,64,64)
                        loss=normalized_mse(pred,g);assert torch.isfinite(loss)
                        (loss*len(selection)/effective).backward();loss_value+=float(loss.detach())*len(selection)/effective
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True);restore_rng(before_rng,generator)
                    if oom_reduced or micro==1:
                        checkpoint('last.pt');write_json(d/'pause.json',dict(reason='GPU_RESOURCE_PAUSE',updates=len(history)));return
                    oom_reduced=True;micro=1;torch.cuda.empty_cache()
                    events.append(dict(updates=len(history),engineering='OOM_REDUCE_TO_BATCH1_ACCUM2'));continue
                named=list(model.named_parameters());grads=group_norm(named,[p.grad for n,p in named])
                if any(p.grad is None or not torch.isfinite(p.grad).all() for n,p in named) or any(v<=0 for v in grads.values()):raise RuntimeError('Missing/nonfinite/zero group gradients')
                grad_total=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
                record=dict(updates=step,loss_nmse=loss_value,indices=batch_ids,stations=STATIONS,
                    gradient_l2=grads,preclip_norm=grad_total,**opt.observations[-1],
                    seconds=time.perf_counter()-step_start,elapsed_seconds=elapsed(),peak_gpu_bytes=torch.cuda.max_memory_allocated(c['device']),
                    queries_per_case=64**3,lr=opt.param_groups[0]['lr'])
                history.append(record)
                if step%50==0:print(json.dumps(dict(stage=stage,updates=step,train_nmse=loss_value,elapsed=elapsed())),flush=True)
                if not smoke and step%100==0:
                    e,preds=evaluate(model,samples,targets,c,stage,step,c['device'],intervene=stage=='B' and step%200==0)
                    e['last_update_gradients']=record['gradient_l2'];e['last_parameter_updates']=record['update_l2'];evals.append(e)
                    state=exit_event(stage,step,[r['loss_nmse'] for r in history],e,c['stages'][stage],state,elapsed(),refs['mean_template']['mean']['nmse'])
                    events.append(state);checkpoint('last.pt');save_records(stage,history,evals,events,guard.summary())
                    print(json.dumps(dict(stage=stage,evaluation=e['mean'],exit=state)),flush=True)
                    if state['reason']:break
            if smoke:
                checkpoint('last.pt');saved=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
                assert saved['updates']==1 and state_hash(saved['model'])==state_hash(model.state_dict())
                result=dict(smoke=True,stage=stage,updates=1,state_roundtrip=True,initial_hash=initial,access=guard.summary())
                write_json(d/'complete.json',result);write_json(ART/f'smoke_{stage}.json',result);print(json.dumps(result));return
            if not evals or evals[-1]['updates']!=len(history):
                e,preds=evaluate(model,samples,targets,c,stage,len(history),c['device'],intervene=stage=='B');evals.append(e)
            if not state.get('reason'):
                state=dict(state,updates=len(history),reason='BUDGET_EXHAUSTED_UNRESOLVED');events.append(state)
            checkpoint('last.pt')
            # Final fixed-case full volumes are derived supervised artifacts, external only.
            with torch.no_grad():
                model.eval();preds=[model(move_batch(projection_pool_collate([s]),c['device'])).cpu() for s in samples]
            np.savez_compressed(d/'final_volumes.npz',prediction=torch.cat(preds).numpy(),gt=torch.stack(targets).numpy())
            result=dict(stage=stage,reason=state['reason'],updates=len(history),elapsed_seconds=elapsed(),
                initial_hash=initial,final_hash=state_hash(model.state_dict()),last_sha256=sha256(path/'last.pt'),
                checkpoint=str(path/'last.pt'),epoch0_sha256=sha256(path/'epoch0.pt'),
                final=evaluations_last(evals),references=refs,peak_gpu_bytes=torch.cuda.max_memory_allocated(c['device']),
                training_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                access=guard.summary(),supervised_diagnostic=True)
            save_records(stage,history,evals,events,guard.summary());write_json(d/'complete.json',result);write_json(ART/stage/'complete.json',result)
            write_json(ART/stage/'references.json',refs)
            print(json.dumps(dict(stage=stage,reason=result['reason'],updates=result['updates'],final=result['final']['mean'])),flush=True)
        except BaseException as exc:
            write_json(d/'engineering_error.json',dict(updates=len(history),exception=repr(exc),traceback=traceback.format_exc(),access=guard.summary()))
            if model is not None and opt is not None and initial is not None:
                # Preserve prior valid last.pt; failed partial step is never promoted.
                write_json(d/'failure_state_identity.json',dict(model_hash=state_hash(model.state_dict())))
            raise
        finally:
            write_json(d/'access.json',guard.summary());write_json(ART/f'{d.name}_access.json',guard.summary())


def evaluations_last(evals):return evals[-1]


def main():
    global STOP
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','smoke','train'],required=True);ap.add_argument('--stage',choices=['A','B'],default='A');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2);c=config()
    def stop(sig,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.phase=='prepare':prepare(c)
    else:train(c,args.stage,args.resume,smoke=args.phase=='smoke')


if __name__=='__main__':main()
