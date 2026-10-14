#!/usr/bin/env python3
"""One registered structural repair, original measurement objective and monitoring."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import torch
import yaml
from torch.utils.data import Subset
from geophy_set3d.training.senseiver_single_repair import NAME,validate,exit_event,verify_resume
from geophy_set3d.training.senseiver_readout import ProbeGuard
from geophy_set3d.models.ray_senseiver_latent_residual import add_latent_residual
from geophy_set3d.training.senseiver_consistency_probe import objective,run_epoch
from geophy_set3d.training.senseiver_init_probe import ObservedAdamW,group_norm
from geophy_set3d.training.selfsup import make_dataset,make_loader,move_dual_batch
from geophy_set3d.training.ray_senseiver import sha256
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import write_json,atomic_checkpoint,rng_state,restore_rng,environment
from geophy_set3d.data.measurement_only import dual_measurement_collate
from run_senseiver_init_probe import fresh,checked_chunk,now
from run_senseiver_consistency_probe import fixed_samples,evaluate
from diagnose_ray_senseiver_learning import spatial,geometry
PRIOR=REPO/'docs/reports/artifacts/ray_senseiver_init_probe_v1'
ART=REPO/'docs/reports/artifacts'/NAME
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons')/NAME
CK=Path('/data/djc/GeoPhy-Set3D/outputs/checkpoints')/NAME


def paths(arm=None,smoke=False):
    out,ck=OUT,CK
    if smoke:out=out/'engineering_smoke';ck=ck/'engineering_smoke'
    if arm:out=out/arm;ck=ck/arm
    out.mkdir(parents=True,exist_ok=True);ck.mkdir(parents=True,exist_ok=True)
    return out,ck


def pair(c,scale):
    control=fresh(c,scale,'treatment');treatment=add_latent_residual(control)
    cs=control.state_dict();ts=treatment.state_dict()
    assert all(torch.equal(v,ts[k]) for k,v in cs.items())
    assert set(ts)-set(cs)=={'decoder.latent_condition.weight'}
    assert not torch.count_nonzero(ts['decoder.latent_condition.weight'])
    return control,treatment


def prepare(c,path):
    out,ck=paths();assert not (out/'reconstruction_registration.json').exists()
    assert json.loads((out/'readout_complete.json').read_text())['readable']==dict(ridge=True,query=True)
    scale=json.loads((PRIOR/'scale.json').read_text())
    with gzip.open(PRIOR/'fixed_inputs.json.gz','rt') as f:inventory=json.load(f)
    ds=make_dataset(c);g=ProbeGuard(c['dataset']['root'],ck,ds.records,8)
    try:
        with g:
            fixed_samples(ds,inventory);a,b=pair(c,scale)
            initial=dict(control=state_hash(a.state_dict()),treatment=state_hash(b.state_dict()),shared_parameters_elementwise_equal=True,
                control_parameters=sum(p.numel() for p in a.parameters()),treatment_parameters=sum(p.numel() for p in b.parameters()),
                added_parameters={'decoder.latent_condition.weight':dict(shape=[64,64],initialization='all zeros',sha256=state_hash({'weight':b.decoder.latent_condition.weight}))},
                no_checkpoint_loaded=True,no_A_or_smoke_warm_start=True)
            assert initial['control']==json.loads((PRIOR/'initialization.json').read_text())['treatment']
            old=json.loads((PRIOR/'registration.json').read_text())
            sources=subprocess.check_output(['git','ls-files','src/**/*.py'],cwd=REPO,text=True).splitlines()+[
                'src/geophy_set3d/models/ray_senseiver_latent_residual.py','src/geophy_set3d/training/senseiver_single_repair.py',
                'scripts/run_senseiver_single_repair.py','tests/test_senseiver_repair_training.py',str(path),
                'scripts/run_senseiver_init_probe.py','scripts/run_senseiver_consistency_probe.py','scripts/diagnose_ray_senseiver_learning.py',
                'docs/ray_senseiver_quick_repair_v1_protocol_zh.md','docs/ray_senseiver_quick_repair_v1_design_zh.md']
            manifest=dict(sha256={p:sha256(REPO/p) for p in sorted(set(sources))})
            history={str(p.relative_to(REPO)):sha256(p) for name in ['signal_path','consistency_probe','relative_probe','init_probe'] for p in (REPO/'docs/reports').glob(f'ray_senseiver_{name}_v1_zh.md')}
            for name in ['readout_registration.json','readout_complete.json']:
                history[str((ART/name).relative_to(REPO))]=sha256(ART/name)
            write_json(out/'reconstruction_initialization.json',initial);write_json(out/'reconstruction_source_manifest.json',manifest)
            for name in ['scale.json','fixed_inputs.json.gz']:(out/name).write_bytes((PRIOR/name).read_bytes())
            reg=dict(at=now(),config=c,initialization=initial,case_order=old['case_order'],batch_groups=old['batch_groups'],
                metadata_sha256=old['metadata_sha256'],projection_sha256=old['projection_sha256'],history_sha256=history,
                source_manifest_sha256=sha256(out/'reconstruction_source_manifest.json'),
                artifacts_sha256={n:sha256(out/n) for n in ['reconstruction_initialization.json','scale.json','fixed_inputs.json.gz']},
                max_epochs=80,max_updates=320,intervention='Only decoder global latent residual; same objective relative=consistency=0',
                go='Last3 health H L1<.90; >=2 dependence cases>=.001; pair variation>=.001; all gradients/updates valid; no anomalies',access=g.summary())
            write_json(out/'reconstruction_registration.json',reg)
    finally:write_json(out/'reconstruction_prepare_access.json',g.summary())
    for n in ['reconstruction_registration.json','reconstruction_initialization.json','reconstruction_source_manifest.json','scale.json','fixed_inputs.json.gz','reconstruction_prepare_access.json']:
        (ART/n).write_bytes((out/n).read_bytes())
    print(json.dumps(initial,indent=2),flush=True)


def verify(c):
    r=json.loads((OUT/'reconstruction_registration.json').read_text());assert r['config']==c
    for n in ['reconstruction_registration.json','reconstruction_source_manifest.json','reconstruction_initialization.json','scale.json','fixed_inputs.json.gz']:
        assert subprocess.check_output(['git','show',f'HEAD:docs/reports/artifacts/{NAME}/{n}'],cwd=REPO)==(OUT/n).read_bytes()
    assert sha256(OUT/'reconstruction_source_manifest.json')==r['source_manifest_sha256']
    for n,h in r['artifacts_sha256'].items():assert sha256(OUT/n)==h
    for p,h in {**json.loads((OUT/'reconstruction_source_manifest.json').read_text())['sha256'],**r['history_sha256']}.items():assert sha256(REPO/p)==h,p
    for p,h in {**r['metadata_sha256'],**r['projection_sha256']}.items():assert sha256(p)==h,p
    with gzip.open(OUT/'fixed_inputs.json.gz','rt') as f:inventory=json.load(f)
    return r,json.loads((OUT/'scale.json').read_text()),inventory


def health(model,ds,full,logits,c):
    rows=[];predictions=[];named=list(model.named_parameters())
    for i in [0,1,2]:
        b=move_dual_batch(dual_measurement_collate([ds[i]]),c['training']['device']);model.train()
        result=objective(model,b,0.);grads=torch.autograd.grad(result[0],tuple(p for n,p in named))
        assert all(torch.isfinite(v).all() for v in grads)
        norms=group_norm(named,grads);assert all(v>0 for v in norms.values())
        pred=result[2][0].detach();predictions.append(pred.cpu())
        row=dict(index=i,case_id=b['A']['case_id'][0],gradient_l2=norms,spatial=spatial(pred))
        del result,grads
        model.eval();zero=dict(b['A'])
        for k in ['input_projection_raw','input_projection']:zero[k]=torch.zeros_like(zero[k])
        assert all(zero[k] is v for k,v in b['A'].items() if k not in ['input_projection_raw','input_projection'])
        with torch.no_grad():normal=model(b['A']);blank=model(zero)
        absolute=float((normal-blank).abs().double().mean());denom=float(normal.abs().double().mean())
        row['measurement_dependence']=dict(absolute_MA=absolute,normal_mean_abs=denom,epsilon=1e-12,relative_MA=absolute/(denom+1e-12),geometry_path_masks_unchanged=True)
        rows.append(row)
    chunk=checked_chunk(model,move_dual_batch(dual_measurement_collate([ds[0]]),c['training']['device'])['A'])
    differences=[dict(a=i,b=j,absolute_MA=float((predictions[i]-predictions[j]).abs().double().mean()),relative_MA=float((predictions[i]-predictions[j]).abs().double().mean()/(predictions[i].abs().double().mean()+1e-12))) for i in range(3) for j in range(i+1,3)]
    assert all(p.grad is None for p in model.parameters())
    return dict(full_fixed_subset=full,logits=logits,diagnostic_cases=rows,case_differences=differences,chunk_check=chunk,optimizer_steps_diagnosis=0)


def smoke(c):
    ds=make_dataset(c);g=ProbeGuard(c['dataset']['root'],CK/'engineering_smoke',ds.records,8);results={}
    try:
        with g:
            reg,scale,inventory=verify(c);a,b=pair(c,scale)
            batch=move_dual_batch(dual_measurement_collate([ds[0],ds[1]]),c['training']['device'])
            a.eval();b.eval()
            with torch.no_grad():pa=a(batch['A']);pb=b(batch['A'])
            torch.testing.assert_close(pa,pb,atol=0,rtol=0)
            for arm,model in [('control',a),('treatment',b)]:
                model.train();seen=[];hook=model.register_forward_pre_hook(lambda m,args:seen.append(id(args[0])))
                try:result=objective(model,batch,0.)
                finally:hook.remove()
                assert seen==[id(batch['A']),id(batch['B'])] and result[2][0].shape==(2,1,64,64,64)
                opt=ObservedAdamW(model,lr=2e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
                result[0].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);sch.step()
                out,ck=paths(arm,True);gen=torch.Generator().manual_seed(20261011)
                atomic_checkpoint(ck/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=rng_state(gen)))
                saved=torch.load(ck/'last.pt',map_location=c['training']['device'],weights_only=False)
                expected=state_hash(model.state_dict());model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sch.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],gen)
                assert state_hash(model.state_dict())==expected and sch.last_epoch==1 and all(int(v['step'])==1 for v in opt.state.values())
                geo=[geometry(batch[k]) for k in ['A','B']];assert all(v['valid_rays']==v['packed_real_rays'] and v['entry_exit_in_domain'] for v in geo)
                grid=model.query_coordinates_m.reshape(64,64,64,3)
                for axis in range(3):
                    index=[0,0,0];index[axis]=32;target=grid.new_zeros(3);target[axis]=32*.6/63
                    torch.testing.assert_close(grid[tuple(index)],target,atol=1e-7,rtol=1e-6)
                chunk=checked_chunk(model,batch['A'])
                results[arm]=dict(passed=True,initial_hash=reg['initialization'][arm],shared_initial_predictions_bitwise_equal=True,
                    optimizer_steps=1,updates=opt.observations,chunk_check=chunk,geometry=geo,XYZ=True,H_excluded=True,resume_state_roundtrip=True)
                del result,opt,sch,saved
            results['access']=g.summary()
    finally:write_json(OUT/'smoke_access.json',g.summary())
    write_json(OUT/'engineering_smoke.json',results);(ART/'engineering_smoke.json').write_bytes((OUT/'engineering_smoke.json').read_bytes())
    print('Independent engineering smoke passed for both arms; formal models will be fresh.',flush=True)


def train(c,arm,resume):
    out,ck=paths(arm);assert not (out/'complete.json').exists(),'Scientifically stopped arm cannot resume'
    assert resume or not (ck/'last.pt').exists(),'Explicit own-last resume required'
    smoke_result=json.loads((OUT/'engineering_smoke.json').read_text());assert all(smoke_result[a]['passed'] for a in ['control','treatment'])
    ds=make_dataset(c);g=ProbeGuard(c['dataset']['root'],ck,ds.records,8);started=time.perf_counter()
    try:
        with g:
            reg,scale,inventory=verify(c);fixed_samples(ds,inventory)
            control,treatment=pair(c,scale);model=control if arm=='control' else treatment
            del control,treatment
            initial=state_hash(model.state_dict());assert initial==reg['initialization'][arm]
            optimizer=ObservedAdamW(model,lr=2e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
            generator=torch.Generator().manual_seed(20261011);loader=make_loader(Subset(ds,list(range(8))),c,False,generator)
            write_json(out/'environment.json',dict(**environment(c),arm=arm,initial_hash=initial,config=c,at=now()))
            history=[];healths=[];events=[];state={};start=0
            def save(epoch):
                atomic_checkpoint(ck/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator),epoch=epoch,updates=optimizer.updates,config=c,arm=arm,relative_l1=0.,consistency_weight=0.,initial_hash=initial,
                    registration_sha256=sha256(OUT/'reconstruction_registration.json'),source_manifest_sha256=sha256(OUT/'reconstruction_source_manifest.json'),history=history,health=healths,events=events,exit_state=state,volumetric_gt_used=False))
            try:
                if resume:
                    saved=torch.load(ck/'last.pt',map_location=c['training']['device'],weights_only=False)
                    verify_resume(saved,c,arm,initial,sha256(OUT/'reconstruction_registration.json'),sha256(OUT/'reconstruction_source_manifest.json'))
                    model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],generator)
                    optimizer.updates=saved['updates'];start=saved['epoch'];history=saved['history'];healths=saved['health'];events=saved['events'];state=saved['exit_state']
                else:
                    full,logits=evaluate(model,loader,c,0.,True);h=health(model,ds,full,logits,c);h.update(epoch=0,updates=0);healths.append(h)
                    save(0);(ck/'epoch0.pt').write_bytes((ck/'last.pt').read_bytes())
                for epoch in range(start+1,81):
                    ds.set_epoch(0);optimizer.observations=[];torch.cuda.reset_peak_memory_stats(c['training']['device']);t=time.perf_counter();lr=optimizer.param_groups[0]['lr']
                    tr=run_epoch(model,loader,c,0.,optimizer);assert optimizer.updates==4*epoch
                    full,logits=evaluate(model,loader,c,0.,epoch%5==0)
                    assert tr['case_station_digest']==full['case_station_digest']==healths[0]['full_fixed_subset']['case_station_digest']
                    scheduler.step();row=dict(epoch=epoch,updates=optimizer.updates,train=tr,eval=full,learning_rate=lr,update_observations=optimizer.observations);history.append(row)
                    if epoch%5==0:
                        h=health(model,ds,full,logits,c);h.update(epoch=epoch,updates=optimizer.updates);healths.append(h)
                        state=exit_event(epoch,history,h,scale['C_init'],state);events.append(state)
                    torch.cuda.synchronize(c['training']['device']);row.update(seconds=time.perf_counter()-t,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(c['training']['device']))
                    save(epoch)
                    for name,value in [('history',history),('health',healths),('events',events),('access',g.summary())]:write_json(out/(name+'.json'),value)
                    print(f"{arm} epoch={epoch}/80 updates={optimizer.updates} H_L1={full['measurement']['heldout']['relative_l1']:.8f} H_L2={full['measurement']['heldout']['relative_l2']:.8f} mean={full['output']['mean']:.8g} reason={state.get('reason')} seconds={row['seconds']:.2f}",flush=True)
                    if state.get('reason'):break
            except BaseException as exc:
                write_json(out/'engineering_error.json',dict(at=now(),error=repr(exc),last_valid_checkpoint=str(ck/'last.pt') if (ck/'last.pt').exists() else None,attempted_updates=optimizer.updates,access=g.summary(),no_silent_fix=True));raise
            assert state.get('reason') and all(p.grad is None for p in model.parameters())
            verify(c)
            summary=dict(arm=arm,epochs=history[-1]['epoch'],updates=optimizer.updates,reason=state['reason'],at=now(),initial_hash=initial,last_checkpoint=str(ck/'last.pt'),
                epoch0_sha256=sha256(ck/'epoch0.pt'),last_sha256=sha256(ck/'last.pt'),model_state_hash=state_hash(model.state_dict()),
                relative_l1=0.,consistency=0.,seconds=time.perf_counter()-started,total_epoch_seconds=sum(r['seconds'] for r in history),peak_gpu_allocated_bytes=max(r['peak_gpu_allocated_bytes'] for r in history),
                grad_buffers_none=True,access=g.summary(),source_history_data_hashes_unchanged=True)
            assert not any(g.summary()[k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
            write_json(out/'complete.json',summary);print(json.dumps(summary,indent=2),flush=True)
    finally:write_json(out/'access_exit.json',g.summary())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','smoke','train'],required=True);p.add_argument('--arm',choices=['control','treatment']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    path='configs/ray_senseiver_single_repair_v1.yaml';c=validate(yaml.safe_load((REPO/path).read_text()));torch.set_num_threads(2)
    if a.phase=='prepare':prepare(c,path)
    elif a.phase=='smoke':smoke(c)
    else:
        if not a.arm:p.error('--arm required')
        train(c,a.arm,a.resume)
