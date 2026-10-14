#!/usr/bin/env python3
"""Register, smoke, and execute the sole two-arm GT-free initialization probe."""
import argparse
import copy
import datetime
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
from geophy_set3d.training.senseiver_init_probe import (NAME,validate,TrainEightGuard,fitted_scale,initialize_pair,exit_event,group_norm,ObservedAdamW)
from geophy_set3d.training.ray_senseiver import build_comparison_model,sha256,tensor_identity,check_prediction
from geophy_set3d.training.selfsup import make_dataset,make_loader,move_dual_batch,objective,projection_scores,run_epoch,assert_disjoint
from geophy_set3d.training.utils import seed_everything,write_json,atomic_checkpoint,rng_state,restore_rng,environment
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from diagnose_ray_senseiver_learning import spatial,moments,geometry,chunk_check

ART=REPO/'docs/reports/artifacts'/NAME


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def paths(c,arm=None,smoke=False):
    base=Path(c['experiment']['output_root'])
    out=base/'comparisons'/NAME
    ck=base/'checkpoints'/NAME
    if smoke:out=out/'engineering_smoke';ck=ck/'engineering_smoke'
    if arm:out=out/arm;ck=ck/arm
    out.mkdir(parents=True,exist_ok=True);ck.mkdir(parents=True,exist_ok=True)
    return out,ck


def fresh(c,scale,arm):
    seed_everything(c['training']['seed'])
    model=build_comparison_model(c)
    if arm=='treatment':
        with torch.no_grad():
            model.decoder.postproc.weight.mul_(.01)
            model.decoder.postproc.bias.fill_(scale['bias_logit'])
    return model


def inventory(ds):
    samples=[ds[i] for i in range(8)]
    entries=[];lengths=[];measurements=[]
    for i,sample in enumerate(samples):
        b=dual_measurement_collate([sample]);assert_disjoint(b)
        row=dict(index=i,case_id=ds.records[i]['case_id'],projection_path=ds.records[i]['train_projection_path'],subsets={})
        for subset in ['A','B','H']:
            v=sample[subset];records=[]
            for j,station in enumerate(v['input_station_indices'].tolist()):
                valid=v['input_valid_ray'][j,0].bool().flatten()
                l=v['input_path_length_m'][j,0].flatten()[valid].double()
                y=v['input_projection_raw'][j,0].flatten()[valid].double()
                records.append(dict(station_id=station,valid_ray_flat_indices=valid.nonzero().flatten().tolist(),valid_count=int(valid.sum()),valid_zero_measurements=int((y==0).sum()),length_measurement_sha256=tensor_identity(l,y)))
                if subset!='H':lengths.append(l);measurements.append(y)
            row['subsets'][subset]=dict(stations=records,batch_identity=tensor_identity(*[v[k] for k in sorted(v) if isinstance(v[k],torch.Tensor)]))
        entries.append(row)
    return entries,fitted_scale(torch.cat(lengths),torch.cat(measurements)),samples


def checked_chunk(model,batch):
    result=chunk_check(model,batch)
    result['relative_only_original_pass']=result['passed']
    result['passed']=result['all_query_gradients_nonzero'] and all(r['finite'] and r['difference_l2']<=3e-5+2e-4*r['full_gradient_l2'] for r in result['encoder_latent_decoder_and_query_gradients']) and result['prediction_max_abs']<=2e-6
    result['criterion']='finite/nonzero query gradients; gradient L2 diff <=3e-5 +2e-4*reference L2; prediction max abs <=2e-6; raw relative-only result retained'
    if not result['passed']:raise RuntimeError('Query chunk gradient check failed')
    return result


def prepare(c,config_path):
    out,ck=paths(c)
    if (out/'registration.json').exists():raise FileExistsError('Registration already exists; no retuning')
    torch.set_num_threads(2);seed_everything(c['training']['seed']);ds=make_dataset(c)
    with TrainEightGuard(c['dataset']['root'],ck,ds.records) as guard:
        inputs,scale,samples=inventory(ds)
        checks=[]
        for sample in samples:
            batch=move_dual_batch(dual_measurement_collate([sample]),c['training']['device'])
            for subset in ['A','B']:
                target=batch[subset];mask=target['input_valid_ray'].bool()&target['view_mask'][:,:,None,None,None]
                constant=torch.full((1,1,64,64,64),scale['C_init'],device=c['training']['device'])
                with torch.no_grad():pic=project_input(constant,target)
                expected=1e6*target['input_path_length_m']*scale['chi_init']
                torch.testing.assert_close(pic[mask],expected[mask],atol=.002,rtol=2e-5)
                checks.append(dict(case_id=target['case_id'][0],subset=subset,valid_rays=int(mask.sum()),max_abs_ppm_m=float((pic[mask]-expected[mask]).abs().max()),relative_l2=float((pic[mask]-expected[mask]).double().norm()/expected[mask].double().norm())))
        seed_everything(c['training']['seed']);base=build_comparison_model(c)
        control,treatment=initialize_pair(base,scale)
        initial=dict(control=state_hash(control.state_dict()),treatment=state_hash(treatment.state_dict()),fresh=state_hash(base.state_dict()),other_parameters_elementwise_equal=True,changed_parameters=['decoder.postproc.weight','decoder.postproc.bias'],nonzero_treatment_weight=int(torch.count_nonzero(treatment.decoder.postproc.weight)),no_checkpoint_read=True)
        scale.update(subsets_used=['A','B'],H_used=False,include_valid_zero=True,method='float64 all-valid-ray LS: y=1e6*path_length*chi',conversion='geophy_set3d.reconstruction.classical.mole_to_mass, methane16.043/air28.965',projector_checks=checks,access=guard.summary())
    with gzip.open(out/'fixed_inputs.json.gz','wt') as f:json.dump(inputs,f)
    write_json(out/'scale.json',scale);write_json(out/'initialization.json',initial)
    root=Path(c['dataset']['root'])
    metadata={str(p):sha256(p) for p in [root/'source_index.jsonl',root/'geometry/train_geometry.npz',root/'train_statistics.json']}
    projections={str(root/r['train_projection_path']):sha256(root/r['train_projection_path']) for r in ds.records[:8]}
    sources=sorted(set(subprocess.check_output(['git','ls-files','src/**/*.py','scripts/diagnose_ray_senseiver_learning.py'],cwd=REPO,text=True).splitlines()+['scripts/run_senseiver_init_probe.py',str(Path(config_path)), 'src/geophy_set3d/training/senseiver_init_probe.py']))
    manifest=dict(sha256={p:sha256(REPO/p) for p in sources})
    write_json(out/'source_manifest.json',manifest)
    reg=dict(at=now(),config=c,metadata_sha256=metadata,projection_sha256=projections,artifact_sha256={p:sha256(out/p) for p in ['fixed_inputs.json.gz','scale.json','initialization.json','source_manifest.json']},case_order=[r['case_id'] for r in ds.records[:8]],batch_groups=[list(range(i,i+2)) for i in range(0,8,2)],initialization_hashes=initial,
             boundaries='Only first8 train/Pool A; no GT64/128, Val/test, Pool B, or foreign checkpoint. H excluded from network and scale, included in original loss. Fixed-subset H is not independent generalization validation.',
             exit_rule_interpretation='Strict after epoch5 => eligible health at10,15,...; after20 =>25,30,...; consecutive eligible checks; window uses post-epoch full-subset H; earliest collapse15 / failure30.',
             go='Treatment last3 health H each<.90; no engineering/collapse; at least one fixed diagnostic pair relative MA >=1e-3 (operational identifiable-case criterion); >=2 diagnostic cases relative MA >=1e-3 with eps1e-12; all3 gradient/update groups positive. No automatic 300-epoch launch.',
             smoke='Separate temporary model/optimizer; each formal arm reinitialized from seed after smoke; no warm start',
             resume='Each arm only own last.pt including model/optimizer/scheduler/RNG/update count/exit state; epoch0.pt archival only; no best selection',
             engineering_error='Stop and save error and last valid atomic checkpoint; never silently modify scientific run')
    write_json(out/'registration.json',reg)
    ART.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.is_file(): (ART/p.name).write_bytes(p.read_bytes())
    print(json.dumps(dict(scale=scale,initial=initial),indent=2),flush=True)


def verify(c):
    out,ck=paths(c);reg=json.loads((out/'registration.json').read_text())
    assert reg['config']==c
    for p,h in {**reg['metadata_sha256'],**reg['projection_sha256']}.items():assert sha256(p)==h,p
    for p,h in reg['artifact_sha256'].items():assert sha256(out/p)==h,p
    for p,h in json.loads((out/'source_manifest.json').read_text())['sha256'].items():assert sha256(REPO/p)==h,p
    for name in ['registration.json','scale.json','initialization.json','source_manifest.json','fixed_inputs.json.gz']:
        assert subprocess.check_output(['git','show',f'HEAD:docs/reports/artifacts/{NAME}/{name}'],cwd=REPO)==(out/name).read_bytes(),name
    return reg,json.loads((out/'scale.json').read_text())


def logits_summary(values):
    x=torch.cat(values).double()
    return dict(**moments(x),fraction_below_minus12=float((x<-12).double().mean()),sigmoid_derivative=moments(x.sigmoid()*(1-x.sigmoid())))


def evaluate(model,loader,c,capture=False):
    logits=[];handle=None
    if capture:handle=model.decoder.postproc.register_forward_hook(lambda m,a,y:logits.append(y.detach().flatten().cpu()))
    try:row=run_epoch(model,loader,c)
    finally:
        if handle:handle.remove()
    return row,logits_summary(logits) if capture else None


def health(model,ds,full,logits,c):
    named=list(model.named_parameters());rows=[];predictions=[]
    for i in c['probe']['diagnostic_case_indices']:
        b=move_dual_batch(dual_measurement_collate([ds[i]]),c['training']['device'])
        model.train();loss,parts,volumes,proj=objective(model,b)
        grads=torch.autograd.grad(loss,tuple(p for n,p in named))
        norms=group_norm(named,grads)
        assert all(torch.isfinite(g).all() for g in grads) and all(v>0 for v in norms.values())
        pred=volumes[0].detach();predictions.append(pred.cpu())
        row=dict(index=i,case_id=b['A']['case_id'][0],gradient_l2=norms,spatial=spatial(pred))
        del loss,parts,volumes,proj,grads
        model.eval();zero=dict(b['A'])
        for k in ['input_projection_raw','input_projection']:zero[k]=torch.zeros_like(zero[k])
        assert all(torch.equal(zero[k],v) for k,v in b['A'].items() if isinstance(v,torch.Tensor) and k not in ['input_projection_raw','input_projection'])
        with torch.no_grad():normal=model(b['A']);blank=model(zero)
        absolute=float((normal-blank).abs().double().mean());denom=float(normal.abs().double().mean())
        row['measurement_dependence']=dict(absolute_MA=absolute,normal_mean_abs=denom,epsilon=1e-12,relative_MA=absolute/(denom+1e-12),geometry_path_masks_unchanged=True)
        rows.append(row)
    chunk=checked_chunk(model,move_dual_batch(dual_measurement_collate([ds[0]]),c['training']['device'])['A'])
    differences=[dict(a=i,b=j,absolute_MA=float((predictions[i]-predictions[j]).abs().double().mean()),relative_MA=float((predictions[i]-predictions[j]).abs().double().mean()/(predictions[i].abs().double().mean()+1e-12))) for i in range(3) for j in range(i+1,3)]
    return dict(full_fixed_subset=full,logits=logits,diagnostic_cases=rows,case_differences=differences,chunk_check=chunk,diagnostic_optimizer_steps=0)


def smoke(c):
    reg,scale=verify(c);ds=make_dataset(c);results={};torch.set_num_threads(2)
    for arm in ['control','treatment']:
        out,ck=paths(c,arm,smoke=True)
        if (out/'result.json').exists():raise FileExistsError('Smoke already done')
        with TrainEightGuard(c['dataset']['root'],ck,ds.records) as guard:
            model=fresh(c,scale,arm);initial=state_hash(model.state_dict());assert initial==reg['initialization_hashes'][arm]
            b=move_dual_batch(dual_measurement_collate([ds[0],ds[1]]),c['training']['device'])
            geometry_checks=[geometry(b[k]) for k in ['A','B']]
            assert all(r['valid_rays']==r['packed_real_rays'] and r['all_tokens_finite'] and r['entry_exit_in_domain'] for r in geometry_checks)
            grid=model.query_coordinates_m.reshape(64,64,64,3)
            torch.testing.assert_close(grid[0,32,0],grid.new_tensor([0,32*.6/63,0]),atol=1e-7,rtol=1e-6)
            opt=ObservedAdamW(model,lr=2e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
            generator=torch.Generator().manual_seed(c['training']['seed'])
            seen=[];hook=model.register_forward_pre_hook(lambda m,a:seen.append(id(a[0])))
            model.train();loss,parts,volumes,proj=objective(model,b);hook.remove()
            assert seen==[id(b['A']),id(b['B'])] and id(b['H']) not in seen
            for v,k in zip(volumes,['A','B']):check_prediction(v,b[k])
            loss.backward();pre=group_norm(opt.named,[p.grad for n,p in opt.named]);assert all(v>0 for v in pre.values())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);sch.step()
            payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=rng_state(generator),updates=opt.updates)
            atomic_checkpoint(ck/'last.pt',payload);expected=state_hash(model.state_dict())
            saved=torch.load(ck/'last.pt',map_location=c['training']['device'],weights_only=False)
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sch.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],generator);opt.updates=saved['updates']
            assert state_hash(model.state_dict())==expected and opt.updates==1 and sch.last_epoch==1
            assert all(int(v['step'])==1 for v in opt.state.values())
            chunk=checked_chunk(model,b['A'])
            results[arm]=dict(passed=True,initial_hash=initial,smoke_steps=1,pre_clip_gradient_l2=pre,updates=opt.observations,geometry_checks=geometry_checks,chunk_check=chunk,axis_XYZ=True,H_not_network=True,full_query_shape=[2,1,64,64,64],restore_state_verified=True,access=guard.summary(),fresh_formal_required=True)
        write_json(out/'result.json',results[arm]);del model,opt,sch,loss,parts,volumes,proj,saved,payload
        torch.cuda.empty_cache()
    out,_=paths(c);write_json(out/'engineering_smoke.json',results)
    (ART/'engineering_smoke.json').write_bytes((out/'engineering_smoke.json').read_bytes())
    print(json.dumps(results,indent=2),flush=True)


def train(c,arm,resume):
    reg,scale=verify(c);parent,_=paths(c);out,ck=paths(c,arm);ds=make_dataset(c)
    assert all(r['passed'] for r in json.loads((parent/'engineering_smoke.json').read_text()).values())
    if (out/'complete.json').exists():raise FileExistsError('Arm already stopped/completed')
    if (ck/'last.pt').exists() and not resume:raise FileExistsError('Explicit own-last resume required')
    torch.set_num_threads(2);began=time.perf_counter()
    with TrainEightGuard(c['dataset']['root'],ck,ds.records) as guard:
        model=fresh(c,scale,arm);initial=state_hash(model.state_dict());assert initial==reg['initialization_hashes'][arm]
        optimizer=ObservedAdamW(model,lr=2e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        generator=torch.Generator().manual_seed(c['training']['seed'])
        loader=make_loader(Subset(ds,list(range(8))),c,False,generator)
        env=environment(c);env.update(arm=arm,at=now(),initial_hash=initial,scientific_config=c,TF32=False)
        write_json(out/'environment.json',env)
        history=[];healths=[];events=[];state={};start=0
        def save(epoch):
            atomic_checkpoint(ck/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator),epoch=epoch,updates=optimizer.updates,config=c,arm=arm,initial_hash=initial,registration_sha256=sha256(parent/'registration.json'),source_manifest_sha256=sha256(parent/'source_manifest.json'),history=history,health=healths,events=events,exit_state=state,volumetric_gt_used=False))
        if resume:
            saved=torch.load(ck/'last.pt',map_location=c['training']['device'],weights_only=False)
            assert saved['config']==c and saved['arm']==arm and saved['initial_hash']==initial and saved['registration_sha256']==sha256(parent/'registration.json') and saved['source_manifest_sha256']==sha256(parent/'source_manifest.json')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],generator);optimizer.updates=saved['updates'];start=saved['epoch']
            history=saved['history'];healths=saved['health'];events=saved['events'];state=saved['exit_state']
            assert optimizer.updates==start*4 and scheduler.last_epoch==start
            if state.get('reason'):raise RuntimeError('Stopped scientific arm cannot be resumed')
        else:
            full,logits=evaluate(model,loader,c,True);h=health(model,ds,full,logits,c);h.update(epoch=0,updates=0)
            healths.append(h);save(0)
            (ck/'epoch0.pt').write_bytes((ck/'last.pt').read_bytes())
        try:
            for epoch in range(start+1,41):
                ds.set_epoch(0);optimizer.observations=[];torch.cuda.reset_peak_memory_stats(c['training']['device']);t=time.perf_counter();lr=optimizer.param_groups[0]['lr']
                gpu_start=torch.cuda.Event(enable_timing=True);gpu_end=torch.cuda.Event(enable_timing=True);gpu_start.record()
                tr=run_epoch(model,loader,c,optimizer)
                assert optimizer.updates==4*epoch
                full,logits=evaluate(model,loader,c,epoch%5==0)
                assert tr['case_station_digest']==full['case_station_digest']==healths[0]['full_fixed_subset']['case_station_digest']
                scheduler.step();row=dict(epoch=epoch,updates=optimizer.updates,train=tr,eval=full,learning_rate=lr,update_observations=optimizer.observations,seconds_before_health=time.perf_counter()-t)
                history.append(row)
                if epoch%5==0:
                    h=health(model,ds,full,logits,c);h.update(epoch=epoch,updates=optimizer.updates)
                    healths.append(h);state=exit_event(epoch,history,h,scale['C_init'],state);events.append(state)
                gpu_end.record();torch.cuda.synchronize(c['training']['device']);row.update(seconds=time.perf_counter()-t,gpu_event_elapsed_ms=gpu_start.elapsed_time(gpu_end),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(c['training']['device']))
                save(epoch)
                write_json(out/'history.json',history);write_json(out/'health.json',healths);write_json(out/'events.json',events);write_json(out/'access.json',guard.summary())
                print(f"{arm} epoch={epoch}/40 updates={optimizer.updates} H={full['measurement']['heldout']['relative_l1']:.8f} mean={full['output']['mean']:.8g} reason={state.get('reason')} seconds={row['seconds']:.2f}",flush=True)
                if state.get('reason'):break
        except Exception as exc:
            write_json(out/'engineering_error.json',dict(at=now(),error=repr(exc),last_valid_checkpoint=str(ck/'last.pt'),attempted_updates=optimizer.updates,access=guard.summary(),no_silent_fix=True))
            raise
        summary=dict(arm=arm,completed_epochs=history[-1]['epoch'],updates=optimizer.updates,reason=state['reason'],at=now(),elapsed_this_process_seconds=time.perf_counter()-began,total_epoch_seconds=sum(r['seconds'] for r in history),peak_gpu_allocated_bytes=max(r['peak_gpu_allocated_bytes'] for r in history),last_checkpoint=str(ck/'last.pt'),last_sha256=sha256(ck/'last.pt'),epoch0_sha256=sha256(ck/'epoch0.pt'),source_manifest_sha256=sha256(parent/'source_manifest.json'),access=guard.summary(),initial_hash=initial,no_GT=True,no_formal_training_started=True)
        assert not any(summary['access'][k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
        write_json(out/'complete.json',summary);print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/ray_senseiver_init_probe_v1.yaml');ap.add_argument('--phase',choices=['prepare','smoke','train'],required=True);ap.add_argument('--arm',choices=['control','treatment']);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    c=validate(yaml.safe_load(Path(a.config).read_text()))
    if a.phase=='prepare':prepare(c,a.config)
    elif a.phase=='smoke':smoke(c)
    else:
        if not a.arm:ap.error('--arm required')
        train(c,a.arm,a.resume)


if __name__=='__main__':main()
