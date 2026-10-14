#!/usr/bin/env python3
"""One registered relative-L1 on/off study; no historical run is resumed."""
import argparse
import copy
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
from geophy_set3d.training.senseiver_relative_probe import NAME,validate,objective,from_projections,exit_event,run_epoch,verify_resume
from geophy_set3d.training.senseiver_init_probe import TrainEightGuard,ObservedAdamW,group_norm
from geophy_set3d.training.selfsup import make_dataset,make_loader,move_dual_batch,assert_disjoint
from geophy_set3d.training.ray_senseiver import sha256,tensor_identity
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything,write_json,atomic_checkpoint,rng_state,restore_rng,environment
from geophy_set3d.data.measurement_only import dual_measurement_collate
from run_senseiver_init_probe import fresh,checked_chunk,logits_summary,now
from diagnose_ray_senseiver_learning import spatial,geometry
from diagnose_senseiver_relative_probe import gradient_split,amplitude_inventory,grouped_constant_contributions,float64_scalar_probe,validate_group_sum

PRIOR=REPO/'docs/reports/artifacts/ray_senseiver_init_probe_v1'
ART=REPO/'docs/reports/artifacts'/NAME


def paths(c,arm=None,smoke=False):
    base=Path(c['experiment']['output_root']);out=base/'comparisons'/NAME;ck=base/'checkpoints'/NAME
    if smoke:out=out/'engineering_smoke';ck=ck/'engineering_smoke'
    if arm:out=out/arm;ck=ck/arm
    out.mkdir(parents=True,exist_ok=True);ck.mkdir(parents=True,exist_ok=True)
    return out,ck


def fixed_samples(ds,inventory):
    samples=[]
    for i in range(8):
        s=ds[i];assert ds.records[i]['case_id']==inventory[i]['case_id']
        assert_disjoint(dual_measurement_collate([s]))
        for k in ['A','B','H']:
            assert tensor_identity(*[s[k][v] for v in sorted(s[k]) if isinstance(s[k][v],torch.Tensor)])==inventory[i]['subsets'][k]['batch_identity']
        samples.append(s)
    return samples


def prepare(c,config_path):
    out,ck=paths(c)
    if (out/'registration.json').exists():raise FileExistsError('No repeated registration/search')
    torch.set_num_threads(2);ds=make_dataset(c)
    with gzip.open(PRIOR/'fixed_inputs.json.gz','rt') as f:inventory=json.load(f)
    scale=json.loads((PRIOR/'scale.json').read_text());priorinit=json.loads((PRIOR/'initialization.json').read_text())
    with TrainEightGuard(c['dataset']['root'],ck,ds.records) as guard:
        fixed_samples(ds,inventory)
        first=fresh(c,scale,'treatment');second=fresh(c,scale,'treatment')
        assert all(torch.equal(x,y) for x,y in zip(first.state_dict().values(),second.state_dict().values()))
        initial=state_hash(first.state_dict());assert initial==priorinit['treatment']
        info=dict(control=initial,treatment=initial,all_parameters_elementwise_equal=True,prior_registered_treatment_hash=initial,no_checkpoint_read=True,scale_reestimated=False,access=guard.summary())
    for name in ['scale.json','fixed_inputs.json.gz']:(out/name).write_bytes((PRIOR/name).read_bytes())
    write_json(out/'initialization.json',info)
    sources=sorted(set(subprocess.check_output(['git','ls-files','src/**/*.py'],cwd=REPO,text=True).splitlines()+['scripts/run_senseiver_init_probe.py','scripts/diagnose_ray_senseiver_learning.py','scripts/run_senseiver_relative_probe.py','scripts/diagnose_senseiver_relative_probe.py','src/geophy_set3d/training/senseiver_relative_probe.py',str(config_path),'docs/ray_senseiver_relative_probe_v1_protocol_zh.md','tests/test_senseiver_relative_probe.py']))
    write_json(out/'source_manifest.json',dict(sha256={p:sha256(REPO/p) for p in sources}))
    old=json.loads((PRIOR/'registration.json').read_text())
    priorfiles=[PRIOR/p for p in ['scale.json','fixed_inputs.json.gz','initialization.json','source_manifest.json','epoch_scalars.csv','registration.json']]
    priorfiles += [REPO/'docs/reports/ray_senseiver_learning_diagnostic_zh.md',REPO/'docs/reports/ray_senseiver_init_probe_v1_zh.md',REPO/'docs/ray_senseiver_init_probe_v1_protocol_zh.md',REPO/'configs/ray_senseiver_init_probe_v1.yaml']
    reg=dict(at=now(),config=c,initial_hash=initial,case_order=old['case_order'],batch_groups=old['batch_groups'],
             reused_prior_sha256={str(p):sha256(p) for p in priorfiles},metadata_sha256=old['metadata_sha256'],projection_sha256=old['projection_sha256'],
             artifacts_sha256={n:sha256(out/n) for n in ['scale.json','fixed_inputs.json.gz','initialization.json','source_manifest.json']},
             only_intervention='Internal measurement_discrepancy relative_l1 .1 vs0 in hold/cross/observed. Outer observed .1 and consistency .05 unchanged; full case sum/sum relative denominator unchanged.',
             stage1='Fresh scale-initialized model and 6 nontraining constant levels on all fixed8 cases; forward/autograd only, no optimizer. Group amplitudes are observations, not true spatial background/plume labels.',
             stopping='Strict after5/20 => eligible10/25. Collapse unchanged. Failure requires last5 H L1>=.98 AND L1 improvement<1% AND L2 improvement<1% AND <2 of fixed case0/1/2 dependence>=1e-3; consecutive2 checks.',
             monitoring='All arms same H/cross/obs L1/L2; relative diagnostics computed even when excluded from objective. Last3 health H<.90 GO; case pair relative MA>=1e-3 as prior operational criterion; >=2 dependence cases; all groups gradient/update valid. Partial iff SmoothL1 or L2 improves but GO fails.',
             boundary='Only first8 train PoolA; no GT64/128, Val/test, PoolB, foreign checkpoint, warm start, search, or automatic full training. H in loss is not independent validation.',
             resume='Own last.pt only, complete state and exit counters; never scientifically stopped arm')
    write_json(out/'registration.json',reg);ART.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.is_file():(ART/p.name).write_bytes(p.read_bytes())
    print(json.dumps(info,indent=2),flush=True)


def verify(c):
    out,_=paths(c);reg=json.loads((out/'registration.json').read_text());assert reg['config']==c
    for p,h in {**reg['reused_prior_sha256'],**reg['metadata_sha256'],**reg['projection_sha256']}.items():assert sha256(p)==h,p
    for n,h in reg['artifacts_sha256'].items():assert sha256(out/n)==h,n
    for p,h in json.loads((out/'source_manifest.json').read_text())['sha256'].items():assert sha256(REPO/p)==h,p
    for n in ['registration.json','source_manifest.json','initialization.json','scale.json','fixed_inputs.json.gz']:
        assert subprocess.check_output(['git','show',f'HEAD:docs/reports/artifacts/{NAME}/{n}'],cwd=REPO)==(out/n).read_bytes()
    with gzip.open(out/'fixed_inputs.json.gz','rt') as f:inventory=json.load(f)
    return reg,json.loads((out/'scale.json').read_text()),inventory


def diagnose(c):
    reg,scale,inventory=verify(c);out,ck=paths(c);target=out/'gradient_diagnosis.json'
    if target.exists():raise FileExistsError('Single preregistered diagnostic only')
    torch.set_num_threads(2);ds=make_dataset(c);began=time.perf_counter();results=[]
    with TrainEightGuard(c['dataset']['root'],ck,ds.records) as guard:
        samples=fixed_samples(ds,inventory);model=fresh(c,scale,'treatment');initial=state_hash(model.state_dict());assert initial==reg['initial_hash']
        model.train()
        for i,sample in enumerate(samples):
            batch=move_dual_batch(dual_measurement_collate([sample]),c['training']['device'])
            result=objective(model,batch,.1)
            split=gradient_split(model,result,batch,check_reference=True)
            row=dict(index=i,case_id=batch['A']['case_id'][0],amplitude=amplitude_inventory(batch),fresh=split,constants=[])
            del result
            for level in [*c['intervention']['constant_levels'],scale['C_init']]:
                value=torch.tensor(level,device=c['training']['device'],requires_grad=True)
                class Constant:
                    def __call__(self,b):return value.expand(len(b['case_id']),1,64,64,64)
                result=objective(Constant(),batch,.1)
                split=gradient_split(None,result,batch,scalar=value)
                contributions=grouped_constant_contributions(result[3],batch,value);validate_group_sum(split,contributions)
                row['constants'].append(dict(C=level,split=split,amplitude_contributions=contributions,float64_scalar_check=float64_scalar_probe(batch,level)))
                del result
            results.append(row);print(f'diagnosed case={i} fresh+6 constants, no optimizer',flush=True)
        assert state_hash(model.state_dict())==initial and all(p.grad is None for p in model.parameters())
        summary=dict(at=now(),initial_hash=initial,model_unchanged=True,optimizer_created=False,optimizer_steps=0,grad_buffers_none=True,case_results=results,seconds=time.perf_counter()-began,access=guard.summary(),group_threshold_normalized=1e-6,positive_derivative='gradient descent lowers constant concentration',relative_definition='per-case sum valid abs residual / clamp_min(sum valid abs target,1e-6); not per-ray',groups_do_not_filter_training=True)
    write_json(target,summary)
    with gzip.open(ART/'gradient_diagnosis.json.gz','wb') as f:f.write(target.read_bytes())
    print(json.dumps({k:v for k,v in summary.items() if k!='case_results'},indent=2),flush=True)


def smoke(c):
    reg,scale,inventory=verify(c);parent,_=paths(c);ds=make_dataset(c);results={};torch.set_num_threads(2)
    assert json.loads((parent/'gradient_diagnosis.json').read_text())['optimizer_steps']==0
    for arm,coefficient in [('control',.1),('treatment',0.)]:
        out,ck=paths(c,arm,True)
        if (out/'result.json').exists():raise FileExistsError('Smoke already exists')
        with TrainEightGuard(c['dataset']['root'],ck,ds.records) as guard:
            model=fresh(c,scale,'treatment');initial=state_hash(model.state_dict());assert initial==reg['initial_hash']
            b=move_dual_batch(dual_measurement_collate([ds[0],ds[1]]),c['training']['device'])
            model.train();seen=[];hook=model.register_forward_pre_hook(lambda m,a:seen.append(id(a[0])))
            result=objective(model,b,coefficient);hook.remove();assert seen==[id(b['A']),id(b['B'])]
            loss=result[0];optimizer=ObservedAdamW(model,lr=2e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
            loss.backward();pre=group_norm(optimizer.named,[p.grad for n,p in optimizer.named]);assert all(v>0 for v in pre.values())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step()
            generator=torch.Generator().manual_seed(c['training']['seed'])
            atomic_checkpoint(ck/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator),updates=1))
            before=state_hash(model.state_dict());saved=torch.load(ck/'last.pt',map_location=c['training']['device'],weights_only=False)
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],generator)
            assert state_hash(model.state_dict())==before and scheduler.last_epoch==1 and all(int(s['step'])==1 for s in optimizer.state.values())
            chunk=checked_chunk(model,b['A']);geo=[geometry(b[k]) for k in ['A','B']]
            assert all(r['valid_rays']==r['packed_real_rays'] and r['entry_exit_in_domain'] for r in geo)
            results[arm]=dict(passed=True,initial_hash=initial,relative_l1=coefficient,steps=1,pre_clip_gradient_l2=pre,update=optimizer.observations,chunk_check=chunk,geometry=geo,H_excluded=True,resume_verified=True,access=guard.summary(),no_warm_start=True)
        write_json(out/'result.json',results[arm]);del model,optimizer,scheduler,result,loss,saved
        torch.cuda.empty_cache()
    write_json(parent/'engineering_smoke.json',results);(ART/'engineering_smoke.json').write_bytes((parent/'engineering_smoke.json').read_bytes())
    print(json.dumps({k:{n:v for n,v in r.items() if n not in ['chunk_check','geometry']} for k,r in results.items()},indent=2),flush=True)


def evaluate(model,loader,c,coefficient,capture=False):
    logits=[];hook=None
    if capture:hook=model.decoder.postproc.register_forward_hook(lambda m,a,y:logits.append(y.detach().flatten().cpu()))
    try:row=run_epoch(model,loader,c,coefficient)
    finally:
        if hook:hook.remove()
    return row,logits_summary(logits) if capture else None


def health(model,ds,full,logits,c):
    rows=[];predictions=[]
    for i in c['probe']['diagnostic_case_indices']:
        b=move_dual_batch(dual_measurement_collate([ds[i]]),c['training']['device']);model.train()
        result=objective(model,b,.1);split=gradient_split(model,result,b)
        pred=result[2][0].detach();predictions.append(pred.cpu())
        row=dict(index=i,case_id=b['A']['case_id'][0],gradient_split=split,spatial=spatial(pred))
        del result
        model.eval();zero=dict(b['A'])
        for k in ['input_projection_raw','input_projection']:zero[k]=torch.zeros_like(zero[k])
        assert all(torch.equal(zero[k],v) for k,v in b['A'].items() if isinstance(v,torch.Tensor) and k not in ['input_projection_raw','input_projection'])
        with torch.no_grad():normal=model(b['A']);blank=model(zero)
        absolute=float((normal-blank).abs().double().mean());denominator=float(normal.abs().double().mean())
        row['measurement_dependence']=dict(absolute_MA=absolute,normal_mean_abs=denominator,epsilon=1e-12,relative_MA=absolute/(denominator+1e-12),geometry_path_masks_unchanged=True)
        rows.append(row)
    chunk=checked_chunk(model,move_dual_batch(dual_measurement_collate([ds[0]]),c['training']['device'])['A'])
    differences=[dict(a=i,b=j,absolute_MA=float((predictions[i]-predictions[j]).abs().double().mean()),relative_MA=float((predictions[i]-predictions[j]).abs().double().mean()/(predictions[i].abs().double().mean()+1e-12))) for i in range(3) for j in range(i+1,3)]
    return dict(full_fixed_subset=full,logits=logits,diagnostic_cases=rows,case_differences=differences,chunk_check=chunk,optimizer_steps_diagnosis=0)


def train(c,arm,resume):
    reg,scale,inventory=verify(c);parent,_=paths(c);out,ck=paths(c,arm);ds=make_dataset(c);coefficient=.1 if arm=='control' else 0.
    assert all(r['passed'] for r in json.loads((parent/'engineering_smoke.json').read_text()).values())
    if (out/'complete.json').exists():raise FileExistsError('Scientific arm already stopped')
    if (ck/'last.pt').exists() and not resume:raise FileExistsError('Own-last explicit resume required')
    torch.set_num_threads(2);began=time.perf_counter()
    with TrainEightGuard(c['dataset']['root'],ck,ds.records) as guard:
        fixed_samples(ds,inventory);model=fresh(c,scale,'treatment');initial=state_hash(model.state_dict());assert initial==reg['initial_hash']
        optimizer=ObservedAdamW(model,lr=2e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        generator=torch.Generator().manual_seed(c['training']['seed']);loader=make_loader(Subset(ds,list(range(8))),c,False,generator)
        write_json(out/'environment.json',dict(**environment(c),arm=arm,relative_l1=coefficient,initial_hash=initial,config=c,at=now(),TF32=False))
        history=[];healths=[];events=[];state={};start=0
        def save(epoch):
            atomic_checkpoint(ck/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator),epoch=epoch,updates=optimizer.updates,config=c,arm=arm,relative_l1=coefficient,initial_hash=initial,registration_sha256=sha256(parent/'registration.json'),source_manifest_sha256=sha256(parent/'source_manifest.json'),history=history,health=healths,events=events,exit_state=state,volumetric_gt_used=False))
        try:
            if resume:
                saved=torch.load(ck/'last.pt',map_location=c['training']['device'],weights_only=False)
                verify_resume(saved,c,arm,initial,sha256(parent/'registration.json'),sha256(parent/'source_manifest.json'))
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],generator);optimizer.updates=saved['updates'];start=saved['epoch'];history=saved['history'];healths=saved['health'];events=saved['events'];state=saved['exit_state']
            else:
                full,logits=evaluate(model,loader,c,coefficient,True);h=health(model,ds,full,logits,c);h.update(epoch=0,updates=0);healths.append(h)
                save(0);(ck/'epoch0.pt').write_bytes((ck/'last.pt').read_bytes())
            for epoch in range(start+1,41):
                ds.set_epoch(0);optimizer.observations=[];torch.cuda.reset_peak_memory_stats(c['training']['device']);t=time.perf_counter();lr=optimizer.param_groups[0]['lr']
                gpu_start=torch.cuda.Event(enable_timing=True);gpu_end=torch.cuda.Event(enable_timing=True);gpu_start.record()
                tr=run_epoch(model,loader,c,coefficient,optimizer);assert optimizer.updates==epoch*4
                full,logits=evaluate(model,loader,c,coefficient,epoch%5==0)
                assert tr['case_station_digest']==full['case_station_digest']==healths[0]['full_fixed_subset']['case_station_digest']
                scheduler.step();row=dict(epoch=epoch,updates=optimizer.updates,train=tr,eval=full,learning_rate=lr,update_observations=optimizer.observations);history.append(row)
                if epoch%5==0:
                    h=health(model,ds,full,logits,c);h.update(epoch=epoch,updates=optimizer.updates);healths.append(h)
                    total_key='total_control' if arm=='control' else 'total_treatment'
                    assert all(all(v>0 for v in r['gradient_split']['branches'][total_key]['parameter_gradient_l2'].values()) for r in h['diagnostic_cases'])
                    state=exit_event(epoch,history,h,scale['C_init'],state);events.append(state)
                gpu_end.record();torch.cuda.synchronize(c['training']['device']);row.update(seconds=time.perf_counter()-t,gpu_event_elapsed_ms=gpu_start.elapsed_time(gpu_end),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(c['training']['device']))
                save(epoch)
                for name,value in [('history',history),('health',healths),('events',events),('access',guard.summary())]:write_json(out/(name+'.json'),value)
                print(f"{arm} epoch={epoch}/40 updates={optimizer.updates} H_L1={full['measurement']['heldout']['relative_l1']:.8f} H_L2={full['measurement']['heldout']['relative_l2']:.8f} mean={full['output']['mean']:.8g} reason={state.get('reason')} seconds={row['seconds']:.2f}",flush=True)
                if state.get('reason'):break
        except Exception as exc:
            write_json(out/'engineering_error.json',dict(at=now(),error=repr(exc),last_valid_checkpoint=str(ck/'last.pt') if (ck/'last.pt').exists() else None,attempted_updates=optimizer.updates,access=guard.summary(),no_silent_fix=True));raise
        summary=dict(arm=arm,relative_l1=coefficient,completed_epochs=history[-1]['epoch'],updates=optimizer.updates,reason=state['reason'],at=now(),elapsed_process_seconds=time.perf_counter()-began,total_epoch_seconds=sum(r['seconds'] for r in history),peak_gpu_allocated_bytes=max(r['peak_gpu_allocated_bytes'] for r in history),initial_hash=initial,last_checkpoint=str(ck/'last.pt'),last_sha256=sha256(ck/'last.pt'),epoch0_sha256=sha256(ck/'epoch0.pt'),access=guard.summary(),no_GT=True,no_full_training_started=True)
        assert not any(summary['access'][k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
        write_json(out/'complete.json',summary);print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/ray_senseiver_relative_probe_v1.yaml');ap.add_argument('--phase',choices=['prepare','diagnose','smoke','train'],required=True);ap.add_argument('--arm',choices=['control','treatment']);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    c=validate(yaml.safe_load(Path(a.config).read_text()))
    if a.phase=='prepare':prepare(c,a.config)
    elif a.phase=='diagnose':diagnose(c)
    elif a.phase=='smoke':smoke(c)
    else:
        if not a.arm:ap.error('--arm required')
        train(c,a.arm,a.resume)


if __name__=='__main__':main()
