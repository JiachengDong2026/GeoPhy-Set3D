#!/usr/bin/env python3
"""Preregistered three-arm scale-fusion replication; measurement-only, exactly20 epochs."""
import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import mrgr_scale_fusion_v2 as candidate
import probe_mrgr_multiscale_v1 as base
import train_mrgr_multiscale_formal_v1 as formal
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.training.utils import move_batch

ROOT=base.ROOT
SEED=20261018
OUT=ROOT/f'outputs/research/mrgr_scale_fusion_v2_seed{SEED}'
ARMS=candidate.ARMS
DEVICES={a:'cuda:0' for a in ARMS}
PANEL=formal.DEFAULT_OUT/'evaluation/frozen_model_test_v1/registration.json'


def read(path):return json.loads(Path(path).read_text())


def sources():
    result=formal.sources()
    for path in (Path(__file__),Path(candidate.__file__),ROOT/'tests/test_mrgr_scale_fusion_v2.py', ROOT/'scripts/mrgr_ocmsf_v1.py',ROOT/'scripts/probe_mrgr_multiscale_v1.py'):
        result[str(path.relative_to(ROOT))]=base.digest(path)
    return result


def effective(p,arm):
    c=copy.deepcopy(p['config']);c['training']['device']=p['arm_device'][arm]
    c['experiment']=dict(name=f'{OUT.name}_{arm}',output_root=str(OUT/arm))
    return c


def verify():
    p=read(OUT/'protocol.json')
    assert base.digest(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    assert sources()==p['sources']
    formal.verify(formal.DEFAULT_OUT);formal.verify_data(p)
    assert base.digest(PANEL)==p['panel_source_sha256']
    for arm in ARMS:
        assert base.digest(OUT/arm/'effective_config.json')==p['effective_config_sha256'][arm]
    return p


def register():
    old=formal.verify(formal.DEFAULT_OUT);formal.verify_data(old)
    assert not OUT.exists(),'Refuse to overwrite existing experiment'
    config=copy.deepcopy(old['config']);config['training']['device']='cuda:0';config['training']['seed']=SEED
    config['experiment']=dict(name=OUT.name,output_root=str(OUT))
    OUT.mkdir(parents=True);(OUT/'checkpoints').mkdir()
    with base.GTReadGuard(config['dataset']['root'],OUT/'checkpoints') as guard:
        models={a:candidate.model_for(config,a,'cpu') for a in ARMS}
        states={a:m.state_dict() for a,m in models.items()}
        for a in ('C_content','E_global'):
            assert all(torch.equal(v,states[a][k]) for k,v in states['B_multiscale'].items())
        assert all(torch.equal(v,states['E_global'][k]) for k,v in states['B_multiscale'].items())
        counts={a:sum(x.numel() for x in m.parameters()) for a,m in models.items()}
        assert counts==dict(B_multiscale=344678,C_content=345417,E_global=345097)
        train=base.selfsup.make_dataset(config);val=base.selfsup.make_dataset(config,validation=True)
        assert (len(train),len(val))==(166,36)
        stations={r['case_id']:{k:val[i][k]['input_station_indices'].tolist() for k in ('A','B','H')} for i,r in enumerate(val.records)}
        settings=[s for s in read(PANEL)['settings'] if s['condition']=='clustered-V4']
        assert sorted(s['draw'] for s in settings)==list(range(5))
        p=dict(registered_at=base.now(),config=config,sources=sources(),data_files=old['data_files'],
            arm_device=DEVICES,parameters=counts,seed=SEED,epochs=20,updates_per_arm=1660,
            train_case_ids=[r['case_id'] for r in train.records],validation_case_ids=[r['case_id'] for r in val.records],
            station_map=stations,cluster_settings=settings,panel_source=str(PANEL),panel_source_sha256=base.digest(PANEL),
            initial_state_sha256={a:base.state_hash(s) for a,s in states.items()},
            architecture='B frozen multiscale; C adds content spatial fusion739params; E adds global3logits plus zero Conv12->32 419params. Both reuse aligned reduce(x32), local16, context8; visibility only, no O numerical values.',
            conditioning='Neither candidate receives O values except f>0 validity. Global softmax shared across samples/voxels; invalid voxels uniform alpha and zero correction. C0/C1 share one fusion module.',
            capture='Existing once-per-model OARI descriptor captured for validity; hook cleanup always_call including errors; no extra eigensolve.',
            initialization='Fresh training, base seed, context seed+1 and isolated new module seed+2; shared tensors exact. Both residual output zero, initial B/C/E outputs bitwise equal. CUDA stochastic ops absent.',
            schedule='Fresh training first20 epochs of frozen cosine T_max300; AdamW2e-4 wd1e-4 batch2 clip1;166train/36val;FP32TF32off',
            objective='Unchanged selfsup hold1,cross0.5,observed0.1,consistency0.05; unchanged P96 projector, normalization and station sampler',
            endpoint='Fixed epoch20 last.pt, no best checkpoint selection for screening; ordinary per-case meanA/B->H',
            promotion_gate=dict(ordinary_mean_ratio_max=.99,ordinary_p90_ratio_max=1.02,cluster5_mean_ratio_max=1.02,
                runtime_ratio_max=1.20,projected300_hours_max=5.,all_correctness=True),
            primary_contrasts=['C_content vs B_multiscale: independent new-seed replication','E_global vs B_multiscale: new global fusion'],exploratory_contrasts=['E_global vs C_content: global vs content spatial gate'],
            statistics='36 paired cases; cluster five draws averaged within case;20000 bootstrap seed20261017; report all six pairwise contrasts without window/epoch search',
            no_gt=True,no_final_test=True,no_automatic_formal300=True,resource_budget_minutes=60,
            resource_rule='GPU3 exclusively, CUDA_VISIBLE_DEVICES=3, logicalcuda0. All arms sequential; no other processes killed.',
            limitations='Single seed and known developmentvalidation; small changes can reflect numeric trajectory variation. Extra32 reduce/GN/SiLU computation and sharedreduce gradients exist in both C/D.',
            registration_audit=guard.summary())
        p['effective_config_sha256']={}
        for arm in ARMS:
            (OUT/arm).mkdir();(OUT/'checkpoints'/arm).mkdir()
            formal.write(OUT/arm/'effective_config.json',effective(p,arm))
            p['effective_config_sha256'][arm]=base.digest(OUT/arm/'effective_config.json')
        formal.write(OUT/'protocol.json',p)
        (OUT/'protocol.sha256').write_text(base.digest(OUT/'protocol.json')+'\n')
        for arm in ARMS:
            base.atomic_checkpoint(OUT/'checkpoints'/arm/'initial.pt',dict(model=states[arm],arm=arm,epoch=0,
                protocol_sha256=base.digest(OUT/'protocol.json'),volumetric_gt_used=False))
    print(json.dumps(dict(stage='registered',parameters=counts)),flush=True)


def full_mask_checks(model,batch):
    model.eval();records=[]
    with torch.no_grad():
        for count in (1,2):
            value={k:v[:count] if isinstance(v,torch.Tensor) else v[:count] if isinstance(v,list) else v for k,v in batch.items()}
            for mode in ('normal','partial_padding','all_padding','no_valid_rays'):
                b={k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in value.items()}
                if mode=='partial_padding':
                    b['view_mask'][:,1:]=False;b['input_valid_ray'][:,1:]=False
                elif mode=='all_padding':
                    b['view_mask'][:]=False;b['input_valid_ray'][:]=False
                elif mode=='no_valid_rays':b['input_valid_ray'][:]=False
                stages=model(b,return_stages=True)
                for field in (stages['C0'],stages['C1']):
                    assert field.shape==(count,1,64,64,64)
                    assert torch.isfinite(field).all() and bool(((field>=0)&(field<=1)).all())
                assert model.decoder._descriptor is None
                assert model.decoder.descriptor_calls==1 and model.decoder.decoder_calls==2
                records.append(dict(batch=count,mode=mode,finite=True,shared_decoder_calls=2,descriptor_calls=1,cache_cleared=True))
    return records


def smoke():
    p=verify();assert not (OUT/'smoke.json').exists()
    unit=subprocess.run(['/scratch/djc/envs/gas-torch/bin/python','-m','unittest','discover','-s','tests','-p','test_mrgr_scale_fusion_v2.py','-v'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'unit_tests.log').write_text(unit.stdout+unit.stderr);assert unit.returncode==0
    config=copy.deepcopy(p['config']);config['training']['device']='cuda:0';config['training']['seed']=SEED;device='cuda:0'
    with base.GTReadGuard(config['dataset']['root'],OUT/'checkpoints') as guard:
        ds=base.selfsup.make_dataset(config)
        batch=base.selfsup.move_dual_batch(base.dual_measurement_collate([ds[0],ds[1]]),device)
        predictions={};records={};mask_checks={};learning={}
        for arm in ARMS:
            model=candidate.model_for(config,arm,device);model.eval()
            with torch.no_grad():predictions[arm]=model(batch['A']).cpu()
            if arm in ('C_content','E_global'):
                mask_checks[arm]=full_mask_checks(model,batch['A'])
            model.train();times=[]
            torch.cuda.reset_peak_memory_stats(device)
            for i in range(3):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize(device);start=time.perf_counter()
                loss,parts,volumes,projections=base.selfsup.objective(model,batch);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                torch.cuda.synchronize(device);times.append(time.perf_counter()-start)
                del loss,parts,volumes,projections
            records[arm]=dict(backward_seconds=times,timed_update_seconds=float(np.mean(times[1:])),gradient_norm=float(norm),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
            if arm in ('C_content','E_global'):
                # A temporary smoke model only; no screened checkpoint is updated.
                opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
                opt.step();opt.zero_grad(set_to_none=True)
                loss,parts,volumes,projections=base.selfsup.objective(model,batch);loss.backward()
                gradients={name:dict(norm=float(param.grad.norm()),finite=bool(torch.isfinite(param.grad).all()),
                    nonzero=bool(param.grad.abs().sum()>0)) for name,param in model.decoder.scale_fusion.named_parameters()}
                assert all(v['finite'] and v['nonzero'] for v in gradients.values())
                opt.step();opt.zero_grad(set_to_none=True)
                model.decoder.scale_fusion.record_diagnostics=True
                with torch.no_grad():model(batch['A'])
                diag=model.decoder.scale_fusion.last_diagnostics
                visible=diag['visible'].expand(-1,3,-1,-1,-1)
                alpha=diag['alpha'].masked_select(visible)
                learning[arm]=dict(after_one_real_smoke_update_gradients_before_second_update=gradients,
                    alpha_abs_difference_from_uniform=float((alpha-1/3).abs().mean()),
                    correction_norm=float(diag['delta'].norm()),smoke_updates_not_part_of_screen=True)
                assert learning[arm]['correction_norm']>0 and learning[arm]['alpha_abs_difference_from_uniform']>0
                del opt,loss,parts,volumes,projections
            del model;gc.collect();torch.cuda.empty_cache()
        assert torch.equal(predictions['B_multiscale'],predictions['C_content'])
        assert torch.equal(predictions['B_multiscale'],predictions['E_global'])
        estimate=sum(x['timed_update_seconds']*93.8*20*1.2 for x in records.values())
        result=dict(passed=True,initial_B_C_E_bitwise_equal=True,unit_tests_passed=True,mask_checks=mask_checks,
            arms=records,learning=learning,estimated_three_arm_total_gpu_minutes=estimate/60,
            estimated_GPU3_candidates_minutes=sum(records[a]['timed_update_seconds']*93.8*20*1.2 for a in ('C_content','E_global'))/60,
            E_to_B_update_ratio=records['E_global']['timed_update_seconds']/records['B_multiscale']['timed_update_seconds'],
            GT_access=guard.summary(),all_smoke_models_discarded=True)
        formal.write(OUT/'smoke.json',result);print(json.dumps(dict(stage='smoke',**result)),flush=True)
        assert result['estimated_GPU3_candidates_minutes']<=p['resource_budget_minutes'],'Candidate estimated resource budget exceeded'


def ordinary_cases(model,loader,config):
    model.eval();records=[]
    if isinstance(model.decoder,candidate.OCMSFDecoder):model.decoder.scale_fusion.record_diagnostics=True
    with torch.no_grad():
        for raw in loader:
            b=base.selfsup.move_dual_batch(raw,config['training']['device']);values={};sanity={};gates={}
            for key in ('A','B'):
                field=model(b[key]);prediction=base.project_input(field,b['H'])
                valid=b['H']['input_valid_ray'].bool()&b['H']['view_mask'][:,:,None,None,None]
                values[key]=[base.relative_projection_metrics(prediction[i],b['H']['input_projection_raw'][i],valid[i]) for i in range(len(field))]
                sanity[key]=[dict(min=float(v.min()),max=float(v.max()),std=float(v.std()),mean=float(v.mean())) for v in field]
                assert all(x['std']>1e-8 and x['max']>x['min'] and all(math.isfinite(t) for t in x.values()) for x in sanity[key])
                if isinstance(model.decoder,candidate.OCMSFDecoder):
                    diag=model.decoder.scale_fusion.last_diagnostics
                    gates[key]=[]
                    for i in range(len(field)):
                        mask=diag['visible'][i,0]
                        alpha=diag['alpha'][i,:,mask]
                        gates[key].append(dict(alpha_mean=alpha.mean(1).tolist(),alpha_spatial_std=alpha.std(1).tolist(),
                            delta_norm=float(diag['delta'][i].norm()),visible_voxels=int(mask.sum())))
            for i,case in enumerate(b['A']['case_id']):
                row=dict(case_id=case,heldout_relative_l1=.5*(values['A'][i]['relative_l1']+values['B'][i]['relative_l1']),
                    heldout_relative_l2=.5*(values['A'][i]['relative_l2']+values['B'][i]['relative_l2']),
                    A=values['A'][i],B=values['B'][i],output_sanity={k:sanity[k][i] for k in sanity},
                    stations={k:b[k]['input_station_indices'][i,b[k]['view_mask'][i]].tolist() for k in ('A','B','H')})
                if gates:row['gate']={k:gates[k][i] for k in gates}
                records.append(row)
    return records


def train(arm):
    p=verify();assert read(OUT/'smoke.json')['passed'];config=effective(p,arm);device=config['training']['device'];run=OUT/arm
    assert not (run/'metrics.jsonl').exists(),'Refuse silent restart of partial/completed arm'
    with base.GTReadGuard(config['dataset']['root'],OUT/'checkpoints') as guard:
        model=candidate.model_for(config,arm,device)
        assert base.state_hash(model.state_dict())==p['initial_state_sha256'][arm]
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
        generator=torch.Generator().manual_seed(SEED)
        ds=base.selfsup.make_dataset(config);vd=base.selfsup.make_dataset(config,validation=True)
        loader=base.selfsup.make_loader(ds,config,True,generator);vl=base.selfsup.make_loader(vd,config)
        metrics=[];began=time.perf_counter();best=float('inf')
        formal.write(run/'environment.json',base.environment(config))
        for epoch in range(20):
            ds.set_epoch(epoch);vd.set_epoch(0);torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device);start=time.perf_counter();lr=opt.param_groups[0]['lr']
            tr=base.selfsup.run_epoch(model,loader,config,opt);va=base.selfsup.run_epoch(model,vl,config)
            torch.cuda.synchronize(device);scheduler.step()
            score=base.selfsup.selection_metric(va);best=min(best,score)
            assert math.isfinite(score) and (tr['sample_count'],va['sample_count'])==(166,36)
            row=dict(epoch=epoch+1,train=tr,val=va,learning_rate=lr,seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                loader_rng_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
            metrics.append(row);base.append(run/'metrics.jsonl',row)
            base.atomic_checkpoint(OUT/'checkpoints'/arm/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                rng=base.rng_state(generator),epoch=epoch+1,updates=(epoch+1)*83,arm=arm,config=config,
                protocol_sha256=base.digest(OUT/'protocol.json'),volumetric_gt_used=False))
            formal.write(run/'gt_access_audit.json',guard.summary())
            formal.write(run/'status.json',dict(state='screening' if epoch<19 else 'finalizing',epoch=epoch+1,max_epochs=20,
                updates=(epoch+1)*83,val_H=score,best_H_diagnostic_only=best,elapsed_seconds=time.perf_counter()-began,
                estimated_remaining_minutes=float(np.mean([x['seconds'] for x in metrics[-5:]])*(19-epoch)/60),updated_at=base.now()))
            print(f'{arm} epoch={epoch+1}/20 valH={score:.8f} seconds={row["seconds"]:.2f}',flush=True)
        cases=ordinary_cases(model,vl,config)
        assert [r['case_id'] for r in cases]==p['validation_case_ids']
        for r in cases:assert r['stations']==p['station_map'][r['case_id']]
        (run/'epoch20_per_case.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in cases))
        audit=guard.summary();assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        result=dict(completed=True,epochs=20,updates=1660,final_H=score,elapsed_seconds=time.perf_counter()-began,
            warm_epoch_seconds=float(np.mean([r['seconds'] for r in metrics[2:]])),gt_audit=audit,
            output_sanity_passed=True,checkpoint_sha256=base.digest(OUT/'checkpoints'/arm/'last.pt'),finished_at=base.now())
        formal.write(run/'summary.json',result);formal.write(run/'status.json',dict(state='complete',epoch=20,**result))
        formal.plot(run,metrics)
    verify();print(json.dumps(dict(stage='arm_complete',arm=arm,**result)),flush=True)


class Replay:
    def __init__(self,s):self.s=s
    def sample(self,station_count,*,epoch,sample_index,case_id):
        return np.asarray(self.s['input_ids'][case_id]),np.asarray(self.s['heldout_ids'][case_id])


def heldout(b):
    h={k:v for k,v in b.items() if not k.startswith(('input_','hold_')) and k!='view_mask'}
    h.update({'input_'+k[5:]:v for k,v in b.items() if k.startswith('hold_') and k!='hold_view_mask'})
    h['view_mask']=b['hold_view_mask'];return h


def panel(arm):
    p=verify();config=effective(p,arm);device=config['training']['device'];run=OUT/arm
    assert not (run/'cluster5_per_case.jsonl').exists()
    summary=read(run/'summary.json');path=OUT/'checkpoints'/arm/'last.pt';assert base.digest(path)==summary['checkpoint_sha256']
    records=[]
    with base.GTReadGuard(config['dataset']['root'],OUT/'checkpoints') as guard:
        ckpt=torch.load(path,map_location='cpu',weights_only=False);assert ckpt['epoch']==20 and ckpt['updates']==1660
        model=candidate.model_for(config,arm,device);model.load_state_dict(ckpt['model']);model.eval()
        with torch.no_grad():
            for s in p['cluster_settings']:
                ds=MeasurementOnlyPoolDataset(config['dataset']['root'],split='val',geometry_pool='train',evaluation_only=True,
                    min_input_views=4,max_input_views=4,hold_views=4,base_seed=SEED,normalization='linear_p99_5',statistics_file='train_statistics.json')
                ds.sampler=Replay(s)
                for first in range(0,len(ds),2):
                    samples=[ds[j] for j in range(first,min(first+2,len(ds)))]
                    b=move_batch(projection_pool_collate(samples),device);inputs={k:v for k,v in b.items() if not k.startswith('hold_')};h=heldout(b)
                    field=model(inputs);pred=base.project_input(field,h);valid=h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]
                    assert torch.isfinite(field).all() and torch.isfinite(pred).all()
                    for j,case in enumerate(inputs['case_id']):
                        assert samples[j]['input_station_indices'].tolist()==s['input_ids'][case]
                        assert samples[j]['hold_station_indices'].tolist()==s['heldout_ids'][case]
                        row=dict(arm=arm,draw=s['draw'],case_id=case,input_ids=s['input_ids'][case],heldout_ids=s['heldout_ids'][case],
                            **base.relative_projection_metrics(pred[j],h['input_projection_raw'][j],valid[j]))
                        records.append(row)
                print(f'{arm} cluster draw{s["draw"]} complete',flush=True)
        audit=guard.summary();assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    assert len(records)==180
    (run/'cluster5_per_case.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    formal.write(run/'cluster5_summary.json',dict(completed=True,gt_audit=audit,
        mean=float(np.mean([np.mean([r['relative_l1'] for r in records if r['case_id']==case]) for case in p['validation_case_ids']]))))


def describe(x):return dict(mean=float(x.mean()),p90=float(np.quantile(x,.9)),maximum=float(x.max()),median=float(np.median(x)))


def paired(a,b):
    delta=b-a;rng=np.random.default_rng(20261017)
    ci=np.quantile(delta[rng.integers(0,36,size=(20000,36))].mean(1),[.025,.975]).tolist()
    return dict(control=describe(a),treatment=describe(b),mean_ratio=float(b.mean()/a.mean()),
        p90_ratio=float(np.quantile(b,.9)/np.quantile(a,.9)),mean_delta=float(delta.mean()),paired_ci95=ci,
        wins=int((delta < -1e-8).sum()),losses=int((delta > 1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()))


def compare():
    p=verify();logs={a:[json.loads(x) for x in (OUT/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    for arm in ARMS:
        assert len(logs[arm])==20
        for x,y in zip(logs['B_multiscale'],logs[arm]):
            assert x['epoch']==y['epoch'] and x['learning_rate']==y['learning_rate'] and x['loader_rng_sha256']==y['loader_rng_sha256']
            for split in ('train','val'):assert x[split]['case_station_digest']==y[split]['case_station_digest']
    ordinary={};hard={};times={};summaries={}
    for arm in ARMS:
        cases=[json.loads(x) for x in (OUT/arm/'epoch20_per_case.jsonl').read_text().splitlines()]
        rows=[json.loads(x) for x in (OUT/arm/'cluster5_per_case.jsonl').read_text().splitlines()]
        assert [x['case_id'] for x in cases]==p['validation_case_ids']
        assert len(rows)==180 and len({(x['case_id'],x['draw']) for x in rows})==180
        ordinary[arm]=np.array([x['heldout_relative_l1'] for x in cases])
        hard[arm]=np.array([np.mean([x['relative_l1'] for x in rows if x['case_id']==case]) for case in p['validation_case_ids']])
        times[arm]=float(np.mean([x['seconds'] for x in logs[arm][2:]]))
        summaries[arm]=read(OUT/arm/'summary.json')
    contrasts={}
    for i,control in enumerate(ARMS):
        for treatment in ARMS[i+1:]:
            key=f'{treatment}_vs_{control}'
            contrasts[key]=dict(ordinary=paired(ordinary[control],ordinary[treatment]),cluster5=paired(hard[control],hard[treatment]),runtime_ratio=times[treatment]/times[control])
    gates={}
    for arm in ('C_content','E_global'):
        x=contrasts[f'{arm}_vs_B_multiscale'];t=p['promotion_gate']
        checks=dict(ordinary_mean=x['ordinary']['mean_ratio']<=t['ordinary_mean_ratio_max'],ordinary_p90=x['ordinary']['p90_ratio']<=t['ordinary_p90_ratio_max'],
            cluster5_mean=x['cluster5']['mean_ratio']<=t['cluster5_mean_ratio_max'],runtime=x['runtime_ratio']<=t['runtime_ratio_max'],
            five_hour=times[arm]*300/3600<=t['projected300_hours_max'],correctness=read(OUT/'smoke.json')['passed'],
            output_sanity=all(s['output_sanity_passed'] for s in summaries.values()))
        gates[arm]=dict(gate_passed=all(checks.values()),checks=checks)
    result=dict(completed=True,seed=SEED,gates=gates,contrasts=contrasts,
        ordinary={a:describe(v) for a,v in ordinary.items()},cluster5={a:describe(v) for a,v in hard.items()},
        parameters=p['parameters'],warm_epoch_seconds=times,projected300_hours={a:times[a]*300/3600 for a in ARMS},
        gt_used=False,final_test_used=False,formal300_started=False,matched_cases_stations_shuffles_lr=True,
        protocol_sha256=base.digest(OUT/'protocol.json'),finished_at=base.now(),limitations=p['limitations'])
    formal.write(OUT/'comparison.json',result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','smoke','train','panel','compare','verify','arm_pipeline'))
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int,choices=(20261018,20261019),default=20261018);args=parser.parse_args();torch.set_num_threads(2)
    global SEED,OUT
    SEED=args.seed;OUT=ROOT/f'outputs/research/mrgr_scale_fusion_v2_seed{SEED}'
    if args.action=='register':register()
    elif args.action=='smoke':smoke()
    elif args.action=='verify':verify();print('Verified')
    elif args.action=='compare':compare()
    elif args.action=='train':assert args.arm;train(args.arm)
    elif args.action=='panel':assert args.arm;panel(args.arm)
    else:
        assert args.arm;train(args.arm);panel(args.arm)


if __name__=='__main__':main()
