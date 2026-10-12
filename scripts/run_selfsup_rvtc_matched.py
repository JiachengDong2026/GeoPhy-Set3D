#!/usr/bin/env python3
"""Register, smoke, train and evaluate one finite, strictly matched RVTC contrast."""
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest
from geophy_set3d.training import selfsup_v2 as v2
from geophy_set3d.training.utils import (load_config, seed_everything, write_json,
    atomic_checkpoint, rng_state, restore_rng, environment)
from geophy_set3d.training.rvtc_matched import (contrast_loss, state_hash, validate_pair,
    ContrastReadGuard, diagnostic_context, paired_effect, applied_gradient_geometry)
from geophy_set3d.evaluation.rvtc_failure_diagnostics import (fixed_sample, collate_sample,
    measurement_rows, gradient_diagnostic, paired_summary)
from geophy_set3d.evaluation.measurement_diagnostics import set_properties

CONFIGS = ['configs/selfsup_rvtc_matched_control.yaml','configs/selfsup_rvtc_matched_treatment.yaml']


def configs():
    pair = [yaml.safe_load((REPO/p).read_text()) for p in CONFIGS]
    validate_pair(pair)
    base = load_config(REPO/pair[0]['base_config']); v2.validate_selfsup_config(base)
    assert base['training']['device'] == pair[0]['device']
    return pair, base, Path(pair[0]['output'])


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def append_json(path, row):
    with path.open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_csv(path, rows):
    if not rows: raise ValueError('No actual rows')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def metadata_snapshot(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,inode=p.stat().st_ino)
            for p in sorted(root.rglob('*')) if p.is_file()}


def inventory(base):
    dataset=Path(base['dataset']['root']); out=Path(base['experiment']['output_root'])
    rows=read_lines(dataset/'source_index.jsonl')
    raw={r['source_target_path']:dict(size=Path(r['source_target_path']).stat().st_size,
        mtime_ns=Path(r['source_target_path']).stat().st_mtime_ns,inode=Path(r['source_target_path']).stat().st_ino) for r in rows}
    frozen=json.loads((REPO/'docs/reports/artifacts/selfsup_mrgr_v2/training_source_manifest.json').read_text())['sha256']
    assert all(digest(REPO/p)==sha for p,sha in frozen.items())
    return dict(dataset=metadata_snapshot(dataset),source_gt128_metadata=raw,
        historical={kind:metadata_snapshot(out/kind) for kind in ('checkpoints','logs','comparisons','diagnostics','figures')},
        original_numerical_source_hashes=frozen,gt_or_foreign_checkpoint_content_opened=False)


def checkpoints(root, arm):
    p=root/'checkpoints'/arm
    return [p/'initial.pt',p/'last.pt']+[p/f'epoch_{e:03d}.pt' for e in range(10,81,10)]


def guard_for(root, base, arms=('Control','Treatment')):
    return ContrastReadGuard(base['dataset']['root'],root/'checkpoints',
        [p for arm in arms for p in checkpoints(root,arm)], root/'access_sessions.jsonl')


def register(pair,base,root):
    if (root/'protocol.json').exists():
        protocol=verify(pair,base,root)
        print('Existing registration verified; will not replace it',flush=True);return
    if root.exists() and any(root.iterdir()):raise FileExistsError('Independent output is not empty; investigate before registration')
    root.mkdir(parents=True,exist_ok=True)
    g=guard_for(root,base)
    with g:
        before=inventory(base)
        station=json.loads((REPO/pair[0]['station_manifest']).read_text())
        datasets={'gate':v2.make_dataset(base),'Val-A':v2.make_dataset(base,validation=True)}
        assert len(station['cases'])==44 and len(datasets['Val-A'])==36
        old=json.loads((REPO/'docs/reports/artifacts/selfsup_mrgr_v2/gate_summary.json').read_text())
        cases=[]
        for record in station['cases']:
            ds=datasets[record['cohort']];i=record['dataset_index']; sample=ds[i]
            actual={k:sample[k]['input_station_indices'].tolist() for k in v2.KEYS}
            assert sample['A']['case_id']==record['case_id'] and actual=={k:record[k] for k in v2.KEYS}
            if record['cohort']=='gate':
                assert actual==old['station_protocol'][record['case_id']]
                assert len(record['U'])==4 and not set(record['U'])&set(record['A']+record['B']+record['H'])
            cases.append(record)
        gate=[r for r in cases if r['cohort']=='gate'];val=[r for r in cases if r['cohort']=='Val-A']
        assert [r['dataset_index'] for r in gate]==list(range(8))
        assert [r['dataset_index'] for r in val]==list(range(36))
        assert set(r['case_id'] for r in gate).isdisjoint(r['case_id'] for r in val)
        states=[]
        for cfg in pair:
            seed_everything(cfg['seed']);model=v2.initialize_fresh_model(base)
            assert sum(p.numel() for p in model.parameters())==332070
            states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()});del model
        identical=all(torch.equal(states[0][k],states[1][k]) for k in states[0])
        assert identical
        init_hash=state_hash(states[0])
        write_json(root/'case_station_protocol.json',dict(cases=cases,train_order=[r['case_id'] for r in gate],
            evaluation_order=[r['case_id'] for r in val],seed=20261011,U_seed=2026101201,station_epoch=0))
        source_files=sorted(set(list((REPO/'src').rglob('*.py'))+list((REPO/'src').rglob('*.cpp'))+
            [Path(__file__),REPO/'tests/test_rvtc_matched.py',*(REPO/p for p in CONFIGS),REPO/pair[0]['base_config']]))
        sources={p.relative_to(REPO).as_posix():digest(p) for p in source_files}
        testlog=root.parent/'selfsup_rvtc_matched_preflight_tests.log'
        if not testlog.exists() or not testlog.read_text().rstrip().endswith('OK'):raise RuntimeError('Required preflight tests missing/failed')
        protocol=dict(status='REGISTERED BEFORE ANY EXPERIMENT OPTIMIZATION',registered_at=now(),
            review_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            configs=pair,base_config=base,source_sha256=sources,preflight_tests_sha256=digest(testlog),
            parameter_count=332070,initial_state_sha256=init_hash,initial_states_tensor_equal=identical,
            station_manifest_sha256=digest(root/'case_station_protocol.json'),train_cases=8,val_cases=36,
            train_loss='hold + 0.5 cross + 0.1 observed + hold_teacher + 0.1 observed_teacher + arm_weight teacher',
            teacher='same network; original v2 target and denominator detach; H/U never network inputs',
            fresh_random=True,external_weights_permitted=False,train_steps_per_epoch=4,total_steps_per_arm=320,
            gradient_cases=[r['case_id'] for r in gate+val[:4]],gradient_epochs=list(range(10,81,10)),
            gradients='fixed checkpoint; applied weight; Control training consistency zero; separate hypothetical 0.1 gradient',
            diagnostics_preserve='model tensors, grads, RNG, training mode; no additional optimizer updates',
            checkpoint_rule='initial; own last each epoch; fixed epoch10/20/30/40/50/60/70/80; evaluate epoch80 only',
            primary='Val-A36 same-case Treatment-Control sparse H relative L1; mean A/B errors, never volume ensemble',
            auxiliary=['Val-A H relative L2','gate U4 sparse relative L1/L2','within-arm teacher vs student','volume discrepancy and denominator','scales/noncollapse/set properties'],
            bootstrap=pair[0]['bootstrap'],conclusion_rules=dict(benefit='mean < 0 and entire 95% CI < 0',
                inconclusive='CI includes zero',harm='mean > 0 and entire 95% CI > 0'),
            interpretation='single-seed exploratory measurement-only; no 3D accuracy, clustered degradation or cross-seed claim',
            engineering=dict(stop_on=['nonfinite values','forbidden reads','protocol mismatch'],
                set_tolerance=1e-5,collapse='original v2 output_collapsed at fixed final, logged; no post-hoc threshold',
                teacher_difference_windows='diagnostic only, never a run gate'),
            no_formal300=True,no_new_model=True,no_gt_pool_b_test=True,original_no_go_retained=True)
        write_json(root/'inventory_before.json',before);write_json(root/'protocol.json',protocol)
        registration_hash=digest(root/'protocol.json');(root/'protocol.sha256').write_text(registration_hash+'\n')
        for cfg,state in zip(pair,states):
            p=root/'checkpoints'/cfg['arm'];p.mkdir(parents=True)
            # Initial states are freshly constructed separately, not loaded from any source.
            atomic_checkpoint(p/'initial.pt',dict(model=state,initialization='fresh random',arm=cfg['arm'],
                initial_state_sha256=init_hash,protocol_sha256=registration_hash,epoch=0,updates=0))
        write_json(root/'registration_audit.json',g.summary())
        print(f'Registered {registration_hash}; identical initial state {init_hash}; 0 updates',flush=True)


def verify(pair,base,root):
    p=json.loads((root/'protocol.json').read_text())
    assert digest(root/'protocol.json')==(root/'protocol.sha256').read_text().strip()
    assert p['configs']==pair and p['base_config']==base
    assert all(digest(REPO/k)==v for k,v in p['source_sha256'].items()),'Registered numerical sources changed'
    assert digest(root/'case_station_protocol.json')==p['station_manifest_sha256']
    return p


def work(base,root):
    ds={'gate':v2.make_dataset(base),'Val-A':v2.make_dataset(base,validation=True)}
    rows=json.loads((root/'case_station_protocol.json').read_text())['cases']
    return [(r,ds[r['cohort']]) for r in rows]


def batch_for(record,ds,device,include_u=False):
    sample=fixed_sample(ds,record['dataset_index'],{k:record[k] for k in v2.KEYS},record['U'] if include_u and record['cohort']=='gate' else None)
    assert sample['A']['case_id']==record['case_id']
    return collate_sample(sample,device)


def smoke(pair,base,root):
    protocol=verify(pair,base,root)
    target=root/'memory_smoke.json'
    if target.exists():assert json.loads(target.read_text())['passed'];print('Existing smoke verified');return
    g=guard_for(root,base);results={}
    with g:
        gate=[item for item in work(base,root) if item[0]['cohort']=='gate']
        index=max(range(8),key=lambda i:len(gate[i][0]['T']))
        for cfg in pair:
            seed_everything(cfg['seed']);model=v2.initialize_fresh_model(base)
            assert state_hash(model.state_dict())==protocol['initial_state_sha256']
            before=state_hash(model.state_dict());torch.cuda.reset_peak_memory_stats(cfg['device'])
            batch=batch_for(*gate[index],cfg['device']);model.train()
            # Allocate AdamW state to estimate full-step memory, without executing step.
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
            for p in model.parameters():optimizer.state[p].update(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
            began=time.perf_counter()
            for _ in range(2):
                _,parts,volumes,projections=v2.objective(model,batch)
                (contrast_loss(parts,cfg['consistency_weight'])/2).backward()
                del parts,volumes,projections
            raw=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.zero_grad(set_to_none=True)
            assert state_hash(model.state_dict())==before
            props={k:set_properties(model,batch[k]) for k in ('A','B','T')}
            assert all(p['passed'] for p in props.values())
            with diagnostic_context(model):branch,vectors=gradient_diagnostic(model,batch)
            assert branch['teacher_consistency_output_gradient_zero'] and branch['student_A_gradient_dot_error']>0 and branch['student_B_gradient_dot_error']>0
            assert branch['teacher_measurement_output_gradient_norm']>0
            geometry=applied_gradient_geometry(vectors,cfg['arm'])
            if cfg['arm']=='Control':assert geometry['applied']['consistency_norm']==0
            results[cfg['arm']]=dict(passed=True,case_id=gate[index][0]['case_id'],max_registered_T_views=len(gate[index][0]['T']),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(cfg['device']),seconds=time.perf_counter()-began,
                raw_accumulated_gradient_norm=float(raw),state_unchanged=True,optimizer_steps=0,
                set_properties=props,branch_checks=branch,gradient_geometry=geometry)
            del model,batch,optimizer,vectors;torch.cuda.empty_cache()
        assert v2.audit_is_clean(g.summary())
        write_json(target,dict(passed=True,results=results,gt_access=g.summary(),protocol_sha256=digest(root/'protocol.json'),
            smoke='two microbatch backward passes, AdamW state buffers, clip; no step; reset before training'))
        print(json.dumps(results,indent=2),flush=True)


def epoch_run(model,gate,device,weight,optimizer=None):
    model.train(optimizer is not None)
    records=[];steps=[];moments={k:[] for k in ('A','B','T')};examples={k:{} for k in moments}
    totals={};scores={};began=time.perf_counter()
    if optimizer:optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(optimizer is not None):
        for index,(r,ds) in enumerate(gate):
            batch=batch_for(r,ds,device)
            _,parts,volumes,projections=v2.objective(model,batch)
            loss=contrast_loss(parts,weight)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite contrast loss')
            if optimizer:
                (loss/2).backward()
                if (index+1)%2==0:
                    raw=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    clipped=float(torch.sqrt(sum(p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None)))
                    steps.append(dict(update_in_epoch=(index+1)//2,raw_gradient_norm=float(raw),post_clip_gradient_norm=clipped,
                        clipping_applied=float(raw)>1.,clip_coefficient=min(1.,1./(float(raw)+1e-6))))
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
            for k,value in dict(total=loss,**parts).items():totals[k]=totals.get(k,0.)+float(value.detach())/8
            for k,values in v2.projection_scores(projections,batch).items():scores.setdefault(k,[]).extend(values)
            ca,cb,ct=(v.detach() for v in volumes)
            row=dict(case_id=r['case_id'],order=index,loss=float(loss.detach()),teacher_relative_discrepancy=float(parts['teacher'].detach()),
                A_output_mean=float(ca.mean()),B_output_mean=float(cb.mean()),T_output_mean=float(ct.mean()),T_output_std=float(ct.std(unbiased=False)))
            records.append(row)
            for k,c in zip(('A','B','T'),(ca,cb,ct)):
                pred=c[0];mean=float(pred.mean());std=float(pred.std(unbiased=False))
                moments[k].append((float(pred.min()),float(pred.max()),mean,std*std+mean*mean,float((pred>=.999).float().mean()),float((pred<=1e-5).float().mean())))
                examples[k][r['case_id']]=pred.cpu()
            del loss,parts,volumes,projections,ca,cb,ct,c,pred,batch
    assert len(records)==8 and (len(steps)==4 if optimizer else len(steps)==0)
    return dict(statistic='within-epoch changing-parameter mean' if optimizer else 'fixed-parameter evaluation',
        loss=totals,measurement={k:{m:sum(v[m] for v in rows)/len(rows) for m in ('relative_l1','relative_l2')} for k,rows in scores.items()},
        output=v2.output_statistics(moments['A']+moments['B'],examples['A']),teacher_output=v2.output_statistics(moments['T'],examples['T']),
        steps=steps,case_order=[r['case_id'] for r in records],microbatch_records=records,seconds=time.perf_counter()-began)


def checkpoint_diagnostics(model,items,cfg,epoch,protocol,sha):
    rows=[]
    with diagnostic_context(model):
        for r,ds in items:
            batch=batch_for(r,ds,cfg['device'])
            values,vectors=gradient_diagnostic(model,batch)
            geom=applied_gradient_geometry(vectors,cfg['arm'])
            rows.append(dict(arm=cfg['arm'],epoch=epoch,case_id=r['case_id'],cohort=r['cohort'],checkpoint_sha256=sha,
                statistic='fixed checkpoint; no updates; weighted raw gradients',
                **{'applied_'+k:v for k,v in geom['applied'].items()},
                **{'hypothetical_0p1_'+k:v for k,v in geom['hypothetical_original_weight_consistency'].items()},
                teacher_detach=values['teacher_consistency_output_gradient_zero'],student_A_direction=values['student_A_gradient_dot_error'],
                student_B_direction=values['student_B_gradient_dot_error'],teacher_measurement_gradient_norm=values['teacher_measurement_output_gradient_norm']))
            assert rows[-1]['teacher_detach'] and rows[-1]['student_A_direction']>0 and rows[-1]['student_B_direction']>0
            assert rows[-1]['teacher_measurement_gradient_norm']>0
            if cfg['arm']=='Control':assert rows[-1]['applied_consistency_norm']==0
            del batch,vectors
    return rows


def train(pair,base,root,arm):
    protocol=verify(pair,base,root);cfg=next(c for c in pair if c['arm']==arm)
    assert json.loads((root/'memory_smoke.json').read_text())['passed']
    folder=root/arm;folder.mkdir(exist_ok=True)
    completed=folder/'training_summary.json'
    if completed.exists():
        s=json.loads(completed.read_text());assert s['completed_epochs']==80 and s['updates']==320
        print(f'{arm} already complete; no repeat training',flush=True);return
    g=guard_for(root,base,(arm,));last=root/'checkpoints'/arm/'last.pt';generator=torch.Generator()
    with g:
        seed_everything(cfg['seed']);model=v2.initialize_fresh_model(base)
        assert state_hash(model.state_dict())==protocol['initial_state_sha256']
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        items=work(base,root);gate=[it for it in items if it[0]['cohort']=='gate']
        diag=[it for it in items if it[0]['case_id'] in protocol['gradient_cases']]
        rows=[];gradient_rows=[];start=0;updates=0;prior=[]
        if last.exists():
            saved=torch.load(last,map_location=cfg['device'],weights_only=False)
            assert saved['arm']==arm and saved['protocol_sha256']==digest(root/'protocol.json') and saved['config']==cfg
            assert saved['updates']==saved['epoch']*4 and saved['epoch']<=80
            assert state_hash(saved['model'])==saved['state_sha256']
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],generator)
            start=saved['epoch'];updates=saved['updates'];rows=saved['metrics'];prior=saved['audit_sessions']
            gradient_rows=read_lines(folder/'gradient_trajectory.jsonl')
            gradient_rows=[r for r in gradient_rows if r['epoch']<=start]
            # A partially written diagnostic is not a completed diagnostic.
            at_boundary=[r for r in gradient_rows if r['epoch']==start]
            if len(at_boundary) != len(diag):
                gradient_rows=[r for r in gradient_rows if r['epoch']!=start]
            print(f'{arm} resume only own last: epoch {start}, updates {updates}',flush=True)
        elif (folder/'training_started.json').exists():raise RuntimeError('Started experiment lacks own resumable checkpoint; do not reinitialize')
        else:
            write_json(folder/'training_started.json',dict(started_at=now(),initial_state_sha256=state_hash(model.state_dict()),protocol_sha256=digest(root/'protocol.json')))
            with diagnostic_context(model):initial=epoch_run(model,gate,cfg['device'],cfg['consistency_weight'])
            write_json(folder/'initial_fixed_evaluation.json',initial)
            atomic_checkpoint(last,dict(arm=arm,config=cfg,epoch=0,updates=0,model=model.state_dict(),optimizer=optimizer.state_dict(),
                rng=rng_state(generator),protocol_sha256=digest(root/'protocol.json'),state_sha256=state_hash(model.state_dict()),metrics=[],audit_sessions=[]))
        # Preserve any interrupted tails explicitly; committed checkpoint metrics govern resume.
        for name,keep in [('metrics.jsonl',rows),('gradient_trajectory.jsonl',gradient_rows)]:
            path=folder/name
            if path.exists() and read_lines(path)!=keep:path.rename(folder/(name+'.interrupted_'+str(time.time_ns())))
            path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in keep))
        write_json(folder/'environment.json',environment(base))
        # Finish an interrupted diagnostic at its own saved boundary before continuing.
        if start in protocol['gradient_epochs'] and not any(r['epoch']==start for r in gradient_rows):
            fixed=root/'checkpoints'/arm/f'epoch_{start:03d}.pt'
            if not fixed.exists():atomic_checkpoint(fixed,torch.load(last,map_location=cfg['device'],weights_only=False))
            drows=checkpoint_diagnostics(model,diag,cfg,start,protocol,digest(fixed))
            for r in drows:append_json(folder/'gradient_trajectory.jsonl',r)
        torch.cuda.reset_peak_memory_stats(cfg['device'])
        for epoch in range(start+1,81):
            verify(pair,base,root)
            tr=epoch_run(model,gate,cfg['device'],cfg['consistency_weight'],optimizer);updates+=4
            assert tr['case_order']==json.loads((root/'case_station_protocol.json').read_text())['train_order']
            assert updates==epoch*4 and v2.audit_is_clean(g.summary())
            row=dict(arm=arm,epoch=epoch,updates=updates,train=tr,learning_rate=optimizer.param_groups[0]['lr'])
            rows.append(row)
            payload=dict(arm=arm,config=cfg,epoch=epoch,updates=updates,model=model.state_dict(),optimizer=optimizer.state_dict(),
                rng=rng_state(generator),protocol_sha256=digest(root/'protocol.json'),state_sha256=state_hash(model.state_dict()),
                metrics=rows,audit_sessions=prior+[g.summary()],fresh_random_initial_state_sha256=protocol['initial_state_sha256'])
            atomic_checkpoint(last,payload);append_json(folder/'metrics.jsonl',row)
            if epoch in protocol['gradient_epochs']:
                path=root/'checkpoints'/arm/f'epoch_{epoch:03d}.pt';atomic_checkpoint(path,payload)
                for r in checkpoint_diagnostics(model,diag,cfg,epoch,protocol,digest(path)):append_json(folder/'gradient_trajectory.jsonl',r)
            write_json(folder/'gt_access_audit.json',dict(current=g.summary(),prior=prior))
            print(f'{arm} {epoch}/80 updates={updates}: sparseH={tr["measurement"]["heldout"]["relative_l1"]:.6f} teacherH={tr["measurement"]["teacher_heldout"]["relative_l1"]:.6f} D={tr["loss"]["teacher"]:.6f} rawGmax={max(s["raw_gradient_norm"] for s in tr["steps"]):.4f}',flush=True)
        assert updates==320 and len(rows)==80
        with diagnostic_context(model):
            final=epoch_run(model,gate,cfg['device'],cfg['consistency_weight'])
            batch=batch_for(*gate[0],cfg['device']);properties={k:set_properties(model,batch[k]) for k in ('A','B','T')}
        assert all(p['passed'] for p in properties.values()) and v2.audit_is_clean(g.summary())
        summary=dict(completed=True,arm=arm,completed_epochs=80,updates=320,final_fixed_evaluation=final,set_properties=properties,
            output_collapsed=v2.output_collapsed(final['output']) or v2.output_collapsed(final['teacher_output']),
            consistency_first10=float(np.mean([r['train']['loss']['teacher'] for r in rows[:10]])),
            consistency_last10=float(np.mean([r['train']['loss']['teacher'] for r in rows[-10:]])),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(cfg['device']),
            final_checkpoint=str(root/'checkpoints'/arm/'epoch_080.pt'),final_checkpoint_sha256=digest(root/'checkpoints'/arm/'epoch_080.pt'),
            final_state_sha256=state_hash(model.state_dict()),initial_state_sha256=protocol['initial_state_sha256'],gt_access=g.summary(),prior_audits=prior,
            protocol_sha256=digest(root/'protocol.json'),finished_at=now(),stopping_reason='fixed epoch80; no selection or early stopping')
        write_json(completed,summary)


def evaluate(pair,base,root):
    protocol=verify(pair,base,root)
    if (root/'paired_summary.json').exists():print('Complete paired evaluation already exists; no repeat');return
    g=guard_for(root,base);measurements=[];summaries={}
    with g:
        items=work(base,root)
        for cfg in pair:
            completed=json.loads((root/cfg['arm']/'training_summary.json').read_text())
            path=Path(completed['final_checkpoint']);assert path==root/'checkpoints'/cfg['arm']/'epoch_080.pt'
            assert digest(path)==completed['final_checkpoint_sha256']
            saved=torch.load(path,map_location=cfg['device'],weights_only=False)
            assert saved['epoch']==80 and saved['updates']==320 and saved['protocol_sha256']==digest(root/'protocol.json')
            seed_everything(cfg['seed']);model=v2.initialize_fresh_model(base);model.load_state_dict(saved['model'])
            with diagnostic_context(model),torch.no_grad():
                for r,ds in items:
                    batch=batch_for(r,ds,cfg['device'],include_u=True)
                    volumes=tuple(model(batch[k]) for k in ('A','B','T'))
                    actual=measurement_rows(volumes,batch,('H','U') if r['cohort']=='gate' else ('H',),r['cohort'])
                    measurements.extend(dict(arm=cfg['arm'],checkpoint_sha256=completed['final_checkpoint_sha256'],**row) for row in actual)
                    del batch,volumes
            del model,saved
            summaries[cfg['arm']]=completed
        write_csv(root/'measurement_per_case.csv',measurements)
        index={(r['arm'],r['cohort'],r['domain'],r['case_id']):r for r in measurements}
        pairs=[];effects={};teacher={}
        for cohort,domain in [('Val-A','H'),('gate','U'),('gate','H')]:
            case_ids=[r['case_id'] for r,ds in items if r['cohort']==cohort]
            for cid in case_ids:
                a,b=[index[(arm,cohort,domain,cid)] for arm in ('Control','Treatment')]
                assert a['target_station_ids']==b['target_station_ids']
                pairs.append(dict(cohort=cohort,domain=domain,case_id=cid,station_ids=a['target_station_ids'],
                    **{f'{arm}_{m}':row['student_'+m] for arm,row in [('Control',a),('Treatment',b)] for m in ('relative_l1','relative_l2')},
                    **{f'Treatment_minus_Control_{m}':b['student_'+m]-a['student_'+m] for m in ('relative_l1','relative_l2')}))
            group=[r for r in pairs if r['cohort']==cohort and r['domain']==domain]
            effects[cohort+'_'+domain]={m:paired_effect([r['Control_'+m] for r in group],[r['Treatment_'+m] for r in group]) for m in ('relative_l1','relative_l2')}
            for arm in ('Control','Treatment'):
                selected=[index[(arm,cohort,domain,cid)] for cid in case_ids]
                teacher[arm+'_'+cohort+'_'+domain]=paired_summary(selected,10000,2026101202)
        write_csv(root/'paired_per_case.csv',pairs)
        inv=json.loads((root/'inventory_before.json').read_text());current=inventory(base)
        assert inv==current,'Original data/artifact/numerical code metadata changed'
        integrity=dict(dataset_metadata_unchanged=True,source_gt128_metadata_unchanged=True,all_historical_artifacts_unchanged=True,
            original_numerical_sources_unchanged=True,original_no_go_retained=not json.loads((REPO/'docs/reports/artifacts/selfsup_mrgr_v2/gate_summary.json').read_text())['passed'],
            GT_content_not_read=True,paired_final_epoch80=True,equal_initial_states=True,equal_steps=True)
        assert v2.audit_is_clean(g.summary())
        write_json(root/'evaluation_audit.json',g.summary());write_json(root/'integrity.json',integrity)
        write_json(root/'paired_summary.json',dict(completed=True,evaluated_at=now(),protocol_sha256=digest(root/'protocol.json'),
            effects=effects,teacher_advantage=teacher,training=summaries,integrity=integrity,evaluation_audit=g.summary(),
            primary=effects['Val-A_H']['relative_l1'],single_seed_exploratory=True,volumetric_accuracy_evaluated=False))
        print(json.dumps(effects,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['register','smoke','train','evaluate'])
    parser.add_argument('--arm',choices=['Control','Treatment'])
    args=parser.parse_args();pair,base,root=configs();torch.set_num_threads(2)
    try:
        if args.phase=='register':register(pair,base,root)
        elif args.phase=='smoke':smoke(pair,base,root)
        elif args.phase=='train':
            if not args.arm:parser.error('train requires --arm')
            train(pair,base,root,args.arm)
        else:evaluate(pair,base,root)
    except BaseException as e:
        if root.exists():write_json(root/f'failure_{args.phase}_{args.arm or "both"}_{time.time_ns()}.json',dict(error=repr(e),at=now(),automatic_tuning=False))
        raise


if __name__=='__main__':main()
