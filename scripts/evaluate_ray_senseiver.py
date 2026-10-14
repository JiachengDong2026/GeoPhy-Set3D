#!/usr/bin/env python3
"""Frozen comparison: measurement-only prediction, separate saved-output GT64."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import pandas as pd
import torch
from geophy_set3d.training.ray_senseiver import (load_comparison_config,verify_registration,sha256,build_comparison_model,check_prediction,tensor_identity,model_identity)
from geophy_set3d.training.utils import seed_everything,move_batch,write_json
from geophy_set3d.training.selfsup import make_dataset,make_loader,run_epoch,selection_metric
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import DiagnosticReadGuard
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.evaluation.ray_senseiver import pair_three,case_means,statistics
from geophy_set3d.models import build_model
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training.metrics import VolumeMetrics
from diagnose_clustered_one_swap import dataset,target_gt


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def verify(c, check_checkpoint=True):
    p=verify_registration(c);out=Path(c['comparison']['output']);freeze=json.loads((out/'frozen_best.json').read_text());complete=json.loads((out/'training_complete.json').read_text())
    assert complete['completed'] and complete['epochs']==freeze['completed_epochs']==300
    assert complete['optimizer_steps']==freeze['optimizer_steps']==24900 and not freeze['volumetric_gt_used']
    if check_checkpoint:assert sha256(freeze['checkpoint'])==freeze['sha256']
    manifest=json.loads((out/'evaluation_source_manifest.json').read_text())
    for path,h in manifest['sha256'].items():assert sha256(REPO/path)==h,path
    assert subprocess.check_output(['git','show','HEAD:docs/reports/artifacts/ray_senseiver_selfsup_v1/evaluation_source_manifest.json'],cwd=REPO)==(out/'evaluation_source_manifest.json').read_bytes()
    return out,p,freeze


def load_baseline(c,freeze,device):
    ck=torch.load(freeze['checkpoint'],map_location=device,weights_only=False)
    assert ck['config']==c and ck['source_manifest_sha256']==freeze['source_manifest_sha256'] and not ck['volumetric_gt_used']
    model=build_comparison_model(c,device);model.load_state_dict(ck['model']);model.eval();return model


def row_token(r):return f"case{r['case_index']}_{r['geometry_id']}"


def rows_saved(out,records):
    results=[]
    for r in records:
        path=out/'prediction_rows'/f'{row_token(r)}.json'
        if path.exists():
            row=json.loads(path.read_text());assert row['record']==r;assert sha256(row['prediction_file'])==row['prediction_sha256'];results.append(row)
    return results


def seal_predictions(out,records,freeze):
    rows=rows_saved(out,records);assert len(rows)==900
    value=dict(rows=900,all_outputs_saved_before_GT=True,checkpoint_sha256=freeze['sha256'],
               prediction_rows_sha256={row_token(r):sha256(out/'prediction_rows'/f'{row_token(r)}.json') for r in records})
    path=out/'predictions_complete.json'
    if path.exists():assert json.loads(path.read_text())==value
    else:write_json(path,value)


def infer(c,device):
    out,p,freeze=verify(c);records=json.loads((out/'station_subsets.json').read_text());existing=rows_saved(out,records);done={row_token(r['record']) for r in existing}
    if len(done)==900:
        seal_predictions(out,records,freeze);print('All 900 saved outputs verified; no duplicate inference',flush=True);return
    (out/'case_images').mkdir(exist_ok=True);(out/'prediction_rows').mkdir(exist_ok=True)
    seed_everything(c['training']['seed']);torch.set_num_threads(2);ref=pd.read_csv(Path(c['comparison']['deepsets_artifact'])/'deepsets_900.csv').set_index(['case_id','group','draw'])
    with DiagnosticReadGuard(c['dataset']['root'],[freeze['checkpoint']],out,'inference') as guard:
        model=load_baseline(c,freeze,device);before=state_hash(model.state_dict());ds=dataset(dict(dataset=c['dataset']['root'],original_seed=c['training']['seed']))
        assert len(ds)==36;began=time.perf_counter()
        with (out/'inference_events.jsonl').open('a',buffering=1) as events,torch.no_grad():
            for index in sorted({r['case_index'] for r in records if row_token(r) not in done}):
                source,projections=ds._projections(index)
                for r in [r for r in records if r['case_index']==index and row_token(r) not in done]:
                    assert source['case_id']==r['case_id'];empty=np.empty(0,dtype=np.int64)
                    b=move_batch(projection_pool_collate([ds._measurement_sample(source,projections,np.asarray(r['station_ids']),empty)]),device)
                    h=move_batch(projection_pool_collate([ds._measurement_sample(source,projections,np.asarray(r['H_station_ids']),empty)]),device)
                    ident=dict(input_identity_sha256=tensor_identity(b['input_projection_raw'],b['input_station_indices'],b['input_station_origin_m'],b['input_path_length_m']),
                               input_ray_view_sha256=tensor_identity(b['view_mask'],b['input_valid_ray']),H_mask_sha256=tensor_identity(h['view_mask'],h['input_valid_ray']))
                    ident['complete_input_sha256']=tensor_identity(*[b[k] for k in sorted(b) if isinstance(b[k],torch.Tensor)])
                    ident['complete_H_sha256']=tensor_identity(*[h[k] for k in sorted(h) if isinstance(h[k],torch.Tensor)])
                    old=ref.loc[(r['case_id'],r['group'],r['draw'])]
                    assert old['checkpoint_sha256']==p['deepsets_checkpoint']['sha256']
                    for key in ['input_identity_sha256','H_mask_sha256','complete_input_sha256','complete_H_sha256']:assert old[key]==ident[key],key
                    token=row_token(r);file=out/'case_images'/f'{token}.npz'
                    reused=file.exists()
                    if reused:
                        recovery=json.loads((out/'prediction_recovery.json').read_text())
                        assert recovery['token']==token and recovery['prediction_sha256']==sha256(file) and recovery['checkpoint_sha256']==freeze['sha256']
                        with np.load(file,allow_pickle=False) as a:pred=check_prediction(torch.from_numpy(a['prediction'].copy())[None,None].to(device),b)
                        seconds=None;peak=None
                        events.write(json.dumps(dict(at=now(),token=token,event='saved_forward_reused'))+'\n')
                    else:
                        events.write(json.dumps(dict(at=now(),token=token,event='forward_started'))+'\n')
                        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);t=time.perf_counter();pred=check_prediction(model(b),b);torch.cuda.synchronize(device)
                        seconds=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated(device)
                    measures={}
                    for prefix,batch in [('observed',b),('heldout',h)]:
                        projected=project_input(pred,batch);valid=batch['input_valid_ray'][0,:,0].bool()&batch['view_mask'][0,:,None,None]
                        metrics=relative_projection_metrics(projected[0,:,0],batch['input_projection_raw'][0,:,0],valid)
                        measures.update({prefix+'_'+k:v for k,v in metrics.items()})
                    file=out/'case_images'/f'{token}.npz';temporary=file.with_suffix('.tmp')
                    if not reused:
                        with temporary.open('wb') as f:np.savez(f,prediction=pred[0,0].cpu().numpy())
                        os.replace(temporary,file)
                    row=dict(record=r,**ident,**measures,prediction_file=str(file),prediction_sha256=sha256(file),checkpoint_sha256=freeze['sha256'],
                             protocol_sha256=sha256(out/'protocol.json'),saved_forward_reused=reused,forward_seconds=seconds,peak_gpu_allocated_bytes=peak,
                             output_min=float(pred.min()),output_max=float(pred.max()),output_mean=float(pred.mean()),output_std=float(pred.std(unbiased=False)))
                    write_json(out/'prediction_rows'/f'{token}.json',row);done.add(token)
                    events.write(json.dumps(dict(at=now(),token=token,event='saved',sha256=row['prediction_sha256']))+'\n')
                print(f'Senseiver: case {index+1}/36, {len(done)}/900 saved, {time.perf_counter()-began:.1f}s',flush=True)
        assert state_hash(model.state_dict())==before and all(q.grad is None for q in model.parameters())
        assert guard.counts['val_gt64_opens']==guard.counts['forbidden_attempts']==0
        write_json(out/'inference_access.json',guard.summary())
    seal_predictions(out,records,freeze)


def selector(c,device):
    out,p,freeze=verify(c)
    if (out/'selector_replay.json').exists():return
    seed_everything(c['training']['seed']);torch.set_num_threads(2)
    with DiagnosticReadGuard(c['dataset']['root'],[freeze['checkpoint']],out,'inference') as guard:
        model=load_baseline(c,freeze,device);ds=make_dataset(c,validation=True)
        runconfig=dict(c,training=dict(c['training'],device=device),loader=dict(num_workers=0,persistent_workers=False,pin_memory=True))
        loader=make_loader(ds,runconfig);result=run_epoch(model,loader,runconfig)
        expected=json.loads((out/'training_complete.json').read_text())['best_val_heldout_relative_l1']
        assert np.isclose(selection_metric(result),expected,atol=1e-8,rtol=1e-5)
        write_json(out/'selector_replay.json',dict(at=now(),passed=True,reference=expected,replayed=selection_metric(result),validation=result,actual_model_forwards=2*len(loader),predicted_case_volumes=2*len(ds),batch_size=runconfig['training']['batch_size'],access=guard.summary()))


def gt(c):
    out=Path(c['comparison']['output'])
    with DiagnosticReadGuard(c['dataset']['root'],[],out,'gt') as guard:
        return gt_saved(c,guard)


def gt_saved(c,guard):
    out,p,freeze=verify(c,check_checkpoint=False);complete=json.loads((out/'predictions_complete.json').read_text());assert complete['rows']==900 and complete['checkpoint_sha256']==freeze['sha256']
    assert json.loads((out/'selector_replay.json').read_text())['passed']
    records=json.loads((out/'station_subsets.json').read_text());saved=rows_saved(out,records);assert len(saved)==900
    for r in records:assert sha256(out/'prediction_rows'/f'{row_token(r)}.json')==complete['prediction_rows_sha256'][row_token(r)]
    if (out/'gt_complete.json').exists():
        assert sha256(out/'paired_900.csv')==json.loads((out/'gt_complete.json').read_text())['paired_sha256']
        print('Existing full GT results retained');return
    rows=[];torch.set_num_threads(2)
    write_json(out/'first_posthoc_gt_boundary.json',dict(at=now(),freeze_sha256=sha256(out/'frozen_best.json'),prediction_manifest_sha256=sha256(out/'predictions_complete.json'),training_epochs=300,optimizer_steps=24900))
    ds=dataset(dict(dataset=c['dataset']['root'],original_seed=c['training']['seed']))
    for index in range(36):
        target=target_gt(ds,index)
        for row in [x for x in saved if x['record']['case_index']==index]:
            with np.load(row['prediction_file'],allow_pickle=False) as file:prediction=torch.from_numpy(file['prediction'].copy())[None,None]
            m=VolumeMetrics();m.update(prediction,target,[row['record']['case_id']])
            flat={**row['record'],**{k:v for k,v in row.items() if k not in ['record','prediction_file']},**{k:v for k,v in m.rows[0].items() if k!='case_id'}};rows.append(flat)
        print(f'GT64 saved-output evaluation {index+1}/36',flush=True)
    assert guard.counts['forbidden_attempts']==0;write_json(out/'gt_access.json',guard.summary())
    baseline=pd.DataFrame(rows);baseline.to_csv(out/'senseiver_900.csv',index=False)
    reference=pd.read_csv(Path(c['comparison']['deepsets_artifact'])/'paired_900.csv')
    deep=pd.read_csv(Path(c['comparison']['deepsets_artifact'])/'deepsets_900.csv')
    paired=pair_three(baseline,reference,deep,records);paired.to_csv(out/'paired_900.csv',index=False)
    cases=case_means(paired);cases.to_csv(out/'case_group_180.csv',index=False)
    stats,tails,contrasts=statistics(cases,c);stats.to_csv(out/'paired_statistics.csv',index=False);tails.to_csv(out/'tails.csv',index=False);write_json(out/'contrasts.json',contrasts)
    write_json(out/'gt_complete.json',dict(at=now(),rows=900,case_groups=180,paired_sha256=sha256(out/'paired_900.csv'),checkpoint_sha256=freeze['sha256'],saved_predictions_only=True))


def timing(c,device):
    out,p,freeze=verify(c)
    if (out/'timing.json').exists():return
    seed_everything(c['training']['seed']);torch.set_num_threads(2);oursfreeze=p['reference_checkpoint'];records=json.loads((out/'station_subsets.json').read_text())
    selected=[r for r in records if r['case_index'] in c['comparison']['representative_cases'] and r['group'] in c['comparison']['representative_groups'] and r['draw']==0];assert len(selected)==9
    import yaml
    with DiagnosticReadGuard(c['dataset']['root'],[freeze['checkpoint'],oursfreeze['checkpoint'],p['deepsets_checkpoint']['checkpoint']],out,'inference') as guard:
        assert sha256(oursfreeze['checkpoint'])==oursfreeze['sha256']
        cfg=oursfreeze['config'];ck=torch.load(oursfreeze['checkpoint'],map_location=device,weights_only=False);assert ck['config']==cfg
        ours=build_model(cfg).to(device);ours.load_state_dict(ck['model']);ours.eval();del ck;deepfreeze=p['deepsets_checkpoint'];assert sha256(deepfreeze['checkpoint'])==deepfreeze['sha256']
        ck=torch.load(deepfreeze['checkpoint'],map_location=device,weights_only=False);deep=build_model(ck['config']).to(device);deep.load_state_dict(ck['model']);deep.eval();del ck
        models={'Ours':ours,'DeepSets':deep,'Senseiver':load_baseline(c,freeze,device)}
        ds=dataset(dict(dataset=c['dataset']['root'],original_seed=c['training']['seed']));rows=[]
        gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)
        with torch.no_grad():
            for r in selected:
                source,pr=ds._projections(r['case_index']);b=move_batch(projection_pool_collate([ds._measurement_sample(source,pr,np.asarray(r['station_ids']),np.empty(0,dtype=np.int64))]),device)
                for model in models.values():
                    for _ in range(c['comparison']['timing_warmup']):model(b)
                times={name:[] for name in models};peaks={name:[] for name in models}
                for repeat in range(c['comparison']['timing_repeats']):
                    names=list(models);offset=repeat%3;order=names[offset:]+names[:offset]
                    for name in order:
                        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);begin=time.perf_counter();pred=models[name](b);torch.cuda.synchronize(device)
                        times[name].append(time.perf_counter()-begin);peaks[name].append(torch.cuda.max_memory_allocated(device));del pred
                for name in models:
                    rows.append(dict(record=r,model=name,seconds=times[name],median_seconds=float(np.median(times[name])),p90_seconds=float(np.quantile(times[name],.9)),peak_allocated_bytes=max(peaks[name])))
        # Both models reside on GPU; subtract the other model's persistent tensor bytes.
        resident={name:sum(t.numel()*t.element_size() for t in list(m.parameters())+list(m.buffers())) for name,m in models.items()}
        for row in rows:row['peak_allocated_excluding_other_model_bytes']=row['peak_allocated_bytes']-sum(v for k,v in resident.items() if k!=row['model'])
        identities={name:dict(parameter_count=sum(x.numel() for x in m.parameters()),fp32_parameter_bytes=sum(x.numel()*x.element_size() for x in m.parameters())) for name,m in models.items()}
        write_json(out/'timing.json',dict(at=now(),device=device,gpu=torch.cuda.get_device_name(device),shared_gpu=True,gpu_before=gpu_before,gpu_after=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True),
                    batch=1,precision='FP32',TF32=False,warmup=10,repeats=30,actual_model_forwards=dict(Ours=360,DeepSets=360,Senseiver=360),runtime=c['runtime'],includes_ours_internal_projector=True,
                    peak_memory_definition='CUDA max allocated minus other co-resident model tensors; same-process allocator/library state may remain',identities=identities,
                    checkpoint_sha256_verified=dict(Ours=oursfreeze['sha256'],DeepSets=deepfreeze['sha256'],Senseiver=freeze['sha256']),checkpoint_bytes=dict(Ours=Path(oursfreeze['checkpoint']).stat().st_size,DeepSets=Path(deepfreeze['checkpoint']).stat().st_size,Senseiver=Path(freeze['checkpoint']).stat().st_size),rows=rows,access=guard.summary()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['selector','infer','gt','timing']);ap.add_argument('--config',default='configs/train_ray_senseiver_selfsup_v1.yaml');ap.add_argument('--device',default='cuda:0');a=ap.parse_args();c=load_comparison_config(a.config)
    if a.phase=='gt':gt(c)
    else:globals()[a.phase](c,a.device)

if __name__=='__main__':main()
