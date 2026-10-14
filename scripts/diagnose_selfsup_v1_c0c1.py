#!/usr/bin/env python3
"""Frozen C0/C1 diagnosis: seal implementation -> infer -> independent GT64."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import assert_no_gt
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import (
    DiagnosticReadGuard, capture_context, stage_amplitudes, METRICS)
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input, Y_SCALE, PROJECTOR_SAMPLES
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import move_batch, seed_everything, write_json
from diagnose_clustered_one_swap import dataset, target_gt, checkpoint_provenance, load_frozen, snapshot, write_csv

ARTIFACT = REPO/'docs/reports/artifacts/selfsup_v1_c0c1_geometry'


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def configuration(path):
    c = yaml.safe_load(Path(path).read_text())
    return c, Path(c['paths']['output'])


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    write_json(temp, value)
    os.replace(temp, path)


def committed(path):
    assert subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(REPO))], cwd=REPO) == path.read_bytes(), 'Commit frozen artifacts before inference'


def verify(c, out, require_implementation=True):
    p = json.loads((out/'protocol.json').read_text())
    assert p['config'] == c
    assert digest(out/'protocol.json') == (out/'protocol.sha256').read_text().strip()
    for section in ['metadata_sha256', 'historical_sha256']:
        for path, h in p[section].items():
            assert digest(path) == h, path
    for path, h in p['protected_source_sha256'].items():
        assert digest(REPO/path) == h, path
    for path, h in p['registration_sha256'].items():
        assert digest(out/path) == h, path
    assert digest(p['freeze']['checkpoint']) == c['checkpoint_sha256']
    committed(ARTIFACT/'protocol.json')
    if require_implementation:
        manifest = json.loads((out/'implementation_manifest.json').read_text())
        assert manifest['protocol_sha256'] == digest(out/'protocol.json')
        for path, h in manifest['source_sha256'].items():
            assert digest(REPO/path) == h, path
        committed(ARTIFACT/'implementation_manifest.json')
    return p


def seal(c, out):
    p = verify(c, out, False)
    models = {'SelfSup':dict(config=c['model_config'], checkpoint=p['freeze']['checkpoint'],
                            sha256=c['checkpoint_sha256'], freeze_record=c['paths']['freeze_record'])}
    with DiagnosticReadGuard(c['paths']['dataset'], [p['freeze']['checkpoint']], out, 'inference') as guard:
        provenance = checkpoint_provenance(dict(models=models))
        assert provenance['SelfSup']['parameter_count'] == 332070
        write_json(out/'checkpoint_provenance.json', provenance)
        write_json(out/'provenance_access.json', guard.summary())
    paths = [REPO/'src/geophy_set3d/evaluation/selfsup_c0c1_diagnostic.py', Path(__file__),
             REPO/'scripts/report_selfsup_v1_c0c1.py', REPO/'tests/test_selfsup_c0c1_diagnostic.py']
    value = dict(sealed_at=now(), protocol_sha256=digest(out/'protocol.json'),
                 provenance_sha256=digest(out/'checkpoint_provenance.json'),
                 source_sha256={str(q.relative_to(REPO)):digest(q) for q in paths},
                 registration_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    path = out/'implementation_manifest.json'
    if path.exists():
        previous=json.loads(path.read_text())
        assert previous['source_sha256']==value['source_sha256']
        return
    write_json(path, value)
    for name in ['implementation_manifest.json','checkpoint_provenance.json','provenance_access.json']:
        shutil.copy2(out/name, ARTIFACT/name)
    print('Implementation sealed; commit manifest before inference', flush=True)


def representative(c, r):
    return (r['case_index'] in c['representative_case_indices'] and
            r['draw'] == c['representative_draw'] and r['group'] in c['representative_groups'])


def token(record, mode):
    return f'{mode}_case{record["case_index"]}_{record["geometry_id"]}'


def expected(c, out, mode):
    records = json.loads((out/'station_subsets.json').read_text())
    return [r for r in records if mode in ('normal','zero') or representative(c,r)]


def valid_saved(row):
    assert digest(row['prediction_file']) == row['prediction_sha256']
    return row


def load_saved(c, out, mode):
    rows = []
    for r in expected(c,out,mode):
        path = out/'prediction_rows'/f'{token(r,mode)}.json'
        if path.exists():
            row = valid_saved(json.loads(path.read_text()))
            assert row['record'] == r and row['mode'] == mode
            rows.append(row)
    return rows


def tensor_hash(*values):
    h = hashlib.sha256()
    for value in values:
        a = value.detach().cpu().contiguous().numpy()
        h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def modes_complete(c, out, modes):
    rows = {m:load_saved(c,out,m) for m in modes}
    for mode in modes:
        assert len(rows[mode]) == len(expected(c,out,mode))
        manifest = out/f'{mode}_predictions_complete.json'
        value = dict(mode=mode,rows=len(rows[mode]),prediction_rows_sha256={
            token(r['record'],mode):digest(out/'prediction_rows'/f'{token(r["record"],mode)}.json') for r in rows[mode]},
            protocol_sha256=digest(out/'protocol.json'),all_outputs_saved_before_GT=True)
        if manifest.exists():
            assert json.loads(manifest.read_text()) == value
        else:
            atomic_json(manifest,value)
    return rows


def infer(c, out, device, mode, smoke=False):
    p = verify(c,out)
    if mode != 'normal':
        assert json.loads((out/'normal_reproduction.json').read_text())['passed']
    records = expected(c,out,mode)
    done = {token(r['record'],mode):r for r in load_saved(c,out,mode)}
    active = [r for r in records if token(r,mode) not in done and
              (not smoke or r['case_index']==0 and r['draw']==0)]
    if not active:
        if not smoke: modes_complete(c,out,[mode])
        print(mode, 'verified existing predictions; no repeated inference', flush=True)
        return
    (out/'case_images').mkdir(exist_ok=True);(out/'prediction_rows').mkdir(exist_ok=True)
    seed_everything(c['original_seed']);torch.set_num_threads(2)
    model_cfg = dict(models={'SelfSup':dict(config=c['model_config'],checkpoint=p['freeze']['checkpoint'],sha256=c['checkpoint_sha256'])},device=device)
    provenance = json.loads((out/'checkpoint_provenance.json').read_text())
    assert digest(out/'checkpoint_provenance.json') == json.loads((out/'implementation_manifest.json').read_text())['provenance_sha256']
    start = time.perf_counter()
    with DiagnosticReadGuard(c['paths']['dataset'],[p['freeze']['checkpoint']],out,'inference') as audit:
        model = load_frozen(model_cfg,'SelfSup',dict(models=provenance))
        before = state_hash(model.state_dict());ds=dataset(dict(dataset=c['paths']['dataset'],original_seed=c['original_seed']))
        assert ds.y_ref == Y_SCALE and PROJECTOR_SAMPLES == 96 and len(ds)==36
        assert [r['case_id'] for r in ds.records] == list(dict.fromkeys(r['case_id'] for r in json.loads((out/'station_subsets.json').read_text())))
        settings = dict(torch_version=str(torch.__version__),cuda=torch.version.cuda,device=device,
                        gpu=torch.cuda.get_device_name(device),FP32=True,TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                        TF32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
                        cudnn_deterministic=torch.backends.cudnn.deterministic,
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),threads=torch.get_num_threads())
        with (out/'inference_events.jsonl').open('a',buffering=1) as ledger, torch.no_grad():
            for index in sorted({r['case_index'] for r in active}):
                source,projections=ds._projections(index)
                for r in [x for x in active if x['case_index']==index]:
                    assert source['case_id']==r['case_id']
                    empty=np.empty(0,dtype=np.int64)
                    batch=move_batch(projection_pool_collate([ds._measurement_sample(source,projections,np.asarray(r['station_ids']),empty)]),device)
                    hb=move_batch(projection_pool_collate([ds._measurement_sample(source,projections,np.asarray(r['H_station_ids']),empty)]),device)
                    assert_no_gt(batch);assert_no_gt(hb)
                    assert batch['input_station_indices'][0].tolist()==r['station_ids']
                    assert hb['input_station_indices'][0].tolist()==r['H_station_ids'] and int(batch['view_mask'].sum())==4
                    ident=token(r,mode);ledger.write(json.dumps(dict(at=now(),token=ident,event='forward_started'))+'\n')
                    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);begin=time.perf_counter()
                    with capture_context(model) as captured:
                        stages=model(batch,return_stages=True,residual_mode=mode)
                    torch.cuda.synchronize(device);seconds=time.perf_counter()-begin
                    ledger.write(json.dumps(dict(at=now(),token=ident,event='forward_returned'))+'\n')
                    for key in ['C0','C1']:
                        value=stages[key]
                        assert value.shape==(1,1,64,64,64) and value.dtype==torch.float32
                        assert torch.isfinite(value).all() and value.min()>=0 and value.max()<=1
                    amplitudes=stage_amplitudes(stages,captured['context0'],batch,c['ratio_epsilon'])
                    visible=stages['visibility_base'] & batch['view_mask'][:,:,None,None,None]
                    local=captured['local_observability'];local_mask=visible.permute(0,2,3,4,1).reshape(-1,4).any(1)
                    local_mean=local[local_mask].double().mean(0).cpu().tolist() if local_mask.any() else [None]*5
                    measures={}
                    for key in ['C0','C1']:
                        measures[key]={}
                        for prefix,b in [('observed',batch),('heldout',hb)]:
                            projected=stages['measurement_prediction_Y0'] if key=='C0' and prefix=='observed' else project_input(stages[key],b)
                            valid=b['input_valid_ray'][0,:,0].bool() & b['view_mask'][0,:,None,None]
                            value=relative_projection_metrics(projected[0,:,0],b['input_projection_raw'][0,:,0],valid)
                            measures[key].update({prefix+'_'+k:v for k,v in value.items()})
                    arrays={k:stages[k][0,0].cpu().numpy() for k in ['C0','C1']}
                    comparison={}
                    if mode!='normal':
                        normal=json.loads((out/'prediction_rows'/f'{token(r,"normal")}.json').read_text())
                        with np.load(normal['prediction_file'],allow_pickle=False) as original:
                            assert np.array_equal(arrays['C0'],original['C0']), 'Shortcut must preserve C0'
                            delta=arrays['C1'].astype(np.float64)-original['C1']
                            comparison=dict(c0_identical_to_normal=True,output_vs_normal_mae=float(np.abs(delta).mean()),
                                            output_vs_normal_rms=float(np.sqrt(np.mean(delta**2))),output_vs_normal_max=float(np.abs(delta).max()))
                    if representative(c,r) and mode=='normal':
                        arrays['visible_view_count_32']=visible[0].sum(0).cpu().numpy()
                    file=out/'case_images'/f'{ident}.npz';temp=file.with_suffix('.tmp')
                    with temp.open('wb') as f: np.savez(f,**arrays)
                    os.replace(temp,file)
                    row=dict(record=r,mode=mode,amplitudes=amplitudes,measurements=measures,comparison=comparison,
                             local_observability_mean=local_mean,prediction_file=str(file),prediction_sha256=digest(file),
                             input_mask_sha256=tensor_hash(batch['view_mask'],batch['input_valid_ray'],stages['visibility_base']),
                             H_mask_sha256=tensor_hash(hb['view_mask'],hb['input_valid_ray']),
                             input_identity_sha256=tensor_hash(batch['input_projection_raw'],batch['input_station_indices'],batch['input_station_origin_m'],batch['input_path_length_m']),
                             checkpoint_sha256=c['checkpoint_sha256'],protocol_sha256=digest(out/'protocol.json'),
                             source_manifest_sha256=digest(out/'implementation_manifest.json'),historical_rows_sha256=p['historical_sha256'][str(Path(c['paths']['historical'])/'rows.jsonl')],
                             forward_seconds=seconds,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),settings=settings)
                    if mode!='normal':
                        for k in ['input_mask_sha256','H_mask_sha256','input_identity_sha256']:assert row[k]==normal[k]
                    atomic_json(out/'prediction_rows'/f'{ident}.json',row)
                    ledger.write(json.dumps(dict(at=now(),token=ident,event='output_saved',prediction_sha256=row['prediction_sha256']))+'\n')
                    done[ident]=row
                print(f'{mode}: case {index+1}/36, {len(done)}/{len(records)} saved, {time.perf_counter()-start:.1f}s',flush=True)
        assert state_hash(model.state_dict())==before and all(q.grad is None for q in model.parameters())
        assert audit.counts['val_gt64_opens']==audit.counts['forbidden_attempts']==0
        sessions=out/'inference_sessions.jsonl'
        with sessions.open('a') as f:f.write(json.dumps(dict(mode=mode,smoke=smoke,actual_forwards=len(active),parameters_buffers_unchanged=True,access=audit.summary(),settings=settings,elapsed_seconds=time.perf_counter()-start))+'\n')
    if not smoke:modes_complete(c,out,[mode])


def evaluate(c, out, modes):
    p=verify(c,out);saved=modes_complete(c,out,modes)
    for mode in modes:
        path=out/f'{mode}_evaluated.jsonl'
        prior=read_rows(path) if path.exists() else []
        done={(r['case_id'],r['geometry_id']) for r in prior};assert len(done)==len(prior)
        saved_by_key={(r['record']['case_id'],r['record']['geometry_id']):r for r in saved[mode]}
        assert done <= set(saved_by_key)
        for r in prior:
            assert r['mode']==mode and r['prediction_sha256']==saved_by_key[(r['case_id'],r['geometry_id'])]['prediction_sha256']
        if len(done)==len(saved[mode]):
            assert digest(path)==json.loads((out/f'{mode}_gt_complete.json').read_text())['rows_sha256']
            print(mode,'GT evaluation verified existing; no repeated evaluation',flush=True)
            continue
        records=[json.loads(l) for l in (Path(c['paths']['dataset'])/'source_index.jsonl').read_text().splitlines()]
        ds=SimpleNamespace(root=Path(c['paths']['dataset']),records=[r for r in records if r['split']=='val'])
        targets={};torch.set_num_threads(2)
        with DiagnosticReadGuard(c['paths']['dataset'],[],out,'gt') as audit, path.open('a',buffering=1) as f, torch.no_grad():
            for row in saved[mode]:
                r=row['record'];key=(r['case_id'],r['geometry_id'])
                if key in done:continue
                assert ds.records[r['case_index']]['case_id']==r['case_id']
                if r['case_index'] not in targets:targets[r['case_index']]=target_gt(ds,r['case_index'])
                target=targets[r['case_index']];value=dict(r,mode=mode,input_station_ids=r['station_ids'],checkpoint_sha256=c['checkpoint_sha256'],
                    protocol_sha256=row['protocol_sha256'],source_manifest_sha256=row['source_manifest_sha256'],prediction_sha256=row['prediction_sha256'],
                    historical_rows_sha256=row['historical_rows_sha256'],input_mask_sha256=row['input_mask_sha256'],H_mask_sha256=row['H_mask_sha256'],
                    input_identity_sha256=row['input_identity_sha256'],**row['amplitudes'],**row['comparison'],forward_seconds=row['forward_seconds'],local_observability_mean=row['local_observability_mean'])
                with np.load(row['prediction_file'],allow_pickle=False) as arrays:
                    for stage in ['C0','C1']:
                        m=VolumeMetrics();m.update(torch.from_numpy(arrays[stage])[None,None],target,[r['case_id']])
                        value.update({stage+'_'+k:v for k,v in m.rows[0].items() if k!='case_id'})
                        value.update({stage+'_'+k:v for k,v in row['measurements'][stage].items()})
                for k in METRICS:value['delta_'+k]=value['C1_'+k]-value['C0_'+k]
                f.write(json.dumps(value,allow_nan=False)+'\n');done.add(key)
            assert audit.counts['forbidden_attempts']==audit.counts['checkpoint_opens']==audit.counts['val_pool_a_projection_opens']==0
            write_json(out/f'{mode}_gt_access.json',audit.summary())
        assert len(done)==len(saved[mode])
        write_json(out/f'{mode}_gt_complete.json',dict(rows=len(done),rows_sha256=digest(path),prediction_manifest_sha256=digest(out/f'{mode}_predictions_complete.json'),saved_predictions_only=True))
    if 'normal' in modes:
        historical={(r['case_id'],r['geometry_id']):r for r in read_rows(Path(c['paths']['historical'])/'rows.jsonl')}
        errors={k:[] for k in METRICS};failures=[]
        for row in read_rows(out/'normal_evaluated.jsonl'):
            old=historical[row['case_id'],row['geometry_id']]
            assert row['input_station_ids']==old['station_ids'] and row['H_station_ids']==old['H_station_ids']
            for k in METRICS:
                delta=abs(row['C1_'+k]-old[k]);tol=c['reproduction_tolerance']['absolute']+c['reproduction_tolerance']['relative']*abs(old[k]);errors[k].append(delta)
                if delta>tol:failures.append(dict(case_id=row['case_id'],geometry_id=row['geometry_id'],metric=k,difference=delta,tolerance=tol))
        value=dict(passed=not failures,rows=900,comparisons=9000,tolerance=c['reproduction_tolerance'],max_absolute_differences={k:max(v) for k,v in errors.items()},failures=failures)
        write_json(out/'normal_reproduction.json',value)
        assert value['passed'], 'C1 historical mismatch; stop interventions and investigate without loosening tolerance'
        print('Normal C1 reproduced all 900 rows / 9000 metrics',json.dumps(value['max_absolute_differences']),flush=True)


def integrity(c,out):
    verify(c,out)
    before=json.loads((out/'inventory_before.json').read_text());parent=out.parents[1]
    assert snapshot(Path(c['paths']['dataset']))==before['dataset']
    for path,record in before['history'].items():assert snapshot(parent/path)==record,path
    for path,record in before['gt128_metadata'].items():
        p=Path(path);assert dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,inode=p.stat().st_ino)==record
    events=read_rows(out/'inference_events.jsonl');counts={}
    for mode in ['normal','zero','signflip','wrong_pairing']:
        rows=load_saved(c,out,mode);assert len(rows)==len(expected(c,out,mode))
        counts[mode]={event:sum(r['event']==event and r['token'].startswith(mode+'_') for r in events) for event in ['forward_started','forward_returned','output_saved']}
        assert all(v==len(rows) for v in counts[mode].values())
    value=dict(passed=True,dataset_metadata_unchanged=True,historical_metadata_unchanged=True,GT128_metadata_only_unchanged=True,
               protected_source_hashes_unchanged=True,original_checkpoint_hash_unchanged=True,
               forward_counts=counts,total_actual_forwards=sum(v['forward_returned'] for v in counts.values()),
               optimizer_steps=0,GT128_content_reads=0,Pool_B_used=False,final_test_used=False,
               protocol_sha256=digest(out/'protocol.json'),source_manifest_sha256=digest(out/'implementation_manifest.json'))
    write_json(out/'integrity.json',value);print(json.dumps(value,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(REPO/'configs/diagnose_selfsup_v1_c0c1.yaml'))
    parser.add_argument('phase',choices=['seal','infer','gt','verify'])
    parser.add_argument('--mode',choices=['normal','zero','signflip','wrong_pairing'],default='normal')
    parser.add_argument('--device',default='cuda:2');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();c,out=configuration(args.config)
    if args.phase=='seal':seal(c,out)
    elif args.phase=='infer':infer(c,out,args.device,args.mode,args.smoke)
    elif args.phase=='gt':evaluate(c,out,[args.mode])
    else:integrity(c,out)

if __name__=='__main__':main()
