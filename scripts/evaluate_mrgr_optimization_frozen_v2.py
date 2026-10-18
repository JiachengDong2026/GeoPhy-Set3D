#!/usr/bin/env python3
"""Batch-sealed matched validation reconstruction; labels only after every prediction."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
import train_mrgr_optimization_formal_v2 as formal
import test_mrgr_multiscale_frozen_v1 as previous

base, ROOT = formal.base, formal.ROOT
DATA = previous.DATA
REFERENCE = previous.OUT


def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def write(path, value): formal.old.write(path, value)
def sha(path): return base.digest(Path(path))
def jsonl(path, values): Path(path).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in values))


def verify(out):
    out = Path(out)
    r = read(out/'registration.json')
    assert sha(out/'registration.json') == (out/'registration.sha256').read_text().strip()
    for path, digest in r['evidence_sha256'].items():
        assert sha(path) == digest, path
    for model in r['models']:
        p = formal.verify(Path(model['run']))
        assert p['candidate'] == model['candidate']
        assert sha(model['checkpoint']) == model['checkpoint_sha256']
    verify_reference(r)
    return r


def register(out, runs, decision):
    out = Path(out).resolve()
    assert not out.exists(), 'Refuse to replace evaluation'
    assert decision and Path(decision).is_file(), 'A frozen promotion decision is required'
    old = read(REFERENCE/'registration.json')
    evidence_paths = [Path(__file__), Path(formal.__file__), Path(previous.__file__),
        ROOT/'scripts/report_historical_reconstruction_v1.py',
        ROOT/'scripts/analyze_frozen_ab_morphology_v1.py',
        REFERENCE/'registration.json', REFERENCE/'per_case_metrics.jsonl',
        REFERENCE/'prediction_manifest.jsonl', REFERENCE/'predictions_complete.json',
        previous.RUN/'metrics.jsonl', previous.RUN/'summary.json',
        DATA/'source_index.jsonl', DATA/'train_statistics.json', DATA/'geometry/train_geometry.npz', Path(decision)]
    source = [x for x in rows(DATA/'source_index.jsonl') if x['split']=='val']
    cases = [x['case_id'] for x in source]
    assert cases == old['cases'] and len(cases)==36
    verify_reference(dict(cases=cases,settings=old['settings']))
    models = []
    for run in map(lambda x:Path(x).resolve(), runs):
        p = formal.verify(run)
        summary = read(run/'summary.json')
        assert summary['completed'] and (summary['epochs'],summary['updates'])==(300,24900)
        assert not summary['final_test_evaluated']
        assert not any(summary['gt_audit'][k] for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        checkpoint = run/'checkpoints/best.pt'
        assert sha(checkpoint)==summary['best_checkpoint_sha256']
        assert set(cases)==set(p['validation_case_ids'])
        models.append(dict(key=run.name, run=str(run), candidate=p['candidate'], seed=p['seed'],
            parameters=p['parameters'], checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            best_epoch=summary['best_epoch'], config=p['config']))
        evidence_paths += [run/'protocol.json', run/'summary.json']
    assert len({m['key'] for m in models})==len(models)
    evidence_paths += [DATA/x['train_projection_path'] for x in source]
    r = dict(registered_at=now(), models=models, cases=cases, settings=copy.deepcopy(old['settings']),
        reconstructions_per_model=612, saved_fields_per_model=1224,
        split='val', geometry_pool='train', grid='64^3 XYZ endpoints',
        primary='ordinary fixedV4 C1 3DrelativeL1 paired vs frozen multiscale and any concurrent control',
        secondary='3DrelativeL2,P90,cluster5 within-case mean,V3/V6/spread,centroid,peak,Dice,background',
        significance_rule='Describe practical significance (>=3% ordinary3D L1 reduction, ordinaryP90/cluster5 no >2% relative worsening) and pairedcase intervals separately; no claim of seed-robust significance from case bootstrap',
        selection='lowest formal validationH only; all candidates/weights/predictions sealed before first newGT score',
        evidence_sha256={str(p.resolve()):sha(p) for p in evidence_paths},
        no_training_or_refinement=True, final_test_evaluated=False,
        statistics='36 pairedcases, five draws averaged within case,20000 bootstrap seed20261017; exploratory no multiplicity correction',
        inference_GT_forbidden=True, GT128_forbidden=True,
        limitations='Development validationGT has been examined historically; no blind-test claim or post-score tuning')
    out.mkdir(parents=True)
    for m in models: (out/m['key']/'predictions').mkdir(parents=True)
    write(out/'registration.json', r)
    (out/'registration.sha256').write_text(sha(out/'registration.json')+'\n')
    print(json.dumps(dict(registered=str(out),models=[m['key'] for m in models])),flush=True)


def predict(out, key, device):
    out = Path(out)
    r = verify(out)
    model_info = next(m for m in r['models'] if m['key']==key)
    target = out/key
    assert not (target/'prediction_manifest.jsonl').exists()
    assert not list((target/'predictions').iterdir()), 'Refuse to overwrite partial inference'
    began = time.perf_counter()
    records = []
    with base.GTReadGuard(DATA, Path(model_info['run'])/'checkpoints') as guard:
        ckpt = torch.load(model_info['checkpoint'], map_location='cpu', weights_only=False)
        assert not ckpt['volumetric_gt_used'] and ckpt['epoch']==model_info['best_epoch']
        assert ckpt['protocol_sha256']==sha(Path(model_info['run'])/'protocol.json')
        model = formal.model_for(ckpt['config'], model_info['candidate'], device)
        model.load_state_dict(ckpt['model'], strict=True)
        model.eval()
        assert sum(p.numel() for p in model.parameters())==model_info['parameters']
        torch.cuda.reset_peak_memory_stats(device)
        for setting in r['settings']:
            ds = previous.MeasurementOnlyPoolDataset(DATA, split='val', geometry_pool='train', evaluation_only=True,
                min_input_views=setting['views'], max_input_views=setting['views'], hold_views=0,
                base_seed=20261011, normalization='linear_p99_5', statistics_file='train_statistics.json')
            ds.sampler = previous.Replay(setting)
            assert [x['case_id'] for x in ds.records]==r['cases']
            with torch.no_grad():
                for i in range(len(ds)):
                    sample = ds[i]
                    case = sample['case_id']
                    assert sample['input_station_indices'].tolist()==setting['input_ids'][case]
                    assert sample['hold_station_indices'].tolist()==setting['heldout_ids'][case]
                    batch = previous.move_batch(previous.projection_pool_collate([sample]), device)
                    inputs = {k:v for k,v in batch.items() if not k.startswith('hold_')}
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    stages = model(inputs, return_stages=True)
                    torch.cuda.synchronize(device)
                    seconds = time.perf_counter()-t0
                    measurement = {}
                    for prefix,label in [('input','observed'),('hold','heldout')]:
                        pred, ref, valid = previous.project_measurements(stages['C1'], batch, prefix)[0]
                        measurement[label] = base.relative_projection_metrics(pred,ref,valid)
                    fields = {k:stages[k][0,0].cpu().numpy() for k in ('C0','C1')}
                    assert all(v.shape==(64,64,64) and np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in fields.values())
                    path = target/'predictions'/f"{setting['condition']}_draw{setting['draw']}_{i:03d}.npz"
                    np.savez_compressed(path, **fields)
                    records.append(dict(case_id=case,condition=setting['condition'],draw=setting['draw'],
                        input_ids=setting['input_ids'][case],heldout_ids=setting['heldout_ids'][case],
                        prediction=str(path.relative_to(target)),sha256=sha(path),measurement=measurement,
                        reconstruct_seconds=seconds))
            print(f"{key} {setting['condition']} draw{setting['draw']} {len(records)}/612",flush=True)
        audit = guard.summary()
        assert not any(audit[k] for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    assert len(records)==612
    jsonl(target/'prediction_manifest.jsonl',records)
    write(target/'predictions_complete.json',dict(completed_at=now(),all_predictions_saved_before_gt=True,
        checkpoint_sha256=model_info['checkpoint_sha256'],registration_sha256=sha(out/'registration.json'),
        manifest_sha256=sha(target/'prediction_manifest.jsonl'), reconstructions=612,fields=1224,audit=audit,
        seconds=time.perf_counter()-began,device=device,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))


def verify_prediction_rows(registration, records):
    settings={(s['condition'],s['draw']):s for s in registration['settings']}
    expected={(s['condition'],s['draw'],c) for s in registration['settings'] for c in registration['cases']}
    assert len(records)==len(expected)==612
    assert {(x['condition'],x['draw'],x['case_id']) for x in records}==expected
    for x in records:
        s=settings[x['condition'],x['draw']]
        assert x['input_ids']==s['input_ids'][x['case_id']]
        assert x['heldout_ids']==s['heldout_ids'][x['case_id']]
        assert not set(x['input_ids']) & set(x['heldout_ids'])


def verify_metric_rows(registration, records):
    expected={(s['condition'],s['draw'],c,stage) for s in registration['settings']
        for c in registration['cases'] for stage in ('C0','C1')}
    assert len(records)==len(expected)==1224
    assert {(x['condition'],x['draw'],x['case_id'],x['stage']) for x in records}==expected


def verify_reference(registration):
    """Explicitly match the frozen comparator, including layouts and score stages."""
    old=previous.verify_registration()
    assert registration['cases']==old['cases'] and registration['settings']==old['settings']
    boundary=read(REFERENCE/'predictions_complete.json')
    assert boundary['all_predictions_saved_before_gt']
    assert boundary['reconstructions']==612 and boundary['fields']==1224
    assert boundary['registration_sha256']==sha(REFERENCE/'registration.json')
    assert boundary['manifest_sha256']==sha(REFERENCE/'prediction_manifest.jsonl')
    assert boundary['checkpoint_sha256']==old['checkpoint_sha256']
    verify_prediction_rows(registration,rows(REFERENCE/'prediction_manifest.jsonl'))
    verify_metric_rows(registration,rows(REFERENCE/'per_case_metrics.jsonl'))


def seal(out):
    out=Path(out)
    assert not (out/'batch_predictions_sealed.json').exists()
    r=verify(out)
    boundaries={}
    for m in r['models']:
        target=out/m['key']
        b=read(target/'predictions_complete.json')
        assert b['all_predictions_saved_before_gt'] and b['reconstructions']==612
        assert b['checkpoint_sha256']==m['checkpoint_sha256']
        assert b['registration_sha256']==sha(out/'registration.json')
        assert b['manifest_sha256']==sha(target/'prediction_manifest.jsonl')
        rr=rows(target/'prediction_manifest.jsonl')
        verify_prediction_rows(r,rr)
        for x in rr: assert sha(target/x['prediction'])==x['sha256']
        boundaries[m['key']]=dict(manifest_sha256=b['manifest_sha256'], boundary_sha256=sha(target/'predictions_complete.json'))
    write(out/'batch_predictions_sealed.json',dict(sealed_at=now(),all_models_complete=True,
        registration_sha256=sha(out/'registration.json'),models=boundaries,
        new_GT_scoring_started=False,final_test_evaluated=False))
    print('All model predictions sealed; only a separate score process may read validationGT.',flush=True)


def paired(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape==(36,)
    d=a-b
    idx=np.random.default_rng(20261017).integers(0,36,(20000,36))
    return dict(mean=float(a.mean()),reference_mean=float(b.mean()),mean_delta=float(d.mean()),
        relative_error_reduction=float(1-a.mean()/b.mean()),ci95=np.quantile(d[idx].mean(1),[.025,.975]).tolist(),
        wins=int((d < -1e-8).sum()),ties=int((abs(d)<=1e-8).sum()),losses=int((d>1e-8).sum()))


def score(out):
    out=Path(out)
    r=verify(out)
    boundary=read(out/'batch_predictions_sealed.json')
    assert boundary['all_models_complete'] and boundary['registration_sha256']==sha(out/'registration.json')
    assert not (out/'summary.json').exists(), 'Refuse to replace completed scoring'
    # Validate the entire batch, including every file, before opening the first label.
    manifests={}
    for m in r['models']:
        target=out/m['key']
        b=boundary['models'][m['key']]
        assert sha(target/'predictions_complete.json')==b['boundary_sha256']
        assert sha(target/'prediction_manifest.jsonl')==b['manifest_sha256']
        rr=rows(target/'prediction_manifest.jsonl')
        verify_prediction_rows(r,rr)
        for x in rr: assert sha(target/x['prediction'])==x['sha256']
        manifests[m['key']]=rr
    source={x['case_id']:x for x in rows(DATA/'source_index.jsonl') if x['split']=='val'}
    gt={c:np.load(DATA/source[c]['gt_volume_64_path'],allow_pickle=False).squeeze().astype(np.float64) for c in r['cases']}
    assert len(gt)==36
    baseline=rows(REFERENCE/'per_case_metrics.jsonl')
    all_records={'frozen_multiscale':baseline}
    for m in r['models']:
        target=out/m['key']
        scored=[]
        for x in manifests[m['key']]:
            with np.load(target/x['prediction'],allow_pickle=False) as fields:
                for stage in ('C0','C1'):
                    pred=fields[stage].astype(np.float64)
                    scored.append(dict(case_id=x['case_id'],condition=x['condition'],draw=x['draw'],stage=stage,
                        volume=previous.volume_metrics(pred,gt[x['case_id']]),morphology=previous.morphology_metrics(pred,gt[x['case_id']])))
        verify_metric_rows(r,scored)
        jsonl(target/'per_case_metrics.jsonl',scored)
        all_records[m['key']]=scored
        print(f"Scored {m['key']}:1224 fields",flush=True)
    indices={key:{(x['condition'],x['draw'],x['case_id'],x['stage']):x for x in rr} for key,rr in all_records.items()}
    panels=[]
    for condition in ('Val-A-V4','V3','V6','spread-V4','clustered-V4'):
        for draws in ([[0]] if condition in ('Val-A-V4','spread-V4') else [[0],list(range(5))]):
            for key,index in indices.items():
                for metric in ('relative_l1','relative_l2'):
                    values=np.array([np.mean([index[condition,d,c,'C1']['volume'][metric] for d in draws]) for c in r['cases']])
                    contrasts={}
                    for other,ref in indices.items():
                        if other==key:continue
                        ref_values=[np.mean([ref[condition,d,c,'C1']['volume'][metric] for d in draws]) for c in r['cases']]
                        contrasts[other]=paired(values,ref_values)
                    panels.append(dict(model=key,condition=condition,draws=draws,metric=metric,mean=float(values.mean()),
                        median=float(np.median(values)),p90=float(np.quantile(values,.9)),maximum=float(values.max()),comparisons=contrasts))
    morphology=[]
    for condition in ('Val-A-V4','clustered-V4'):
        for key,index in indices.items():
            for stage in ('C0','C1'):
                rr=[index[condition,0,c,stage]['morphology'] for c in r['cases']]
                values={k:float(np.mean([x[k] for x in rr])) for k in rr[0] if np.isscalar(rr[0][k])}
                values['absolute_peak_ratio_error']=float(np.mean([abs(x['peak_ratio']-1) for x in rr]))
                morphology.append(dict(model=key,condition=condition,draw=0,stage=stage,mean=values))
    summary=dict(completed_at=now(),panels=panels,morphology=morphology,models=r['models'],
        sources_preserved=True,development_validation_only=True,final_test_evaluated=False,
        GT_access=dict(first_phase='separate score after fullbatch seal',GT64_files=36,GT128_loaded=False,test_loaded=False,
            GT64_sha256={str(DATA/source[c]['gt_volume_64_path']):sha(DATA/source[c]['gt_volume_64_path']) for c in r['cases']}),
        boundary_sha256=sha(out/'batch_predictions_sealed.json'))
    write(out/'summary.json',summary)
    print('Frozen reconstruction scoring complete.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','predict','seal','score','verify'))
    parser.add_argument('--out',required=True)
    parser.add_argument('--runs',nargs='*')
    parser.add_argument('--decision')
    parser.add_argument('--key')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if args.action=='register':register(args.out,args.runs,args.decision)
    elif args.action=='predict':predict(args.out,args.key,args.device)
    elif args.action=='seal':seal(args.out)
    elif args.action=='score':score(args.out)
    else:verify(args.out);print('Evaluation registration, numerical sources and checkpoints verified')


if __name__=='__main__':main()
