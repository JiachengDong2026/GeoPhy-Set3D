#!/usr/bin/env python3
"""Independent frozen A/B validation on historical draws 1..4; never test.

Register all layouts, save all 864 fields, then score in another process.
The five-draw summary aggregates layouts within case before paired bootstrap.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest, timestamp, write_json
from geophy_set3d.evaluation.research import frozen_record, EvaluationWithHoldout
from geophy_set3d.training.gt_read_guard import GTReadGuard

BASE = ROOT / 'outputs/research/ab_remaining_draws_v1'
REFERENCE = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v1/selfsup_oari_mrgr_v1')
RUNS = dict(A=ROOT / 'outputs/research/scheme_a_budget_v4_wide_seed20261011',
            B=ROOT / 'outputs/research/scheme_b_perf_v2_formal_seed20261011')
CONDITIONS = dict(V3=3, V6=6, **{'clustered-V4':4})


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


class Replay:
    def __init__(self, setting):
        self.setting = setting
        self.views, self.draw_id, self.pool, self.protocol = setting['views'], setting['draw'], 'train', 'matched_'+setting['condition']

    def sample(self, station_count, *, epoch, sample_index, case_id):
        ids = np.asarray(self.setting['input_ids'][case_id], dtype=np.int64)
        assert len(ids)==self.views and ids.max()<station_count
        return ids, np.empty(0,dtype=np.int64)


def register():
    frozen = {m:frozen_record(r) for m,r in RUNS.items()}
    configs = {m:yaml.safe_load((r/'config_resolved.yaml').read_text()) for m,r in RUNS.items()}
    root = Path(configs['A']['dataset']['root'])
    assert configs['B']['dataset']['root'] == str(root)
    source = [r for r in rows(root/'source_index.jsonl') if r['split']=='val']
    expected = {r['case_id'] for r in source}; assert len(expected)==36
    settings = []
    for condition,views in CONDITIONS.items():
        for draw in range(1,5):
            reference = REFERENCE/condition/f'draw{draw}'/'C1/per_case_metrics.jsonl'
            r = rows(reference)
            assert len(r)==36 and {x['case_id'] for x in r}==expected
            ids = {x['case_id']:x['selected_station_ids'] for x in r}
            assert all(len(v)==len(set(v))==views and min(v)>=0 and max(v)<64 for v in ids.values())
            setting = dict(condition=condition,views=views,draw=draw,
                reference=str(reference),reference_sha256=digest(reference),input_ids=ids)
            sampler = EvaluationWithHoldout(Replay(setting))
            setting['heldout_ids'] = {case:sampler.sample(64,epoch=0,sample_index=i,case_id=case)[1].tolist()
                for i,case in enumerate(sorted(expected))}
            settings.append(setting)
    draw0 = []
    for condition,views in CONDITIONS.items():
        reference = REFERENCE/condition/'draw0/C1/per_case_metrics.jsonl'
        historical = {x['case_id']:x for x in rows(reference)}
        assert set(historical)==expected
        record = dict(condition=condition,draw=0,reference=str(reference),reference_sha256=digest(reference),models={})
        for model,run in RUNS.items():
            file = run/'evaluation'/f'val_train_matched_{condition}_v{views}_draw0'/'per_case_metrics.jsonl'
            r = rows(file)
            assert len(r)==36 and {x['case_id'] for x in r}==expected
            for x in r:
                assert x['selected_station_ids']==historical[x['case_id']]['selected_station_ids']
            record['models'][model] = dict(path=str(file),sha256=digest(file))
        draw0.append(record)
    geometry = np.load(root/'geometry/train_geometry.npz',allow_pickle=False)
    azimuths = geometry['station_azimuth_deg']
    for setting in settings:
        spans=[]
        for ids in setting['input_ids'].values():
            a=np.sort(np.mod(azimuths[np.array(ids)],360)); spans.append(float(360-np.max(np.diff(np.r_[a,a[0]+360]))))
        setting['angular_span_deg'] = dict(min=min(spans),max=max(spans),mean=float(np.mean(spans)))
    BASE.mkdir(parents=True,exist_ok=False)
    data_files=[root/'source_index.jsonl',root/'train_statistics.json',root/'geometry/train_geometry.npz']
    data_files.extend(root/r['train_projection_path'] for r in source)
    write_json(BASE/'registration.json',dict(
        registered_at=timestamp(),script_sha256=digest(Path(__file__)),settings=settings,draw0=draw0,
        models={m:dict(run=str(RUNS[m]),checkpoint=frozen[m]['checkpoint'],checkpoint_sha256=frozen[m]['sha256'],best_epoch=frozen[m]['best_epoch']) for m in RUNS},
        dataset_root=str(root),data_sha256={str(p):digest(p) for p in data_files},
        cases=36,new_layouts=12,new_fields=864,new_draws=[1,2,3,4],
        statistics='For all5 summary, average 5 draw errors within each case, then paired case bootstrap: n=36, resamples=10000, seed=20261016. Do not treat 180 layouts as independent cases.',
        budget='<=300 seconds measured prediction for both models; GPU1 only after release; no architecture/precision changes; no extra training.',
        development_validation_only=True,final_test_evaluated=False,gt_for_selection=False,gt_for_training=False,
        all_new_fields_saved_before_gt_required=True))
    print('Registered all 12 historical remaining-draw layouts x 36 cases x 2 models = 864 fields.',flush=True)


def predict(model_name,device):
    from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
    from geophy_set3d.data.collate import projection_pool_collate
    from geophy_set3d.training.utils import move_batch
    from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
    registration=json.loads((BASE/'registration.json').read_text())
    assert digest(Path(__file__))==registration['script_sha256']
    for p,h in registration['data_sha256'].items():assert digest(p)==h
    record=registration['models'][model_name]
    assert frozen_record(record['run'])['sha256']==record['checkpoint_sha256']
    output=BASE/model_name;output.mkdir(exist_ok=False)
    began=time.monotonic();manifest=[]
    torch.set_num_threads(2)
    with GTReadGuard(registration['dataset_root'],Path(record['run'])/'checkpoints') as guard:
        checkpoint=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
        config=checkpoint['config']
        if model_name=='A':
            from plan_a_budget_v4 import build_budget_model
            from plan_a_runtime_candidates import configure_backend
            configure_backend(config['execution_runtime']);model=build_budget_model(config)
        else:
            from research.plan_b.training import build_model
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            model=build_model(config)
        model.load_state_dict(checkpoint['model']);model=model.to(device).eval()
        del checkpoint
        previous_seconds=sum(json.loads(p.read_text())['elapsed_seconds'] for p in BASE.glob('*/prediction_boundary.json'))
        with torch.no_grad():
            for setting in registration['settings']:
                assert digest(setting['reference'])==setting['reference_sha256']
                views=setting['views']
                dataset=MeasurementOnlyPoolDataset(registration['dataset_root'],split='val',geometry_pool='train',
                    evaluation_only=True,min_input_views=views,max_input_views=views,hold_views=0,
                    base_seed=20261011,normalization='linear_p99_5',statistics_file='train_statistics.json')
                dataset.sampler=EvaluationWithHoldout(Replay(setting))
                for index in range(len(dataset)):
                    assert time.monotonic()-began+previous_seconds<300,'Registered 5-minute prediction budget exhausted'
                    sample=dataset[index];case=sample['case_id']
                    assert sample['input_station_indices'].tolist()==setting['input_ids'][case]
                    assert sample['hold_station_indices'].tolist()==setting['heldout_ids'][case]
                    batch=move_batch(projection_pool_collate([sample]),device)
                    inputs={k:v for k,v in batch.items() if not k.startswith('hold_')}
                    heldout={k:v for k,v in batch.items() if not k.startswith(('input_','hold_')) and k!='view_mask'}
                    heldout.update({'input_'+k[5:]:v for k,v in batch.items() if k.startswith('hold_') and k!='hold_view_mask'})
                    heldout['view_mask']=batch['hold_view_mask']
                    torch.cuda.synchronize();start=time.monotonic()
                    if model_name=='A':
                        field=model(inputs);observed=model.operator(field,inputs);hpred=model.operator(field,heldout)
                    else:
                        context=model.encode(inputs);field=model.sample_grid(context,64)
                        observed=model.integrate(context,inputs);hpred=model.integrate(context,heldout)
                    torch.cuda.synchronize();seconds=time.monotonic()-start
                    assert all(bool(torch.isfinite(t).all()) for t in (field,observed,hpred))
                    value=field[0,0].cpu().numpy();assert value.shape==(64,64,64) and value.min()>=0 and value.max()<=1
                    name=f"{setting['condition']}_draw{setting['draw']}_{index:03d}.npz";path=output/name
                    np.savez_compressed(path,field=value)
                    metrics={}
                    for domain,pred,target in [('observed',observed,inputs),('heldout',hpred,heldout)]:
                        valid=target['input_valid_ray'].bool()&target['view_mask'][:,:,None,None,None]
                        metrics[domain]=relative_projection_metrics(pred[0],target['input_projection_raw'][0],valid[0])
                    manifest.append(dict(model=model_name,condition=setting['condition'],draw=setting['draw'],case_id=case,
                        prediction=str(path.relative_to(BASE)),sha256=digest(path),measurement=metrics,seconds=seconds,
                        input_ids=setting['input_ids'][case],heldout_ids=setting['heldout_ids'][case]))
                print(f"{model_name} {setting['condition']} draw{setting['draw']} 36/36; elapsed={time.monotonic()-began:.1f}s",flush=True)
        audit=guard.summary()
    manifest_path=output/'manifest.jsonl';manifest_path.write_text(''.join(json.dumps(x)+'\n' for x in manifest))
    assert len(manifest)==432
    write_json(output/'prediction_boundary.json',dict(completed_at=timestamp(),elapsed_seconds=time.monotonic()-began,
        cases=36,fields=432,manifest_sha256=digest(manifest_path),checkpoint_sha256=record['checkpoint_sha256'],
        gt_read_audit=audit,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    print(f'Model {model_name} complete; no GT read.',flush=True)


def seal():
    r=json.loads((BASE/'registration.json').read_text())
    count=0;elapsed=0
    for m in RUNS:
        b=json.loads((BASE/m/'prediction_boundary.json').read_text());p=BASE/m/'manifest.jsonl'
        assert digest(p)==b['manifest_sha256']
        values=rows(p);assert len(values)==432
        assert not any(b['gt_read_audit'][k] for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        for x in values:assert digest(BASE/x['prediction'])==x['sha256']
        count+=len(values);elapsed+=b['elapsed_seconds']
    write_json(BASE/'all_predictions_complete.json',dict(completed_at=timestamp(),fields=count,
        registration_sha256=digest(BASE/'registration.json'),elapsed_seconds=elapsed,
        all_predictions_saved_before_gt=True,final_test_evaluated=False))
    print('Sealed 864 predictions; separate score process may now read validation GT.',flush=True)


def paired(values,reference):
    values,reference=np.asarray(values),np.asarray(reference);d=values-reference
    rng=np.random.default_rng(20261016)
    boot=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1)
    return dict(mean_delta=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        wins=int((d < -1e-8).sum()),losses=int((d > 1e-8).sum()),ties=int((abs(d)<=1e-8).sum()),
        relative_error_reduction=float(1-values.mean()/reference.mean()))


def score():
    from geophy_set3d.evaluation.research import volume_metrics
    r=json.loads((BASE/'registration.json').read_text());boundary=json.loads((BASE/'all_predictions_complete.json').read_text())
    assert boundary['fields']==864 and boundary['all_predictions_saved_before_gt']
    assert digest(BASE/'registration.json')==boundary['registration_sha256']
    index={x['case_id']:x for x in rows(Path(r['dataset_root'])/'source_index.jsonl')}
    # Validate the full boundary, checkpoints and references before opening GT.
    manifests=[]
    for m in RUNS:
        assert frozen_record(r['models'][m]['run'])['sha256']==r['models'][m]['checkpoint_sha256']
        b=json.loads((BASE/m/'prediction_boundary.json').read_text());p=BASE/m/'manifest.jsonl'
        assert digest(p)==b['manifest_sha256'];manifests.extend(rows(p))
    for s in r['settings']+r['draw0']:assert digest(s['reference'])==s['reference_sha256']
    for x in manifests:assert digest(BASE/x['prediction'])==x['sha256']
    scored=[];cache={}
    for x in manifests:
        case=x['case_id'];assert index[case]['split']=='val'
        if case not in cache:cache[case]=np.load(Path(r['dataset_root'])/index[case]['gt_volume_64_path'],allow_pickle=False)
        with np.load(BASE/x['prediction'],allow_pickle=False) as f:
            scored.append(dict(x,volume=volume_metrics(f['field'],cache[case])))
    (BASE/'per_case_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in scored))
    values={};historical={}
    for s in r['settings']:
        for x in rows(s['reference']):historical[(s['condition'],s['draw'],x['case_id'])]=x['relative_l1']
    for x in scored:values[(x['model'],x['condition'],x['draw'],x['case_id'])]=x['volume']['relative_l1']
    for s in r['draw0']:
        old={x['case_id']:x for x in rows(s['reference'])}
        for x in old.values():historical[(s['condition'],0,x['case_id'])]=x['relative_l1']
        for m,p in s['models'].items():
            assert digest(p['path'])==p['sha256']
            for x in rows(p['path']):
                assert x['selected_station_ids']==old[x['case_id']]['selected_station_ids']
                values[(m,s['condition'],0,x['case_id'])]=x['relative_l1']
    case_ids=sorted(cache);summary=[]
    for condition in CONDITIONS:
        for draw in list(range(5))+['all5']:
            draws=range(5) if draw=='all5' else [draw]
            h=np.array([np.mean([historical[(condition,d,c)] for d in draws]) for c in case_ids])
            a=np.array([np.mean([values[('A',condition,d,c)] for d in draws]) for c in case_ids])
            b=np.array([np.mean([values[('B',condition,d,c)] for d in draws]) for c in case_ids])
            summary.append(dict(condition=condition,draw=draw,independent_cases=36,draws_per_case=len(list(draws)),
                relative_l1=dict(history=float(h.mean()),A=float(a.mean()),B=float(b.mean())),
                case_averaged_l1_p90=dict(history=float(np.quantile(h,.9)),A=float(np.quantile(a,.9)),B=float(np.quantile(b,.9))),
                A_versus_history=paired(a,h),B_versus_history=paired(b,h),B_versus_A=paired(b,a)))
    write_json(BASE/'summary.json',dict(results=summary,prediction_elapsed_seconds=boundary['elapsed_seconds'],
        cases=36,seed=20261011,new_fields=864,all5_uses_existing_draw0=True,
        final_test_evaluated=False,development_validation_only=True,gt_for_training=False,gt_for_selection=False))
    for x in summary:
        if x['draw']=='all5':print(json.dumps(x),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('register','predict','seal','score'))
    p.add_argument('--model',choices=('A','B'));p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    {'register':register,'predict':lambda:predict(a.model,a.device),'seal':seal,'score':score}[a.action]()
