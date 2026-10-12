#!/usr/bin/env python3
"""One preregistered frozen-model clustered V4 replacement diagnostic; no training."""
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.data.projection_pool_dataset import artifact_path
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models import build_model
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training.utils import load_config,move_batch,write_json,seed_everything
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler
from geophy_set3d.evaluation.clustered_one_swap import (paired_station_sets,geometry_description,
    paired_statistics,ClusteredEvaluationReadGuard)

CONFIG=REPO/'configs/diagnose_clustered_one_swap.yaml'
METRICS=['relative_l1','relative_l2','mae','rmse','heldout_relative_l1','heldout_relative_l2']


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_lines(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_csv(path,rows):
    if not rows:raise ValueError('No actual rows')
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def configuration():
    cfg=yaml.safe_load(CONFIG.read_text())
    assert (cfg['split'],cfg['pool'],cfg['views'],cfg['draws'],cfg['precision'],cfg['parameter_count'])==('val','train',4,list(range(5)),'fp32',332070)
    assert cfg['center_m']==[.3,.3,.3] and cfg['bootstrap']['samples']==10000
    return cfg,Path(cfg['output'])


def read_guard(cfg,root,geometry_only=False):
    return ClusteredEvaluationReadGuard(cfg['dataset'],[r['checkpoint'] for r in cfg['models'].values()],root,geometry_only)


def dataset(cfg):
    return MeasurementOnlyPoolDataset(cfg['dataset'],split='val',geometry_pool='train',min_input_views=4,max_input_views=4,hold_views=0,
        base_seed=cfg['original_seed'],normalization='linear_p99_5',statistics_file='train_statistics.json')


def snapshot(path):
    path=Path(path)
    return {p.relative_to(path).as_posix():dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,inode=p.stat().st_ino)
            for p in sorted(path.rglob('*')) if p.is_file()}


def inventory(cfg):
    output=Path(cfg['output']).parents[1]
    raw={r['source_target_path']:dict(size=Path(r['source_target_path']).stat().st_size,mtime_ns=Path(r['source_target_path']).stat().st_mtime_ns,inode=Path(r['source_target_path']).stat().st_ino)
         for r in read_lines(Path(cfg['dataset'])/'source_index.jsonl')}
    return dict(dataset=snapshot(cfg['dataset']),source_gt128_metadata=raw,
        history={k:snapshot(output/k) for k in ('checkpoints','logs','figures','comparisons','experiments')},
        old_diagnostics={p.name:snapshot(p) for p in (output/'diagnostics').iterdir() if p.is_dir() and p.resolve()!=Path(cfg['output']).resolve()})


def history(cfg,model,draw):
    path=Path(cfg['models'][model]['historical_root'])/f'draw{draw}'/'C1/per_case_metrics.jsonl'
    if not path.exists():raise FileNotFoundError(path)
    return read_lines(path)


def checkpoint_provenance(cfg):
    result={};architecture=None
    for name,m in cfg['models'].items():
        path=Path(m['checkpoint'])
        if not path.is_file():raise FileNotFoundError(path)
        if digest(path)!=m['sha256']:raise ValueError('Frozen checkpoint hash mismatch')
        freeze=json.loads(Path(m['freeze_record']).read_text())
        sha=freeze['sha256'] if name=='SelfSup' else freeze['checkpoint_sha256']
        assert freeze['checkpoint']==str(path) and sha==m['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        base=load_config(REPO/m['config']);assert saved['config']==base
        if architecture is None:architecture=base['model']
        assert architecture==base['model'] and base['normalization']==dict(method='linear_p99_5',statistics_file='train_statistics.json')
        model=build_model(base);model.load_state_dict(saved['model']);model.eval()
        assert sum(p.numel() for p in model.parameters())==332070
        if name=='SelfSup':
            assert freeze['completed_epochs']==300 and not freeze['volumetric_gt_used']
            assert saved['volumetric_gt_used'] is False and saved['training_signal']=='TDLAS measurements only'
            tr=json.loads((path.parents[2]/'logs/selfsup_oari_mrgr_v1/training_summary.json').read_text())
            assert tr['completed_epochs']==300 and tr['completed'] and not tr['gt_access']['gt_open_attempts']
            assert tr['selection_metric']=='mean held-out measurement relative L1'
        result[name]=dict(checkpoint=str(path),sha256=sha,config=base,canonical_state_sha256=state_hash(model.state_dict()),
            parameter_count=332070,epoch_zero_based=saved['epoch'],training_source_commit=saved['git_commit'],freeze_record=str(m['freeze_record']),
            freeze_record_sha256=digest(m['freeze_record']))
    return result


def register(cfg,root):
    if (root/'protocol.json').exists():verify(cfg,root);print('Registration already exists; verified, not replaced');return
    if root.exists() and any(root.iterdir()):raise FileExistsError('Independent directory nonempty; do not overwrite')
    root.mkdir(parents=True)
    with read_guard(cfg,root,geometry_only=True) as audit:
        ds=dataset(cfg);assert len(ds)==36
        models=checkpoint_provenance(cfg)
        records=[];sources={}
        for draw in cfg['draws']:
            histories={name:history(cfg,name,draw) for name in cfg['models']}
            assert all(len(rows)==36 for rows in histories.values())
            lookup={name:{r['case_id']:r for r in rows} for name,rows in histories.items()}
            sampler=EvaluationViewSampler(ds.geometry['station_azimuth_deg'],4,draw,'train',cfg['original_seed'],'clustered')
            for index,row in enumerate(ds.records):
                original=lookup['SelfSup'][row['case_id']]['selected_station_ids']
                assert lookup['Supervised'][row['case_id']]['selected_station_ids']==original
                assert sampler.select(row['case_id']).tolist()==original
                sets=paired_station_sets(original,ds.geometry['station_origin_m'],row['case_id'],draw,cfg['random_seed'],cfg['heldout_seed'],cfg['center_m'])
                geom={condition:dict(station_ids=sets[condition],origins_m=ds.geometry['station_origin_m'][sets[condition]].tolist(),
                    azimuth_deg=ds.geometry['station_azimuth_deg'][sets[condition]].tolist(),
                    **geometry_description(sets[condition],ds.geometry['station_origin_m'],cfg['center_m'])) for condition in ('Original','Targeted','Random')}
                records.append(dict(case_id=row['case_id'],case_index=index,draw=draw,**sets,geometry=geom,
                    H_origins_m=ds.geometry['station_origin_m'][sets['H']].tolist(),H_azimuth_deg=ds.geometry['station_azimuth_deg'][sets['H']].tolist()))
            for name,m in cfg['models'].items():
                for file in ('per_case_metrics.jsonl','metrics.json'):
                    p=Path(m['historical_root'])/f'draw{draw}'/'C1'/file;sources[str(p)]=digest(p)
        write_json(root/'station_protocol.json',dict(records=records,selection_uses='frozen geometry and Original IDs only; never measurement/GT/model outputs',
            random_seed=cfg['random_seed'],heldout_seed=cfg['heldout_seed']))
        before=inventory(cfg);write_json(root/'inventory_before.json',before)
        test=Path(cfg['output']).parent/'clustered_one_swap_tests.log'
        if not test.exists() or not test.read_text().rstrip().endswith('OK'):raise RuntimeError('Required tests missing/failed')
        numeric=sorted(set(list((REPO/'src').rglob('*.py'))+list((REPO/'src').rglob('*.cpp'))+
            [Path(__file__),CONFIG,REPO/'tests/test_clustered_one_swap.py']+[REPO/m['config'] for m in cfg['models'].values()]))
        protocol=dict(status='REGISTERED BEFORE MODEL INFERENCE',registered_at=now(),review_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            config=cfg,models=models,source_sha256={p.relative_to(REPO).as_posix():digest(p) for p in numeric},historical_sha256=sources,
            station_protocol_sha256=digest(root/'station_protocol.json'),train_updates=0,optimizer_permitted=False,
            cases=[r['case_id'] for r in ds.records],draws=5,conditions=['Original','Targeted','Random'],
            primary='SelfSup C1 volume relative L1: Targeted-Original, mean per case after averaging five draws',
            secondary=METRICS,comparisons=['Targeted-Original','Targeted-Random','Random-Original'],
            bootstrap=cfg['bootstrap'],decisions=dict(improvement='mean difference < 0 and CI entirely < 0',
                inconclusive='CI includes zero',worsening='mean difference > 0 and CI entirely > 0',
                targeted_extra_advantage='requires Targeted-Random clearly negative; Original improvement alone is insufficient'),
            reproduction=dict(tolerance=cfg['reproduction_tolerance'],scope='both models all180 per-case four voxel metrics and two observed errors, plus draw mean/pooled',
                before_new_contrast=True,stop_on_mismatch=True,tolerance_basis='reuse frozen reference absolute1e-8 + relative1e-5; not fitted to this replay'),
            descriptors='acute crossing angle = acos(abs dot); Q = mean outer(d,d), eigenvalues ascending; descriptive only',
            worst_case_rule='all cases retained; report five largest Targeted case-mean relative L1, no failure cutoff',
            geometry_relation='case-mean score change versus case-mean voxel error change; Pearson correlation descriptive, no independence/significance claim',
            figures=cfg['figures'],GT_policy='GT64 only after station/protocol freeze; never for selection; GT128/Pool B/test denied',
            normalization_scale_ppm_m=ds.y_ref,tests_sha256=digest(test),original_v2_no_go_retained=True,
            limitations=['single checkpoint/seed per training mode','same Val cases, possibly correlated operating families','supervised checkpoint selected on Val supervised loss','geometry input changed, not increased model ability on Original'])
        write_json(root/'protocol.json',protocol);(root/'protocol.sha256').write_text(digest(root/'protocol.json')+'\n')
        assert audit.counts['forbidden_attempts']==0 and audit.counts['val_gt64_opens']==0 and audit.counts['val_pool_a_projection_opens']==0
        write_json(root/'registration_audit.json',audit.summary())
        print(f'Registered {digest(root/"protocol.json")}: 180 paired station rows; no model inference or measurement/GT reads',flush=True)


def verify(cfg,root):
    p=json.loads((root/'protocol.json').read_text());assert p['config']==cfg
    assert digest(root/'protocol.json')==(root/'protocol.sha256').read_text().strip()
    assert all(digest(REPO/k)==sha for k,sha in p['source_sha256'].items()),'Numerical sources changed after registration'
    assert all(digest(k)==sha for k,sha in p['historical_sha256'].items()),'Historical records changed'
    assert digest(root/'station_protocol.json')==p['station_protocol_sha256']
    return p


def load_frozen(cfg,name,protocol):
    m=cfg['models'][name];assert digest(m['checkpoint'])==m['sha256']
    saved=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
    assert saved['config']==protocol['models'][name]['config']
    model=build_model(saved['config']).to(cfg['device']);model.load_state_dict(saved['model']);model.eval();model.requires_grad_(False)
    assert state_hash(model.state_dict())==protocol['models'][name]['canonical_state_sha256']
    return model


def samples_for(ds,record,condition):
    row,p=ds._projections(record['case_index']);assert row['case_id']==record['case_id']
    empty=np.empty(0,dtype=np.int64)
    sample=ds._measurement_sample(row,p,np.asarray(record[condition],dtype=np.int64),empty)
    h=ds._measurement_sample(row,p,np.asarray(record['H'],dtype=np.int64),empty)
    return sample,h


def target_gt(ds,index):
    p=artifact_path(ds.root,ds.records[index]['gt_volume_64_path'])
    gt=np.load(p,mmap_mode='r',allow_pickle=False)
    if gt.shape!=(64,64,64) or gt.dtype!=np.float32 or not np.isfinite(gt).all() or np.any((gt<0)|(gt>1)):raise ValueError('Invalid real GT64')
    return torch.from_numpy(np.array(gt,copy=True))[None,None]


def infer_row(cfg,ds,record,condition,model,name):
    sample,h=samples_for(ds,record,condition)
    batch=move_batch(projection_pool_collate([sample]),cfg['device']);hb=move_batch(projection_pool_collate([h]),cfg['device'])
    with torch.inference_mode():
        prediction=model(batch)
        if prediction.dtype!=torch.float32 or not torch.isfinite(prediction).all():raise FloatingPointError('Finite FP32 C1 required')
        gt=target_gt(ds,record['case_index']).to(cfg['device'])
        metric=VolumeMetrics();metric.update(prediction,gt,[record['case_id']])
        valid=hb['input_valid_ray'][0,:,0].bool() & hb['view_mask'][0,:,None,None]
        score=relative_projection_metrics(project_input(prediction,hb)[0,:,0],hb['input_projection_raw'][0,:,0],valid)
        observed_valid=batch['input_valid_ray'][0,:,0].bool() & batch['view_mask'][0,:,None,None]
        obs=relative_projection_metrics(project_input(prediction,batch)[0,:,0],batch['input_projection_raw'][0,:,0],observed_valid)
        result=dict(model=name,condition=condition,case_id=record['case_id'],case_index=record['case_index'],draw=record['draw'],
            input_station_ids=record[condition],H_station_ids=record['H'],checkpoint_sha256=cfg['models'][name]['sha256'],
            **{k:v for k,v in metric.rows[0].items() if k!='case_id'},
            heldout_relative_l1=score['relative_l1'],heldout_relative_l2=score['relative_l2'],heldout_valid_rays=score['valid_rays'],
            observed_relative_l1=obs['relative_l1'],observed_relative_l2=obs['relative_l2'],
            geometry_score=record['geometry'][condition]['score'],crossing_angle_mean_deg=record['geometry'][condition]['crossing_angle_mean_deg'],
            direction_tensor_eigenvalues=record['geometry'][condition]['direction_tensor_eigenvalues'],
            output_mean=float(prediction.mean()),output_std=float(prediction.std(unbiased=False)),output_max=float(prediction.max()))
    return result,prediction.detach().cpu().numpy()[0,0],gt.detach().cpu().numpy()[0,0]


def original(cfg,root):
    protocol=verify(cfg,root)
    if (root/'original_reproduction.json').exists():
        assert json.loads((root/'original_reproduction.json').read_text())['passed'];print('Original already reproduced; not repeated');return
    with (root/'original_started.json').open('x') as f:json.dump(dict(at=now()),f)
    rows=[];comparisons=[]
    with read_guard(cfg,root) as audit:
        ds=dataset(cfg);records=json.loads((root/'station_protocol.json').read_text())['records']
        for name in cfg['models']:
            model=load_frozen(cfg,name,protocol);before=state_hash(model.state_dict())
            for r in records:
                row,pred,gt=infer_row(cfg,ds,r,'Original',model,name);rows.append(row)
                historic={x['case_id']:x for x in history(cfg,name,r['draw'])}[r['case_id']]
                tol=cfg['reproduction_tolerance']
                for metric in ['mae','rmse','relative_l1','relative_l2','observed_relative_l1','observed_relative_l2']:
                    expected=historic['observed'][metric.replace('observed_','')] if metric.startswith('observed_') else historic[metric]
                    diff=abs(row[metric]-expected);limit=tol['absolute']+tol['relative']*abs(expected)
                    comparisons.append(dict(model=name,case_id=r['case_id'],draw=r['draw'],metric=metric,expected=expected,actual=row[metric],absolute_difference=diff,tolerance=limit,passed=diff<=limit))
                if r['draw']==cfg['figures']['draw'] and r['case_index'] in cfg['figures']['case_indices']:
                    (root/'case_images').mkdir(exist_ok=True)
                    np.savez_compressed(root/'case_images'/f'{name}_val{r["case_index"]}_Original.npz',GT=gt,Original=pred)
                if r['case_index']==35:print(f'Original {name} draw{r["draw"]} completed',flush=True)
            assert state_hash(model.state_dict())==before and all(p.grad is None for p in model.parameters())
        for name in cfg['models']:
            for draw in cfg['draws']:
                actual=[r for r in rows if r['model']==name and r['draw']==draw]
                frozen=json.loads((Path(cfg['models'][name]['historical_root'])/f'draw{draw}'/'C1/metrics.json').read_text())
                for metric in ['mae','rmse','relative_l1','relative_l2']:
                    expected=frozen['mean_per_case'][metric];value=float(np.mean([r[metric] for r in actual]));limit=tol['absolute']+tol['relative']*abs(expected)
                    comparisons.append(dict(model=name,case_id='DRAW_MEAN',draw=draw,metric=metric,expected=expected,actual=value,absolute_difference=abs(value-expected),tolerance=limit,passed=abs(value-expected)<=limit))
                pooled=dict(mae=sum(r['absolute_sum'] for r in actual)/sum(r['voxel_count'] for r in actual),
                    rmse=np.sqrt(sum(r['squared_sum'] for r in actual)/sum(r['voxel_count'] for r in actual)),
                    relative_l1=sum(r['absolute_sum'] for r in actual)/sum(r['gt_absolute_sum'] for r in actual),
                    relative_l2=np.sqrt(sum(r['squared_sum'] for r in actual)/sum(r['gt_squared_sum'] for r in actual)))
                for metric,value in pooled.items():
                    expected=frozen['pooled'][metric];limit=tol['absolute']+tol['relative']*abs(expected)
                    comparisons.append(dict(model=name,case_id='DRAW_POOLED',draw=draw,metric=metric,expected=expected,actual=value,absolute_difference=abs(value-expected),tolerance=limit,passed=abs(value-expected)<=limit))
        write_csv(root/'original_per_case_draw.csv',rows);write_json(root/'original_rows.json',rows)
        write_csv(root/'original_reproduction_comparisons.csv',comparisons)
        passed=all(r['passed'] for r in comparisons)
        write_json(root/'original_reproduction.json',dict(passed=passed,comparison_count=len(comparisons),max_absolute_difference=max(r['absolute_difference'] for r in comparisons),
            models={name:dict(actual_volume_relative_l1=float(np.mean([r['relative_l1'] for r in rows if r['model']==name])),historical_volume_relative_l1=float(np.mean([r['relative_l1'] for d in cfg['draws'] for r in history(cfg,name,d)]))) for name in cfg['models']},
            state_and_buffers_unchanged=True,gt_access=audit.summary()))
        if not passed:raise AssertionError('Original reproduction exceeds registered tolerance; stop new contrasts')
        assert audit.counts['forbidden_attempts']==0
        print('All Original reproduction comparisons passed',flush=True)


def evaluate(cfg,root):
    p=verify(cfg,root);assert json.loads((root/'original_reproduction.json').read_text())['passed']
    if (root/'summary.json').exists():print('Complete same-protocol evaluation exists; not repeated');return
    with (root/'evaluation_started.json').open('x') as f:json.dump(dict(at=now()),f)
    rows=json.loads((root/'original_rows.json').read_text())
    with read_guard(cfg,root) as audit:
        ds=dataset(cfg);records=json.loads((root/'station_protocol.json').read_text())['records']
        for name in cfg['models']:
            model=load_frozen(cfg,name,p);before=state_hash(model.state_dict())
            for record in records:
                for condition in ('Targeted','Random'):
                    row,pred,gt=infer_row(cfg,ds,record,condition,model,name);rows.append(row)
                    if record['draw']==cfg['figures']['draw'] and record['case_index'] in cfg['figures']['case_indices']:
                        np.savez_compressed(root/'case_images'/f'{name}_val{record["case_index"]}_{condition}.npz',**{condition:pred})
                if record['case_index']==35:print(f'{name} new paired conditions draw{record["draw"]} completed',flush=True)
            assert before==state_hash(model.state_dict()) and all(p.grad is None for p in model.parameters())
        write_csv(root/'per_case_draw.csv',rows);write_json(root/'per_case_draw.json',rows)
        stats=[paired_statistics(rows,metric,name,left,right,p['bootstrap']['samples'],p['bootstrap']['seed']) for name in cfg['models'] for metric in METRICS
               for left,right in [('Targeted','Original'),('Targeted','Random'),('Random','Original')]]
        write_json(root/'paired_statistics.json',stats)
        case_rows=[dict(model=s['model'],metric=s['metric'],comparison=s['comparison'],**r) for s in stats for r in s['per_case']]
        write_csv(root/'paired_case_means.csv',case_rows)
        draw_rows=[dict(model=name,condition=condition,draw=draw,**{metric:float(np.mean([r[metric] for r in rows if r['model']==name and r['condition']==condition and r['draw']==draw])) for metric in METRICS})
            for name in cfg['models'] for condition in ('Original','Targeted','Random') for draw in cfg['draws']]
        write_csv(root/'per_draw_summary.csv',draw_rows)
        correlations=[]
        for name in cfg['models']:
            for comparison in ('Targeted-Original','Random-Original','Targeted-Random'):
                effect=next(s for s in stats if s['model']==name and s['metric']=='relative_l1' and s['comparison']==comparison)
                left,right=comparison.split('-');gx=[];dy=[]
                for case in effect['per_case']:
                    relevant=[r for r in records if r['case_id']==case['case_id']]
                    gx.append(float(np.mean([r['geometry'][left]['score']-r['geometry'][right]['score'] for r in relevant])));dy.append(case['delta'])
                corr=float(np.corrcoef(gx,dy)[0,1]) if np.std(gx)>1e-12 and np.std(dy)>1e-12 else None
                correlations.append(dict(model=name,comparison=comparison,case_mean_score_delta=gx,case_mean_error_delta=dy,
                    pearson=corr,constant_geometry_across_cases=np.std(gx)<=1e-12,interpretation='descriptive only; case/family dependence, no p-value'))
        before=json.loads((root/'inventory_before.json').read_text());assert before==inventory(cfg),'Original data/history metadata changed'
        assert all(digest(cfg['models'][name]['checkpoint'])==cfg['models'][name]['sha256'] for name in cfg['models'])
        assert audit.counts['forbidden_attempts']==0
        integrity=dict(parameters_and_buffers_unchanged=True,checkpoint_hashes_unchanged=True,dataset_metadata_unchanged=True,
            source_gt128_metadata_unchanged=True,historical_metadata_unchanged=True,historical_records_hashes_unchanged=True,
            numerical_sources_hashes_unchanged=True,optimizer_created=False,optimizer_steps=0,GT64_evaluation_only=True,GT128_opened=False,
            original_v2_no_go_retained=True)
        verify(cfg,root)
        primary=next(s for s in stats if s['model']=='SelfSup' and s['metric']=='relative_l1' and s['comparison']=='Targeted-Original')
        extra=next(s for s in stats if s['model']=='SelfSup' and s['metric']=='relative_l1' and s['comparison']=='Targeted-Random')
        worst={}
        for name in cfg['models']:
            s=next(s for s in stats if s['model']==name and s['metric']=='relative_l1' and s['comparison']=='Targeted-Original')
            worst[name]=sorted(s['per_case'],key=lambda r:r['left_mean'],reverse=True)[:5]
        write_json(root/'integrity.json',integrity);write_json(root/'evaluation_audit.json',audit.summary())
        write_json(root/'summary.json',dict(completed=True,completed_at=now(),protocol_sha256=digest(root/'protocol.json'),
            row_count=len(rows),primary=primary,targeted_vs_random=extra,paired_statistics=stats,per_draw=draw_rows,
            geometry_correlations=correlations,worst_targeted_cases=worst,random_targeted_coincidences=sum(r['random_equals_targeted'] for r in records),
            integrity=integrity,audit=audit.summary(),original_reproduction=json.loads((root/'original_reproduction.json').read_text())))
        print(json.dumps(dict(primary=primary,targeted_vs_random=extra),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['register','original','evaluate']);args=parser.parse_args()
    cfg,root=configuration();seed_everything(20261011);torch.set_num_threads(2)
    try:globals()[args.phase](cfg,root)
    except BaseException as error:
        if root.exists():write_json(root/f'failure_{args.phase}_{__import__("time").time_ns()}.json',dict(error=repr(error),at=now(),no_training_or_substitution=True))
        raise


if __name__=='__main__':main()
