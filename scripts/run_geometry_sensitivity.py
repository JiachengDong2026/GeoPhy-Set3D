#!/usr/bin/env python3
"""Preregister, smoke, and resume frozen-v1 V4 geometry-only interventions."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.evaluation.clustered_one_swap import ClusteredEvaluationReadGuard, keyed_rng
from geophy_set3d.evaluation.geometry_catalog import GeometryCatalog
from geophy_set3d.evaluation.geometry_strata import population
from geophy_set3d.evaluation.geometry_sensitivity import circular_span, group_masks, spatial_descriptors, station_gram, spectrum_from_gram
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything, move_batch, write_json
from diagnose_clustered_one_swap import dataset, target_gt, checkpoint_provenance, load_frozen, write_csv, snapshot


def configuration(path):
    return yaml.safe_load(Path(path).read_text())


def archive(out):
    destination=REPO/'docs/reports/artifacts/geometry_sensitivity_v1'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ['protocol.json','protocol.sha256','geometry_layouts.json','geometry_descriptors.csv',
                 'spectra.json','station_subsets.json','registration_access_audit.json','operator_cost.json']:
        shutil.copy2(out/name,destination/name)


def inventory(cfg):
    paths=cfg['paths']; output=Path(paths['output']).resolve(); parent=output.parents[1]
    records=[json.loads(l) for l in (Path(paths['dataset'])/'source_index.jsonl').read_text().splitlines()]
    return dict(dataset=snapshot(paths['dataset']),
        gt128_metadata={r['source_target_path']:dict(size=Path(r['source_target_path']).stat().st_size,
            mtime_ns=Path(r['source_target_path']).stat().st_mtime_ns,inode=Path(r['source_target_path']).stat().st_ino) for r in records},
        history={str(p.relative_to(parent)):snapshot(p) for section in ['checkpoints','logs','comparisons','experiments','diagnostics','evaluations']
            for p in sorted((parent/section).iterdir()) if p.is_dir() and p.resolve()!=output})


def verify(cfg, committed=True):
    out=Path(cfg['paths']['output']);p=json.loads((out/'protocol.json').read_text())
    assert p['config']==cfg
    assert digest(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    for k,v in p['source_sha256'].items():assert digest(REPO/k)==v,k
    for k,v in p['registration_sha256'].items():assert digest(out/k)==v,k
    for k,v in p['historical_sha256'].items():assert digest(k)==v,k
    for k,v in p['metadata_sha256'].items():assert digest(k)==v,k
    assert digest(cfg['paths']['checkpoint'])==cfg['checkpoint_sha256']
    if committed:
        relative='docs/reports/artifacts/geometry_sensitivity_v1/protocol.json'
        saved=subprocess.check_output(['git','show','HEAD:'+relative],cwd=REPO)
        assert saved==(out/'protocol.json').read_bytes(),'Protocol must be committed before inference'
    return p


def register(cfg):
    out=Path(cfg['paths']['output'])
    if (out/'protocol.json').exists():
        verify(cfg,False);print('Complete existing registration verified; not regenerated');return
    if out.exists() and any(out.iterdir()):raise FileExistsError('Nonempty independent output; inspect, never overwrite')
    out.mkdir(parents=True)
    write_json(out/'inventory_before.json',inventory(cfg))
    with ClusteredEvaluationReadGuard(cfg['paths']['dataset'],[],out,geometry_only=True) as audit:
        geompath=Path(cfg['paths']['dataset'])/'geometry/train_geometry.npz'
        g=dict(np.load(geompath,allow_pickle=False));origins=g['station_origin_m']
        cases=[r for r in map(json.loads,(Path(cfg['paths']['dataset'])/'source_index.jsonl').read_text().splitlines()) if r['split']=='val']
        assert len(cases)==36 and len(origins)==64 and cfg['views']==4 and cfg['draws']==5
        ids,scores,pop=population(origins,4,cfg['population_seed'])
        spans=circular_span(g['station_azimuth_deg'][ids]);boundaries=np.quantile(scores,cfg['quantiles'])
        masks=group_masks(scores,spans,boundaries,cfg['cluster_span_max_deg'])
        rng=np.random.default_rng(cfg['subset_seed']);selected=[]
        for group,mask in masks.items():
            eligible=np.flatnonzero(mask);assert len(eligible)>=cfg['draws']
            for draw,j in enumerate(rng.choice(eligible,cfg['draws'],replace=False)):
                selected.append(dict(group=group,draw=draw,geometry_id=f'V4-{group}-d{draw}',
                    station_ids=rng.permutation(ids[j]).astype(int).tolist(),
                    center_direction_score=float(scores[j]),eligible_population_size=int(len(eligible))))
        # Only after layout selection is complete: all K96 rays and coverage.
        cat=GeometryCatalog(g,cfg['coverage_resolution'],cfg['samples']);gram={};cost=[]
        for station in sorted({j for r in selected for j in r['station_ids']}):
            gram[station],timing=station_gram(g,station,cfg['spectral_resolution'],cfg['samples']);cost.append(timing)
        descriptors=[];spectra=[]
        for record in selected:
            s=record['station_ids'];start=time.perf_counter()
            spec=spectrum_from_gram(sum(gram[j] for j in s),cfg['spectral_relative_sigma_tolerance'])
            elapsed=time.perf_counter()-start
            spectra.append(dict(geometry_id=record['geometry_id'],**spec))
            d=cat.describe(s)
            d.update(spatial_descriptors(origins[s],g['station_azimuth_deg'][s]))
            d.update({k:v for k,v in spec.items() if k!='singular_values_ppm_m'})
            d.update(actual_ray_abs_cosine_redundancy=1-d['actual_ray_direction_score_sampled'],
                coarse_resolution=cfg['spectral_resolution'],coarse_unknowns=cfg['spectral_resolution']**3,
                full64_nullity_lower_bound=64**3-d['valid_ray_count'],spectral_seconds=elapsed,
                spectral_model='linear nodal mole fraction chi; all valid rays; K96; geometry-only; NOT nonlinear P_diff(C) Jacobian',
                **record)
            descriptors.append(d)
        excluded={j for r in selected for j in r['station_ids']};allowed=sorted(set(range(64))-excluded)
        # Five draws per group can use >60 stations: a common H may be impossible.
        # Same H across groups per draw, outside union of that draw's five inputs.
        records=[]
        for index,case in enumerate(cases):
            for draw in range(cfg['draws']):
                excluded={j for r in selected if r['draw']==draw for j in r['station_ids']}
                available=sorted(set(range(64))-excluded);assert len(available)>=4
                h=sorted(keyed_rng(cfg['heldout_seed'],draw,case['case_id'],'sensitivity-common-H4').choice(available,4,replace=False).tolist())
                for r in selected:
                    if r['draw']==draw:
                        records.append(dict(case_id=case['case_id'],case_index=index,H_station_ids=h,**r))
        write_json(out/'geometry_layouts.json',dict(population=pop,quantiles=cfg['quantiles'],boundaries=boundaries.tolist(),
            groups={k:int(m.sum()) for k,m in masks.items()},excluded_count=int((~np.stack(list(masks.values())).any(0)).sum()),
            descriptors=descriptors,layouts=selected))
        write_csv(out/'geometry_descriptors.csv',descriptors);write_json(out/'spectra.json',spectra)
        write_json(out/'station_subsets.json',records);write_json(out/'operator_cost.json',dict(stations=cost,
            construction_seconds=sum(r['build_seconds'] for r in cost),gram_cache_peak_bytes=len(gram)*cfg['spectral_resolution']**6*8,
            caching='station sparse operators discarded after Gram; station Grams reused across layouts; no persistent operator cache'))
        write_json(out/'registration_access_audit.json',audit.summary())
        assert audit.counts['val_pool_a_projection_opens']==audit.counts['val_gt64_opens']==audit.counts['checkpoint_opens']==0
    historic=[]
    for root,files in [('historical_strata',['SelfSup_rows.jsonl','protocol.json','SelfSup_checkpoint_provenance.json']),
                       ('historical_swap',['per_case_draw.json','station_protocol.json','protocol.json'])]:
        historic.extend(Path(cfg['paths'][root])/name for name in files)
    frozen=['geometry_layouts.json','geometry_descriptors.csv','spectra.json','station_subsets.json','operator_cost.json']
    sources=sorted(list((REPO/'src').rglob('*.py'))+list((REPO/'src').rglob('*.cpp'))+
        [Path(__file__),REPO/'configs/geometry_sensitivity_v1.yaml',REPO/'tests/test_geometry_sensitivity.py'])
    p=dict(status='REGISTERED BEFORE CHECKPOINT LOAD AND NEW INFERENCE',config=cfg,
        baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_sha256={str(p.relative_to(REPO)):digest(p) for p in sources},
        registration_sha256={k:digest(out/k) for k in frozen},historical_sha256={str(p):digest(p) for p in historic},
        metadata_sha256={str(p):digest(p) for p in [geompath,Path(cfg['paths']['dataset'])/'source_index.jsonl',Path(cfg['paths']['dataset'])/'train_statistics.json',Path(cfg['paths']['freeze_record'])]},
        inventory_sha256=digest(out/'inventory_before.json'),
        statistics='36 cases, average five draws before paired 10000 case bootstrap; 25 geometry means for Pearson/Spearman, case-bootstrap CI; 10000 geometry permutations descriptive, Holm across descriptors within correlation type; no repeated row p-values. Measurement fit association: raw descriptive, within each group, two-way-centered case/layout residual, case bootstrap.',
        score='Post-result strongest directed nonconstant metric by absolute Spearman over 25 geometry means; explicitly exploratory GT-informed, not used for groups, training or checkpoint selection; minmax geometry normalization.',
        smoke='case index 0/draw 0/all five groups; assert finite C in [0,1], frozen state, correct IDs/masks/shape; compare historical if exact inputs exist; write smoke rows once then full resumes',
        stop='Frozen inference only; no optimizer. No score or group changes after results.',
        scope='36 Val Pool A/GT64 only; no train content, Pool B, final test or GT128; original v1 checkpoint')
    write_json(out/'protocol.json',p);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n');archive(out)
    print(json.dumps(dict(rows=len(records),layouts=len(selected),groups=p['config']['quantiles'],boundaries=boundaries.tolist(),eligible={k:int(m.sum()) for k,m in masks.items()}),indent=2))


def evaluate(cfg,device,smoke):
    seed_everything(cfg['original_seed']);torch.set_num_threads(2);p=verify(cfg);out=Path(cfg['paths']['output'])
    records=json.loads((out/'station_subsets.json').read_text());result=out/'rows.jsonl'
    rows=[json.loads(l) for l in result.read_text().splitlines()] if result.exists() else []
    key=lambda r:(r['case_id'],r['geometry_id']);done={key(r) for r in rows};assert len(done)==len(rows)
    expected={(r['case_id'],r['geometry_id']) for r in records};assert done<=expected
    if not smoke:
        assert json.loads((out/'smoke_complete.json').read_text())['passed']
        if done==expected:
            assert (out/'complete.json').exists();print('Full completed evaluation exists; verified, not rerun');return
    active=[r for r in records if (not smoke or (r['case_index']==0 and r['draw']==0)) and key(r) not in done]
    desc={r['geometry_id']:r for r in json.loads((out/'geometry_layouts.json').read_text())['descriptors']}
    models={'SelfSup':dict(config=cfg['model_config'],checkpoint=cfg['paths']['checkpoint'],sha256=cfg['checkpoint_sha256'],freeze_record=cfg['paths']['freeze_record'])}
    old=[json.loads(l) for l in (Path(cfg['paths']['historical_strata'])/'SelfSup_rows.jsonl').read_text().splitlines()]
    historical={(r['case_id'],tuple(r['input_station_ids']),tuple(r['H_station_ids'])):r for r in old}
    cf=dict(dataset=cfg['paths']['dataset'],models=models,original_seed=cfg['original_seed'],device=device)
    (out/'case_images').mkdir(exist_ok=True);start=time.perf_counter();comparisons=[]
    with ClusteredEvaluationReadGuard(cfg['paths']['dataset'],[cfg['paths']['checkpoint']],out) as guard:
        provenance=checkpoint_provenance(cf);write_json(out/'checkpoint_provenance.json',provenance)
        model=load_frozen(cf,'SelfSup',dict(models=provenance));before=state_hash(model.state_dict());ds=dataset(cf)
        assert [r['case_id'] for r in ds.records]==list(dict.fromkeys(r['case_id'] for r in records))
        with result.open('a',buffering=1) as f,torch.inference_mode():
            for index in sorted({r['case_index'] for r in active}):
                source,projections=ds._projections(index)
                for r in [r for r in active if r['case_index']==index]:
                    sample=ds._measurement_sample(source,projections,np.asarray(r['station_ids']),np.empty(0,dtype=np.int64))
                    batch=move_batch(projection_pool_collate([sample]),device)
                    assert batch['input_station_indices'][0].tolist()==r['station_ids']
                    assert int(batch['view_mask'].sum())==4
                    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);t=time.perf_counter()
                    prediction=model(batch);torch.cuda.synchronize(device);forward=time.perf_counter()-t
                    assert prediction.shape==(1,1,64,64,64) and prediction.dtype==torch.float32
                    assert torch.isfinite(prediction).all() and prediction.min()>=0 and prediction.max()<=1
                    # No GT or H content used until prediction has been produced.
                    t=time.perf_counter();gt=target_gt(ds,index).to(device)
                    hs=ds._measurement_sample(source,projections,np.asarray(r['H_station_ids']),np.empty(0,dtype=np.int64))
                    hb=move_batch(projection_pool_collate([hs]),device);metric=VolumeMetrics();metric.update(prediction,gt,[r['case_id']])
                    valid=batch['input_valid_ray'][0,:,0].bool()&batch['view_mask'][0,:,None,None]
                    hv=hb['input_valid_ray'][0,:,0].bool()&hb['view_mask'][0,:,None,None]
                    obs=relative_projection_metrics(project_input(prediction,batch)[0,:,0],batch['input_projection_raw'][0,:,0],valid)
                    held=relative_projection_metrics(project_input(prediction,hb)[0,:,0],hb['input_projection_raw'][0,:,0],hv)
                    torch.cuda.synchronize(device);evaluation=time.perf_counter()-t
                    row=dict(**r,checkpoint_sha256=cfg['checkpoint_sha256'],V=4,split='val',pool='train',model='SelfSup-v1',
                        **{k:v for k,v in metric.rows[0].items() if k!='case_id'},observed_relative_l1=obs['relative_l1'],observed_relative_l2=obs['relative_l2'],
                        heldout_relative_l1=held['relative_l1'],heldout_relative_l2=held['relative_l2'],valid_input_rays=obs['valid_rays'],
                        valid_heldout_rays=held['valid_rays'],forward_seconds=forward,evaluation_seconds=evaluation,
                        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),device=device,result_source='actual frozen inference')
                    hist=historical.get((r['case_id'],tuple(r['station_ids']),tuple(r['H_station_ids'])))
                    if hist:
                        for k in ['relative_l1','relative_l2','mae','rmse','observed_relative_l1','observed_relative_l2','heldout_relative_l1','heldout_relative_l2']:
                            diff=abs(row[k]-hist[k]);assert diff<=1e-8+1e-5*abs(hist[k]);comparisons.append(dict(case_id=r['case_id'],geometry_id=r['geometry_id'],metric=k,difference=diff))
                    if index in cfg['representative_case_indices'] and r['draw']==cfg['representative_draw'] and r['group'] in cfg['representative_groups']:
                        np.savez_compressed(out/'case_images'/f'case{index}_{r["group"]}.npz',prediction=prediction[0,0].cpu().numpy(),GT=gt[0,0].cpu().numpy())
                    f.write(json.dumps(row,allow_nan=False)+'\n');done.add(key(r))
                print(f'case {index+1}/36: {len(done)}/900 rows, elapsed {time.perf_counter()-start:.1f}s',flush=True)
        assert state_hash(model.state_dict())==before and all(q.grad is None for q in model.parameters())
        assert digest(cfg['paths']['checkpoint'])==cfg['checkpoint_sha256']
        write_json(out/('smoke_access_audit.json' if smoke else 'full_access_audit.json'),guard.summary())
        write_json(out/('smoke_complete.json' if smoke else 'complete.json'),dict(passed=True,rows=len(done),state_unchanged=True,optimizer_steps=0,
            historical_same_input_comparisons=comparisons,elapsed_seconds=time.perf_counter()-start,
            full=not smoke,smoke_station_sets_distinct=len({tuple(r['station_ids']) for r in records if r['case_index']==0 and r['draw']==0})==5))
    verify(cfg)
    if not smoke:assert len(done)==900


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(REPO/'configs/geometry_sensitivity_v1.yaml'))
    parser.add_argument('--stage',choices=['register','smoke','full'],required=True);parser.add_argument('--device',default='cuda:2')
    args=parser.parse_args();cfg=configuration(args.config)
    if args.stage=='register':register(cfg)
    else:evaluate(cfg,args.device,args.stage=='smoke')

if __name__=='__main__':main()
