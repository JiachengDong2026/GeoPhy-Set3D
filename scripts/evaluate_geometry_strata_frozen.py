#!/usr/bin/env python3
"""Resumable registered paired frozen evaluation. No optimizer or training."""
import argparse
import json
import sys
import time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.evaluation.clustered_one_swap import ClusteredEvaluationReadGuard
from geophy_set3d.evaluation.geometry_strata import validate_shared_inputs
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything,move_batch,write_json
from diagnose_clustered_one_swap import dataset,target_gt,checkpoint_provenance,load_frozen,write_csv
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/evaluations/geometry_strata_frozen_v1')


def verify(out=OUT):
    p=json.loads((out/'protocol.json').read_text())
    assert digest(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    for k,v in p['source_sha256'].items():assert digest(REPO/k)==v, k
    for k,v in p['registration_sha256'].items():assert digest(out/k)==v,k
    for k,v in p['historical_sha256'].items():assert digest(k)==v,k
    # Formal registration must already exist in committed Git, before inference.
    import subprocess
    committed=subprocess.check_output(['git','show','HEAD:docs/reports/artifacts/geometry_strata_v2/protocol.json'],cwd=REPO)
    assert committed==(out/'protocol.json').read_bytes()
    return p


def identity(row):return (row['section'],row['V'],row['stratum'],row['case_id'],row['draw'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['SelfSup','Supervised'],required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    seed_everything(20261011);torch.set_num_threads(2);p=verify();c=p['config'];name=args.model
    output=OUT/f'{name}_rows.jsonl'
    done=[json.loads(l) for l in output.read_text().splitlines()] if output.exists() else []
    completed={identity(r) for r in done};assert len(completed)==len(done)
    records=json.loads((OUT/'station_subsets.json').read_text())['records']
    if len(done)==len(records):print(f'{name}: complete registered results exist; not repeated');return
    cfg=dict(dataset=c['dataset'],models=p['models'],original_seed=c['natural_seed'],device=args.device)
    desc={tuple(r['station_ids']):r for r in json.loads((OUT/'geometry_descriptors.json').read_text())}
    clustered=json.loads(Path('/data/djc/GeoPhy-Set3D/outputs/diagnostics/clustered_v4_one_swap/per_case_draw.json').read_text())
    clusterlookup={(r['model'],r['case_id'],r['draw']):r for r in clustered if r['condition']=='Original'}
    naturallookup={}
    for views in c['views']:
        for draw in range(c['draws']):
            path=Path(p['models'][name]['historical_root']).parent/f'V{views}/draw{draw}/C1/per_case_metrics.jsonl'
            for r in map(json.loads,path.read_text().splitlines()):naturallookup[views,r['case_id'],draw]=r
    # Provenance loads both checkpoints once; input evaluation uses only selected frozen model.
    with ClusteredEvaluationReadGuard(c['dataset'],[m['checkpoint'] for m in p['models'].values()],OUT) as audit:
        provenance=checkpoint_provenance(cfg);write_json(OUT/f'{name}_checkpoint_provenance.json',provenance)
        model=load_frozen(cfg,name,dict(models=provenance));before=state_hash(model.state_dict());ds=dataset(cfg)
        assert [r['case_id'] for r in ds.records]==list(dict.fromkeys(r['case_id'] for r in records))
        comparisons=[];startall=time.perf_counter()
        with output.open('a',buffering=1) as f,torch.inference_mode():
            for index in range(len(ds)):
                relevant=[r for r in records if r['case_index']==index and identity(r) not in completed]
                if not relevant:continue
                source,projections=ds._projections(index);gt=target_gt(ds,index).to(args.device)
                for record in relevant:
                    ids=record['station_ids'];hids=record['H_station_ids'];empty=np.empty(0,dtype=np.int64)
                    sample=ds._measurement_sample(source,projections,np.asarray(ids),empty)
                    hs=ds._measurement_sample(source,projections,np.asarray(hids),empty)
                    batch=move_batch(projection_pool_collate([sample]),args.device);hb=move_batch(projection_pool_collate([hs]),args.device)
                    torch.cuda.synchronize(args.device);torch.cuda.reset_peak_memory_stats(args.device);t=time.perf_counter()
                    prediction=model(batch);torch.cuda.synchronize(args.device);forward=time.perf_counter()-t;t=time.perf_counter()
                    assert prediction.dtype==torch.float32 and prediction.shape==(1,1,64,64,64) and torch.isfinite(prediction).all()
                    metrics=VolumeMetrics();metrics.update(prediction,gt,[record['case_id']])
                    observed_valid=batch['input_valid_ray'][0,:,0].bool()&batch['view_mask'][0,:,None,None]
                    hvalid=hb['input_valid_ray'][0,:,0].bool()&hb['view_mask'][0,:,None,None]
                    obs=relative_projection_metrics(project_input(prediction,batch)[0,:,0],batch['input_projection_raw'][0,:,0],observed_valid)
                    held=relative_projection_metrics(project_input(prediction,hb)[0,:,0],hb['input_projection_raw'][0,:,0],hvalid)
                    torch.cuda.synchronize(args.device);evaluation=time.perf_counter()-t
                    geometry=desc[tuple(ids)]
                    row=dict(model=name,section=record['section'],V=record['V'],stratum=record['stratum'],draw=record['draw'],
                        case_id=record['case_id'],case_index=index,geometry_id=record['geometry_id'],input_station_ids=ids,H_station_ids=hids,
                        checkpoint_sha256=p['models'][name]['sha256'],**{k:v for k,v in metrics.rows[0].items() if k!='case_id'},
                        observed_relative_l1=obs['relative_l1'],observed_relative_l2=obs['relative_l2'],
                        heldout_relative_l1=held['relative_l1'],heldout_relative_l2=held['relative_l2'],heldout_valid_rays=held['valid_rays'],
                        forward_seconds=forward,evaluation_seconds=evaluation,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                        GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),device=args.device,
                        **{k:geometry[k] for k in c['association_descriptors']})
                    if record['section'] in ('natural','clustered'):
                        historic=naturallookup[record['V'],record['case_id'],record['draw']] if record['section']=='natural' else clusterlookup[name,record['case_id'],record['draw']]
                        historic_ids=historic.get('selected_station_ids',historic.get('input_station_ids'))
                        assert ids==historic_ids,(record, historic_ids)
                        for metric in ['relative_l1','relative_l2','mae','rmse','observed_relative_l1','observed_relative_l2']:
                            expected=historic['observed'][metric.replace('observed_','')] if record['section']=='natural' and metric.startswith('observed_') else historic[metric]
                            diff=abs(row[metric]-expected);tol=p['reproduction_tolerance'];limit=tol['absolute']+tol['relative']*abs(expected)
                            assert diff<=limit,(name,record['section'],record['case_id'],metric,diff,limit)
                            comparisons.append(dict(model=name,section=record['section'],V=record['V'],case_id=record['case_id'],draw=record['draw'],metric=metric,actual=row[metric],expected=expected,difference=diff,passed=True))
                    f.write(json.dumps(row,allow_nan=False)+'\n');completed.add(identity(row))
                print(f'{name}: case {index+1}/36, {len(completed)}/{len(records)} acquisition forwards, elapsed={time.perf_counter()-startall:.1f}s',flush=True)
        assert state_hash(model.state_dict())==before and all(q.grad is None for q in model.parameters())
        assert all(digest(m['checkpoint'])==m['sha256'] for m in p['models'].values())
        write_json(OUT/f'{name}_access_audit.json',audit.summary())
        write_json(OUT/f'{name}_reproduction_session_{time.time_ns()}.json',comparisons)
        write_json(OUT/f'{name}_complete.json',dict(rows=len(completed),state_unchanged=True,optimizer_created=False,optimizer_steps=0,checkpoint_hashes_unchanged=True))
    verify();rows=[json.loads(l) for l in output.read_text().splitlines()];assert len(rows)==2340
    other=OUT/f'{"Supervised" if name=="SelfSup" else "SelfSup"}_rows.jsonl'
    if other.exists():validate_shared_inputs(rows+[json.loads(l) for l in other.read_text().splitlines()])

if __name__=='__main__':main()
