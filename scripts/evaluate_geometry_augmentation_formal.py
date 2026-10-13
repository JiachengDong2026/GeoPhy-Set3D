#!/usr/bin/env python3
"""Both300/best freeze -> measurements -> independent post-hoc Val GT64."""
import argparse,json,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models import build_model
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.evaluation.geometry_stratified_sampler import SamplerTrainingReadGuard
from geophy_set3d.evaluation.clustered_one_swap import ClusteredEvaluationReadGuard
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.utils import move_batch,seed_everything,write_json
from geophy_set3d.training.rvtc_matched import state_hash
from diagnose_clustered_one_swap import dataset,target_gt,write_csv
from run_geometry_augmentation_formal import configuration,verify


def freeze(c,out):
    p=verify(c,out);frozen={}
    for arm in c['arms']:
        row=json.loads((out/arm/'complete.json').read_text());ck=out/arm/'checkpoints/best.pt'
        assert row['epochs']==300 and row['optimizer_steps']==24900 and digest(ck)==row['best_sha256']
        assert row['access']['gt_open_attempts']==row['access']['forbidden_npy_attempts']==row['access']['supervised_checkpoint_open_attempts']==0
        frozen[arm]=dict(checkpoint=str(ck),sha256=row['best_sha256'],initial_state_sha256=row['initial_state_sha256'])
    assert frozen['Control']['initial_state_sha256']==frozen['Treatment']['initial_state_sha256']
    path=out/'frozen_best.json'
    if path.exists():assert json.loads(path.read_text())==frozen
    else:write_json(path,frozen)
    return p,frozen


def measurements(c,out,device):
    p,frozen=freeze(c,out);root=out/'posthoc_evaluation';root.mkdir(exist_ok=True);(root/'case_images').mkdir(exist_ok=True)
    records=json.loads((out/'confirmatory_geometries/evaluation_records.json').read_text());data=p['configs']['Control']['dataset']['root']
    for arm in c['arms']:
        path=root/f'{arm}_measurement_rows.jsonl';done=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        keys={(r['case_id'],r['section'],r['V'],r['stratum'],r['draw']) for r in done};assert len(keys)==len(done)
        if len(done)==2880:print(arm,'measurement complete; no repeat');continue
        with SamplerTrainingReadGuard(data,out/arm/'checkpoints',root) as audit:
            seed_everything(20261011);torch.set_num_threads(2);saved=torch.load(frozen[arm]['checkpoint'],map_location='cpu',weights_only=False)
            model=build_model(saved['config']).to(device);model.load_state_dict(saved['model']);model.eval();model.requires_grad_(False);before=state_hash(model.state_dict())
            ds=dataset(dict(dataset=data,original_seed=20261011));cfg=dict(device=device)
            with path.open('a',buffering=1) as f,torch.inference_mode():
                for index in range(36):
                    relevant=[r for r in records if r['case_index']==index and (r['case_id'],r['section'],r['V'],r['stratum'],r['draw']) not in keys]
                    if not relevant:continue
                    source,proj=ds._projections(index)
                    for record in relevant:
                        empty=np.empty(0,dtype=np.int64);ids=record['station_ids'];h=record['H_station_ids']
                        b=move_batch(projection_pool_collate([ds._measurement_sample(source,proj,np.asarray(ids),empty)]),device)
                        hb=move_batch(projection_pool_collate([ds._measurement_sample(source,proj,np.asarray(h),empty)]),device)
                        torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);began=time.perf_counter();pred=model(b);torch.cuda.synchronize(device);seconds=time.perf_counter()-began
                        assert pred.shape==(1,1,64,64,64) and torch.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
                        assert b['input_station_indices'][0].tolist()==ids and hb['input_station_indices'][0].tolist()==h and not set(ids)&set(h)
                        observed=relative_projection_metrics(project_input(pred,b)[0,:,0],b['input_projection_raw'][0,:,0],b['input_valid_ray'][0,:,0].bool())
                        held=relative_projection_metrics(project_input(pred,hb)[0,:,0],hb['input_projection_raw'][0,:,0],hb['input_valid_ray'][0,:,0].bool())
                        token=f'{arm}_{index}_{record["section"]}_V{record["V"]}_{record["stratum"]}_{record["draw"]}'
                        volume=root/'case_images'/f'{token}.npz';np.savez_compressed(volume,C=pred.cpu().numpy()[0,0])
                        row=dict(geometry_id=record.get('geometry_id',f'natural-V4-d{record["draw"]}'),group=record.get('group',record['stratum']),model=arm,case_id=record['case_id'],case_index=index,section=record['section'],V=record['V'],stratum=record['stratum'],draw=record['draw'],
                            input_station_ids=ids,H_station_ids=h,checkpoint_sha256=frozen[arm]['sha256'],observed_relative_l1=observed['relative_l1'],observed_relative_l2=observed['relative_l2'],
                            heldout_relative_l1=held['relative_l1'],heldout_relative_l2=held['relative_l2'],forward_seconds=seconds,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),prediction_file=str(volume),prediction_sha256=digest(volume))
                        f.write(json.dumps(row,allow_nan=False)+'\n')
                    print(arm,'measurement case',index+1,'/36',flush=True)
            assert state_hash(model.state_dict())==before
            write_json(root/f'{arm}_measurement_access_audit.json',audit.summary());assert audit.summary()['gt_open_attempts']==0
    selector_audit(c,out,device)
    rows=[json.loads(l) for arm in c['arms'] for l in (root/f'{arm}_measurement_rows.jsonl').read_text().splitlines()];assert len(rows)==5760
    write_json(root/'measurement_complete.json',dict(rows=5760,both_epochs=300,GT_open_attempts=0,frozen_best_sha256=digest(out/'frozen_best.json'),
        row_sha256={a:digest(root/f'{a}_measurement_rows.jsonl') for a in c['arms']},parameters_buffers_unchanged=True))


def selector_audit(c,out,device):
    from geophy_set3d.training.selfsup import make_dataset,make_loader,run_epoch,selection_metric
    import copy
    root=out/'posthoc_evaluation';path=root/'selector_audit.json'
    if path.exists():assert json.loads(path.read_text())['passed'];return
    _,frozen=freeze(c,out);rows={}
    for arm in c['arms']:
        with SamplerTrainingReadGuard(c['paths']['dataset'],out/arm/'checkpoints',root) as guard:
            saved=torch.load(frozen[arm]['checkpoint'],map_location='cpu',weights_only=False);cfg=copy.deepcopy(saved['config']);cfg['training']['device']=device
            seed_everything(c['seed']);model=build_model(cfg).to(device);model.load_state_dict(saved['model']);before=state_hash(model.state_dict())
            val=make_dataset(cfg,validation=True);metrics=run_epoch(model,make_loader(val,cfg),cfg);score=selection_metric(metrics)
            historical=[json.loads(l) for l in (out/arm/'metrics.jsonl').read_text().splitlines()];best=min(historical,key=lambda r:selection_metric(r['val']))
            delta=score-selection_metric(best['val']);assert abs(delta)<=1e-7+1e-6*abs(score)
            assert best['epoch']==saved['epoch'] and state_hash(model.state_dict())==before
            rows[arm]=dict(best_epoch=saved['epoch'],original_logged_score=selection_metric(best['val']),replayed_score=score,delta=delta,metrics=metrics,access=guard.summary())
    write_json(path,dict(passed=True,rows=rows,GT_reads=0,selector='unchanged natural Val dual A4/B4 H4 measurement relative L1'))


def gt(c,out):
    p,frozen=freeze(c,out);root=out/'posthoc_evaluation';completed=json.loads((root/'measurement_complete.json').read_text());assert completed['rows']==5760
    for arm,sha in completed['row_sha256'].items():assert digest(root/f'{arm}_measurement_rows.jsonl')==sha
    if (root/'posthoc_complete.json').exists():print('Post-hoc GT complete, not repeated');return
    data=p['configs']['Control']['dataset']['root'];rows=[]
    with ClusteredEvaluationReadGuard(data,[],root) as audit:
        ds=dataset(dict(dataset=data,original_seed=20261011));targets={}
        for arm in c['arms']:
            for row in map(json.loads,(root/f'{arm}_measurement_rows.jsonl').read_text().splitlines()):
                index=row['case_index']
                if index not in targets:targets[index]=target_gt(ds,index)
                path=Path(row['prediction_file']);assert digest(path)==row['prediction_sha256'];pred=torch.from_numpy(np.load(path)['C'])[None,None]
                m=VolumeMetrics();m.update(pred,targets[index],[row['case_id']]);rows.append(dict(row,**{k:v for k,v in m.rows[0].items() if k!='case_id'}))
        write_csv(root/'per_case_draw.csv',rows);write_json(root/'per_case_draw.json',rows);write_json(root/'posthoc_access_audit.json',audit.summary())
        assert audit.summary()['unique_val_gt64_files']==36 and audit.summary()['forbidden_attempts']==0
        write_json(root/'posthoc_complete.json',dict(rows=5760,GT_evaluation_only=True,both300_before_any_GT_read=True,
            measurement_complete_sha256=digest(root/'measurement_complete.json'),frozen_best_sha256=digest(out/'frozen_best.json')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['measurements','gt']);parser.add_argument('--device',default='cuda:0');parser.add_argument('--config');args=parser.parse_args();c,out=configuration(args.config)
    if args.phase=='measurements':measurements(c,out,args.device)
    else:gt(c,out)

if __name__=='__main__':main()
