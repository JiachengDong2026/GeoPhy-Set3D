#!/usr/bin/env python3
"""Prospective explicit augmentation sampler registration and real GT-free backward smoke.

This preparation has no optimizer/training branch. It cannot launch formal training.
"""
import argparse
import copy
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.measurement_only import DisjointABHSampler,dual_measurement_collate
from geophy_set3d.evaluation.geometry_augmentation_sampler import GeometryAugmentationABHSampler,MODES
from geophy_set3d.evaluation.geometry_strata import population
from geophy_set3d.evaluation.geometry_stratified_sampler import PlannedABHSampler,SamplerTrainingReadGuard
from geophy_set3d.training.selfsup import make_dataset,initialize_fresh_model,objective,move_dual_batch
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything,write_json


def base_config(c,device):
    cfg=copy.deepcopy(yaml.safe_load((REPO/c['base_config']).read_text()))
    cfg['dataset']['root']=c['paths']['dataset'];cfg['training']['device']=device
    cfg['loader']['num_workers']=0;cfg['loader']['persistent_workers']=False
    return cfg


def verify(c,committed=False):
    out=Path(c['paths']['output']);p=json.loads((out/'protocol.json').read_text())
    assert p['config']==c and digest(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    for k,v in p['source_sha256'].items():assert digest(REPO/k)==v,k
    for k,v in p['input_sha256'].items():assert digest(out/k)==v,k
    if committed:
        saved=subprocess.check_output(['git','show','HEAD:docs/reports/artifacts/geometry_augmentation_preparation_v1/protocol.json'],cwd=REPO)
        assert saved==(out/'protocol.json').read_bytes()
    return p


def prepare(c):
    assert not c['formal_training'];out=Path(c['paths']['output'])
    if (out/'protocol.json').exists():verify(c);print('Preparation exists; verified, not regenerated');return
    if out.exists() and any(out.iterdir()):raise FileExistsError('Nonempty preparation directory; inspect first')
    decision=Path(c['paths']['sensitivity'])/'decision.json'
    assert json.loads(decision.read_text())['augmentation_sampler_implementation_justified']
    out.mkdir(parents=True);cfg=base_config(c,'cpu')
    with SamplerTrainingReadGuard(cfg['dataset']['root'],out/'checkpoints',out) as guard:
        ds=make_dataset(cfg);assert len(ds)==166
        bounds={};population_stats={}
        for v in (3,4,5):
            ids,scores,stats=population(ds.geometry['station_origin_m'],v,c['V5_population_seed'],c['V5_population_size'])
            bounds[str(v)]=np.quantile(scores,c['moderate_quantiles']).tolist();population_stats[str(v)]=stats
        sampler=GeometryAugmentationABHSampler(ds.geometry['station_origin_m'],ds.geometry['station_azimuth_deg'],bounds,
            seed=c['seed'],probabilities=c['probabilities'],hard_span_deg=c['hard_span_max_deg'],max_attempts=c['rejection_cap'])
        original=DisjointABHSampler();plans=[];joint=np.zeros((3,3),int);views_match=True;max_attempts=0
        mode_freq={k:0 for k in MODES};stations={k:np.zeros(64,int) for k in MODES};smoke=[];seen=set()
        old=json.loads((Path(c['paths']['existing_control'])/'Control_station_plan.json').read_text())
        assert len(old)==c['planned_epochs']
        for epoch in range(c['planned_epochs']):
            ep=[]
            for index,case in enumerate(ds.records):
                identity=dict(epoch=epoch,sample_index=index,case_id=case['case_id'])
                control=original.sample(ds.station_count,**identity)
                assert all(control[j].tolist()==old[epoch][index][k] for j,k in enumerate(('A','B','H')))
                sets,meta=sampler.sample_with_metadata(ds.station_count,**identity)
                assert (len(sets[0]),len(sets[1]))==(len(control[0]),len(control[1]))
                assert not set(sets[0])&set(sets[1]) and not set(sets[2])&(set(sets[0])|set(sets[1]))
                row=dict(case_id=case['case_id'],case_index=index,epoch=epoch,A=sets[0].tolist(),B=sets[1].tolist(),H=sets[2].tolist(),**meta)
                ep.append(row);a,b=MODES.index(meta['A_mode']),MODES.index(meta['B_mode']);joint[a,b]+=1
                for j,key in enumerate(('A_mode','B_mode')):mode_freq[meta[key]]+=1;stations[meta[key]][sets[j]]+=1
                max_attempts=max(max_attempts,meta['joint_attempts'])
                pair=(meta['A_mode'],meta['B_mode'])
                if pair not in seen:seen.add(pair);smoke.append(row)
            plans.append(ep)
        assert len(seen)==9
        # Fully materialized future schedule, not a completed training claim.
        write_json(out/'Treatment_station_plan.json',plans)
        write_json(out/'boundaries.json',dict(boundaries=bounds,population=population_stats,
            hard_span_deg=c['hard_span_max_deg'],hard_candidate_counts={str(k):len(v) for k,v in sampler.hard.items()}))
        write_json(out/'sampler_audit.json',dict(planned_pairs=len(ds)*c['planned_epochs'],planned_inputs=len(ds)*c['planned_epochs']*2,
            mode_input_counts=mode_freq,AB_joint_mode_counts=joint.tolist(),all_control_original_IDs_exact=True,
            V_counts_match_control=True,pairwise_disjoint=True,joint_max_attempts=max_attempts,
            station_frequency_by_mode={k:v.tolist() for k,v in stations.items()},
            distribution='A/B modes independently .50/.25/.25 before joint rejection; station marginals conditional on disjointness; modes never rerolled',
            existing_control_plan_sha256=digest(Path(c['paths']['existing_control'])/'Control_station_plan.json')))
        write_json(out/'smoke_cases.json',smoke)
        seed_everything(c['seed']);model=initialize_fresh_model(cfg);initial=state_hash(model.state_dict());del model
        oldp=json.loads((Path(c['paths']['existing_control'])/'protocol.json').read_text())
        assert initial==oldp['initial_state_sha256']['Control']
        assert guard.summary()['projection_open_count']==guard.summary()['gt_open_attempts']==guard.summary()['supervised_checkpoint_open_attempts']==0
        write_json(out/'registration_access_audit.json',guard.summary())
    files=['Treatment_station_plan.json','boundaries.json','sampler_audit.json','smoke_cases.json']
    sources=[Path(__file__),REPO/'src/geophy_set3d/evaluation/geometry_augmentation_sampler.py',REPO/'configs/geometry_augmentation_preparation_v1.yaml',REPO/'tests/test_geometry_augmentation_sampler.py']
    sources+=sorted((REPO/'src').rglob('*.py'));sources=list(set(sources))
    p=dict(status='PREPARATION ONLY; NO FORMAL OPTIMIZATION AUTHORIZED BY THIS ENTRY',config=c,
        rationale_sha256=digest(decision),source_sha256={str(q.relative_to(REPO)):digest(q) for q in sorted(sources)},
        input_sha256={k:digest(out/k) for k in files},initial_state_sha256=initial,
        smoke='Nine prespecified mode pairs, fresh v1 model, measurement-only original objective backward; no optimizer/parameter steps',
        future_control_reuse='Before formal training independently validate complete old Control numeric protocol/order/initial state/selector; sampler plans alone do not establish full equivalence',
        unchanged='v1 architecture/loss/natural Val selector; no GT-informed exploratory score, curriculum or weighting',
        formal_training_run=False)
    write_json(out/'protocol.json',p);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    dest=REPO/'docs/reports/artifacts/geometry_augmentation_preparation_v1';dest.mkdir(parents=True,exist_ok=True)
    for name in ['protocol.json','protocol.sha256','boundaries.json','sampler_audit.json','smoke_cases.json','registration_access_audit.json']:shutil.copy2(out/name,dest/name)
    print(json.dumps(json.loads((out/'sampler_audit.json').read_text()),indent=2))


def smoke(c,device):
    out=Path(c['paths']['output']);p=verify(c,True)
    if (out/'smoke_result.json').exists():
        assert json.loads((out/'smoke_result.json').read_text())['passed'];print('Complete smoke exists; no repeated execution');return
    rows=json.loads((out/'smoke_cases.json').read_text());cfg=base_config(c,device);start=time.perf_counter()
    with SamplerTrainingReadGuard(cfg['dataset']['root'],out/'checkpoints',out) as audit:
        seed_everything(c['seed']);torch.set_num_threads(2);model=initialize_fresh_model(cfg);model.eval()
        assert state_hash(model.state_dict())==p['initial_state_sha256'];ds=make_dataset(cfg)
        plan=json.loads((out/'Treatment_station_plan.json').read_text());ds.sampler=PlannedABHSampler(plan);results=[]
        for r in rows:
            ds.set_epoch(r['epoch']);sample=ds[r['case_index']]
            batch=move_dual_batch(dual_measurement_collate([sample]),device)
            assert batch['A']['input_station_indices'][0].tolist()==r['A'] and batch['B']['input_station_indices'][0].tolist()==r['B']
            model.zero_grad(set_to_none=True);loss,parts,volumes,_=objective(model,batch);loss.backward()
            grads=[q.grad for q in model.parameters() if q.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(float(g.abs().max())>0 for g in grads)
            assert all(torch.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in volumes)
            results.append(dict(case_id=r['case_id'],A_mode=r['A_mode'],B_mode=r['B_mode'],loss=float(loss.detach()),
                loss_parts={k:float(v.detach()) for k,v in parts.items()},finite_nonzero_gradient=True))
        model.zero_grad(set_to_none=True);assert state_hash(model.state_dict())==p['initial_state_sha256']
        assert audit.summary()['gt_open_attempts']==audit.summary()['supervised_checkpoint_open_attempts']==0
        result=dict(passed=True,mode_pairs=9,actual_objective_backward_passes=9,optimizer_created=False,optimizer_steps=0,
            parameters_buffers_unchanged=True,initial_state_sha256=p['initial_state_sha256'],seconds=time.perf_counter()-start,
            audit=audit.summary(),rows=results,formal_training_run=False)
        write_json(out/'smoke_result.json',result)
    dest=REPO/'docs/reports/artifacts/geometry_augmentation_preparation_v1';shutil.copy2(out/'smoke_result.json',dest/'smoke_result.json')
    print(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(REPO/'configs/geometry_augmentation_preparation_v1.yaml'))
    ap.add_argument('--stage',choices=['prepare','smoke'],required=True);ap.add_argument('--device',default='cuda:2');a=ap.parse_args()
    c=yaml.safe_load(Path(a.config).read_text())
    if a.stage=='prepare':prepare(c)
    else:smoke(c,a.device)

if __name__=='__main__':main()
