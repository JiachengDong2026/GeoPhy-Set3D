#!/usr/bin/env python3
"""Register then run fresh, GT-free matched300; only sampler differs."""
import argparse
import copy
import json
import sys
import time
import subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
import yaml
from torch.utils.data import Subset
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.measurement_only import DisjointABHSampler
from geophy_set3d.evaluation.geometry_strata import population
from geophy_set3d.evaluation.geometry_strata import membership
from geophy_set3d.evaluation.geometry_stratified_sampler import StratifiedABHSampler,PlannedABHSampler,SamplerTrainingReadGuard
from geophy_set3d.training.selfsup import make_dataset,make_loader,run_epoch,selection_metric,initialize_fresh_model,validate_selfsup_config
from geophy_set3d.training.utils import seed_everything,write_json,atomic_checkpoint,rng_state,restore_rng
from geophy_set3d.training.rvtc_matched import state_hash
from diagnose_clustered_one_swap import write_csv
CONFIG=REPO/'configs/geometry_stratified_selfsup_matched_v1.yaml'


def configuration():
    c=yaml.safe_load(CONFIG.read_text());return c,Path(c['output'])


def arm_config(c,arm):
    cfg=yaml.safe_load((REPO/c['base_config']).read_text());cfg=copy.deepcopy(cfg)
    cfg['experiment']['name']=c['name']+'_'+arm
    cfg['experiment']['output_root']=c['output']
    cfg['training']['device']=c['arms'][arm]['device'];cfg['loader']['num_workers']=0;cfg['loader']['persistent_workers']=False
    validate_selfsup_config(cfg);return cfg


def register(c,out):
    if (out/'protocol.json').exists():verify(c,out,committed=False);print('Matched registration exists, verified');return
    stage=Path(c['stage_c']);assert json.loads((stage/'decision.json').read_text())['decision']=='GO'
    out.mkdir(parents=True,exist_ok=True);cfg=arm_config(c,'Control')
    with SamplerTrainingReadGuard(cfg['dataset']['root'],out/'registration_checkpoints',out) as audit:
        ds=make_dataset(cfg);val=make_dataset(cfg,validation=True);assert len(ds)==166 and len(val)==36
        g=ds.geometry;bounds=json.loads((stage/'boundaries.json').read_text())['V']
        _,_,v5=population(g['station_origin_m'],5,c['V5_population_seed'],c['V5_population_size'])
        v5.update(source_geometry_sha256=digest(ds.root/'geometry/train_geometry.npz'),source_code_sha256=digest(REPO/'src/geophy_set3d/evaluation/geometry_strata.py'))
        write_json(out/'V5_boundaries.json',v5)
        boundaries={str(v):bounds[str(v)]['boundaries'] for v in (3,4)};boundaries['5']=v5['boundaries']
        treatment=StratifiedABHSampler(g['station_origin_m'],boundaries,max_attempts=c['sampler_rejection_cap'])
        control=DisjointABHSampler();plans={a:[] for a in c['arms']};orders=[];gen=torch.Generator().manual_seed(c['seed'])
        freqs={a:np.zeros((3,len(g['station_origin_m'])),dtype=int) for a in c['arms']}
        distributions={a:dict(scores=[],separation_deg=[],target_strata={'low':0,'middle':0,'high':0},AB_stratum_joint_counts=np.zeros((3,3),int)) for a in c['arms']}
        epochs=[]
        for epoch in range(c['epochs']):
            orders.append(torch.randperm(len(ds),generator=gen).tolist());ep={a:[] for a in c['arms']}
            for index,case in enumerate(ds.records):
                ids=control.sample(ds.station_count,epoch=epoch,sample_index=index,case_id=case['case_id'])
                ta,tb,th,q=treatment.sample(ds.station_count,epoch=epoch,sample_index=index,case_id=case['case_id'])
                assert (len(ids[0]),len(ids[1]))==(len(ta),len(tb))
                for arm,sets in [('Control',ids),('Treatment',(ta,tb,th))]:
                    qa,qb=[('low','middle','high').index(membership(treatment.score(s),boundaries[str(len(s))])) for s in sets[:2]]
                    distributions[arm]['AB_stratum_joint_counts'][qa,qb]+=1
                    ep[arm].append(dict(case_id=case['case_id'],A=sets[0].tolist(),B=sets[1].tolist(),H=sets[2].tolist(),target_stratum=q if arm=='Treatment' else None))
                    for j,subset in enumerate(sets):freqs[arm][j,subset]+=1
                    for subset in sets[:2]:
                        distributions[arm]['scores'].append(treatment.score(subset))
                        az=g['station_azimuth_deg'][subset];a,b=np.triu_indices(len(subset),1)
                        distributions[arm]['separation_deg'].extend(np.abs(az[a]-az[b]).tolist())
                    if arm=='Treatment':distributions[arm]['target_strata'][q]+=1
            for arm in c['arms']:plans[arm].append(ep[arm])
            epochs.append(dict(epoch=epoch,case_order_sha256=__import__('hashlib').sha256(json.dumps(orders[-1]).encode()).hexdigest(),
                matched_view_counts=True,Control_station_sha256=__import__('hashlib').sha256(json.dumps(ep['Control']).encode()).hexdigest(),
                Treatment_station_sha256=__import__('hashlib').sha256(json.dumps(ep['Treatment']).encode()).hexdigest()))
        for arm in c['arms']:write_json(out/f'{arm}_station_plan.json',plans[arm])
        write_json(out/'case_orders.json',orders);write_csv(out/'epoch_plan_audit.csv',epochs)
        frequency=[];distribution_summary={}
        for arm in c['arms']:
            for i in range(ds.station_count):frequency.append(dict(arm=arm,station_id=i,A=int(freqs[arm][0,i]),B=int(freqs[arm][1,i]),H=int(freqs[arm][2,i]),
                azimuth_deg=float(g['station_azimuth_deg'][i]),radius_m=float(np.linalg.norm(g['station_origin_m'][i,[0,2]]-.3)),height_m=float(g['station_origin_m'][i,1])))
            rows=[r for r in frequency if r['arm']==arm];w=np.asarray([r['A']+r['B'] for r in rows]);summary={}
            for key,edges in [('azimuth_deg',np.linspace(0,180,19)),('radius_m',np.linspace(5.5,6.5,11)),('height_m',np.linspace(.25,.35,11))]:
                summary[key]=dict(edges=edges.tolist(),counts=np.histogram([r[key] for r in rows],bins=edges,weights=w)[0].tolist())
            for key,edges in [('scores',np.linspace(0,1,21)),('separation_deg',np.linspace(0,180,19))]:
                vals=np.asarray(distributions[arm][key]);summary[key]=dict(edges=edges.tolist(),counts=np.histogram(vals,bins=edges)[0].tolist(),mean=float(vals.mean()),std=float(vals.std()))
            summary['target_strata']=distributions[arm]['target_strata'];distribution_summary[arm]=summary
            summary['AB_stratum_joint_counts']=distributions[arm]['AB_stratum_joint_counts'].tolist()
        write_csv(out/'sampler_station_frequency.csv',frequency);write_json(out/'sampler_distribution_audit.json',distribution_summary)
        val_ids=[]
        for index,r in enumerate(val.records):
            a,b,h=val.sampler.sample(val.station_count,epoch=0,sample_index=index,case_id=r['case_id'])
            val_ids.append(dict(case_id=r['case_id'],A=a.tolist(),B=b.tolist(),H=h.tolist()))
        write_json(out/'selector_station_ids.json',val_ids)
        # Two independently constructed fresh initial states; no checkpoint is read.
        hashes={}
        for arm in c['arms']:
            seed_everything(c['seed']);model=initialize_fresh_model(arm_config(c,arm));hashes[arm]=state_hash(model.state_dict());del model;torch.cuda.empty_cache()
        assert len(set(hashes.values()))==1
        write_json(out/'registration_access_audit.json',audit.summary());assert audit.summary()['gt_open_attempts']==audit.summary()['projection_open_count']==0
    sourcefiles=list((REPO/'src').rglob('*.py'))+[Path(__file__),CONFIG,REPO/'scripts/evaluate_geometry_stratified_matched.py',REPO/'tests/test_geometry_stratified_matched.py']
    files=['V5_boundaries.json','Control_station_plan.json','Treatment_station_plan.json','case_orders.json','epoch_plan_audit.csv',
        'sampler_station_frequency.csv','sampler_distribution_audit.json','selector_station_ids.json']
    protocol=dict(status='REGISTERED BEFORE OPTIMIZATION',config=c,configs={a:arm_config(c,a) for a in c['arms']},initial_state_sha256=hashes,
        source_sha256={str(q.relative_to(REPO)):digest(q) for q in sourcefiles},input_sha256={q:digest(out/q) for q in files},
        stage_c_decision_sha256=digest(stage/'decision.json'),evaluation_station_subsets_sha256=digest(stage/'station_subsets.json'),
        optimizer_steps=24900,view_counts_exactly_matched=True,case_orders_shared=True,initialization='fresh seed20261011; no checkpoint reads; own last only for resume',
        objective='unchanged original selfsup.objective; L_hold+.5L_cross+.1L_obs+.05L_cons',
        selector='unchanged original Val-A natural random dual-input H4 held-out mean relative L1; same frozen A/B/H for both',
        only_intervention='A/B input geometry subset distribution; same V per case/epoch, q uniform and both in q; H remains uniform disjoint',
        stage_D_gate=dict(natural_macro='Treatment-Control 95%CI lower must not exceed0',low_macro='mean delta<0; CI reported',
            high_macro='reject material harm if mean relative increase>5% AND CI lower>0',clustered='secondary only'),
        posthoc='Both300complete and bothbest hashes frozen -> measurement only all frozen C inputs -> separate GT64 evaluator; no GT before this boundary',
        limitations='single matched seed; station marginals may change; shared4090 timing not hardware benchmark; no GT tuning')
    write_json(out/'protocol.json',protocol);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    archive=REPO/'docs/reports/artifacts/geometry_stratified_selfsup_matched_v1';archive.mkdir(parents=True,exist_ok=True)
    import shutil
    for f in ['protocol.json','protocol.sha256','V5_boundaries.json','case_orders.json','epoch_plan_audit.csv','sampler_station_frequency.csv','sampler_distribution_audit.json','selector_station_ids.json','registration_access_audit.json']:
        shutil.copy2(out/f,archive/f)
    shutil.copy2(stage/'station_subsets.json',archive/'evaluation_station_subsets.json')
    print('Matched300 registered; identical initial state',hashes,'no measurement/GT opens',flush=True)


def verify(c,out,committed=True):
    p=json.loads((out/'protocol.json').read_text());assert p['config']==c
    assert digest(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    for key,sha in p['source_sha256'].items():assert digest(REPO/key)==sha,key
    for key,sha in p['input_sha256'].items():assert digest(out/key)==sha,key
    assert digest(Path(c['stage_c'])/'decision.json')==p['stage_c_decision_sha256']
    assert digest(Path(c['stage_c'])/'station_subsets.json')==p['evaluation_station_subsets_sha256']
    if committed:
        data=subprocess.check_output(['git','show','HEAD:docs/reports/artifacts/geometry_stratified_selfsup_matched_v1/protocol.json'],cwd=REPO)
        assert data==(out/'protocol.json').read_bytes()
    return p


def train(c,out,arm):
    p=verify(c,out);cfg=arm_config(c,arm);root=out/arm;root.mkdir(exist_ok=True);ck=root/'checkpoints';ck.mkdir(exist_ok=True)
    if (root/'complete.json').exists():assert json.loads((root/'complete.json').read_text())['epochs']==300;print(arm,'complete; no repeated training');return
    with SamplerTrainingReadGuard(cfg['dataset']['root'],ck,root) as audit:
        seed_everything(c['seed']);torch.set_num_threads(2);model=initialize_fresh_model(cfg)
        assert state_hash(model.state_dict())==p['initial_state_sha256'][arm]
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        ds=make_dataset(cfg);val=make_dataset(cfg,validation=True)
        plans=json.loads((out/f'{arm}_station_plan.json').read_text());ds.sampler=PlannedABHSampler(plans)
        orders=json.loads((out/'case_orders.json').read_text());generator=torch.Generator().manual_seed(c['seed'])
        vloader=make_loader(val,cfg);start=0;best=float('inf');count={'steps':0}
        last=ck/'last.pt'
        if last.exists():
            saved=torch.load(last,map_location=cfg['training']['device'],weights_only=False)
            assert saved['config']==cfg and saved['protocol_sha256']==digest(out/'protocol.json')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            restore_rng(saved['rng'],generator);start=saved['epoch']+1;best=saved['best'];count['steps']=saved['optimizer_steps']
            rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines() if json.loads(l)['epoch']<start]
            (root/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        optimizer.register_step_post_hook(lambda *args:count.update(steps=count['steps']+1))
        for epoch in range(start,300):
            ds.set_epoch(epoch);val.set_epoch(0);began=time.perf_counter();steps_before=count['steps'];lr=optimizer.param_groups[0]['lr']
            loader=make_loader(Subset(ds,orders[epoch]),cfg,shuffle=False,generator=generator)
            torch.cuda.reset_peak_memory_stats(cfg['training']['device']);tr=run_epoch(model,loader,cfg,optimizer);va=run_epoch(model,vloader,cfg)
            assert count['steps']-steps_before==83 and tr['sample_count']==166 and va['sample_count']==36
            score=selection_metric(va);improved=score<best;best=min(best,score);scheduler.step()
            row=dict(epoch=epoch,arm=arm,train=tr,val=va,learning_rate=lr,seconds=time.perf_counter()-began,optimizer_steps=count['steps'],
                case_order=orders[epoch],GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(cfg['training']['device']))
            payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,
                optimizer_steps=count['steps'],config=cfg,protocol_sha256=digest(out/'protocol.json'),rng=rng_state(generator),volumetric_gt_used=False)
            atomic_checkpoint(last,payload)
            if improved:atomic_checkpoint(ck/'best.pt',payload)
            with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            write_json(root/'access_audit.json',audit.summary())
            print(f'{arm} {epoch+1}/300 steps={count["steps"]} valH_L1={score:.6f} best={best:.6f} seconds={row["seconds"]:.1f}',flush=True)
        assert count['steps']==24900
        write_json(root/'complete.json',dict(epochs=300,optimizer_steps=count['steps'],best_sha256=digest(ck/'best.pt'),last_sha256=digest(last),
            initial_state_sha256=p['initial_state_sha256'][arm],GT_used=False,access=audit.summary(),selector='original natural held-out mean measurement L1',
            protocol_sha256=digest(out/'protocol.json')))
    verify(c,out)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','train']);parser.add_argument('--arm',choices=['Control','Treatment']);args=parser.parse_args()
    c,out=configuration()
    if args.phase=='register':register(c,out)
    else:
        assert args.arm;train(c,out,args.arm)

if __name__=='__main__':main()
