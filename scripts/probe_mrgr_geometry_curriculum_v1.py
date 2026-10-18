#!/usr/bin/env python3
"""Registered low-dose geometry curriculum; two fresh matched 20-epoch arms."""
import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import probe_mrgr_multiscale_v1 as base
import train_mrgr_multiscale_formal_v1 as formal
import run_mrgr_local_detail_v1 as helper
from geophy_set3d.data.measurement_only import DisjointABHSampler

ROOT=base.ROOT
OUT=ROOT/'outputs/research/mrgr_geometry_curriculum_v1'
ARMS=('control','curriculum')
ORIGINAL_DATASET=base.selfsup.make_dataset
ORIGINAL_MODEL=base.model_for
ACTIVE_ARM='control'

def read(p):return json.loads(Path(p).read_text())
def span(angles):
    a=np.sort(np.asarray(angles)%360)
    return float(360-np.diff(np.r_[a,a[0]+360]).max())

class CurriculumSampler:
    """Only geometry + case identity; exact deterministic quota, never target data."""
    def __init__(self,original,azimuth,case_ids):
        self.original=original;self.azimuth=np.asarray(azimuth)
        self.case_ids=list(case_ids);self.candidates={}
        order=np.argsort(self.azimuth%360);a=self.azimuth[order]%360
        for n in (3,4,5):
            values=[]
            for i in range(len(a)):
                for j in range(i+n-1,len(a)):
                    if not 20<=a[j]-a[i]<=40:continue
                    rng=np.random.default_rng(np.random.SeedSequence([2026101703,n,i,j]))
                    middle=rng.choice(order[i+1:j],n-2,replace=False)
                    ids=np.r_[order[i],np.sort(middle),order[j]].astype(np.int64)
                    assert 20-1e-8<=span(self.azimuth[ids])<=40+1e-8
                    values.append(ids)
            assert values,f'No medium-span local station group for V{n}'
            self.candidates[n]=values
        self.quota_cache={}

    def quota(self,epoch):
        # Zero-index epochs 0..4 untouched; 5..19 ramp to floor(10%*N).
        return math.floor(len(self.case_ids)*.10*max(0,min(15,epoch-4))/15)

    def selected(self,epoch):
        if epoch not in self.quota_cache:
            ranked=sorted(range(len(self.case_ids)),key=lambda i:hashlib.sha256(
                f'curriculum-quota-v1|20261011|{epoch}|{self.case_ids[i]}'.encode()).digest())
            self.quota_cache[epoch]=set(ranked[:self.quota(epoch)])
        return self.quota_cache[epoch]

    def sample(self,station_count,*,epoch=0,sample_index=0,case_id=''):
        original=self.original.sample(station_count,epoch=epoch,sample_index=sample_index,case_id=case_id)
        if sample_index not in self.selected(epoch):return original
        assert self.case_ids[sample_index]==case_id
        seed=np.frombuffer(hashlib.sha256(f'curriculum-stations-v1|{epoch}|{case_id}'.encode()).digest(),dtype='<u4')
        rng=np.random.default_rng(seed)
        na,nb,nh=map(len,original)
        a=self.candidates[na][int(rng.integers(len(self.candidates[na])))].copy()
        remaining=np.setdiff1d(np.arange(station_count),a)
        other=rng.choice(remaining,nb+nh,replace=False).astype(np.int64)
        return a,other[:nb],other[nb:]

def model_for(config,arm,device):return ORIGINAL_MODEL(config,'multiscale',device)

def dataset(config,validation=False,fixed_gate=False):
    ds=ORIGINAL_DATASET(config,validation=validation,fixed_gate=fixed_gate)
    if not validation and ACTIVE_ARM=='curriculum':
        ds.sampler=CurriculumSampler(ds.sampler,ds.geometry['station_azimuth_deg'],[r['case_id'] for r in ds.records])
    return ds

def sources():
    paths=[*sorted((ROOT/'src').rglob('*.py')),*sorted((ROOT/'src').rglob('*.cpp')),
        Path(__file__),Path(base.__file__),Path(formal.__file__),Path(helper.__file__),
        ROOT/'configs/train_selfsup_oari_mrgr_v1.yaml']
    return {str(p.relative_to(ROOT)):base.digest(p) for p in paths}

def verify():
    p=read(OUT/'protocol.json')
    assert base.digest(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    assert p['sources']==sources(),'Registered source changed'
    formal.verify_data(p)
    return p

def register():
    assert not OUT.exists(),'Refuse overwrite'
    p0=base.verify();config=copy.deepcopy(p0['config']);config['training']['device']='cuda:3'
    OUT.mkdir(parents=True)
    with base.GTReadGuard(config['dataset']['root'],OUT) as guard:
        train=ORIGINAL_DATASET(config);val=ORIGINAL_DATASET(config,validation=True)
        ids=[r['case_id'] for r in train.records]
        sampler=CurriculumSampler(train.sampler,train.geometry['station_azimuth_deg'],ids)
        audit=[];epoch_stats=[]
        for epoch in range(20):
            changed=[]
            for i,case in enumerate(ids):
                old=train.sampler.sample(train.station_count,epoch=epoch,sample_index=i,case_id=case)
                new=sampler.sample(train.station_count,epoch=epoch,sample_index=i,case_id=case)
                again=sampler.sample(train.station_count,epoch=epoch,sample_index=i,case_id=case)
                assert all(np.array_equal(x,y) for x,y in zip(new,again))
                assert list(map(len,old))==list(map(len,new))
                assert len(set(np.concatenate(new)))==sum(map(len,new))
                replaced=i in sampler.selected(epoch)
                if not replaced:assert all(np.array_equal(x,y) for x,y in zip(old,new))
                if replaced:changed.append(span(sampler.azimuth[new[0]]))
                audit.append(dict(epoch=epoch+1,case_id=case,replaced=replaced,
                    original={k:v.tolist() for k,v in zip(('A','B','H'),old)},
                    candidate={k:v.tolist() for k,v in zip(('A','B','H'),new)},
                    candidate_A_span_deg=span(sampler.azimuth[new[0]])))
            assert len(changed)==sampler.quota(epoch)
            epoch_stats.append(dict(epoch=epoch+1,replacements=len(changed),ratio=len(changed)/166,
                spans_deg=changed,mean_span_deg=float(np.mean(changed)) if changed else None))
        (OUT/'station_plan.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in audit))
        base.write_json(OUT/'sampler_audit.json',dict(passed=True,epoch_stats=epoch_stats,
            candidate_counts={str(k):len(v) for k,v in sampler.candidates.items()},
            deterministic=True,pairwise_disjoint=True,unchanged_samples_exact=True,view_counts_exact=True))
        model=model_for(config,'control','cpu');initial=base.state_hash(model.state_dict())
        for arm in ARMS:
            other=model_for(config,arm,'cpu');assert base.state_hash(other.state_dict())==initial
            path=OUT/'checkpoints'/arm;path.mkdir(parents=True)
            base.atomic_checkpoint(path/'initial.pt',dict(model=other.state_dict(),arm=arm,epoch=0))
        data=read(ROOT/'outputs/research/optimization_screening_20261016/data_identity_postflight.json')['files']
        p=dict(registered_at=base.now(),config=config,sources=sources(),data_files=data,epochs=20,
            train_case_ids=ids,validation_case_ids=[r['case_id'] for r in val.records],
            train_cases=166,val_cases=36,updates_per_epoch=83,updates_per_arm=1660,initial_sha256=initial,
            parameter_count=sum(v.numel() for v in model.parameters()),
            intervention='Only train station sampler: epochs1-5 unchanged; epochs6-20 ramp exact per-epoch quota floor(166*0.1*(epoch-5)/15). Replace A by n stations in local azimuth20-40deg window; preserve original A/B/H counts, redraw B/H from remainder. No GT/physical metadata.',
            caveat='Station marginal distribution also shifts; not isolated angular coverage causality. One preset strategy, no ratio/span search.',
            prior='Old fixed 50/25/25 augmentation had natural 3D error degradation11.13%; this distinct low-dose curriculum is exploratory, not confirmatory.',
            sampler_plan_sha256=base.digest(OUT/'station_plan.jsonl'),
            precision='fp32 unchanged',schedule='First20 of cosine T_max300, fresh identical model weights and epoch shuffle',
            loss='Unchanged selfsup v1',gt_allowed=False,final_test_allowed=False,
            checkpoint_rule='Fixed epoch20 last.pt for all screening metrics; no best selection',
            promotion_gate=dict(ordinary_H_max_ratio=.99,ordinary_P90_max_ratio=1.02,clustered_H_max_ratio=1.02,runtime_max_ratio=1.20,estimated300_max_hours=5.),
            hard_panel_source=str(ROOT/'outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/registration.json'),
            hard_panel_source_sha256=base.digest(ROOT/'outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/registration.json'),
            registration_audit=guard.summary(),no_automatic_formal_training=True)
        formal.verify_data(p);base.write_json(OUT/'protocol.json',p)
        (OUT/'protocol.sha256').write_text(base.digest(OUT/'protocol.json')+'\n')
    print(json.dumps(dict(stage='registered',parameters=p['parameter_count'],replacements=sum(x['replacements'] for x in epoch_stats))),flush=True)

def smoke():
    p=verify();config=p['config'];device=config['training']['device']
    with base.GTReadGuard(config['dataset']['root'],OUT) as guard:
        ds=dataset(config);ds.set_epoch(19)
        chosen=CurriculumSampler(ds.sampler,ds.geometry['station_azimuth_deg'],p['train_case_ids']).selected(19)
        first=sorted(chosen)[:2]
        # Candidate variable-view padded batch at final intervention strength.
        cs=CurriculumSampler(ds.sampler,ds.geometry['station_azimuth_deg'],p['train_case_ids']);ds.sampler=cs
        batch=base.selfsup.move_dual_batch(base.dual_measurement_collate([ds[i] for i in first]),device)
        base.selfsup.assert_disjoint(batch);records={};outputs={}
        for arm in ARMS:
            m=model_for(config,arm,device);m.eval()
            with torch.no_grad():outputs[arm]=m(batch['A']).cpu()
            before=base.state_hash(m.state_dict());m.train();times=[]
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(3):
                m.zero_grad(set_to_none=True);torch.cuda.synchronize(device);began=time.perf_counter()
                loss,parts,vols,projs=base.selfsup.objective(m,batch);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
                torch.cuda.synchronize(device);times.append(time.perf_counter()-began)
                assert all(v.std()>1e-8 and torch.isfinite(v).all() for v in vols)
                del loss,parts,vols,projs
            assert before==base.state_hash(m.state_dict())
            records[arm]=dict(seconds=times,gradient_norm=float(norm),peak_gib=torch.cuda.max_memory_allocated(device)/1024**3)
            del m;gc.collect();torch.cuda.empty_cache()
        assert torch.equal(outputs['control'],outputs['curriculum'])
        estimate=sum(np.mean(v['seconds'][1:]) for v in records.values())*93.8*20*1.15
        result=dict(passed=True,arms=records,estimated_two_arm_minutes=float(estimate/60),
            requires_parent_resource_notice=estimate>45*60,initial_forward_exact=True,gt_audit=guard.summary())
        base.write_json(OUT/'smoke.json',result);print(json.dumps(result),flush=True)

def train(arm):
    global ACTIVE_ARM
    p=verify();assert read(OUT/'smoke.json')['passed'];ACTIVE_ARM=arm
    # Frozen shared training loop; only the explicitly registered dataset factory differs.
    base.OUT=OUT;base.model_for=model_for;base.selfsup.make_dataset=dataset
    return base.train_arm(p,arm)

def panel():
    p=verify();helper.ARMS=ARMS;helper.verify=lambda out:p
    helper.candidate.model_for=model_for
    for arm in ARMS:
        link=OUT/arm/'checkpoints'
        if not link.exists():link.symlink_to(Path('../checkpoints')/arm,target_is_directory=True)
    helper.hard_panel(OUT)
    # Independent final-field sanity, measurement-only, no GT.
    config=p['config'];device=config['training']['device'];records=[]
    with base.GTReadGuard(config['dataset']['root'],OUT) as guard:
        val=ORIGINAL_DATASET(config,validation=True)
        for arm in ARMS:
            ckpt=torch.load(OUT/'checkpoints'/arm/'last.pt',map_location='cpu',weights_only=False)
            m=model_for(config,arm,device);m.load_state_dict(ckpt['model']);m.eval()
            for start in range(0,len(val),2):
                b=base.selfsup.move_dual_batch(base.dual_measurement_collate([val[i] for i in range(start,min(start+2,len(val)))]),device)
                with torch.no_grad():v=m(b['A'])
                for j,case in enumerate(b['A']['case_id']):
                    r=dict(arm=arm,case_id=case,min=float(v[j].min()),max=float(v[j].max()),std=float(v[j].std()),mean=float(v[j].mean()))
                    assert r['std']>1e-8 and r['max']>r['min'] and all(math.isfinite(x) for k,x in r.items() if k not in ('arm','case_id'))
                    records.append(r)
            del m,ckpt;gc.collect();torch.cuda.empty_cache()
        base.write_json(OUT/'output_sanity.json',dict(passed=True,records=records,gt_audit=guard.summary()))

def compare():
    p=verify();logs={a:[json.loads(x) for x in (OUT/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    for x,y in zip(logs['control'],logs['curriculum']):
        assert x['epoch']==y['epoch'] and x['learning_rate']==y['learning_rate'] and x['loader_rng_sha256']==y['loader_rng_sha256']
        assert x['val']['case_station_digest']==y['val']['case_station_digest']
        if x['epoch']<=5:assert x['train']['case_station_digest']==y['train']['case_station_digest']
        else:assert x['train']['case_station_digest']!=y['train']['case_station_digest']
        for split,n in [('train',166),('val',36)]:assert x[split]['sample_count']==y[split]['sample_count']==n
    data={a:[json.loads(x) for x in (OUT/a/'final_per_case.jsonl').read_text().splitlines()] for a in ARMS}
    assert all([x['case_id'] for x in data[a]]==p['validation_case_ids'] for a in ARMS)
    arrays={a:np.array([x['heldout_relative_l1'] for x in data[a]]) for a in ARMS}
    times={a:float(np.mean([x['seconds'] for x in logs[a][2:]])) for a in ARMS}
    hard=read(OUT/'hard_panel.json')['means'];ratios=dict(ordinary_H=arrays['curriculum'].mean()/arrays['control'].mean(),
        ordinary_P90=np.quantile(arrays['curriculum'],.9)/np.quantile(arrays['control'],.9),
        clustered_H=hard['curriculum']/hard['control'],runtime=times['curriculum']/times['control'])
    t=p['promotion_gate'];checks={k:bool(v<=t[k+'_max_ratio']) for k,v in ratios.items()}
    checks.update(five_hour=times['curriculum']*300/3600<=t['estimated300_max_hours'],smoke=read(OUT/'smoke.json')['passed'],output_sanity=read(OUT/'output_sanity.json')['passed'])
    delta=arrays['curriculum']-arrays['control'];rng=np.random.default_rng(2026101703)
    ci=np.quantile(delta[rng.integers(0,36,(20000,36))].mean(1),[.025,.975]).tolist()
    hardrows=[json.loads(x) for x in (OUT/'hard_panel_per_case.jsonl').read_text().splitlines()]
    h={a:np.array([np.mean([x['relative_l1'] for x in hardrows if x['arm']==a and x['case_id']==c]) for c in p['validation_case_ids']]) for a in ARMS}
    hd=h['curriculum']-h['control'];hci=np.quantile(hd[rng.integers(0,36,(20000,36))].mean(1),[.025,.975]).tolist()
    result=dict(completed=True,passed=all(checks.values()),checks=checks,ratios={k:float(v) for k,v in ratios.items()},
        ordinary={a:dict(mean=float(v.mean()),p90=float(np.quantile(v,.9))) for a,v in arrays.items()},
        clustered_five_draw=hard,paired_ordinary_ci95=ci,wins=int((delta<0).sum()),losses=int((delta>0).sum()),
        paired_cluster_ci95=hci,cluster_wins=int((hd<0).sum()),cluster_losses=int((hd>0).sum()),
        warm_epoch_seconds=times,estimated300_hours=times['curriculum']*300/3600,
        same_initialization=True,same_shuffle=True,same_validation_stations=True,train_station_changes_registered=True,
        gt_used=False,formal_training_started=False,interpretation='Single-seed20-epoch measurement-only development screening, no converged or3D superiority claim',finished_at=base.now())
    base.write_json(OUT/'comparison.json',result);print(json.dumps(result),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','smoke','train','panel','compare','verify'));parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.action=='register':register()
    elif args.action=='smoke':smoke()
    elif args.action=='train':assert args.arm;train(args.arm)
    elif args.action=='panel':panel()
    elif args.action=='compare':compare()
    else:verify();print('Sources/data verified')

if __name__=='__main__':main()
