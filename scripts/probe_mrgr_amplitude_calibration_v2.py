#!/usr/bin/env python3
"""Exact nonlinear observed-ray amplitude calibration of a previously frozen field."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import test_mrgr_multiscale_frozen_v1 as ev

ROOT, SOURCE, DATA = ev.ROOT, ev.OUT, ev.DATA
OUT = ROOT/'outputs/research/mrgr_amplitude_calibration_v2'
SCALES = np.linspace(.9,1.1,9).tolist()


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sha(path): return ev.sha(path)


def verify():
    p=read(OUT/'protocol.json')
    assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for path,digest in p['sources'].items(): assert sha(path)==digest,path
    ev.formal.verify(ev.RUN)
    return p


def register():
    assert not OUT.exists(), 'Refuse existing experiment'
    r=read(SOURCE/'registration.json')
    settings=[s for s in r['settings'] if s['condition'] in ('Val-A-V4','clustered-V4')]
    assert len(settings)==6
    manifest=[x for x in rows(SOURCE/'prediction_manifest.jsonl') if x['condition'] in ('Val-A-V4','clustered-V4')]
    assert len(manifest)==216
    files=[Path(__file__),Path(ev.__file__),SOURCE/'registration.json',SOURCE/'prediction_manifest.jsonl',
        SOURCE/'predictions_complete.json',DATA/'source_index.jsonl',DATA/'geometry/train_geometry.npz',
        ROOT/'src/geophy_set3d/physics/projectors.py',ROOT/'src/geophy_set3d/training/projection_losses.py']
    files += [SOURCE/x['prediction'] for x in manifest]
    p=dict(registered_at=ev.now(),sources={str(p):sha(p) for p in files},cases=r['cases'],settings=settings,
        scale_grid=SCALES,model='previously frozen formal multiscale best278; no new network training',
        mechanism='Global mass-fraction amplitude adjustment only. Does not change spatial localization when unclipped.',
        selection='For each input field choose feasible scale with lowest exact observed-ray relativeL1; ties prefer identity. H never used to select percase scale.',
        forward='Unchanged nonlinear mass_to_mole, XYZ64 interpolation, midpointK96; do NOT assume projection(alpha*C)=alpha*projection(C).',
        bounds='Only grid values .9:.025:1.1 with alpha*max(C)<=1; identity always feasible; no clipping or postselection changes',
        gate=dict(mean_ratio_max=.99,p90_ratio_max=1.02,cluster5_mean_ratio_max=1.02),
        no_GT=True,no_final_test=True,no_formal_training_needed=True,
        runtime='CPU2threads; postprocessing cost measured separately, no direct GPU timing comparison',
        limitations='Deterministic frozen-model developmentvalidation screen, no3D benefit inferred. Cannot repair missing peaks or centroid displacement.')
    OUT.mkdir(parents=True)
    ev.write(OUT/'protocol.json',p)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')


def calibrate():
    p=verify()
    assert not (OUT/'per_case.jsonl').exists(), 'Refuse replacing results'
    original={(x['condition'],x['draw'],x['case_id']):x for x in rows(SOURCE/'prediction_manifest.jsonl')}
    records=[]
    start=time.perf_counter()
    with ev.GTReadGuard(DATA,ev.RUN/'checkpoints') as guard, torch.no_grad():
        for setting in p['settings']:
            ds=ev.MeasurementOnlyPoolDataset(DATA,split='val',geometry_pool='train',evaluation_only=True,
                min_input_views=4,max_input_views=4,hold_views=0,base_seed=20261011,
                normalization='linear_p99_5',statistics_file='train_statistics.json')
            ds.sampler=ev.Replay(setting)
            for i in range(len(ds)):
                sample=ds[i]
                case=sample['case_id']
                saved=original[setting['condition'],setting['draw'],case]
                assert sha(SOURCE/saved['prediction'])==saved['sha256']
                assert sample['input_station_indices'].tolist()==saved['input_ids']
                assert sample['hold_station_indices'].tolist()==saved['heldout_ids']
                with np.load(SOURCE/saved['prediction'],allow_pickle=False) as f:
                    field=torch.from_numpy(f['C1'].copy())[None,None]
                batch=ev.projection_pool_collate([sample])
                assert field.shape==(1,1,64,64,64)
                assert torch.isfinite(field).all() and 0<=field.min() and field.max()<=1
                t0=time.perf_counter()
                feasible=sorted([a for a in p['scale_grid'] if a*float(field.max())<=1+1e-8],key=lambda a:(abs(a-1),a))
                assert 1. in feasible
                scores=[]
                for alpha in feasible:
                    pred,ref,valid=ev.project_measurements(field*alpha,batch,'input')[0]
                    scores.append((alpha,ev.relative_projection_metrics(pred,ref,valid)))
                selected=min(scores,key=lambda x:x[1]['relative_l1'])[0]
                identity=next(v for a,v in scores if a==1.)
                chosen=next(v for a,v in scores if a==selected)
                assert chosen['relative_l1']<=identity['relative_l1']+1e-12
                heldout={}
                for key,alpha in [('baseline',1.),('calibrated',selected)]:
                    pred,ref,valid=ev.project_measurements(field*alpha,batch,'hold')[0]
                    heldout[key]=ev.relative_projection_metrics(pred,ref,valid)
                assert abs(heldout['baseline']['relative_l1']-saved['measurement']['heldout']['relative_l1'])<2e-5
                records.append(dict(case_id=case,condition=setting['condition'],draw=setting['draw'],
                    selected_alpha=selected,observed_baseline=identity,observed_calibrated=chosen,
                    heldout=heldout,all_observed_scores=[dict(alpha=a,**v) for a,v in scores],
                    seconds=time.perf_counter()-t0,input_ids=saved['input_ids'],heldout_ids=saved['heldout_ids']))
            print(f"{setting['condition']} draw{setting['draw']} {len(records)}/216 elapsed={time.perf_counter()-start:.1f}s",flush=True)
        audit=guard.summary()
        assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    ev.jsonl(OUT/'per_case.jsonl',records)
    summaries={}
    for condition in ('Val-A-V4','clustered-V4'):
        values={}
        for key in ('baseline','calibrated'):
            values[key]=np.array([np.mean([x['heldout'][key]['relative_l1'] for x in records
                if x['case_id']==c and x['condition']==condition]) for c in p['cases']])
        a,b=values['baseline'],values['calibrated']
        summaries[condition]=dict(baseline_mean=float(a.mean()),calibrated_mean=float(b.mean()),
            mean_ratio=float(b.mean()/a.mean()),baseline_p90=float(np.quantile(a,.9)),calibrated_p90=float(np.quantile(b,.9)),
            p90_ratio=float(np.quantile(b,.9)/np.quantile(a,.9)),paired=ev.paired(b,a))
    gate=p['gate'];normal=summaries['Val-A-V4'];hard=summaries['clustered-V4']
    checks=dict(mean=normal['mean_ratio']<=gate['mean_ratio_max'],p90=normal['p90_ratio']<=gate['p90_ratio_max'],
        cluster=hard['mean_ratio']<=gate['cluster5_mean_ratio_max'])
    result=dict(completed=True,settings=summaries,gate_checks=checks,gate_passed=all(checks.values()),
        seconds=time.perf_counter()-start,records=216,changed_fields=sum(x['selected_alpha']!=1 for x in records),
        GT_access=audit,GT_used=False,test_used=False,protocol_sha256=sha(OUT/'protocol.json'))
    ev.write(OUT/'summary.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','run','verify'))
    args=parser.parse_args();torch.set_num_threads(2)
    if args.action=='register':register()
    elif args.action=='run':calibrate()
    else:verify()


if __name__=='__main__':main()
