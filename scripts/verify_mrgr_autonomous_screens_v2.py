#!/usr/bin/env python3
"""Parent-level scalar/protocol verification, independent of candidate comparators."""
import argparse
import hashlib
import json
from pathlib import Path
import math

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research/mrgr_autonomous_optimization_v2'


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)
def audit(value):
    for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'):assert value[k]==0


def verify_screen(path,kind):
    p=read(path/'protocol.json')
    assert sha(path/'protocol.json')==(path/'protocol.sha256').read_text().strip()
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['data_files'].items():assert sha(Path(p['config']['dataset']['root'])/name)==digest,name
    comp=read(path/'comparison.json')
    assert comp['completed'] and comp['protocol_sha256']==sha(path/'protocol.json')
    arms=p.get('arms',list(p.get('arm_device',{})))
    control='control' if kind=='parameters' else 'B_one' if kind=='two_step' else 'B_multiscale'
    ordinary,hard,logs,cost={},{},{},{}
    cases=p['validation_case_ids']
    for arm in arms:
        run=path/arm
        config=read(run/'effective_config.json')
        config_hashes=p.get('config_sha256',p.get('effective_config_sha256'))
        assert sha(run/'effective_config.json')==config_hashes[arm]
        logs[arm]=rows(run/'metrics.jsonl')
        assert len(logs[arm])==20
        lr=config['optimizer']['lr']
        for i,row in enumerate(logs[arm]):
            assert row['epoch']==i+1
            close(row['learning_rate'],lr*.5*(1+math.cos(math.pi*i/300)))
            assert row['train']['sample_count']==166 and row['val']['sample_count']==36
            for split in ('train','val'):
                assert not row[split]['gt_used']
                loss=row[split]['loss']
                expected=sum(config['loss'][k]*loss[k] for k in ('hold','cross','observed','consistency'))
                # Components and the batch total were independently rounded in FP32.
                assert math.isclose(expected,loss['total'],rel_tol=2e-6,abs_tol=2e-7),(expected,loss['total'])
        raw=rows(run/'epoch20_per_case.jsonl')
        assert [x['case_id'] for x in raw]==cases
        assert all(x['stations']==p['station_map'][x['case_id']] for x in raw)
        ordinary[arm]=np.array([x['heldout_relative_l1'] for x in raw])
        close(ordinary[arm].mean(),logs[arm][-1]['val']['measurement']['heldout']['relative_l1'])
        cluster=rows(run/'cluster5_per_case.jsonl')
        assert len(cluster)==180 and {(x['case_id'],x['draw']) for x in cluster}=={(c,d) for c in cases for d in range(5)}
        settings={s['draw']:s for s in p['cluster_settings']}
        for x in cluster:
            s=settings[x['draw']]
            assert x['input_ids']==s['input_ids'][x['case_id']] and x['heldout_ids']==s['heldout_ids'][x['case_id']]
            assert not set(x['input_ids'])&set(x['heldout_ids'])
        hard[arm]=np.array([np.mean([x['relative_l1'] for x in cluster if x['case_id']==c]) for c in cases])
        cost[arm]=np.mean([x['seconds'] for x in logs[arm][2:]])
        close(cost[arm],comp['warm_epoch_seconds'][arm])
        for group,values in [('ordinary',ordinary[arm]),('cluster5',hard[arm])]:
            close(values.mean(),comp[group][arm]['mean'])
            close(np.quantile(values,.9),comp[group][arm]['p90'])
        summary=read(run/'summary.json');assert summary['completed'] and summary['epochs']==20 and summary['updates']==1660
        audit(summary['gt_audit'])
        checkpoint=(run/'checkpoints/last.pt') if kind=='parameters' else (path/'checkpoints'/arm/'last.pt')
        assert sha(checkpoint)==summary['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['epoch']==20 and saved['updates']==1660 and not saved['volumetric_gt_used']
        assert saved['config']==config and saved['protocol_sha256']==sha(path/'protocol.json')
        for row,ref in zip(logs[arm],logs.get(control,logs[arm])):
            assert row['loader_rng_sha256']==ref['loader_rng_sha256']
            for split in ('train','val'):assert row[split]['case_station_digest']==ref[split]['case_station_digest']
    differences={}
    for key,value in comp['contrasts'].items():
        if kind=='parameters':candidate,reference=key,control
        else:candidate,reference=key.split('_vs_')
        for label,arrays in [('ordinary',ordinary),('cluster5',hard)]:
            a,b=arrays[reference],arrays[candidate]
            v=value[label]
            close(b.mean()/a.mean(),v['mean_ratio'])
            close(np.quantile(b,.9)/np.quantile(a,.9),v['p90_ratio'])
            delta=b-a
            idx=np.random.default_rng(20261017).integers(0,36,(20000,36))
            np.testing.assert_allclose(np.quantile(delta[idx].mean(1),[.025,.975]),v['paired_ci95'],atol=1e-12,rtol=1e-10)
        checks=dict(mean=ordinary[candidate].mean()/ordinary[reference].mean()<=.99,
            p90=np.quantile(ordinary[candidate],.9)/np.quantile(ordinary[reference],.9)<=1.02,
            cluster=hard[candidate].mean()/hard[reference].mean()<=1.02,
            runtime=cost[candidate]/cost[reference]<=(1.50 if kind=='two_step' else 1.20),
            five_hour=cost[candidate]*300/3600<=5)
        # Primary gate keys differ across independent runners; derive their shared numerical rules here.
        if kind=='parameters':assert all(checks.values())==value['gate_passed']
        elif kind=='scale' and reference==control:assert all(checks.values())==comp['gates'][candidate]['gate_passed']
        elif kind=='geometry' and candidate=='D_geometry':assert all(checks.values())==comp['gates'][reference]['gate_passed']
        elif kind=='two_step':assert all(checks.values())==comp['gates'][reference]['gate_passed']
        differences[key]=dict(mean_ratio=ordinary[candidate].mean()/ordinary[reference].mean(),
            p90_ratio=np.quantile(ordinary[candidate],.9)/np.quantile(ordinary[reference],.9),
            cluster_ratio=hard[candidate].mean()/hard[reference].mean(),numerical_gate=all(checks.values()))
    return dict(path=str(path),kind=kind,passed=True,arms=arms,epochs=len(arms)*20,updates=len(arms)*1660,
        ordinary_mean={a:float(v.mean()) for a,v in ordinary.items()},differences=differences,protocol_sha256=sha(path/'protocol.json'))


def verify_calibration():
    path=ROOT/'outputs/research/mrgr_amplitude_calibration_v2'
    p=read(path/'protocol.json');s=read(path/'summary.json');rr=rows(path/'per_case.jsonl')
    assert sha(path/'protocol.json')==(path/'protocol.sha256').read_text().strip()==s['protocol_sha256']
    for name,digest in p['sources'].items():assert sha(name)==digest,name
    assert len(rr)==216
    for x in rr:
        preferred=sorted(x['all_observed_scores'],key=lambda a:(abs(a['alpha']-1),a['alpha']))
        best=min(preferred,key=lambda a:a['relative_l1'])
        assert x['selected_alpha']==best['alpha']
        assert x['observed_calibrated']['relative_l1']<=x['observed_baseline']['relative_l1']+1e-12
    ratios={}
    for condition in ('Val-A-V4','clustered-V4'):
        arrays={key:np.array([np.mean([x['heldout'][key]['relative_l1'] for x in rr
            if x['case_id']==c and x['condition']==condition]) for c in p['cases']]) for key in ('baseline','calibrated')}
        a,b=arrays['baseline'],arrays['calibrated']
        summary=s['settings'][condition]
        close(a.mean(),summary['baseline_mean']);close(b.mean(),summary['calibrated_mean'])
        close(b.mean()/a.mean(),summary['mean_ratio'])
        close(np.quantile(b,.9)/np.quantile(a,.9),summary['p90_ratio'])
        ratios[condition]=summary['mean_ratio']
    audit(s['GT_access'])
    assert not s['GT_used'] and not s['test_used'] and not s['gate_passed']
    return dict(passed=True,records=216,ratios=ratios,gate_passed=False,protocol_sha256=s['protocol_sha256'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='independent_verification.json')
    args=parser.parse_args();torch.set_num_threads(2)
    records=[]
    for kind,pattern in [('geometry','mrgr_adaptive_geometry_v2_seed*/comparison.json'),
                         ('scale','mrgr_scale_fusion_v2_seed*/comparison.json'),
                         ('parameters','mrgr_parameter_optimization_v2/seed*/comparison.json'),
                         ('two_step','mrgr_two_step_v2_seed*/comparison.json')]:
        for path in sorted((ROOT/'outputs/research').glob(pattern)):records.append(verify_screen(path.parent,kind))
    assert {'geometry','scale','parameters'} <= {r['kind'] for r in records},'All three first-seed screens must finish'
    result=dict(passed=True,records=records,calibration=verify_calibration(),total_epochs=sum(r['epochs'] for r in records),
        total_updates=sum(r['updates'] for r in records),verifier_sha256=sha(Path(__file__)),
        scope='Raw case metrics and pairedCI,all epochs LR/loss,stations,shuffle,source/data/config/checkpoint hashes,GT audits; no new3D labels')
    path=OUT/args.output
    assert not path.exists(), 'Preserve previous verification artifact'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
