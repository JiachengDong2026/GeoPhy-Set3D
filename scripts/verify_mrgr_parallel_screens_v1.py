#!/usr/bin/env python3
"""Independent scalar-only reconstruction of parallel screen means and gates."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT/'outputs/research'
OUT=RESEARCH/'mrgr_parallel_optimization_20261017'


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def close(a,b):assert np.isclose(a,b,rtol=1e-12,atol=1e-14),(a,b)
def describe(a):return dict(mean=float(np.mean(a)),p90=float(np.quantile(a,.9)))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_audit(audit):
    assert audit['gt_file_opened'] is False
    for key in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'):
        assert audit[key]==0,(key,audit[key])


def verify_averaging():
    out=RESEARCH/'mrgr_checkpoint_average_v1'
    rr=rows(out/'per_case_metrics.jsonl');summary=read(out/'summary.json')
    p=read(out/'protocol.json')
    assert digest(ROOT/'scripts/probe_mrgr_checkpoint_average_v1.py')==p['script_sha256']
    for path,expected in p['source_sha256'].items():assert digest(path)==expected,path
    for path,expected in p['frozen_formal_sources'].items():assert digest(ROOT/path)==expected,path
    assert digest(out/'protocol.json')==summary['protocol_sha256']
    assert digest(out/'per_case_metrics.jsonl')==summary['raw_metrics_sha256']
    check_audit(summary['gt_audit'])
    assert summary['gt_evaluation_performed'] is False and summary['final35_test_evaluated'] is False
    assert len(rr)==432
    arrays={}
    for arm in ('best278','average250_275_300'):
        ordinary=[r for r in rr if r['arm']==arm and r['panel']=='ordinary']
        assert len(ordinary)==36
        assert {r['case_id'] for r in ordinary}==set(p['validation_case_ids'])
        for r in ordinary:close(r['relative_l1'],.5*(r['A']['relative_l1']+r['B']['relative_l1']))
        clusters=[r for r in rr if r['arm']==arm and r['panel']!='ordinary']
        assert len(clusters)==180 and len({(r['draw'],r['case_id']) for r in clusters})==180
        cases=[x['case_id'] for x in ordinary]
        arrays[arm]=dict(ordinary=np.array([x['relative_l1'] for x in ordinary]),
            clustered=np.array([np.mean([r['relative_l1'] for r in clusters if r['case_id']==c]) for c in cases]))
    for panel in ('ordinary','clustered'):
        s=summary['comparisons'][panel]
        for arm,label in (('best278','reference'),('average250_275_300','candidate')):
            a=describe(arrays[arm][panel]);close(a['mean'],s[label+'_mean']);close(a['p90'],s[label+'_p90'])
    s=summary['comparisons'];checks=dict(ordinary_mean=s['ordinary']['candidate_mean']/s['ordinary']['reference_mean']<=.99,
        ordinary_p90=s['ordinary']['candidate_p90']/s['ordinary']['reference_p90']<=1.02,
        cluster5_mean=s['clustered']['candidate_mean']/s['clustered']['reference_mean']<=1.02)
    assert checks==summary['checks'];assert all(checks.values())==summary['promoted']
    return dict(passed_verification=True,gate_passed=summary['promoted'],ordinary=s['ordinary'],clustered=s['clustered'])


def verify_training(name,arms,casefile,summaryfile):
    out=RESEARCH/name;p=read(out/'protocol.json')
    assert hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()==(out/'protocol.sha256').read_text().strip()
    for source_name,d in p['sources'].items():assert digest(ROOT/source_name)==d,source_name
    assert p['gt_allowed'] is False and p['final_test_allowed'] is False
    assert p['updates_per_epoch']==83 and p['updates_per_arm']==1660
    check_audit(p['registration_audit']);check_audit(read(out/'smoke.json')['gt_audit'])
    s=read(out/summaryfile);metrics={a:rows(out/a/'metrics.jsonl') for a in arms}
    ordinary={a:rows(out/a/casefile) for a in arms}
    ids=p['validation_case_ids'];assert len(ids)==36
    arrays={}
    for arm in arms:
        check_audit(read(out/arm/'gt_access_audit.json'))
        assert len(metrics[arm])==20 and [x['epoch'] for x in metrics[arm]]==list(range(1,21))
        for record in metrics[arm]:
            for split,count in (('train',166),('val',36)):
                assert record[split]['sample_count']==count and record[split]['gt_used'] is False
        assert [x['case_id'] for x in ordinary[arm]]==ids
        arrays[arm]=np.array([x['heldout_relative_l1'] for x in ordinary[arm]])
        close(arrays[arm].mean(),metrics[arm][-1]['val']['measurement']['heldout']['relative_l1'])
    hard=rows(out/'hard_panel_per_case.jsonl');assert len(hard)==360
    check_audit(read(out/'hard_panel.json')['gt_audit'])
    hardarrays={}
    for arm in arms:
        selected=[x for x in hard if x['arm']==arm]
        assert len({(x['case_id'],x['draw']) for x in selected})==180
        hardarrays[arm]=np.array([np.mean([x['relative_l1'] for x in selected if x['case_id']==case]) for case in ids])
    for x,y in zip(metrics[arms[0]],metrics[arms[1]]):
        assert x['loader_rng_sha256']==y['loader_rng_sha256'] and x['learning_rate']==y['learning_rate']
        assert x['val']['case_station_digest']==y['val']['case_station_digest']
        if name.endswith('projection_detail_v1') or x['epoch']<=5:assert x['train']['case_station_digest']==y['train']['case_station_digest']
    a,b=arms
    ratios=dict(ordinary=arrays[b].mean()/arrays[a].mean(),p90=np.quantile(arrays[b],.9)/np.quantile(arrays[a],.9),cluster=hardarrays[b].mean()/hardarrays[a].mean())
    checks=dict(ordinary_H=ratios['ordinary']<=.99,ordinary_P90=ratios['p90']<=1.02,clustered_H=ratios['cluster']<=1.02)
    times={arm:float(np.mean([x['seconds'] for x in metrics[arm][2:]])) for arm in arms}
    checks['runtime']=times[b]/times[a]<=1.2
    checks['five_hour_budget' if name.endswith('projection_detail_v1') else 'five_hour']=times[b]*300/3600<=5
    checks['smoke']=read(out/'smoke.json')['passed']
    if not name.endswith('projection_detail_v1'):checks['output_sanity']=read(out/'output_sanity.json')['passed']
    for k,v in checks.items():assert v==s['checks'][k],(name,k)
    assert set(checks)==set(s['checks']) and bool(all(checks.values()))==s['passed']
    if name.endswith('projection_detail_v1'):
        for arm in arms:
            close(arrays[arm].mean(),s['ordinary_H'][arm]);close(np.quantile(arrays[arm],.9),s['ordinary_P90'][arm]);close(hardarrays[arm].mean(),s['clustered_H'][arm])
    else:
        amendment=read(out/'serialization_amendment.json')
        assert amendment['before_any_optimizer_updates'] and amendment['numerical_script_unchanged']
        assert digest(ROOT/'scripts/run_mrgr_geometry_curriculum_v1.py')==amendment['runner_sha256']
        assert digest(out/'protocol.json')==amendment['protocol_sha256']
        assert digest(out/'station_plan.jsonl')==p['sampler_plan_sha256']
        for arm in arms:
            close(arrays[arm].mean(),s['ordinary'][arm]['mean']);close(np.quantile(arrays[arm],.9),s['ordinary'][arm]['p90']);close(hardarrays[arm].mean(),s['clustered_five_draw'][arm])
        plan=rows(out/'station_plan.jsonl');assert len(plan)==20*166
        for x in plan:
            old,new=x['original'],x['candidate']
            assert {k:len(v) for k,v in old.items()}=={k:len(v) for k,v in new.items()}
            assert len(set(sum(new.values(),[])))==sum(map(len,new.values()))
            if not x['replaced']:assert old==new
            else:assert x['epoch']>=6 and 20-1e-8<=x['candidate_A_span_deg']<=40+1e-8
        assert sum(x['replaced'] for x in plan)==126
        assert len({(x['epoch'],x['case_id']) for x in plan})==3320
        quotas=[sum(x['replaced'] for x in plan if x['epoch']==e) for e in range(1,21)]
        assert quotas==[0]*5+[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16]
        prefix=[dict(epoch=x['epoch'],ordinary_H_delta=float(y['val']['measurement']['heldout']['relative_l1']-x['val']['measurement']['heldout']['relative_l1']),
            train_loss_delta=float(y['train']['loss']['total']-x['train']['loss']['total']))
            for x,y in zip(metrics[a][:5],metrics[b][:5])]
    return dict(passed_verification=True,gate_passed=s['passed'],ordinary={arm:describe(v) for arm,v in arrays.items()},
        clustered={arm:describe(v) for arm,v in hardarrays.items()},ratios={k:float(v) for k,v in ratios.items()},checks=s['checks'],
        warm_epoch_seconds=times,projected_candidate_300_hours=times[b]*300/3600,
        preintervention_prefix=prefix if not name.endswith('projection_detail_v1') else None)


def main():
    result=dict(checkpoint_average=verify_averaging(),
        projection_detail=verify_training('mrgr_projection_detail_v1',('multiscale','detail'),'epoch20_per_case.jsonl','gate.json'),
        geometry_curriculum=verify_training('mrgr_geometry_curriculum_v1',('control','curriculum'),'final_per_case.jsonl','comparison.json'),
        metric='heldout projection relative L1; no GT arrays opened',case_unit=36,final_test_used=False,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Two contemporaneous controls provide a diagnostic of execution drift,
    # not a seed-variance estimate or an alteration to the registered gates.
    ca=rows(RESEARCH/'mrgr_projection_detail_v1/multiscale/metrics.jsonl')
    cb=rows(RESEARCH/'mrgr_geometry_curriculum_v1/control/metrics.jsonl')
    for a,b in zip(ca,cb):
        assert a['loader_rng_sha256']==b['loader_rng_sha256'] and a['learning_rate']==b['learning_rate']
        for split in ('train','val'):assert a[split]['case_station_digest']==b[split]['case_station_digest']
    va=ca[-1]['val']['measurement']['heldout']['relative_l1']
    vb=cb[-1]['val']['measurement']['heldout']['relative_l1']
    result['repeat_control_diagnostic']=dict(projection_route=va,curriculum_route=vb,
        relative_difference=float(va/vb-1),absolute_difference=float(va-vb),
        shuffle_and_station_digests_equal=True,
        limitation='Two processes/devices/training entry points, same seed; descriptive execution drift, not isolated GPU causality or seed variance')
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
