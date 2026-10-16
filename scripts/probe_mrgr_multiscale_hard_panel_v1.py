#!/usr/bin/env python3
"""Post-hoc measurement-only clustered-layout panel; fixed short-probe endpoint."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
import probe_mrgr_multiscale_v1 as probe
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.training.utils import move_batch

OUT = probe.OUT/'hard_panel'
SOURCE = probe.ROOT/'outputs/research/ab_remaining_draws_v1/registration.json'


class Replay:
    def __init__(self, setting): self.setting=setting
    def sample(self, station_count, *, epoch, sample_index, case_id):
        return (np.asarray(self.setting['input_ids'][case_id], dtype=np.int64),
                np.asarray(self.setting['heldout_ids'][case_id], dtype=np.int64))


def prepare():
    if (OUT/'registration.json').exists():
        return verify()
    protocol=probe.verify(); old=json.loads(SOURCE.read_text())
    settings=[copy.deepcopy(s) for s in old['settings'] if s['condition']=='clustered-V4']
    assert sorted(s['draw'] for s in settings)==[1,2,3,4]
    record=next(s for s in old['draw0'] if s['condition']=='clustered-V4')
    manifest=Path(record['models']['A']['path']).parent/'predictions_manifest.jsonl'
    rows=[json.loads(l) for l in manifest.read_text().splitlines()]
    settings.insert(0,dict(condition='clustered-V4',views=4,draw=0,
        input_ids={r['case_id']:r['selected_station_ids'] for r in rows},
        heldout_ids={r['case_id']:r['heldout_station_ids'] for r in rows}))
    expected=set(settings[0]['input_ids']); assert len(expected)==36
    for s in settings:
        assert set(s['input_ids'])==set(s['heldout_ids'])==expected
        for case in expected:
            a,h=s['input_ids'][case],s['heldout_ids'][case]
            assert len(a)==len(set(a))==len(h)==len(set(h))==4
            assert not set(a)&set(h) and min(a+h)>=0 and max(a+h)<64
    OUT.mkdir(exist_ok=False)
    reg=dict(registered_at=probe.now(),classification='post-hoc exploratory measurement panel, not the original primary endpoint',
        script_sha256=probe.digest(Path(__file__)),short_probe_protocol_sha256=probe.digest(probe.OUT/'protocol.json'),
        source_registration=str(SOURCE),source_registration_sha256=probe.digest(SOURCE),
        draw0_layout_manifest=str(manifest),draw0_layout_manifest_sha256=probe.digest(manifest),
        settings=settings,device='cuda:2',cases=36,draws=5,models=list(probe.ARMS),
        checkpoint_rule='each arm fixed final epoch20 last.pt; do not use best.pt or any geometry-dependent selection',
        measurements=['single input-field to fixed disjoint H4 relative L1','same input observed relative L1'],
        statistics='average five draw errors within each case; then mean/P90 and paired n36 wins/bootstrap, never 180 independent cases',
        selection_changed=False,gt_for_training=False,gt_for_selection=False,gt_scoring=False,
        budget_seconds=120,config=protocol['config'])
    probe.write_json(OUT/'registration.json',reg)
    (OUT/'registration.sha256').write_text(probe.digest(OUT/'registration.json')+'\n')
    print(json.dumps(dict(stage='hard_panel_registered',cases=36,draws=5,fields=360)),flush=True)
    return reg


def verify():
    reg=json.loads((OUT/'registration.json').read_text())
    assert probe.digest(OUT/'registration.json')==(OUT/'registration.sha256').read_text().strip()
    assert probe.digest(Path(__file__))==reg['script_sha256']
    assert probe.digest(SOURCE)==reg['source_registration_sha256']
    assert probe.digest(reg['draw0_layout_manifest'])==reg['draw0_layout_manifest_sha256']
    probe.verify()
    return reg


def heldout(batch):
    value={k:v for k,v in batch.items() if not k.startswith(('input_','hold_')) and k!='view_mask'}
    value.update({'input_'+k[5:]:v for k,v in batch.items() if k.startswith('hold_') and k!='hold_view_mask'})
    value['view_mask']=batch['hold_view_mask']
    return value


def run():
    reg=verify(); assert not (OUT/'metrics.jsonl').exists()
    for arm in probe.ARMS:
        summary=json.loads((probe.OUT/arm/'summary.json').read_text())
        assert summary['epochs']==20 and summary['completed_short_probe']
        assert probe.digest(probe.OUT/'checkpoints'/arm/'last.pt')==summary['final_checkpoint_sha256']
    config=reg['config']; device=reg['device']; records=[]; checkpoint_hashes={}
    guard=probe.GTReadGuard(config['dataset']['root'],probe.OUT/'checkpoints')
    began=time.perf_counter()
    with guard:
        for arm in probe.ARMS:
            path=probe.OUT/'checkpoints'/arm/'last.pt'; checkpoint_hashes[arm]=probe.digest(path)
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            assert checkpoint['epoch']==20 and checkpoint['updates']==1660
            model=probe.model_for(config,arm,device)
            model.load_state_dict(checkpoint['model']); model.eval(); del checkpoint
            with torch.no_grad():
                for setting in reg['settings']:
                    ds=MeasurementOnlyPoolDataset(config['dataset']['root'],split='val',geometry_pool='train',
                        evaluation_only=True,min_input_views=4,max_input_views=4,hold_views=4,
                        base_seed=20261011,normalization='linear_p99_5',statistics_file='train_statistics.json')
                    ds.sampler=Replay(setting)
                    for first in range(0,len(ds),2):
                        assert time.perf_counter()-began < reg['budget_seconds'], '120 second panel budget exhausted'
                        samples=[ds[i] for i in range(first,min(first+2,len(ds)))]
                        batch=move_batch(projection_pool_collate(samples),device)
                        inputs={k:v for k,v in batch.items() if not k.startswith('hold_')}
                        h=heldout(batch); field=model(inputs)
                        observed=probe.project_input(field,inputs); hpred=probe.project_input(field,h)
                        assert all(bool(torch.isfinite(t).all()) for t in (field,observed,hpred))
                        for i,s in enumerate(samples):
                            case=s['case_id']
                            assert s['input_station_indices'].tolist()==setting['input_ids'][case]
                            assert s['hold_station_indices'].tolist()==setting['heldout_ids'][case]
                            measures={}
                            for name,pred,t in [('heldout',hpred,h),('observed',observed,inputs)]:
                                valid=t['input_valid_ray'].bool()&t['view_mask'][:,:,None,None,None]
                                measures[name]=probe.relative_projection_metrics(pred[i],t['input_projection_raw'][i],valid[i])
                            row=dict(arm=arm,draw=setting['draw'],case_id=case,measurement=measures,
                                input_ids=setting['input_ids'][case],heldout_ids=setting['heldout_ids'][case])
                            records.append(row); probe.append(OUT/'metrics.jsonl',row)
                    print(f'hard panel {arm} draw{setting["draw"]} complete; elapsed={time.perf_counter()-began:.2f}s',flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        audit=guard.summary()
    assert len(records)==360
    case_ids=sorted(reg['settings'][0]['input_ids']); summaries={}; paired={}
    for domain in ('heldout','observed'):
        arrays={}
        for arm in probe.ARMS:
            arrays[arm]=np.array([np.mean([r['measurement'][domain]['relative_l1'] for r in records if r['arm']==arm and r['case_id']==case]) for case in case_ids])
            summaries.setdefault(arm,{})[domain]=dict(mean=float(arrays[arm].mean()),p90=float(np.quantile(arrays[arm],.9)))
        delta=arrays['multiscale']-arrays['reference']; rng=np.random.default_rng(20261016)
        boot=delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(1)
        paired[domain]=dict(mean_delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
            relative_improvement=float(1-arrays['multiscale'].mean()/arrays['reference'].mean()),
            wins=int((delta < -1e-8).sum()),losses=int((delta > 1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()))
    result=dict(completed=True,exploratory=True,selection_changed=False,gt_used=False,
        elapsed_seconds=time.perf_counter()-began,cases=36,draws_per_case=5,
        aggregation='five draws averaged within each case before case statistics',summary=summaries,paired=paired,
        checkpoint_sha256=checkpoint_hashes,gt_access=audit,registration_sha256=probe.digest(OUT/'registration.json'),
        metrics_sha256=probe.digest(OUT/'metrics.jsonl'),limitations='measurement-only, short20 endpoint, no converged/3D/extreme-geometry guarantee')
    assert not any(audit[k] for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    probe.write_json(OUT/'summary.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'))
    a=p.parse_args();torch.set_num_threads(2)
    prepare() if a.action=='prepare' else run()


if __name__=='__main__': main()
