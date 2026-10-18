#!/usr/bin/env python3
"""Independently verify complete formal logs and frozen 3D formulas after scoring."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import evaluate_mrgr_optimization_frozen_v2 as ev


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    torch.set_num_threads(2)
    assert not (out/'independent_verification.json').exists()
    registration=ev.verify(out);summary=ev.read(out/'summary.json')
    seal=ev.read(out/'batch_predictions_sealed.json')
    assert seal['all_models_complete'] and summary['boundary_sha256']==ev.sha(out/'batch_predictions_sealed.json')
    source={x['case_id']:x for x in ev.rows(ev.DATA/'source_index.jsonl') if x['split']=='val'}
    for path,digest in summary['GT_access']['GT64_sha256'].items():assert ev.sha(path)==digest,path
    # This verifier runs only after the completed label phase; it does not select models.
    gt={c:np.load(ev.DATA/source[c]['gt_volume_64_path'],allow_pickle=False).astype(np.float64).squeeze()
        for c in registration['cases']}
    records={};training=[];field_count=0;max_error=0.;morph_count=0
    chosen=ev.read(ev.previous.AB_RUNS['A']/'evaluation/frozen_model_test/slice_selection.json')
    selected={x['case_id'] for x in chosen}
    for model in registration['models']:
        run=Path(model['run']);p=ev.formal.verify(run);s=ev.read(run/'summary.json')
        rr=ev.rows(run/'metrics.jsonl')
        assert len(rr)==300 and s['completed'] and s['updates']==24900
        config=p['config'];lr=config['optimizer']['lr']
        for i,x in enumerate(rr):
            assert x['epoch']==i+1
            close(x['learning_rate'],lr*.5*(1+math.cos(math.pi*i/300)))
            for split in ('train','val'):
                y=x[split];assert y['sample_count']==(166 if split=='train' else 36) and not y['gt_used']
                loss=y['loss'];expected=sum(config['loss'][k]*loss[k] for k in ('hold','cross','observed','consistency'))
                assert math.isclose(expected,loss['total'],rel_tol=2e-6,abs_tol=2e-7)
                close(y['standard_total'],sum(ev.base.selfsup.FROZEN_LOSS[k]*loss[k]
                    for k in ('hold','cross','observed','consistency')))
        best=min(rr,key=lambda x:x['val']['measurement']['heldout']['relative_l1'])
        assert best['epoch']==s['best_epoch']==model['best_epoch']
        close(best['val']['measurement']['heldout']['relative_l1'],s['best_val_H'])
        assert ev.sha(run/'checkpoints/best.pt')==s['best_checkpoint_sha256']
        assert ev.sha(run/'checkpoints/last.pt')==s['last_checkpoint_sha256']
        last=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        best_state=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        assert last['epoch']==300 and last['updates']==24900 and last['metrics']==rr
        assert ev.base.state_hash(best_state['model'])==ev.base.state_hash(last['best_model'])
        assert not last['volumetric_gt_used'] and not best_state['volumetric_gt_used']
        assert not any(s['gt_audit'][k] for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        training.append(dict(key=model['key'],epochs=300,updates=24900,best_epoch=s['best_epoch'],passed=True))
        target=out/model['key'];boundary=ev.read(target/'predictions_complete.json')
        assert ev.sha(target/'predictions_complete.json')==seal['models'][model['key']]['boundary_sha256']
        assert ev.sha(target/'prediction_manifest.jsonl')==boundary['manifest_sha256']
        manifests=ev.rows(target/'prediction_manifest.jsonl');ev.verify_prediction_rows(registration,manifests)
        scored=ev.rows(target/'per_case_metrics.jsonl')
        ev.verify_metric_rows(registration,scored)
        index={(x['condition'],x['draw'],x['case_id'],x['stage']):x for x in scored}
        assert len(index)==len(scored)==1224
        for x in manifests:
            assert ev.sha(target/x['prediction'])==x['sha256']
            g=gt[x['case_id']]
            with np.load(target/x['prediction'],allow_pickle=False) as fields:
                for stage in ('C0','C1'):
                    pred=fields[stage].astype(np.float64);error=pred-g
                    assert pred.shape==g.shape==(64,64,64) and np.isfinite(pred).all() and 0<=pred.min()<=pred.max()<=1
                    values=dict(relative_l1=float(np.sum(np.abs(error))/max(np.sum(np.abs(g)),1e-6)),
                        relative_l2=float(np.linalg.norm(error.ravel())/max(np.linalg.norm(g.ravel()),1e-6)))
                    saved=index[x['condition'],x['draw'],x['case_id'],stage]
                    for k,v in values.items():
                        close(v,saved['volume'][k]);max_error=max(max_error,abs(v-saved['volume'][k]))
                    field_count+=1
                    if x['draw']==0 and x['condition'] in ('Val-A-V4','clustered-V4') and x['case_id'] in selected:
                        coords=np.linspace(0,.6,64)
                        def centroid(v):
                            return np.array([np.dot(v.sum(axis=tuple(j for j in range(3) if j!=i)),coords)/v.sum() for i in range(3)])
                        close(np.linalg.norm(centroid(pred)-centroid(g))*1000,saved['morphology']['centroid_distance_mm'])
                        a=pred>=.5*g.max();b=g>=.5*g.max()
                        close(2*np.logical_and(a,b).sum()/(a.sum()+b.sum()),saved['morphology']['dice_0.5'])
                        morph_count+=1
        records[model['key']]=index
        print(f"Verified {model['key']}:300 epochs,1224 fields",flush=True)
    old=ev.rows(ev.REFERENCE/'per_case_metrics.jsonl')
    ev.verify_metric_rows(registration,old)
    records['frozen_multiscale']={(x['condition'],x['draw'],x['case_id'],x['stage']):x for x in old}
    # The old scorer recorded GT paths, not GT64 hashes. Recompute its saved fields
    # against this same GT cache to verify that the historical comparator is valid.
    new_field_count=field_count;reference_field_count=0
    for x in ev.rows(ev.REFERENCE/'prediction_manifest.jsonl'):
        path=ev.REFERENCE/x['prediction'];assert ev.sha(path)==x['sha256']
        g=gt[x['case_id']]
        with np.load(path,allow_pickle=False) as fields:
            for stage in ('C0','C1'):
                pred=fields[stage].astype(np.float64);error=pred-g
                values=dict(relative_l1=float(np.sum(np.abs(error))/max(np.sum(np.abs(g)),1e-6)),
                    relative_l2=float(np.linalg.norm(error.ravel())/max(np.linalg.norm(g.ravel()),1e-6)))
                saved=records['frozen_multiscale'][x['condition'],x['draw'],x['case_id'],stage]['volume']
                for k,v in values.items():
                    close(v,saved[k]);max_error=max(max_error,abs(v-saved[k]))
                reference_field_count+=1;field_count+=1
    assert reference_field_count==1224
    panels_verified=0
    rng_indices=np.random.default_rng(20261017).integers(0,36,(20000,36))
    for panel in summary['panels']:
        condition,draws,metric=panel['condition'],panel['draws'],panel['metric']
        def values(key):
            return np.array([np.mean([records[key][condition,d,c,'C1']['volume'][metric] for d in draws]) for c in registration['cases']])
        v=values(panel['model'])
        for k,value in [('mean',v.mean()),('median',np.median(v)),('p90',np.quantile(v,.9)),('maximum',v.max())]:close(value,panel[k])
        for reference,pair in panel['comparisons'].items():
            ref=values(reference);delta=v-ref
            close(v.mean()/ref.mean(),1-pair['relative_error_reduction'])
            close(delta.mean(),pair['mean_delta'])
            assert pair['wins']==int((delta < -1e-8).sum())
            assert pair['ties']==int((np.abs(delta)<=1e-8).sum())
            assert pair['losses']==int((delta > 1e-8).sum())
            np.testing.assert_allclose(np.quantile(delta[rng_indices].mean(1),[.025,.975]),pair['ci95'],atol=1e-12,rtol=1e-10)
        panels_verified+=1
    result=dict(passed=True,formal_training=training,fields_independently_recomputed=field_count,
        new_fields_independently_recomputed=new_field_count,reference_fields_independently_recomputed=reference_field_count,
        panels_verified=panels_verified,morphology_fields_independently_checked=morph_count,
        max_L1_L2_abs_difference=max_error,verifier_sha256=ev.sha(Path(__file__)),
        phase='Post-score independent verification; no model selection/training',
        GT64_files=36,GT128_loaded=False,final_test_loaded=False)
    ev.write(out/'independent_verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
