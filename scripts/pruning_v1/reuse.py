"""Audit and reuse only historical development predictions and scalar records."""
from runtime import *
import copy
import numpy as np
import torch
from pruned_models import model_for
import probe_mrgr_multiscale_v1 as probe
from evaluation import validate_rows

def audit():
    if (BASE/'reuse_audit.json').exists():
        assert verified(BASE/'reuse_audit.json')['passed'];return
    reg=verified(BASE/'registration.json');settings=verified(BASE/'layouts.json')['val']
    all_metrics=rows(HISTORY/'evaluation/val/metrics.jsonl');methods=[];evidence={};records=[];combined=[]
    for arm in ('multiscale','uniform_fusion','no_observability'):
        for seed in SEEDS:
            key=f'{arm}_seed{seed}';legacy=arm=='multiscale' and seed==SEEDS[0]
            run=ORIGINAL if legacy else HISTORY/'training'/key
            protocol=run/'protocol.json';p=read(protocol)
            side=run/'protocol.sha256' if legacy else protocol.with_suffix('.json.sha256')
            assert sha(protocol)==side.read_text().strip()
            sources={(str(HISTORICAL/k) if not Path(k).is_absolute() else k):v for k,v in p['sources'].items()}
            verify_hashes(sources)
            summary=read(run/'summary.json');log=rows(run/'metrics.jsonl')
            assert summary['completed'] and summary['epochs']==300 and summary['updates']==24900
            assert len(log)==300 and [r['epoch'] for r in log]==list(range(1,301))
            assert all(summary['gt_audit'][k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
            best=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
            assert best['epoch']==summary['best_epoch'] and best['val']['measurement']['heldout']['relative_l1']==summary['best_val_H']
            assert p.get('train_case_ids',p.get('train_cases'))==reg['cases']['train']
            assert p.get('validation_case_ids',p.get('validation_cases'))==reg['cases']['val']
            ck=run/'checkpoints/best.pt';last=run/'checkpoints/last.pt'
            assert sha(ck)==summary['best_checkpoint_sha256'] and sha(last)==summary['last_checkpoint_sha256']
            with DevGuard(checkpoints=[ck,last]) as guard:
                saved=torch.load(ck,map_location='cpu',weights_only=False)
                assert saved['config']==p['config'] and saved['epoch']==best['epoch'] and not saved['volumetric_gt_used']
                assert saved['protocol_sha256']==sha(protocol)
                m=model_for(p['config'],arm,'cpu');m.load_state_dict(saved['model'],strict=True)
                initial=probe.state_hash(model_for(p['config'],arm,'cpu').state_dict())
                expected=p['initial_model_sha256'] if legacy else read(run/'environment.json')['initial_model_sha256']
                assert initial==expected,(key,initial,expected)
            pred=OLD_EVAL if legacy else HISTORY/'evaluation/val'/key
            manifest=pred/('prediction_manifest.jsonl' if legacy else 'manifest.jsonl')
            rr=rows(manifest);validate_rows(reg['cases']['val'],settings,rr)
            hashes={str(pred/r['prediction']):r['sha256'] for r in rr}
            for path,digest in hashes.items():assert sha(path)==digest,path
            # Explicit additional allowlist contains registered dev predictions only.
            with DevGuard() as g:
                g.allowed.update({str(Path(path).resolve()):'registered_development_prediction' for path in hashes})
                for path in hashes:
                    with np.load(path,allow_pickle=False) as z:
                        assert set(z.files)=={'C0','C1'}
                        for stage in z.files:
                            a=z[stage];assert a.shape==(64,)*3 and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
            if legacy:
                metric_path=OLD_EVAL/'per_case_metrics.jsonl';mm={(r['condition'],r['draw'],r['case_id']):r for r in rr};metrics=[]
                for r in rows(metric_path):
                    y=copy.deepcopy(r);meta=mm[r['condition'],r['draw'],r['case_id']]
                    y.update(method=key,measurement=meta['measurement'] if y['stage']=='C1' else {},reconstruct_seconds=meta['reconstruct_s'],peak_allocated_bytes=None)
                    y['morphology']['absolute_peak_ratio_error']=abs(y['morphology']['peak_ratio']-1);metrics.append(y)
            else:
                metric_path=HISTORY/'evaluation/val/metrics.jsonl';metrics=[r for r in all_metrics if r['method']==key]
            assert len(metrics)==1224
            assert {(r['condition'],r['draw'],r['case_id'],r['stage']) for r in metrics}=={(r['condition'],r['draw'],r['case_id'],s) for r in rr for s in ('C0','C1')}
            combined.extend(metrics)
            method=dict(key=key,kind='neural',run=str(run),checkpoint=str(ck),checkpoint_sha256=sha(ck),best_epoch=summary['best_epoch'],arm=arm,
                config=p['config'],parameters=p['parameters'],seed=seed,prediction_root=str(pred),manifest=str(manifest),reused=True)
            methods.append(method)
            for path in (protocol,run/'summary.json',run/'metrics.jsonl',ck,last,manifest,metric_path):evidence[str(path)]=sha(path)
            records.append(dict(key=key,valid=True,epochs=300,updates=24900,best_epoch=best['epoch'],best_val_H=summary['best_val_H'],
                initial_model_sha256=initial,predictions=len(rr),prediction_hashes_checked=len(hashes),source_files_checked=len(sources),
                checkpoint_sha256=sha(ck),audit=g.summary(),training_seconds=summary['elapsed_seconds'],peak_training_bytes=max(r['peak_allocated_bytes'] for r in log)))
            print('Reused and verified',key,flush=True)
    freeze(BASE/'reuse_methods.json',methods);jsonl(BASE/'reuse_metrics.jsonl',combined)
    evidence[str(BASE/'reuse_metrics.jsonl')]=sha(BASE/'reuse_metrics.jsonl')
    freeze(BASE/'reuse_audit.json',dict(at=now(),passed=True,runs=records,evidence=evidence,test_GT_read=False,GT128_read=False))

if __name__=='__main__':audit()
