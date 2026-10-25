"""Frozen 900-row inference and separate-process saved-output GT64 evaluation."""
import argparse,subprocess,time
from context import *
import pandas as pd
from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import DiagnosticReadGuard
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.evaluation.deep_comparison import pair_reference,case_means,statistics
from diagnose_clustered_one_swap import dataset,target_gt
HIST=PROJECT/'docs/reports/artifacts/deepsets_selfsup_v1'

def require_frozen_formal(arm,check_checkpoint=True):
    root=ROOT/arm/'formal';p=read(root/'training_complete.json');f=read(root/'frozen_best.json')
    assert p['epochs']==f['completed_epochs']==300 and p['optimizer_steps']==f['optimizer_steps']==24900 and p['completed']
    assert not f['pilot_only'] and not f['volumetric_gt_used']
    if check_checkpoint:assert sha(f['checkpoint'])==f['sha256']
    assert read(ART/(arm+'_pilot.json'))['status']=='GO'
    review=read(ART/(arm+'_independent_review.json'));assert review['all_passed'] and review['scientific_GO']
    return root,f

def token(r):return f"case{r['case_index']}_{r['geometry_id']}"

def load(arm,c,freeze):
    ck=torch.load(freeze['checkpoint'],map_location='cuda:0',weights_only=False)
    assert ck['config']==c and not ck['pilot_only'] and not ck['volumetric_gt_used']
    assert ck['source_manifest_sha256']==sha(ART/'source_manifest.json')
    m=build_model(c);m.load_state_dict(ck['model']);return m.eval()

def selector(arm):
    c=load_config(arm);root,f=require_frozen_formal(arm);out=ROOT/arm
    if (out/'selector_replay.json').exists():assert read(out/'selector_replay.json')['passed'];return
    with DiagnosticReadGuard(c['dataset']['root'],[f['checkpoint']],out,'inference') as guard:
        m=load(arm,c,f);result=run_epoch(m,make_loader(make_dataset(c,validation=True),c),c)
        expected=read(root/'training_complete.json')['best_val_heldout_relative_l1']
        assert np.isclose(selection_metric(result),expected,atol=1e-8,rtol=1e-5)
        write(out/'selector_replay.json',dict(passed=True,reference=expected,replayed=selection_metric(result),access=guard.summary()))

def infer(arm,primary=False):
    c=load_config(arm);formal,freeze=require_frozen_formal(arm);out=ROOT/('primary' if primary else arm)
    assert read(ROOT/arm/'selector_replay.json')['passed']
    if primary:freeze=read(ART/'history_audit.json')['current_primary'];freeze=dict(freeze,sha256=freeze['checkpoint_sha256'])
    out.mkdir(exist_ok=True);(out/'case_images').mkdir(exist_ok=True);(out/'prediction_rows').mkdir(exist_ok=True)
    records=read(HIST/'station_subsets.json');assert len(records)==900
    historical=pd.read_csv(HIST/'deepsets_900.csv').set_index(['case_id','group','draw'])
    with DiagnosticReadGuard(c['dataset']['root'],[freeze['checkpoint']],out,'inference') as guard:
        assert sha(freeze['checkpoint'])==freeze['sha256']
        if primary:
            sys.path.insert(0,str(PROJECT/'scripts/deep_baselines_v1'))
            # Legacy scripts use the generic module name context. Isolate that
            # import so their original factory receives its original context.
            import importlib
            own_context=sys.modules.pop('context',None)
            try:
                bc=importlib.import_module('baseline_context')
            finally:
                if own_context is not None:sys.modules['context']=own_context
            m=bc.c.model_for(freeze['config'],bc.c.PLAIN,'cuda:0').eval()
            ck=torch.load(freeze['checkpoint'],map_location='cuda:0',weights_only=False);m.load_state_dict(ck['model']);del ck
        else:m=load(arm,c,freeze)
        ds=dataset(dict(dataset=c['dataset']['root'],original_seed=20261011));before=state_hash(m.state_dict())
        with torch.no_grad():
            for r in records:
                rowpath=out/'prediction_rows'/f'{token(r)}.json'
                if rowpath.exists():
                    saved=read(rowpath);assert saved['record']==r and saved['checkpoint_sha256']==freeze['sha256'] and sha(saved['prediction_file'])==saved['prediction_sha256'];continue
                source,projections=ds._projections(r['case_index']);assert source['case_id']==r['case_id'];empty=np.empty(0,dtype=np.int64)
                b=move_batch(projection_pool_collate([ds._measurement_sample(source,projections,np.asarray(r['station_ids']),empty)]),'cuda:0')
                h=move_batch(projection_pool_collate([ds._measurement_sample(source,projections,np.asarray(r['H_station_ids']),empty)]),'cuda:0')
                ident=dict(input_identity_sha256=tensor_identity(b['input_projection_raw'],b['input_station_indices'],b['input_station_origin_m'],b['input_path_length_m']),
                    H_mask_sha256=tensor_identity(h['view_mask'],h['input_valid_ray']),complete_input_sha256=tensor_identity(*[b[k] for k in sorted(b) if isinstance(b[k],torch.Tensor)]),
                    complete_H_sha256=tensor_identity(*[h[k] for k in sorted(h) if isinstance(h[k],torch.Tensor)]))
                old=historical.loc[(r['case_id'],r['group'],r['draw'])]
                for k,v in ident.items():assert old[k]==v,k
                # Legacy lifting visibility is identity metadata only, never a network feature.
                ident['input_mask_sha256']=str(old['input_mask_sha256'])
                path=out/'case_images'/f'{token(r)}.npz';assert not path.exists(),'Orphan output needs explicit identity recovery; no duplicate forward'
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter();pred=check_prediction(m(b),b);torch.cuda.synchronize();elapsed=time.perf_counter()-began
                metrics={}
                for name,t in [('observed',b),('heldout',h)]:
                    v=t['input_valid_ray'].bool()&t['view_mask'][:,:,None,None,None]
                    metrics.update({name+'_'+k:x for k,x in relative_projection_metrics(project_input(pred,t),t['input_projection_raw'],v).items()})
                np.savez(path,prediction=pred[0,0].cpu().numpy())
                write(rowpath,dict(record=r,**ident,**metrics,prediction_file=str(path),prediction_sha256=sha(path),checkpoint_sha256=freeze['sha256'],
                    forward_seconds=elapsed,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        assert before==state_hash(m.state_dict())
        write(out/'inference_access.json',guard.summary())
    manifest={token(r):sha(out/'prediction_rows'/f'{token(r)}.json') for r in records}
    write(out/'predictions_complete.json',dict(rows=900,checkpoint_sha256=freeze['sha256'],all_outputs_saved_before_GT=True,rows_sha256=manifest))

def gt(arm):
    c=load_config(arm);_,freeze=require_frozen_formal(arm,check_checkpoint=False)
    assert read(ROOT/arm/'selector_replay.json')['passed']
    records=read(HIST/'station_subsets.json');tables={}
    # Both full manifests must exist and be verified before either GT64 read.
    for name in [arm,'primary']:
        out=ROOT/name;seal=read(out/'predictions_complete.json');assert seal['rows']==900 and seal['all_outputs_saved_before_GT']
        for r in records:
            p=out/'prediction_rows'/f'{token(r)}.json';assert sha(p)==seal['rows_sha256'][token(r)]
            row=read(p);assert row['record']==r and sha(row['prediction_file'])==row['prediction_sha256']
    for name in [arm,'primary']:
        out=ROOT/name
        if (out/'gt_complete.json').exists():tables[name]=pd.read_csv(out/'volume_900.csv');continue
        with DiagnosticReadGuard(c['dataset']['root'],[],out,'gt') as guard:
            ds=dataset(dict(dataset=c['dataset']['root'],original_seed=20261011));rows=[]
            write(out/'first_gt_boundary.json',dict(formal_checkpoint_sha256=freeze['sha256'],both_manifests_sealed=True,epochs=300,updates=24900))
            for i in range(36):
                target=target_gt(ds,i)
                for r in [r for r in records if r['case_index']==i]:
                    saved=read(out/'prediction_rows'/f'{token(r)}.json')
                    with np.load(saved['prediction_file'],allow_pickle=False) as z:pred=torch.from_numpy(z['prediction'].copy())[None,None]
                    vm=VolumeMetrics();vm.update(pred,target,[r['case_id']])
                    rows.append({**r,**{k:v for k,v in saved.items() if k not in ['record','prediction_file']},**{k:v for k,v in vm.rows[0].items() if k!='case_id'}})
            tables[name]=pd.DataFrame(rows);tables[name].to_csv(out/'volume_900.csv',index=False)
            write(out/'gt_complete.json',dict(rows=900,table_sha256=sha(out/'volume_900.csv'),access=guard.summary()))
    # Reuse original pairing/bootstrap unchanged; rename the legacy baseline label in exported tables.
    current=tables['primary'].copy()
    from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import METRICS
    current.rename(columns={m:'C1_'+m for m in METRICS},inplace=True)
    for name,table in [(arm,tables[arm]),('geolift_deepsets_sum',pd.read_csv(HIST/'deepsets_900.csv'))]:
        paired=pair_reference(table,current,records);cases=case_means(paired);stats,tails,contrasts=statistics(cases,c)
        dest=ROOT/arm/'comparisons'/name;dest.mkdir(parents=True,exist_ok=False)
        for tabledata,filename in [(paired,'paired_900.csv'),(cases,'case_group_180.csv'),(stats,'paired_statistics.csv'),(tails,'tails.csv')]:
            tabledata.rename(columns=lambda k:k.replace('DeepSets_','Baseline_').replace('deepsets_','baseline_')).to_csv(dest/filename,index=False)
        write(dest/'contrasts.json',contrasts);write(dest/'labels.json',dict(Ours='frozen current Plain',Baseline=name,legacy_statistics_function_reused=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['evaluate','selector','infer','infer_primary','gt']);ap.add_argument('arm',choices=['deeponet','fno']);a=ap.parse_args()
    verify();setup()
    if a.phase=='evaluate':
        selector(a.arm);infer(a.arm);infer(a.arm,True)
        subprocess.run([sys.executable,__file__,'gt',a.arm],check=True)
    elif a.phase=='selector':selector(a.arm)
    elif a.phase=='infer':infer(a.arm)
    elif a.phase=='infer_primary':infer(a.arm,True)
    else:gt(a.arm)
if __name__=='__main__':main()
