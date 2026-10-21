"""Seal all new outputs, score authorized GT64, independently check paired stats."""
from stage import *
from predict import methods
import csv
import shutil
import gzip
import time
import evaluation as official

METRICS=('relative_l1','relative_l2','peak_ratio','absolute_peak_ratio_error','peak_absolute_error',
    'peak_location_error_mm','centroid_error_mm','centroid_distance_mm','Dice50')


def csv_write(path,records):
    assert records and not path.exists()
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)),lineterminator='\n');w.writeheader();w.writerows(records)


def predictions():
    mm=verified(BASE/'references.json')+methods();out=[]
    for m in mm:
        root=Path(m['prediction_root']) if 'prediction_root' in m else BASE/'evaluation'/m['key']
        complete=read(root/'complete.json');rr=rows(root/'manifest.jsonl')
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        if 'prediction_root' in m:
            assert complete['checkpoint_sha256']==m['checkpoint_sha256']
            historical_seal=verified(root.parent/'all_predictions_sealed.json')
            assert sha(root/'manifest.jsonl')==historical_seal['manifests'][str(root/'manifest.jsonl')]
        assert sha(root/'manifest.jsonl')==complete['manifest_sha256']
        rr=[r for r in rr if r['condition'] in ('Val-A-V4','clustered-V4')]
        assert len(rr)==210
        for r in rr:
            path=Path(r['prediction']);path=path if path.is_absolute() else root/path
            out.append(dict(case_id=r['case_id'],condition=r['condition'],draw=r['draw'],method=m['key'],seed=m['seed'],
                checkpoint_sha256=m['checkpoint_sha256'],prediction=str(path),prediction_sha256=r.get('prediction_sha256',r.get('sha256')),
                input_ids=r['input_ids'],input_sha256=r.get('input_sha256'),reused='prediction_root' in m,
                seconds=r.get('seconds',r.get('reconstruct_seconds')),peak_allocated_bytes=r['peak_allocated_bytes']))
    frozen_root=BASE/'frozen';complete=verified(frozen_root/'complete.json');assert sha(frozen_root/'manifest.jsonl')==complete['manifest_sha256']
    for r in rows(frozen_root/'manifest.jsonl'):out.append(dict(r,seed=20261019))
    old=BASE.parent/'mrgr_oari_mechanism_v2';plans={p['plan_id']:p for p in verified(old/'plans.json')}
    old_protocol=verified(old/'protocol.json')
    for r in verified(old/'prediction_manifest.json'):
        if r['group']!='obs_token' or r['condition'] not in ('Val-A-V4','clustered-V4'):continue
        p=plans[r['plan_id']];unit=verified(old/'units'/f'p{p["plan_id"]:03d}'/'obs_token.json')
        assert r['checkpoint_sha256']==unit['checkpoint_sha256']==old_protocol['model']['checkpoint_sha256']
        assert unit['case_id']==r['case_id']==p['case_id'] and unit['condition']==r['condition']==p['condition']
        assert unit['group']==r['group'] and unit['input_sha256']==r['input_sha256']
        assert unit['prediction']==r['prediction'] and unit['prediction_sha256']==r['prediction_sha256']
        assert unit['no_grad'] and unit['eval']
        out.append(dict(case_id=r['case_id'],condition=r['condition'],draw=0,method='obs_token_seed20261011',seed=20261011,
            checkpoint_sha256=r['checkpoint_sha256'],prediction=r['prediction'],prediction_sha256=r['prediction_sha256'],
            input_ids=p['input_ids'],input_sha256=r['input_sha256'],reused=True,seconds=None,peak_allocated_bytes=None,
            source_unit=str(old/'units'/f'p{p["plan_id"]:03d}'/'obs_token.json')))
    assert len(out)==1400 and len({(r['case_id'],r['condition'],r['draw'],r['method']) for r in out})==1400
    return mm,out


def allow_guard(manifest,gt=False):
    g=AccessGuard(splits=[] if gt else ['test'],gt_splits=['test'] if gt else [],output=BASE)
    for r in manifest:g.allowed[str(Path(r['prediction']).resolve())]='sealed_prediction'
    return g


def seal():
    if (BASE/'all_predictions_sealed.json').exists():return verified(BASE/'prediction_manifest.json')
    assert verified(BASE/'training_complete.json')['completed']
    verify_hashes(verified(BASE/'protocol.json')['sources'])
    mm,manifest=predictions();cases=verified(BASE/'cases.json');settings=verified(BASE/'layouts.json')
    lookup={(r['case_id'],r['condition'],r['draw'],r['method']):r for r in manifest}
    with allow_guard(manifest) as guard:
        for s in settings:
            ds=dataset('test',s);assert [r['case_id'] for r in ds.records]==cases
            for i,case in enumerate(cases):
                sample=ds[i]
                from geophy_set3d.data.collate import projection_pool_collate
                b={k:v for k,v in projection_pool_collate([sample]).items() if not k.startswith('hold_')};ih=input_hash(b)
                rr=[r for r in manifest if (r['case_id'],r['condition'],r['draw'])==(case,s['condition'],s['draw'])]
                assert len(rr)==(8 if s['draw']==0 else 6)
                for r in rr:
                    assert r['input_ids']==s['input_ids'][case]
                    if r['input_sha256'] is not None:assert r['input_sha256']==ih
                    r['input_sha256']=ih
                    assert sha(r['prediction'])==r['prediction_sha256']
                    with np.load(r['prediction'],allow_pickle=False) as z:
                        assert set(z.files)=={'C0','C1'}
                        assert all(z[k].shape==(64,)*3 and z[k].dtype==np.float32 and np.isfinite(z[k]).all() and z[k].min()>=0 and z[k].max()<=1 for k in z.files)
        assert not guard.denied
    freeze(BASE/'model_manifest.json',mm);freeze(BASE/'prediction_manifest.json',manifest)
    sources={str(p):sha(p) for p in HERE.glob('*.py')}
    sources[str(PROJECT/'src/geophy_set3d/models/geolift_plain_set_mrgr.py')]=sha(PROJECT/'src/geophy_set3d/models/geolift_plain_set_mrgr.py')
    for relative in ('configs/mrgr_uniform_plain_set_interaction_v1.json',
                     'docs/reports/mrgr_oari_targeted_v1_design_zh.md',
                     'scripts/oari_mechanism_v2/runtime.py','scripts/oari_mechanism_v2/interventions.py',
                     'scripts/report_historical_reconstruction_v1.py',
                     'src/geophy_set3d/evaluation/research.py'):
        sources[str(PROJECT/relative)]=sha(PROJECT/relative)
    import platform
    import importlib.metadata
    freeze(BASE/'dependency_environment.json',dict(at=now(),python=platform.python_version(),
        executable=sys.executable,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),packages=sorted(
            f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    freeze(BASE/'evaluation_lock.json',dict(at=now(),protocol_sha256=sha(BASE/'protocol.json'),sources=sources,
        models=mm,GT64_authorization='current user requests this registered stage, scoring only after all new outputs sealed; same previously used35cases',
        dependency_environment_sha256=sha(BASE/'dependency_environment.json'),
        GT128_forbidden=True,stats=verified(BASE/'protocol.json')['stats']))
    freeze(BASE/'all_predictions_sealed.json',dict(at=now(),count=1400,new_predictions=700,reused_predictions=700,
        C0_C1_fields=2800,manifest_sha256=sha(BASE/'prediction_manifest.json'),model_manifest_sha256=sha(BASE/'model_manifest.json'),
        evaluation_lock_sha256=sha(BASE/'evaluation_lock.json'),all_predictions_saved_before_GT=True,GT64_read_in_phase=False,GT128_read=False,audit=guard.summary()))
    return manifest


def flat(volume,morph):
    return dict(relative_l1=volume['relative_l1'],relative_l2=volume['relative_l2'],peak_ratio=morph['peak_ratio'],
        absolute_peak_ratio_error=abs(morph['peak_ratio']-1),peak_absolute_error=volume['peak_absolute_error'],
        peak_location_error_mm=volume['peak_location_error_mm'],centroid_error_mm=volume['centroid_error_mm'],
        centroid_distance_mm=morph['centroid_distance_mm'],Dice50=morph['dice_0.5'])


def score(manifest):
    if (BASE/'metrics.csv').exists():return
    sealed=verified(BASE/'all_predictions_sealed.json');assert sealed['all_predictions_saved_before_GT']
    assert sha(BASE/'prediction_manifest.json')==sealed['manifest_sha256']
    verify_hashes(verified(BASE/'evaluation_lock.json')['sources'])
    cases=verified(BASE/'cases.json');source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')}
    old_gt=read(PROJECT/'docs/reports/artifacts/mrgr_observability_error_pairing_v1/GT64_access.json')['GT64_sha256']
    rr=[];gt_hashes={};first_gt=None;archived={}
    for m in verified(BASE/'references.json'):
        for r in rows(m['source_metrics']):
            if r['method']==m['key'] and r['condition'] in ('Val-A-V4','clustered-V4'):
                archived[r['method'],r['case_id'],r['condition'],r['draw'],r['stage']]=flat(r['volume'],r['morphology'])
    # Prior seed11 frozen scores provide additional exact draw0 verification.
    with (PROJECT/'docs/reports/artifacts/mrgr_observability_error_pairing_v1/case_metrics.csv').open() as f:
        for r in csv.DictReader(f):
            if r['model']=='obs_token':archived['obs_token_seed20261011',r['case_id'],r['condition'],0,r['stage']]={k:float(r[k]) for k in METRICS}
    maxdiff=0.;verified_fields=0
    with allow_guard(manifest,gt=True) as guard:
        for r in manifest:assert sha(r['prediction'])==r['prediction_sha256']
        for case in cases:
            path=DATA/source[case]['gt_volume_64_path'];assert source[case]['split']=='test'
            if first_gt is None:first_gt=now()
            target=np.load(path,allow_pickle=False).astype(np.float64).squeeze();gt_hashes[str(path)]=sha(path)
            assert gt_hashes[str(path)]==old_gt[str(path)]
            selected=[r for r in manifest if r['case_id']==case];ff={}
            for r in selected:
                with np.load(r['prediction'],allow_pickle=False) as z:
                    ff[r['method'],r['condition'],r['draw']]={k:z[k].astype(np.float64) for k in ('C0','C1')}
            for r in selected:
                basekey=f"uniform_fusion_seed{r['seed']}"
                for stage,pred in ff[r['method'],r['condition'],r['draw']].items():
                    metrics=flat(official.volume_metrics(pred,target),official.safe_shape(pred,target))
                    assert all(v is not None and np.isfinite(v) for v in metrics.values())
                    ref=ff[basekey,r['condition'],r['draw']][stage]
                    d=float(np.abs(pred-ref).sum()/(np.abs(ref).sum()+1e-6))
                    previous=archived.get((r['method'],case,r['condition'],r['draw'],stage))
                    if previous:
                        verified_fields+=1
                        for k,v in metrics.items():
                            diff=abs(previous[k]-v);maxdiff=max(maxdiff,diff);assert diff<1e-11,(r,stage,k,diff)
                    rr.append(dict(case_id=case,condition=r['condition'],draw=r['draw'],method=r['method'],seed=r['seed'],stage=stage,
                        D=d,seconds=r.get('seconds'),peak_allocated_bytes=r.get('peak_allocated_bytes'),**metrics))
            print('GT64 scoring',len(gt_hashes),'/35; fields',len(rr),'/2800',flush=True)
        assert not guard.denied
    csv_write(BASE/'metrics.csv',rr)
    freeze(BASE/'GT64_access.json',dict(at=now(),first_GT64_at=first_gt,seal_sha256=sha(BASE/'all_predictions_sealed.json'),
        unique_GT64_files=35,GT64_sha256=gt_hashes,GT128_read=False,audit=guard.summary(),
        archived_fields_verified=verified_fields,maximum_archived_difference=maxdiff))


def statistics():
    if (BASE/'statistics.json').exists():return verified(BASE/'statistics.json')
    records=list(csv.DictReader((BASE/'metrics.csv').open()));cases=verified(BASE/'cases.json')
    lookup={(r['method'],r['case_id'],r['condition'],int(r['draw']),r['stage']):r for r in records};assert len(lookup)==2800
    idx=np.random.default_rng(SEED).integers(0,35,(20000,35));summaries=[];case_rows=[]
    comparisons=[(f'{arm}_seed{s}',s) for arm in ('uniform_plain_set_interaction','uniform_no_ray_relations','obs_token') for s in (20261011,20261019)]
    for model,seed in comparisons:
        basekey=f'uniform_fusion_seed{seed}'
        settings=[('Val-A-V4',[0]),('clustered-V4',[0])]
        if not model.startswith('obs_token'):settings.append(('clustered-V4',list(range(5))))
        for condition,draws in settings:
            for stage in ('C0','C1'):
                values={k:[] for k in (*METRICS,'D')};basevals={k:[] for k in METRICS};treatvals={k:[] for k in METRICS}
                for case in cases:
                    aa=[lookup[basekey,case,condition,d,stage] for d in draws];bb=[lookup[model,case,condition,d,stage] for d in draws]
                    row=dict(case_id=case,model=model,reference=basekey,seed=seed,condition=condition,draws=','.join(map(str,draws)),stage=stage)
                    for metric in METRICS:
                        a=float(np.mean([float(x[metric]) for x in aa]));b=float(np.mean([float(x[metric]) for x in bb]));delta=b-a
                        row[metric]=b;row['baseline_'+metric]=a;row['delta_'+metric]=delta
                        values[metric].append(delta);basevals[metric].append(a);treatvals[metric].append(b)
                    row['D']=float(np.mean([float(x['D']) for x in bb]));values['D'].append(row['D']);case_rows.append(row)
                for metric,v in values.items():
                    x=np.array(v);ci=np.quantile(x[idx].mean(1),[.025,.975])
                    summaries.append(dict(model=model,seed=seed,reference=basekey,condition=condition,draws=draws,stage=stage,metric=metric,n_cases=35,
                        mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std(ddof=1)),ci95=ci.tolist(),
                        minimum=float(x.min()),maximum=float(x.max()),
                        baseline_mean=float(np.mean(basevals[metric])) if metric!='D' else None,
                        treatment_mean=float(np.mean(treatvals[metric])) if metric!='D' else None,
                        increased=int((x>1e-8).sum()),decreased=int((x < -1e-8).sum()),unchanged=int((np.abs(x)<=1e-8).sum())))
    csv_write(BASE/'paired_case_metrics.csv',case_rows)
    csv_write(BASE/'summary.csv',[dict(r,draws=','.join(map(str,r['draws'])),ci95_low=r['ci95'][0],ci95_high=r['ci95'][1]) for r in summaries])
    freeze(BASE/'statistics.json',dict(at=now(),seed=SEED,resamples=20000,n_cases=35,summaries=summaries,
        intervals='unadjusted exploratory pairedcase mean percentile; not trainingseed uncertainty'))
    return verified(BASE/'statistics.json')


def independently_verify():
    if (BASE/'independent_verification.json').exists():return
    cases=verified(BASE/'cases.json');raw=list(csv.DictReader((BASE/'metrics.csv').open()))
    lookup={(r['method'],r['case_id'],r['condition'],int(r['draw']),r['stage']):r for r in raw}
    idx=np.random.default_rng(SEED).integers(0,35,(20000,35));weights=np.array([np.bincount(x,minlength=35) for x in idx],dtype=float)
    maximum=0.;stat=verified(BASE/'statistics.json')
    def percentile(v,q):
        v=sorted(v);position=(len(v)-1)*q;i=int(position);f=position-i;return v[i]*(1-f)+v[min(i+1,len(v)-1)]*f
    for r in stat['summaries']:
        vector=[]
        for case in cases:
            b=[float(lookup[r['model'],case,r['condition'],d,r['stage']][r['metric']]) for d in r['draws']]
            if r['metric']=='D':v=sum(b)/len(b)
            else:
                a=[float(lookup[r['reference'],case,r['condition'],d,r['stage']][r['metric']]) for d in r['draws']]
                v=sum(bb-aa for aa,bb in zip(a,b))/len(a)
            vector.append(v)
        x=np.array(vector);bs=weights@x/35
        for a,b in [(sum(vector)/35,r['mean']),(percentile(vector,.5),r['median']),(percentile(bs,.025),r['ci95'][0]),(percentile(bs,.975),r['ci95'][1])]:
            maximum=max(maximum,abs(a-b));assert abs(a-b)<1e-10,(r,a,b)
        assert sum(v>1e-8 for v in vector)==r['increased'] and sum(v < -1e-8 for v in vector)==r['decreased']
    for r in verified(BASE/'prediction_manifest.json'):assert sha(r['prediction'])==r['prediction_sha256']
    for m in verified(BASE/'model_manifest.json'):assert sha(m['checkpoint'])==m['checkpoint_sha256']
    freeze(BASE/'independent_verification.json',dict(at=now(),passed=True,summary_vectors=len(stat['summaries']),
        bootstrap='independent multinomial count weights with independently sorted interpolated percentiles',
        maximum_difference=maximum,prediction_hashes_verified=1400,GT64_read_in_verification=False,GT128_read=False))


def report_export():
    from final_report import create_report
    create_report()
    ART.mkdir(parents=True,exist_ok=True)
    for p in BASE.rglob('*'):
        if not p.is_file() or p.suffix in ('.pt','.npz','.npy','.lock') or p.name in ('controller_launch.log',):continue
        if 'scheduling' in p.parts and p.name=='resources.jsonl':
            out=ART/p.relative_to(BASE);out=out.with_suffix(out.suffix+'.gz');out.parent.mkdir(parents=True,exist_ok=True)
            with p.open('rb') as f,gzip.open(out,'wb') as g:shutil.copyfileobj(f,g)
            continue
        if p.name=='acceptance.json' or p.name=='acceptance.json.sha256':continue
        out=ART/p.relative_to(BASE);out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():assert sha(out)==sha(p)
        else:shutil.copy2(p,out)
    exported={str(p.relative_to(PROJECT)):sha(p) for p in ART.rglob('*') if p.is_file()}
    exported.update({str(p.relative_to(PROJECT)):sha(p) for p in HERE.glob('*.py')})
    exported[str(REPORT.relative_to(PROJECT))]=sha(REPORT)
    acceptance=dict(at=now(),completed=True,training_runs=3,epochs_each=300,updates_each=24900,new_predictions=700,
        reused_predictions=700,test_cases=35,GT128_read=False,independent_confirmation=False,UniformFusion_all_models=True,
        uninterrupted=False,resource_incidents=1,controlled_resume_epoch=4,foreign_processes_signalled=0,
        engineering_checks_passed=True,independent_verification_passed=True,report=str(REPORT),report_sha256=sha(REPORT),exports=exported)
    freeze(BASE/'acceptance.json',acceptance);shutil.copy2(BASE/'acceptance.json',ART/'acceptance.json');shutil.copy2(BASE/'acceptance.json.sha256',ART/'acceptance.json.sha256')


def main():
    backend();manifest=seal();score(manifest);statistics();independently_verify();report_export()


if __name__=='__main__':main()
