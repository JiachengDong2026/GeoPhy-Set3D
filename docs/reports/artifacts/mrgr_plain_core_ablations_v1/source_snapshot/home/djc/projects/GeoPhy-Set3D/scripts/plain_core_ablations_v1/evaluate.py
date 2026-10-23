"""Strict global prediction seal, bounded authorized GT64 and original metrics."""
from common_stage import *
from paired_statistics import METRICS,csv_write,statistics,independently_verify
import evaluation as official
import csv
import platform
import importlib.metadata


def register_evaluation():
    if (BASE/'evaluation_lock.json').exists():return verified(BASE/'evaluation_lock.json')
    assert verified(BASE/'training_complete.json')['completed']
    mm=verified(BASE/'references.json')+methods()
    sources=training_sources()
    for name in ('scripts/report_historical_reconstruction_v1.py','src/geophy_set3d/evaluation/research.py','scripts/paper_final_v1/evaluation.py'):
        sources[str(PROJECT/name)]=sha(PROJECT/name)
    freeze(BASE/'evaluation_lock.json',dict(at=now(),models=mm,sources=sources,protocol_sha256=sha(BASE/'protocol.json'),
        cases_sha256=sha(BASE/'cases.json'),layouts_sha256=sha(BASE/'layouts.json'),
        python=platform.python_version(),executable=sys.executable,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        packages=sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
        authorization=verified(BASE/'protocol.json')['authorization'],GT128_forbidden=True))
    return verified(BASE/'evaluation_lock.json')


def predictions():
    mm=register_evaluation()['models'];out=[]
    for m in mm:
        root=Path(m['prediction_root']) if 'prediction_root' in m else BASE/'evaluation'/m['key']
        complete=verified(root/'complete.json');rr=rows(root/'manifest.jsonl')
        assert sha(m['checkpoint'])==m['checkpoint_sha256']==complete['method']['checkpoint_sha256']
        assert sha(root/'manifest.jsonl')==complete['manifest_sha256'] and len(rr)==210
        for r in rr:
            assert r['checkpoint_sha256']==m['checkpoint_sha256']
            out.append(dict(r,seed=m['seed'],reused='prediction_root' in m))
    assert len(out)==1680 and len({(r['case_id'],r['condition'],r['draw'],r['method']) for r in out})==1680
    return mm,out


def allow_guard(manifest,gt=False):
    g=AccessGuard(splits=[] if gt else ['test'],gt_splits=['test'] if gt else [],output=BASE)
    for r in manifest:g.allowed[str(Path(r['prediction']).resolve())]='sealed_prediction'
    if gt:
        cases=set(verified(BASE/'cases.json'))
        permitted={str((DATA/r['gt_volume_64_path']).resolve()) for r in rows(DATA/'source_index.jsonl') if r['case_id'] in cases and r['split']=='test'}
        assert len(permitted)==35
        for path in list(g.allowed):
            if g.allowed[path]=='GT64_test' and path not in permitted:del g.allowed[path]
    return g


def seal():
    if (BASE/'all_predictions_sealed.json').exists():return verified(BASE/'prediction_manifest.json')
    lock=register_evaluation();verify_hashes(lock['sources']);mm,manifest=predictions()
    cases=verified(BASE/'cases.json');settings=verified(BASE/'layouts.json')
    from geophy_set3d.data.collate import projection_pool_collate
    with allow_guard(manifest) as guard:
        for setting in settings:
            ds=dataset('test',setting);assert [r['case_id'] for r in ds.records]==cases
            for i,case in enumerate(cases):
                b={k:v for k,v in projection_pool_collate([ds[i]]).items() if not k.startswith('hold_')};ih=input_hash(b)
                rr=[r for r in manifest if (r['case_id'],r['condition'],r['draw'])==(case,setting['condition'],setting['draw'])]
                assert len(rr)==8
                for r in rr:
                    assert r['input_ids']==setting['input_ids'][case] and r['input_sha256']==ih
                    assert sha(r['prediction'])==r['prediction_sha256']
                    with np.load(r['prediction'],allow_pickle=False) as z:
                        assert set(z.files)=={'C0','C1'}
                        assert all(z[k].shape==(64,)*3 and z[k].dtype==np.float32 and np.isfinite(z[k]).all() and z[k].min()>=0 and z[k].max()<=1 for k in z.files)
        assert not guard.denied
    freeze(BASE/'model_manifest.json',mm);freeze(BASE/'prediction_manifest.json',manifest)
    freeze(BASE/'all_predictions_sealed.json',dict(at=now(),count=1680,new_predictions=1260,reused_predictions=420,C0_C1_fields=3360,
        manifest_sha256=sha(BASE/'prediction_manifest.json'),models_sha256=sha(BASE/'model_manifest.json'),
        evaluation_lock_sha256=sha(BASE/'evaluation_lock.json'),all_predictions_saved_before_GT=True,GT64_read=False,GT128_read=False,audit=guard.summary()))
    return manifest


def flat(volume,morph):
    return dict(relative_l1=volume['relative_l1'],relative_l2=volume['relative_l2'],peak_ratio=morph['peak_ratio'],
        absolute_peak_ratio_error=abs(morph['peak_ratio']-1),peak_absolute_error=volume['peak_absolute_error'],
        peak_location_error_mm=volume['peak_location_error_mm'],centroid_error_mm=volume['centroid_error_mm'],
        centroid_distance_mm=morph['centroid_distance_mm'],Dice50=morph['dice_0.5'])


def score(manifest):
    if (BASE/'GT64_access.json').exists():
        assert (BASE/'metrics.csv').exists();return
    assert not (BASE/'metrics.csv').exists(),'Partial scoring output; audit before continuation'
    sealed=verified(BASE/'all_predictions_sealed.json');assert sealed['all_predictions_saved_before_GT']
    assert sha(BASE/'prediction_manifest.json')==sealed['manifest_sha256'];verify_hashes(verified(BASE/'evaluation_lock.json')['sources'])
    cases=verified(BASE/'cases.json');source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')}
    known_gt=verified(PRIOR/'GT64_access.json')['GT64_sha256']
    archived={(r['method'],r['case_id'],r['condition'],int(r['draw']),r['stage']):r for r in csv.DictReader((PRIOR/'metrics.csv').open()) if r['method'].startswith(PLAIN)}
    rr=[];gt_hashes={};first=None;maximum=0.;fields=0
    with allow_guard(manifest,gt=True) as guard:
        for r in manifest:assert sha(r['prediction'])==r['prediction_sha256']
        for case in cases:
            path=DATA/source[case]['gt_volume_64_path'];assert source[case]['split']=='test'
            if first is None:first=now()
            target=np.load(path,allow_pickle=False).astype(np.float64).squeeze();gt_hashes[str(path)]=sha(path)
            assert gt_hashes[str(path)]==known_gt[str(path)]
            selected=[r for r in manifest if r['case_id']==case];ff={}
            for r in selected:
                with np.load(r['prediction'],allow_pickle=False) as z:ff[r['method'],r['condition'],r['draw']]={k:z[k].astype(np.float64) for k in ('C0','C1')}
            for r in selected:
                basekey=f"{PLAIN}_seed{r['seed']}"
                for stage,pred in ff[r['method'],r['condition'],r['draw']].items():
                    metrics=flat(official.volume_metrics(pred,target),official.safe_shape(pred,target));assert all(v is not None and np.isfinite(v) for v in metrics.values())
                    ref=ff[basekey,r['condition'],r['draw']][stage];d=float(np.abs(pred-ref).sum()/(np.abs(ref).sum()+1e-6))
                    prev=archived.get((r['method'],case,r['condition'],r['draw'],stage))
                    if prev:
                        fields+=1
                        for k,v in metrics.items():
                            diff=abs(float(prev[k])-v);maximum=max(maximum,diff);assert diff<1e-11
                    rr.append(dict(case_id=case,condition=r['condition'],draw=r['draw'],method=r['method'],seed=r['seed'],stage=stage,D=d,
                        seconds=r['seconds'],peak_allocated_bytes=r['peak_allocated_bytes'],**metrics))
            print('GT64 scoring',len(gt_hashes),'/35; fields',len(rr),'/3360',flush=True)
        assert not guard.denied
    csv_write(BASE/'metrics.csv',rr)
    freeze(BASE/'GT64_access.json',dict(at=now(),first_GT64_at=first,seal_sha256=sha(BASE/'all_predictions_sealed.json'),
        unique_GT64_files=35,GT64_sha256=gt_hashes,GT128_read=False,audit=guard.summary(),archived_fields_verified=fields,maximum_archived_difference=maximum))


def main():
    backend();manifest=seal();score(manifest);statistics();independently_verify()
    from finish import main as finish
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','score'],default='score',nargs='?');a=p.parse_args()
    if a.action=='register':register_evaluation()
    else:main()
