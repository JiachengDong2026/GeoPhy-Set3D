"""Frozen formal U-Net: registered35test x17layouts, prediction seal before GT64."""
from sparse_guard_v2_context import *
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.evaluation.research import volume_metrics
import argparse,csv


def as_input(batch,prefix):
    target={('input_'+k[len(prefix)+1:] if k.startswith(prefix+'_') else k):v for k,v in batch.items()
        if k not in ['view_mask','hold_view_mask'] and (not k.startswith(('input_','hold_')) or k.startswith(prefix+'_'))}
    target['view_mask']=batch['view_mask' if prefix=='input' else 'hold_view_mask']
    return target


class Replay:
    def __init__(self,setting):self.setting=setting
    def sample(self,station_count,*,epoch,sample_index,case_id):
        return np.array(self.setting['input_ids'][case_id]),np.array(self.setting['heldout_ids'][case_id])


def dataset(setting):
    ds=MeasurementOnlyPoolDataset.__new__(MeasurementOnlyPoolDataset)
    ProjectionPoolDataset.__init__(ds,DATA,split='test',geometry_pool='train',min_input_views=setting['views'],max_input_views=setting['views'],hold_views=0,
        base_seed=20261011,normalization='linear_p99_5',statistics_file='train_statistics.json')
    ds.measurement_only=True;ds.sampler=Replay(setting)
    return ds


class TestGuard:
    def __init__(self,phase):
        self.phase=phase;self.active=False;self.touched=set();self.denied=[]
        source=[json.loads(s) for s in (DATA/'source_index.jsonl').read_text().splitlines()]
        self.gt={str((DATA/r['gt_volume_64_path']).resolve()) for r in source}
        self.gt128={str(Path(r['source_target_path']).resolve()) for r in source}
        self.allowed={str((DATA/r['train_projection_path']).resolve()) for r in source if r['split']=='test'} if phase=='predict' else {str((DATA/r['gt_volume_64_path']).resolve()) for r in source if r['split']=='test'}
    def audit(self,event,args):
        if not self.active or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();path=str(p);mode,flags=args[1:3]
        writing=any(c in mode for c in 'wa+') if isinstance(mode,str) else bool(flags&(os.O_WRONLY|os.O_RDWR))
        reason=None
        if writing:
            if p.is_relative_to(DATA):reason='dataset_write'
        elif path in self.gt128:reason='GT128_forbidden'
        elif path in self.allowed:self.touched.add(path)
        elif path in self.gt:reason='GT64_forbidden_in_this_phase'
        elif p.suffix in ['.npy','.npz'] and not p.is_relative_to(ROOT) and p!=DATA/'geometry/train_geometry.npz':reason='array_outside_phase'
        elif p.suffix in ['.pt','.pth','.ckpt'] and (self.phase!='predict' or p!=Path(read(ART/'unet_test_lock.json')['checkpoint'])):reason='checkpoint_outside_phase'
        if reason:self.denied.append(dict(path=path,reason=reason));raise PermissionError(str(self.denied[-1]))
    def __enter__(self):sys.addaudithook(self.audit);self.active=True;return self
    def __exit__(self,*args):self.active=False


def registration():
    verify()
    for path,digest in read(ART/'test_registration.json')['sources'].items():assert sha(PROJECT/path)==digest,path
    assert read(ART/'unet_validation_review.json')['passed']
    summary=read(ART/'unet_summary.json');assert summary['formal_complete'] and summary['physical_updates']==24900
    checkpoint=Path(summary['best_checkpoint']);assert sha(checkpoint)==summary['best_checkpoint_sha256']
    layouts=read(ART/'test_layouts.json');cases=read(ART/'test_cases.json');assert len(layouts)==17 and len(cases)==35
    records=[json.loads(s) for s in (ROOT/'unet/metrics.jsonl').read_text().splitlines()]
    physical=[r for r in records if r['stage']=='physical']
    assert len(physical)==300 and [r['epoch'] for r in physical]==list(range(1,301))
    chosen=min(physical,key=lambda r:selection_metric(r['val']))
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['epoch']==chosen['epoch'] and ck['best_val_H']==summary['best_val_H']
    write(ART/'unet_test_lock.json',dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),best_epoch=ck['epoch'],
        selected_by='minimum complete36Val H, no test input or GT used',layouts_sha256=sha(ART/'test_layouts.json'),
        cases_sha256=sha(ART/'test_cases.json'),training_summary_sha256=sha(ART/'unet_summary.json'),
        test_source_sha256=sha(PROJECT/'scripts/classical_feasible_v1/test_sparse_guard_v2.py')))
    print('FROZEN_FOR_TEST epoch',ck['epoch'],'ValH',summary['best_val_H'],flush=True)


def predict():
    setup();verify();lock=read(ART/'unet_test_lock.json')
    assert sha(lock['checkpoint'])==lock['sha256']
    assert sha(ART/'test_layouts.json')==lock['layouts_sha256'] and sha(ART/'test_cases.json')==lock['cases_sha256']
    out=ROOT/'test';out.mkdir(exist_ok=False);ledger=[]
    guard=TestGuard('predict')
    with guard,torch.inference_mode():
        model=build('unet').to('cuda:0');ck=torch.load(lock['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval();del ck
        for setting in read(ART/'test_layouts.json'):
            ds=dataset(setting);assert [r['case_id'] for r in ds.records]==read(ART/'test_cases.json')
            for index,record in enumerate(ds.records):
                sample=ds[index];batch=move_batch(projection_pool_collate([sample]),'cuda:0')
                torch.cuda.synchronize();started=time.perf_counter();field=model(as_input(batch,'input'));torch.cuda.synchronize();seconds=time.perf_counter()-started
                assert field.shape==(1,1,64,64,64) and torch.isfinite(field).all() and field.min()>=0 and field.max()<=1
                metrics={}
                for name,prefix in [('observed','input'),('heldout','hold')]:
                    target=as_input(batch,prefix)
                    mask=target['input_valid_ray'].bool()&target['view_mask'][:,:,None,None,None]
                    metrics[name]=relative_projection_metrics(project_input(field,target),target['input_projection_raw'],mask)
                name=f"{setting['condition']}_d{setting['draw']}_{index:03d}.npz";path=out/name
                np.savez_compressed(path,concentration=field[0,0].cpu().numpy())
                row=dict(case_id=record['case_id'],condition=setting['condition'],draw=setting['draw'],input_ids=setting['input_ids'][record['case_id']],
                    heldout_ids=setting['heldout_ids'][record['case_id']],prediction=str(path),sha256=sha(path),measurement=metrics,seconds=seconds)
                ledger.append(row);append(out/'predictions.jsonl',row)
            print('TEST_PREDICT',setting['condition'],setting['draw'],'completed',len(ledger),'/595',flush=True)
    assert len(ledger)==595 and not guard.denied
    write(ART/'unet_test_prediction_seal.json',dict(all_predictions_saved_before_GT64=True,checkpoint_sha256=lock['sha256'],
        predictions=ledger,manifest_sha256=sha(out/'predictions.jsonl'),test_projection_files=len(guard.touched),denied=guard.denied))


def score():
    setup();seal=read(ART/'unet_test_prediction_seal.json');assert seal['all_predictions_saved_before_GT64']
    assert sha(ROOT/'test/predictions.jsonl')==seal['manifest_sha256']
    for r in seal['predictions']:assert sha(r['prediction'])==r['sha256']
    source={r['case_id']:r for r in [json.loads(s) for s in (DATA/'source_index.jsonl').read_text().splitlines()] if r['split']=='test'}
    gt={};metrics=[];guard=TestGuard('score')
    with guard:
        for r in seal['predictions']:
            case=r['case_id']
            if case not in gt:gt[case]=np.load(DATA/source[case]['gt_volume_64_path'],allow_pickle=False)
            with np.load(r['prediction'],allow_pickle=False) as arrays:p=arrays['concentration']
            g=gt[case].squeeze();volume=volume_metrics(p,g)
            morphology=dict(peak_ratio=float(p.max()/max(g.max(),1e-30)),mass_ratio=float(p.sum()/max(g.sum(),1e-30)))
            for fraction in [.01,.1,.5]:
                pm=p>=fraction*g.max();gm=g>=fraction*g.max();morphology[f'dice_{fraction}']=float(2*(pm&gm).sum()/max(pm.sum()+gm.sum(),1))
            row=dict(method='UNet3D-feasible-preheat-sparseguard-v2',condition=r['condition'],draw=r['draw'],case_id=case,
                volume=volume,morphology=morphology,measurement=r['measurement'],seconds=r['seconds'])
            metrics.append(row);append(ROOT/'test/metrics.jsonl',row)
    assert len(guard.touched)==35 and not guard.denied
    summary=[]
    for condition in sorted({r['condition'] for r in metrics}):
        rows=[r for r in metrics if r['condition']==condition]
        for metric in ['relative_l1','relative_l2','mae','rmse','centroid_error_mm','peak_location_error_mm']:
            per_case=[float(np.mean([r['volume'][metric] for r in rows if r['case_id']==case and r['volume'][metric] is not None])) for case in source]
            values=np.asarray(per_case);rng=np.random.default_rng(2026102701)
            bootstrap=values[rng.integers(0,35,size=(10000,35))].mean(1)
            summary.append(dict(condition=condition,metric=metric,cases=35,mean=float(values.mean()),ci95_low=float(np.quantile(bootstrap,.025)),ci95_high=float(np.quantile(bootstrap,.975))))
    with (ART/'unet_test_summary.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
    write(ART/'unet_test_score.json',dict(GT64_files=35,GT128_reads=0,predictions=595,GT64_after_prediction_seal=True,
        metrics_path=str(ROOT/'test/metrics.jsonl'),metrics_sha256=sha(ROOT/'test/metrics.jsonl'),denied=guard.denied,summary=summary))
    print('TEST_SCORED 595 predictions,35cases,GT64 after seal',flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='freeze' and not (ART/'unet_validation_review.json').exists():
        from review_validation import main as review_validation
        review_validation()
    {'freeze':registration,'predict':predict,'score':score}[action]()
