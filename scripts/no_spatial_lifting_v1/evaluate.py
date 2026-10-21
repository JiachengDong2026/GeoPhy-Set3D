"""Locked all-layout prediction, global sealing, then authorized GT64 scoring."""
from runtime import *
import argparse
import csv
import math
import time
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
import evaluation as canonical

METRICS=('relative_l1','relative_l2','dice50','absolute_peak_ratio_error','centroid_distance_mm')


def csv_write(path,records):
    assert not path.exists()
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)),lineterminator='\n');w.writeheader();w.writerows(records)


def sources():
    paths=[HERE/n for n in ('evaluate.py','analysis_statistics.py','verify_statistics.py','report.py')]
    paths += [PROJECT/'scripts/lifting_mapping_v1/analysis.py',PROJECT/'scripts/paper_final_v1/evaluation.py',PROJECT/'scripts/report_historical_reconstruction_v1.py']
    return {str(p):sha(p) for p in paths}


def validate_source():
    verify_hashes(verified(BASE/'protocol.json')['sources'])
    verify_hashes(verified(BASE/'evaluation_lock.json')['sources'])


def register():
    backend();prelock=verified(BASE/'analysis_source_prelock.json');verify_hashes(prelock['sources']);assert sources()==prelock['sources']
    tc=verified(BASE/'training_complete.json');verify_hashes(tc['evidence'])
    assert verified(BASE/'engineering_checks.json')['passed']
    methods=[]
    for seed in SEEDS:
        target=run_protocol(seed);p=verified(target/'protocol.json');summary=read(target/'summary.json');rr=rows(target/'metrics.jsonl')
        assert summary['completed'] and summary['epochs']==300 and summary['updates']==24900 and len(rr)==300
        assert [r['epoch'] for r in rr]==list(range(1,301))
        chosen=min(rr,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        assert chosen['epoch']==summary['best_epoch'] and chosen['val']['measurement']['heldout']['relative_l1']==summary['best_val_H']
        assert summary['parameters']==342725 and p['initial_model_sha256']==read(target/'environment.json')['initial_model_sha256']
        assert all(summary['gt_audit'][k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        assert not read(target/'boundary_audit.json')['denied']
        ck=target/'checkpoints/best.pt';assert sha(ck)==summary['best_checkpoint_sha256']
        with AccessGuard(checkpoints=[ck],output=BASE) as guard:
            checkpoint=torch.load(ck,map_location='cpu',weights_only=False)
            assert checkpoint['epoch']==summary['best_epoch'] and checkpoint['config']==p['config'] and checkpoint['protocol_sha256']==sha(target/'protocol.json')
            assert not checkpoint['volumetric_gt_used']
            model=model_for(p['config'],ARM,'cpu');model.load_state_dict(checkpoint['model'],strict=True)
        methods.append(dict(seed=seed,run=str(target),config=p['config'],checkpoint=str(ck),checkpoint_sha256=sha(ck),best_epoch=summary['best_epoch'],parameters=342725,fusion='Uniform Fusion'))
    # Training and model choice are complete; no GT has been accessed.
    old=verified(OUT/'final_lock.json')
    freeze(BASE/'evaluation_lock.json',dict(at=now(),sources=sources(),methods=methods,
        training_complete_sha256=sha(BASE/'training_complete.json'),data_identity=old['data_identity'],GT64_stat_only=old['gt64_stat_only'],
        metric_definitions='canonical volume_metrics+safe_shape; independently recomputed float64 NumPy; Dice threshold fixed at0.5*GTpeak; centroid endpoint aligned0..0.6m',
        primary='Val-A-V4 C1 relativeL1, ablation-full; per-seed paired35cases plus descriptive2fixedseed case mean',
        secondary='clustered-V4 C1, five draws averaged within case; others exploratory',
        bootstrap=dict(seed=20261019,n=20000,unit='case',confidence=.95,conservative_reference_family=14),GT64_read=False,GT128_read=False))
    for n in ('evaluation_lock.json','evaluation_lock.json.sha256'):(BASE/n).chmod(0o444)


def method(seed):return next(m for m in verified(BASE/'evaluation_lock.json')['methods'] if m['seed']==seed)


def input_hash(batch):
    h=hashlib.sha256()
    for k,v in sorted(batch.items()):
        h.update(k.encode())
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous();h.update(str((t.shape,t.dtype)).encode());h.update(t.numpy().tobytes())
        else:h.update(repr(v).encode())
    return h.hexdigest()


def predict(seed):
    backend();validate_source();lock=verified(BASE/'evaluation_lock.json');verify_hashes(lock['data_identity']);m=method(seed)
    target=BASE/'evaluation'/str(seed);target.mkdir(parents=True,exist_ok=False)
    layouts=verified(BASE/'layouts.json');cases=verified(BASE/'cases.json');records=[]
    assert sha(m['checkpoint'])==m['checkpoint_sha256']
    with AccessGuard(splits=['test'],checkpoints=[m['checkpoint']],output=BASE) as guard:
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
        model=model_for(m['config'],ARM,'cuda:0');model.load_state_dict(ck['model'],strict=True);model.eval();model.requires_grad_(False);del ck
        start_hash=state_hash(model.state_dict());calls=[]
        def count(module,args,out):calls.append(args[0].shape[2])
        handle=model.lifting.register_forward_hook(count)
        with torch.inference_mode():
            for layout in layouts:
                ds=dataset('test',layout);assert [r['case_id'] for r in ds.records]==cases
                for i,case in enumerate(cases):
                    raw=projection_pool_collate([ds[i]]);batch=move_batch(raw,'cuda:0');inputs={k:v for k,v in batch.items() if not k.startswith('hold_')}
                    assert raw['input_station_indices'][0].tolist()==layout['input_ids'][case]
                    before=input_hash(inputs);calls.clear();torch.cuda.synchronize();t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                    st=model(inputs,return_stages=True);torch.cuda.synchronize();elapsed=time.perf_counter()-t
                    assert calls==[32,16] and not model.training and all(p.grad is None and not p.requires_grad for p in model.parameters())
                    assert input_hash(inputs)==before
                    fields={k:st[k][0,0].cpu().numpy() for k in ('C0','C1')}
                    assert all(a.shape==(64,)*3 and a.dtype==np.float32 and np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for a in fields.values())
                    assert torch.equal(st['visibility_base'],st['visibility_residual'])
                    path=target/'predictions'/f'{layout["condition"]}_d{layout["draw"]}_{i:03d}.npz';path.parent.mkdir(exist_ok=True);assert not path.exists()
                    np.savez_compressed(path,**fields)
                    records.append(dict(seed=seed,condition=layout['condition'],draw=layout['draw'],case_id=case,input_ids=layout['input_ids'][case],heldout_ids=layout['heldout_ids'][case],
                        prediction=str(path.relative_to(target)),sha256=sha(path),checkpoint_sha256=m['checkpoint_sha256'],input_sha256=before,
                        forward_calls=1,lifting_channels=calls.copy(),fusion='Uniform Fusion',reconstruct_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')))
                    jsonl(target/'manifest.jsonl',records);del st,fields,batch,inputs,raw
                canonical.validate_rows(cases,[layout],[r for r in records if (r['condition'],r['draw'])==(layout['condition'],layout['draw'])])
                for r in records:assert sha(target/r['prediction'])==r['sha256']
                assert state_hash(model.state_dict())==start_hash
                append(target/'layout_audits.jsonl',dict(at=now(),condition=layout['condition'],draw=layout['draw'],total=len(records),finite_valid_range=True,weights_unchanged=True,hashes_verified=True))
                print(f'PREDICT seed{seed} {layout["condition"]} draw{layout["draw"]}: {len(records)}/595',flush=True)
        handle.remove();assert not guard.denied
    canonical.validate_rows(cases,layouts,records)
    freeze(target/'complete.json',dict(at=now(),completed=True,reconstructions=595,fields=1190,forward_calls=595,lifting_calls=1190,
        manifest_sha256=sha(target/'manifest.jsonl'),checkpoint_sha256=m['checkpoint_sha256'],state_sha256=start_hash,audit=guard.summary(),finite_valid_range=True))


class ScoringGuard(AccessGuard):
    def __init__(self):
        super().__init__(gt_splits=['test'],output=BASE)
        for r in verified(BASE/'baseline_manifest.json'):self.allowed[str(Path(r['path']).resolve())]='sealed_baseline_array'


def seal():
    validate_source();manifests={};files={}
    for seed in SEEDS:
        root=BASE/'evaluation'/str(seed);c=verified(root/'complete.json');rr=rows(root/'manifest.jsonl')
        assert sha(root/'manifest.jsonl')==c['manifest_sha256'] and c['reconstructions']==595
        canonical.validate_rows(verified(BASE/'cases.json'),verified(BASE/'layouts.json'),rr)
        manifests[str(root/'manifest.jsonl')]=sha(root/'manifest.jsonl')
        for r in rr:
            path=root/r['prediction'];assert sha(path)==r['sha256'];files[str(path)]=r['sha256']
            with np.load(path,allow_pickle=False) as z:
                assert set(z.files)=={'C0','C1'}
                assert all(z[k].shape==(64,)*3 and z[k].dtype==np.float32 and np.isfinite(z[k]).all() and z[k].min()>=0 and z[k].max()<=1 for k in z.files)
    for r in verified(BASE/'baseline_manifest.json'):assert sha(Path(r['path']))==r['sha256']
    assert len(files)==1190
    freeze(BASE/'prediction_inventory.json',dict(at=now(),files=files,manifests=manifests))
    freeze(BASE/'all_predictions_sealed.json',dict(at=now(),new_predictions=1190,new_fields=2380,reused_predictions=1190,
        manifests=manifests,inventory_sha256=sha(BASE/'prediction_inventory.json'),all_predictions_saved_before_GT64=True,GT64_read=False,GT128_read=False))
    for path in [*map(Path,files),*map(Path,manifests),BASE/'prediction_inventory.json',BASE/'all_predictions_sealed.json']:path.chmod(0o444)


def metric_functions():
    # This existing module has no import-time scoring. Import with our runtime;
    # only its pure metric functions are reused, independently verified below.
    import importlib.util
    spec=importlib.util.spec_from_file_location('pure_lifting_metrics',PROJECT/'scripts/lifting_mapping_v1/analysis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.canonical_metrics,module.independent_metrics


def score():
    backend();seal();boundary=verified(BASE/'all_predictions_sealed.json');verify_hashes(boundary['manifests'])
    lock=verified(BASE/'evaluation_lock.json');validate_source();verify_hashes(lock['data_identity'])
    for info in lock['GT64_stat_only'].values():
        s=Path(info['path']).stat();assert (s.st_size,s.st_mtime_ns)==(info['size'],info['mtime_ns'])
    canonical_metric,independent_metric=metric_functions();source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')}
    records=verified(BASE/'baseline_manifest.json')
    for seed in SEEDS:
        root=BASE/'evaluation'/str(seed)
        records.extend(dict(r,path=str(root/r['prediction']),variant='ablation') for r in rows(root/'manifest.jsonl'))
    for r in records:r.setdefault('variant','full')
    assert len(records)==2380
    old={(r['method'],r['condition'],r['draw'],r['case_id'],r['stage']):r for r in rows(OUT/'evaluation/test/metrics.jsonl') if r['method'] in [m['key'] for m in verified(BASE/'controls.json')]}
    controls={m['seed']:m['key'] for m in verified(BASE/'controls.json')}
    absolute=[];disagreements={};gt_hashes={};first_gt=None;missing=[]
    with ScoringGuard() as guard:
        for ci,case in enumerate(verified(BASE/'cases.json')):
            gp=DATA/source[case]['gt_volume_64_path'];first_gt=first_gt or now();g=np.load(gp,allow_pickle=False).squeeze().astype(np.float64)
            assert g.shape==(64,)*3 and np.isfinite(g).all() and g.sum()>0 and g.min()>=0 and g.max()<=1
            gt_hashes[str(gp)]=sha(gp)
            for r in [r for r in records if r['case_id']==case]:
                path=Path(r['path']);assert sha(path)==r['sha256']
                with np.load(path,allow_pickle=False) as z:
                    for stage in ('C0','C1'):
                        pred=z[stage].astype(np.float64);a=canonical_metric(pred,g);b=independent_metric(pred,g)
                        for k in METRICS:
                            assert (a[k] is None and b[k] is None) or math.isclose(a[k],b[k],rel_tol=1e-8,abs_tol=1e-9),(r,k,a,b)
                            if a[k] is not None:disagreements[k]=max(disagreements.get(k,0),abs(a[k]-b[k]))
                            else:missing.append(dict(seed=r['seed'],case=case,condition=r['condition'],draw=r['draw'],stage=stage,variant=r['variant'],metric=k))
                        if r['variant']=='full':
                            archived=old[controls[r['seed']],r['condition'],r['draw'],case,stage]
                            ref=dict(relative_l1=archived['volume']['relative_l1'],relative_l2=archived['volume']['relative_l2'],dice50=archived['morphology']['dice_0.5'],absolute_peak_ratio_error=archived['morphology']['absolute_peak_ratio_error'],centroid_distance_mm=archived['morphology']['centroid_distance_mm'])
                            for k in METRICS:assert (a[k] is None and ref[k] is None) or math.isclose(a[k],ref[k],rel_tol=1e-8,abs_tol=1e-9)
                        absolute.append(dict(seed=r['seed'],case_id=case,condition=r['condition'],draw=r['draw'],stage=stage,variant=r['variant'],prediction_path=str(path),prediction_sha256=r['sha256'],checkpoint_sha256=r['checkpoint_sha256'],**a))
            print(f'SCORED and independent metric check {ci+1}/35',flush=True)
        assert not guard.denied
    assert len(absolute)==4760
    csv_write(BASE/'per_case_metrics.csv',absolute)
    freeze(BASE/'metric_verification.json',dict(at=now(),passed=True,fields=4760,independent_formulas=5,max_disagreement=disagreements,undefined=missing,
        archived_controls_match=True,historical_metrics_sha256=sha(OUT/'evaluation/test/metrics.jsonl'),audit=guard.summary(),first_GT64_access=first_gt,seal_at=boundary['at'],GT64_hashes=gt_hashes,GT128_read=False))
    import analysis_statistics as stats
    stats.summarize(absolute)
    subprocess.run([sys.executable,str(HERE/'verify_statistics.py')],cwd=PROJECT,check=True)
    import report
    report.finish()


def benchmark():
    backend();validate_source();p=verified(BASE/'protocol.json');config=copy.deepcopy(p['controls'][0]['config'])
    from geophy_set3d.training import selfsup
    from geophy_set3d.data.measurement_only import dual_measurement_collate
    result=[]
    with TrainGuard() as guard:
        ds=selfsup.make_dataset(config,validation=True);batch=selfsup.move_dual_batch(dual_measurement_collate([ds[0]]),'cuda:0')['A']
        assert int(batch['view_mask'].sum())==4
        for arm in ('uniform',ARM):
            model=model_for(config,arm,'cuda:0').eval().requires_grad_(False)
            macs=[0];hooks=[]
            def hook(mod,args,out):
                if isinstance(mod,(torch.nn.Conv2d,torch.nn.Conv3d)):
                    macs[0]+=out.numel()*mod.in_channels//mod.groups*int(np.prod(mod.kernel_size))
                elif isinstance(mod,torch.nn.Linear):macs[0]+=out.numel()*mod.in_features
            for mod in model.modules():
                if isinstance(mod,(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear)):hooks.append(mod.register_forward_hook(hook))
            with torch.inference_mode():model(batch,return_stages=True)
            for h in hooks:h.remove()
            elapsed=[]
            with torch.inference_mode():
                for i in range(50):
                    torch.cuda.synchronize();t=time.perf_counter();st=model(batch,return_stages=True);torch.cuda.synchronize()
                    if i>=10:elapsed.append(time.perf_counter()-t)
                    del st
            result.append(dict(arm=arm,fusion='Uniform Fusion',parameters=sum(p.numel() for p in model.parameters()),trainable_architecture_parameters=342725,
                partial_Conv_Linear_MACs=macs[0],mean_seconds=float(np.mean(elapsed)),median_seconds=float(np.median(elapsed)),timed_iterations=40,warmups=10))
            del model;torch.cuda.empty_cache()
    freeze(BASE/'benchmark.json',dict(at=now(),passed=True,results=result,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=torch.cuda.get_device_name(),batch=1,views=4,precision='FP32',
        includes='C0+C1 and physical feedback, excludes input/output I/O; fresh paired initialization for architecture-only benchmark',
        compute_limit='partial MACs count only Conv2d/Conv3d/Linear; excludes projector/grid_sample/interpolation/attention matmul/normalization and elementwise ops; not total FLOPs',audit=guard.summary()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','predict','benchmark','score']);parser.add_argument('--seed',type=int);args=parser.parse_args()
    try:
        if args.command=='predict':predict(args.seed)
        else:globals()[args.command]()
    except BaseException:
        import traceback
        write(BASE/f'evaluation_failure_{args.command}.json',dict(at=now(),traceback=traceback.format_exc()));raise
