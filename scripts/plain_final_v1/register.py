"""Freeze identities before supplemental inference; no optimizer or GT access."""
from context import *
from collections import Counter


def independent_audit():
    root=Path('/data/djc/GeoPhy-Set3D/data');raw=root/'raw/Fluent/static_final_frame_partial_v1'
    source=rows(DATA/'source_index.jsonl');manifest=rows(raw/'manifest.jsonl')
    assert len(source)==len(manifest)==237
    known={str(Path(r['source_target_path']).resolve()) for r in source}
    listed=[dict(path=str(p.resolve()),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in (root/'raw').rglob('*.npy') if 'targets' in p.parts]
    unknown=[r for r in listed if r['path'] not in known]
    rawsets=sorted(str(p.parent) for p in (root/'raw').rglob('manifest.jsonl'))
    processed=sorted(str(p.parent) for p in (root/'processed').rglob('dataset_build_manifest.json'))
    metadata=[];runs={};gthashes={};frames=[]
    for r in source:
        meta=read(r['source_metadata_path']);run=meta['source_run_dir'];assert meta['sample_id']==r['case_id']
        runs.setdefault(run,[]).append(r['case_id']);gthashes.setdefault(r['source_gt_sha256'],[]).append(r['case_id'])
        metadata.append(dict(case_id=r['case_id'],split=r['split'],source_run=run,source_gt_sha256=r['source_gt_sha256'],
            metadata_sha256=sha(r['source_metadata_path']),target_stat=next(x for x in listed if x['path']==r['source_target_path'])))
        fp=DATA/'samples'/r['case_id']/'frame_manifest.jsonl';f=rows(fp);assert len(f)==1
        frames.append(dict(path=str(fp),sha256=sha(fp),frame_count=len(f)))
    assert len(runs)==len(gthashes)==237
    assert Counter(r['split'] for r in source)==dict(train=166,val=36,test=35)
    assert not unknown and rawsets==[str(raw)] and processed==[str(DATA)],'Potential new dataset: requires separate pre-test registration'
    freeze(BASE/'independent_data_audit.json',dict(at=now(),status='BLOCKED_NO_INDEPENDENT_DATA',scope=str(root),
        metadata_only=True,GT128_opened=False,GT64_opened=False,Pool_B_arrays_opened=False,
        source_index_sha256=sha(DATA/'source_index.jsonl'),raw_manifest_sha256=sha(raw/'manifest.jsonl'),
        raw_datasets=rawsets,processed_datasets=processed,targets_listed_without_opening=listed,unregistered_targets=unknown,
        split_counts=dict(Counter(r['split'] for r in source)),unique_source_runs=len(runs),unique_source_GT_hashes=len(gthashes),
        metadata=metadata,frame_manifests=frames,
        split_audit_files={str(p):sha(p) for folder in ('splits','audits') for p in (DATA/folder).rglob('*') if p.is_file()},
        debug_source_index_sha256=sha(DATA/'debug/source_index.jsonl'),debug_not_independent=True,
        historical_GT_access_evidence={str(p):sha(p) for p in (HISTORY/'evaluation/test/gt_access.json',TARGETED/'GT64_access.json',CORE/'GT64_access.json')},
        finding='All project-owned cases registered and previously assigned; historical35 used in development. Layout/draw/PoolB do not create independent cases. No claim about unavailable external datasets.',
        future_plan=dict(initial_case_plan=35,target_case_plan=[50,100],not_a_power_analysis=True,
            requirements=['new source runs/cases deduplicated against all237 by case/run/GT provenance hash','pre-register case-level sample-size precision/power justification',
            'freeze both model identities, layouts, normalization, endpoints, stopping and missing/failure handling before blind scoring',
            'separate custodian verifies GT64 identities and access history, prediction seal precedes GT scoring','no GT128 or PoolB use without separate explicit authorization/protocol'])))


def main():
    backend();assert not BASE.exists(),'Preserve existing registration; inspect before recovery'
    BASE.mkdir();(BASE/'scheduling').mkdir();(BASE/'checks').mkdir()
    freeze(BASE/'initial_resources.json',resources())
    historical=verified(HISTORY/'final_lock.json');seal=verified(HISTORY/'evaluation/test/all_predictions_sealed.json')
    assert seal['all_predictions_saved_before_GT'] and seal['final_lock_sha256']==sha(HISTORY/'final_lock.json')
    verify_hashes(historical['sources']);verify_hashes(historical['data_identity']);verify_hashes(historical['evidence'])
    verify_hashes(verified(HISTORY/'classical/implementation_amendment_v2.json')['sources'])
    sart_registration=verified(HISTORY/'sart/registration.json')
    sart_amendment=verified(HISTORY/'sart/implementation_amendment_v2.json')
    assert sart_amendment['registration_sha256']==sha(HISTORY/'sart/registration.json')
    verify_hashes(sart_amendment['sources'])
    verify_hashes({p:h for p,h in sart_registration['sources'].items() if p not in sart_amendment['sources']})
    settings=verified(HISTORY/'layouts.json')['test'];cases=verified(HISTORY/'registration.json')['cases']['test']
    assert len(settings)==17 and len(cases)==35 and cases==historical['case_ids']
    freeze(BASE/'cases.json',cases);freeze(BASE/'layouts.json',settings)
    methods=[]
    for key in (MAIN,SECOND):
        old=next(m for m in verified(TARGETED/'model_manifest.json') if m['key']==key)
        m=core.validate_training(old['run']);assert all(m[k]==old[k] for k in m)
        assert m['checkpoint_sha256']==core.EXPECTED[m['seed']] and m['parameters']==342209
        p=verified(Path(m['run'])/'protocol.json');verify_hashes(p['sources'])
        verify_hashes({str(DATA/k) if not Path(k).is_absolute() else k:h for k,h in p['data_files'].items()})
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
        assert ck['epoch']==m['best_epoch'] and ck['config']==m['config'] and not ck['volumetric_gt_used'] and ck['protocol_sha256']==sha(Path(m['run'])/'protocol.json')
        model=model_for(m['config'],PLAIN,'cpu').eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True)
        assert sum(p.numel() for p in model.parameters())==342209
        m.update(role='primary_user_designated' if key==MAIN else 'second_seed_reference',file_size=Path(m['checkpoint']).stat().st_size,
            state_sha256=canonical_worker.probe.state_hash(model.state_dict()),config_sha256=hashlib.sha256(json.dumps(m['config'],sort_keys=True).encode()).hexdigest(),
            protocol_sha256=sha(Path(m['run'])/'protocol.json'),training_summary_sha256=sha(Path(m['run'])/'summary.json'),
            training_log_sha256=sha(Path(m['run'])/'metrics.jsonl'),train_cases=p['train_cases'],validation_cases=p['validation_cases'],training_sources=p['sources'],
            selection=p['selection'],GT_free_training_and_epoch_selection=True,development_independent_of_historical_test=False)
        methods.append(m);freeze(BASE/'configs'/f'{key}.json',m['config'])
    model= model_for(methods[0]['config'],PLAIN,'cpu').eval().requires_grad_(False)
    model.load_state_dict(torch.load(methods[0]['checkpoint'],map_location='cpu',weights_only=False)['model'],strict=True)
    state=canonical_worker.probe.state_hash(model.state_dict())
    with io_guard(splits=['val']) as guard,torch.inference_mode():
        b=batch_for(dataset('val',verified(HISTORY/'layouts.json')['val'][0]),0)
        stages=model(b,return_stages=True)
        for k in ('C0','C1'):
            v=stages[k];assert v.shape==(1,1,64,64,64) and torch.isfinite(v).all() and v.min()>=0 and v.max()<=1
        assert canonical_worker.probe.state_hash(model.state_dict())==state
    freeze(BASE/'checks/cpu.json',dict(at=now(),passed=True,checkpoint_load_strict=True,output_shape=[1,1,64,64,64],model_state_unchanged=True,no_training=True,audit=guard.summary()))
    origins=[];predictions=[];evidence={}
    for key in (MAIN,SECOND):
        root=TARGETED/'evaluation'/key;complete=verified(root/'complete.json');rr=rows(root/'manifest.jsonl')
        assert len(rr)==210 and sha(root/'manifest.jsonl')==complete['manifest_sha256']
        m=next(m for m in methods if m['key']==key);assert complete['method']['checkpoint_sha256']==m['checkpoint_sha256']
        origins.append(dict(method=key,count=210,manifest=str(root/'manifest.jsonl'),manifest_sha256=sha(root/'manifest.jsonl'),kind='unchanged prior predictions'))
        for r in rr:assert r['checkpoint_sha256']==m['checkpoint_sha256']
        predictions.extend(dict(r,reused=True,origin_manifest=str(root/'manifest.jsonl')) for r in rr)
    selected=verified(HISTORY/'classical/selected.json')['selected']
    expected={(s['condition'],s['draw'],c) for s in settings for c in cases};bysetting={(s['condition'],s['draw']):s for s in settings}
    for method in CLASSICS:
        root=HISTORY/'classical/test'/method/'frozen';mp=root/'manifest.jsonl';rr=rows(mp);complete=read(root/'complete.json')
        assert len(rr)==595 and sha(mp)==complete['manifest_sha256']==seal['manifests'][str(mp)] and not complete['audit']['denied']
        config=complete['config'];assert complete['method']==method and complete['split']=='test'
        if method!='SART':assert config==selected[method]['config']
        else:assert config==next(m for m in historical['methods'] if m['key']==method)['config']
        assert {(r['condition'],r['draw'],r['case_id']) for r in rr}==expected
        # Historical row order is retained exactly, and matches its frozen layout/case order.
        assert [(r['condition'],r['draw'],r['case_id']) for r in rr]==[(s['condition'],s['draw'],c) for s in settings for c in cases]
        for r in rr:
            s=bysetting[r['condition'],r['draw']];c=r['case_id'];assert r['input_ids']==s['input_ids'][c] and r['heldout_ids']==s['heldout_ids'][c]
            assert r['config']==config and 0<r['iterations']<=config['iterations']
            curve=read(r['curve']);assert curve['iterations']==r['iterations'] and curve['status']==r['solver_status'] and len(curve['curve'])==r['iterations']
            evidence[r['curve']]=sha(r['curve'])
            predictions.append(dict(r,prediction=str(root/r['prediction']),prediction_sha256=r['sha256'],method=method,reused=True,origin_manifest=str(mp)))
        origins.append(dict(method=method,count=595,manifest=str(mp),manifest_sha256=sha(mp),config=config,
            complete_sha256=sha(root/'complete.json'),statuses=dict(Counter(r['solver_status'] for r in rr)),
            operator='linear nodal mole CSR (supplementary)' if method=='SART' else 'P_diff mass interpolation then mole K96',reuse=True))
    input_records=[]
    with io_guard(splits=['test'],predictions=predictions) as guard:
        for r in predictions:
            assert sha(r['prediction'])==r['prediction_sha256'];check_fields(r['prediction'],['C1'] if r['method'] in CLASSICS else ['C0','C1'])
        for s in settings:
            ds=dataset('test',s);assert [r['case_id'] for r in ds.records]==cases
            for i,c in enumerate(cases):
                b=batch_for(ds,i);ih=input_hash(b)
                for r in predictions:
                    if r['method'] not in CLASSICS and (r['case_id'],r['condition'],r['draw'])==(c,s['condition'],s['draw']):assert r['input_sha256']==ih
                input_records.append(dict(case_id=c,case_index=i,condition=s['condition'],draw=s['draw'],input_ids=s['input_ids'][c],heldout_ids=s['heldout_ids'][c],
                    input_sha256=ih,tensor_hashes={k:tensor_hash(v) for k,v in b.items() if isinstance(v,torch.Tensor)}))
        independent_audit()
    freeze(BASE/'input_manifest.json',input_records);freeze(BASE/'reuse_predictions.json',predictions)
    freeze(BASE/'model_manifest.json',dict(at=now(),primary=MAIN,reference=SECOND,models=methods,
        factory="scripts/oari_targeted_v1/variants.py:model_for(config,'uniform_plain_set_interaction',device)",
        inherited_config_name_not_factory=True,architecture=['Encoder','GeometryLifting','Plain Set Interaction','Multi-scale MRGR','Uniform Fusion','PRF'],
        normalization=dict(method='linear_p99_5',scale=32284.894580078173,statistics_path=str(DATA/'train_statistics.json'),sha256=sha(DATA/'train_statistics.json')),
        projector='P_diff,64^3 endpoint XYZ[0,.6]m,96 midpoint samples, interpolate mass then convert mole,ppm*m',
        output=dict(C0='initial decode',C1='decode after signed measurement residual feedback',shape=[1,1,64,64,64],quantity='CH4 mass fraction',bounds=[0,1]),
        selection='user-designated primary seed11; strict minimum validation H4 only over full300epochs; no test-based seed promotion',
        checkpoint_immutable=True,no_training=True,independent_confirmation=False))
    freeze(BASE/'comparison_manifest.json',dict(at=now(),origins=origins,case_count=35,layouts=17,expected_per_method=595,
        classical_predictions_reused=2380,classical_recomputations=0,primary_predictions_reused=210,primary_missing_to_infer=385,
        second_seed_scope='210 existing predictions retained, core paired robustness only; no new inference',
        source_metrics=dict(path=str(HISTORY/'evaluation/test/metrics.jsonl'),sha256=sha(HISTORY/'evaluation/test/metrics.jsonl')),
        historical_seal_sha256=sha(HISTORY/'evaluation/test/all_predictions_sealed.json'),historical_final_lock_sha256=sha(HISTORY/'final_lock.json'),
        input_manifest_sha256=sha(BASE/'input_manifest.json'),curve_hashes=evidence,data_identity=historical['data_identity'],audit=guard.summary()))
    src=sources()
    freeze(BASE/'protocol.json',dict(at=now(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        sources=src,protocol_document_sha256=sha(PROJECT/'docs/mrgr_plain_final_v1_protocol_zh.md'),
        model_manifest_sha256=sha(BASE/'model_manifest.json'),comparison_manifest_sha256=sha(BASE/'comparison_manifest.json'),
        environment=dict(executable=sys.executable,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
            packages=sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())),
        statistics=dict(seed=20261019,resamples=20000,unit='case, draws averaged within case',primary='Val-A-V4 C1 volume.relative_l1',family=4,historical_conservative_family=20,tie=1e-8),
        authorized_GT64_cases=cases,GT128_forbidden=True,Pool_B_forbidden=True,GT64_after_prediction_seal_only=True,exploratory=True))
    snapshots={}
    for path in src:
        p=Path(path)
        relative=p.relative_to(PROJECT) if p.is_relative_to(PROJECT) else Path('external')/p.relative_to('/')
        dst=BASE/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        assert sha(dst)==src[path];snapshots[path]=dict(snapshot=str(dst),sha256=sha(dst))
    freeze(BASE/'source_snapshot_manifest.json',snapshots)
    print('REGISTERED',BASE,flush=True)


if __name__=='__main__':main()
