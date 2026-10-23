"""Audit available controls and freeze exactly six full-budget training arms."""
from common_stage import *


def normalized(c):
    c=copy.deepcopy(c);c.pop('experiment');return c


def main():
    backend()
    if BASE.exists():raise FileExistsError('Refuse existing stage; inspect evidence before any continuation: '+str(BASE))
    assert verified(PRIOR/'acceptance.json')['completed']
    selected=[m for m in verified(PRIOR/'model_manifest.json') if m['arm']==PLAIN]
    assert len(selected)==2
    cases=verified(PRIOR/'cases.json');layouts=verified(PRIOR/'layouts.json')
    identities=[];refs=[]
    for m in selected:
        assert sha(m['checkpoint'])==m['checkpoint_sha256']==EXPECTED[m['seed']]
        root=Path(m['run']);p=verified(root/'protocol.json');verify_hashes(p['sources'])
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
        assert ck['epoch']==m['best_epoch'] and ck['config']==m['config'] and not ck['volumetric_gt_used']
        assert ck['protocol_sha256']==sha(root/'protocol.json')
        model=model_for(m['config'],PLAIN,'cpu');model.load_state_dict(ck['model'],strict=True)
        log=rows(root/'metrics.jsonl');summary=read(root/'summary.json')
        chosen=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        assert len(log)==300 and chosen['epoch']==m['best_epoch'] and summary['updates']==24900
        assert summary['completed'] and sha(root/'checkpoints/last.pt')==summary['last_checkpoint_sha256']
        predroot=PRIOR/'evaluation'/m['key'];complete=verified(predroot/'complete.json')
        assert sha(predroot/'manifest.jsonl')==complete['manifest_sha256']
        assert complete['method']['checkpoint_sha256']==m['checkpoint_sha256']
        rr=rows(predroot/'manifest.jsonl');assert len(rr)==210
        for r in rr:assert sha(r['prediction'])==r['prediction_sha256']
        refs.append(dict(m,prediction_root=str(predroot),source_metrics=str(PRIOR/'metrics.csv')))
        identities.append(dict(key=m['key'],seed=m['seed'],best_epoch=m['best_epoch'],parameters=m['parameters'],checkpoint_sha256=m['checkpoint_sha256'],
            selected_by='strict minimum validation H4 only',epochs=300,updates=24900,protocol_sha256=sha(root/'protocol.json'),prediction_manifest_sha256=sha(predroot/'manifest.jsonl')))
    candidates=[]
    names={f'{a}_seed{s}' for a,s in TASKS}
    for p in BASE.parent.glob('*/training/*'):
        if p.name in names:candidates.append(dict(path=str(p),checkpoint_exists=any((p/f'checkpoints/{n}.pt').exists() for n in ('best','last'))))
    assert not any(p['checkpoint_exists'] for p in candidates),candidates
    BASE.mkdir()
    freeze(BASE/'references.json',refs);freeze(BASE/'cases.json',cases);freeze(BASE/'layouts.json',layouts)
    history=verified(HISTORY/'registration.json');data_files=verified(HISTORY/'model_manifest.json')['training_data_sha256']
    sources=training_sources();shared=[]
    for source,digest in sources.items():
        snapshot=BASE/'source_snapshot'/Path(source).relative_to('/')
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(Path(source).read_bytes());assert sha(snapshot)==digest
    for arm,seed in TASKS:
        ref=next(m for m in refs if m['seed']==seed);target=BASE/'training'/f'{arm}_seed{seed}'
        cfg=copy.deepcopy(ref['config']);cfg['experiment']=dict(name=target.name,output_root=str(target))
        assert normalized(cfg)==normalized(ref['config'])
        full=model_for(cfg,PLAIN,'cpu');rng=torch.get_rng_state().clone()
        model=model_for(cfg,arm,'cpu');assert torch.equal(rng,torch.get_rng_state())
        a,b=full.state_dict(),model.state_dict();assert all(k in a and torch.equal(v,a[k]) for k,v in b.items())
        removed=sorted(set(a)-set(b));assert all(k.startswith(('residual_encoder.','residual_adapter.')) for k in removed)
        assert bool(removed)==(arm==NO_PRF)
        count=sum(p.numel() for p in model.parameters())
        freeze(target/'protocol.json',dict(registered_at=now(),arm=arm,seed=seed,config=cfg,epochs=300,planned_updates=24900,parameters=count,
            sources=sources,train_cases=history['cases']['train'],validation_cases=history['cases']['val'],data_files=data_files,subset_smoke=False,
            fresh_initialization=True,initial_model_sha256=canonical_worker.probe.state_hash(model.state_dict()),
            shared_initialization_exact=True,constructor_CPU_RNG_identical=True,removed_state_keys=removed,
            selection='strict minimum validation projection H4 over all300epochs; no GT',
            resume='unchanged canonical last.pt epoch commit, optimizer/scheduler/Python/NumPy/Torch/CUDA/loader RNG and metrics',
            nondeterminism='FP32 TF32off; canonical CUDA grid_sample backward atomics may be nondeterministic',
            matched_reference=ref['key'],matched_checkpoint_sha256=ref['checkpoint_sha256'],structure=str(model)))
        write(target/'config.json',cfg)
        shared.append(dict(key=target.name,parameters=count,parameter_delta=count-ref['parameters'],all_retained_state_bit_exact=True,CPU_RNG_exact=True,removed_keys=removed))
    # Bind context-free gaps to live workflow parent identities, not GPU utilization.
    snap=resources();table=subprocess.check_output(['ps','-eo','user,pid,ppid,etime,args'],text=True)
    proc={int(line.split(None,4)[1]):line.split(None,4) for line in table.splitlines()[1:] if len(line.split(None,4))==5}
    foreign=[]
    for line in snap['gpu_processes'].splitlines():
        if not line.strip():continue
        uuid,pid,*rest=[x.strip() for x in line.split(',')];pid=int(pid);info=proc[pid];parent=int(info[2]);pi=proc.get(parent)
        # Only pin a workflow parent if it is an identified job runner.
        owner=parent if pi and ('run_cv_resume_worker.sh' in pi[4] or 'run_jump_stable_cv_gpu3.sh' in pi[4]) else pid
        start=Path(f'/proc/{owner}/stat').read_text().split(') ',1)[1].split()[19]
        foreign.append(dict(gpu_uuid=uuid,pid=pid,user=info[0],command=info[4],workflow_pid=owner,workflow_start_ticks=start,workflow_command=proc[owner][4]))
    freeze(BASE/'resource_registration.json',dict(at=now(),snapshot=snap,process_table=table,foreign_workflows=foreign,
        policy='do not use GPU while pinned foreign workflow is alive, including CUDA-context-free gaps; double15s snapshots and UUID locks always required'))
    freeze(BASE/'audit.json',dict(at=now(),baseline_checks=identities,initialization=shared,candidates=candidates,
        historical_reuse='two plain baseline checkpoints/420 predictions; OARI module ablation results are incompatible and not reused as current results',
        formal_definitions=dict(geometry=str(PROJECT/'scripts/no_spatial_lifting_v1/runtime.py'),single=str(PROJECT/'scripts/paper_final_v1/models.py'),prf='historical NoFeedback adapted only to PlainSetStage0'),
        GT64_read=False,GT128_read=False,conflicts=[]))
    freeze(BASE/'protocol.json',dict(at=now(),sources=sources,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tasks=[dict(arm=a,seed=s,batch=1 if i<4 else 2) for i,(a,s) in enumerate(TASKS)],epochs=300,updates=24900,
        baseline_sha256=EXPECTED,test_cases=35,layouts=6,new_predictions=1260,reused_predictions=420,
        stats=dict(seed=SEED,resamples=20000,unit='pairedcase; clustered five draws averaged within eachcase',tie=1e-8),
        authorization='Current user explicitly requests exactly six ablation trainings and paired evaluation on existing35cases ordinaryV4+clusteredV4; only required authorized GT64 after all predictions sealed; no GT128',
        GT128_forbidden=True,training_GT64_forbidden=True,test_previously_used=True,independent_confirmation=False,
        scheduling='batch1 four geometry/single tasks max4; batch2 twoPRF only after batch1 accepted; wait for unoccupied GPUs, no co-tenancy or foreign signals'))
    update_tracker();print(BASE,flush=True)


if __name__=='__main__':main()
