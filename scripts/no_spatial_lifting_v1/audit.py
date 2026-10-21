"""Read-only provenance audit followed by immutable experimental registration."""
from runtime import *
import platform


def register():
    backend();assert BASE.parent.is_dir();BASE.mkdir(exist_ok=False)
    manifest=verified(OUT/'model_manifest.json')
    identity=verified(UNIFORM_BASE/'model_identity.json')
    decision=verified(UNIFORM_BASE/'model_decision_after_stage_v1.json')
    catalog=read(UNIFORM_BASE/'model_catalog.json')
    assert identity['parameters']==342725
    assert sha(Path(identity['config_path']))==identity['config_sha256']
    verify_hashes(identity['training_sources'])
    controls=[next(m for m in catalog['seeds'] if m['seed']==seed) for seed in SEEDS]
    assert identity['main']==controls[0] and decision['checkpoint_sha256']==EXPECTED[20261011]
    assert decision['original_backbone_OARI_and_original_feedback_retained']
    baseline=[];evidence={};init=[]
    import evaluation
    cases=verified(OUT/'registration.json')['cases']['test'];layouts=verified(OUT/'layouts.json')['test']
    assert len(layouts)==17 and len(cases)==35
    for m in controls:
        seed=m['seed'];run=Path(m['run']);cp=Path(m['checkpoint'])
        assert seed in SEEDS and m['parameters']==342725 and sha(cp)==m['checkpoint_sha256']==EXPECTED[seed]
        proto=read(run/'protocol.json')
        side=(run/'protocol.json.sha256') if (run/'protocol.json.sha256').exists() else run/'protocol.sha256'
        assert sha(run/'protocol.json')==side.read_text().strip()
        src={str(PROJECT/k) if not Path(k).is_absolute() else k:v for k,v in proto['sources'].items()}
        verify_hashes(src)
        summary=read(run/'summary.json');log=rows(run/'metrics.jsonl')
        assert summary['completed'] and summary['epochs']==300 and summary['updates']==24900 and len(log)==300
        chosen=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        assert chosen['epoch']==m['best_epoch']==summary['best_epoch']
        assert chosen['val']['measurement']['heldout']['relative_l1']==summary['best_val_H']
        with AccessGuard(checkpoints=[str(cp)],output=BASE) as guard:
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            assert ck['epoch']==m['best_epoch'] and ck['config']==m['config'] and not ck['volumetric_gt_used']
            assert ck['protocol_sha256']==sha(run/'protocol.json')
            official=official_model(m['config'],'uniform','cpu');official.load_state_dict(ck['model'],strict=True)
            fresh=official_model(m['config'],'uniform','cpu');alternate=model_for(m['config'],ARM,'cpu')
            assert fresh.state_dict().keys()==alternate.state_dict().keys()
            assert all(torch.equal(v,alternate.state_dict()[k]) for k,v in fresh.state_dict().items())
            initial=state_hash(fresh.state_dict())
            assert initial==read(run/'environment.json')['initial_model_sha256']
            archived_initial='official Uniform control historical environment.json initial state hash'
            assert sum(p.numel() for p in official.parameters())==342725
            from module_models_v2 import UniformFusion
            assert type(official.fusion) is UniformFusion and len(list(official.fusion.parameters()))==0
            assert not guard.denied
        init.append(dict(seed=seed,state_sha256=initial,parameters=342725,trainable_parameters=342725,
            shared_state_keys=len(fresh.state_dict()),all_shared_tensors_equal=True,original_initial_identity=archived_initial,
            tensor_sha256={k:hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest() for k,v in fresh.state_dict().items()},
            added_parameters=0,removed_parameters=0,new_module_initialization='none; no trainable replacement parameters'))
        root=Path(m['prediction_root']);complete=read(root/'complete.json')
        rr=rows(root/'manifest.jsonl');evaluation.validate_rows(cases,layouts,rr)
        assert complete['checkpoint_sha256']==m['checkpoint_sha256']
        assert sha(root/'manifest.jsonl')==complete['manifest_sha256']
        seal=verified(OUT/'evaluation/test/all_predictions_sealed.json')
        assert seal['manifests'][str(root/'manifest.jsonl')]==complete['manifest_sha256']
        assert not complete['audit']['denied'] and not any(k.startswith('GT64') for k in complete['audit']['counts'])
        for r in rr:
            assert sha(root/r['prediction'])==r['sha256']
            baseline.append(dict(r,control=m['key'],seed=seed,path=str(root/r['prediction']),checkpoint_sha256=m['checkpoint_sha256']))
        evidence.update({str(p):sha(p) for p in [UNIFORM_BASE/'model_catalog.json',UNIFORM_BASE/'model_identity.json',UNIFORM_BASE/'model_decision_after_stage_v1.json',Path(identity['config_path']),cp,run/'protocol.json',run/'metrics.jsonl',run/'summary.json',root/'manifest.jsonl',root/'complete.json']})
    data_files=manifest['training_data_sha256']
    assert len(data_files)==205
    with TrainGuard() as guard:
        for path,h in data_files.items():assert sha(DATA/path)==h
        assert not guard.denied
    assert all(controls[0]['config'][k]==controls[1]['config'][k] for k in controls[0]['config'] if k not in ('training','experiment'))
    t0=dict(controls[0]['config']['training']);t1=dict(controls[1]['config']['training']);t0.pop('seed');t1.pop('seed');assert t0==t1
    config=controls[0]['config']
    assert config['training']['batch_size']==2 and config['training']['gradient_accumulation']==1 and config['training']['max_epochs']==300
    assert config['optimizer']==dict(name='adamw',lr=2e-4,weight_decay=1e-4)
    assert config['scheduler']==dict(name='cosine',T_max=300) and config['training']['gradient_clip_norm']==1.
    freeze(BASE/'controls.json',controls);freeze(BASE/'baseline_manifest.json',baseline)
    freeze(BASE/'layouts.json',layouts);freeze(BASE/'cases.json',cases);freeze(BASE/'initialization_audit.json',init)
    freeze(BASE/'historical_audit.json',dict(at=now(),passed=True,controls=controls,checkpoint_and_training_evidence=evidence,
        historical_predictions_verified=len(baseline),historical_inference_repeated=False,GT64_read=False,GT128_read=False,
        previous_frozen_diagnostic_acceptance_sha256=sha(PROJECT/'docs/reports/artifacts/mrgr_lifting_mapping_v1/acceptance.json'),
        previous_experiments_unchanged=True,training_config_differences_from_request=[],
        original_family_size=14,final_uniform_identity_verified=True,identity_source='final model catalog, final model identity, post-stage model decision, config, actual checkpoint and training/test products; not diagnostic metadata',
        note='same seed11 checkpoint was also used in prior diagnostics, but its formal status is independently established by the final model registration',
        original_comparison_schema='Metadata audit inspected original score file header and first historical LR reference comparison before training; not consumed by model design, engineering, training or selection. No new model test scores exist.',
        source_dependencies=dict(lifting='src/geophy_set3d/models/lifting.py',forward='src/geophy_set3d/models/geolift_oari_mrgr_res3d.py',
            OARI='src/geophy_set3d/models/geolift_oari_res3d.py',geometry_descriptor='src/geophy_set3d/models/geolift_gca_res3d.py',
            official_factory='scripts/uniform_final_v1/uniform_models.py:model_for(uniform), loaded with actual final runtime -> original multiscale + UniformFusion',
            train='scripts/paper_final_v1/train.py',objective='src/geophy_set3d/training/selfsup.py',
            tensor_shapes='features[B,V,32,64,64] then residual[B,V,16,64,64] -> [B,V,C,32,32,32]; visibility[B,V,32,32,32]; FP32',
            geometry='station origins/axes/angle bounds for visibility+descriptors; ray directions/path lengths for physics; all original')))
    freeze(BASE/'candidate_comparison.json',dict(at=now(),candidates=[
        dict(name='global_pool_broadcast',selected=False,parameters=0,information='erases all detector-local variation',
             isolation='confounds spatial mapping with destruction of2D local information',interface='compatible',stability='trainable but broad information bottleneck'),
        dict(name='fixed_planar_resize_extrusion',selected=True,parameters=0,information='keeps32x32 local detector variation; no view/voxel geometry in feature values',
             isolation='removes acquisition-conditioned spatial correspondence; also changes sample coverage/density, imposes constant depth before geometry tokens',
             interface='same shapes/channels/dtype and original visibility',compute='1024 bilinear samples and32768 broadcast positions per feature; original32768 irregular samples',
             stability='convex bilinear interpolation, bounded by input feature range, ordinary autograd'),
        dict(name='fixed_permutation_or_flip',selected=False,parameters=0,information='retains invertible wrong mapping',
             isolation='still uses spatial geometric sampling and could learn inverse permutation; does not remove target mechanism')],
        selected='fixed_planar_resize_extrusion',scope='both main and residual feature-lifting calls',
        limitations=['not a pure operator-only effect','changes detector sampling density and depth structure',
                     'geometry descriptors, OARI, Uniform Fusion and physics still provide geometric information and may compensate',
                     'feedback branch retained but its feature lifting is intentionally replaced by same factor']))
    runs=[]
    for seed in SEEDS:
        runs.append(str(register_run(seed)))
        for name in ('uninterrupted','resumed'):register_run(seed,name)
    freeze(BASE/'dependencies.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),
        packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)))
    freeze(BASE/'initial_resources.json',resources())
    freeze(BASE/'protocol.json',dict(at=now(),id=BASE.name,arm=ARM,seeds=list(SEEDS),fusion='Uniform Fusion in both full and ablated models',controls=controls,runs=runs,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),sources=training_identity(),
        instruction_sha256=sha(Path('/home/djc/.codex/attachments/e5981615-79b2-4fbb-96cc-379c4e535855/已粘贴的文本.txt')),
        design='fixed geometry-free planar bilinear resize/extrusion, both lifting calls, inherited angular_grid unchanged and used for original visibility/descriptors only',
        axes='detector horizontal -> world Y, vertical -> world Z, repeated along world X; fixed for all views/cases/seeds; no search',
        unchanged=['encoder inputs and path channels','geometry_descriptor','visibility','ray relations','observability','OARI','Uniform Fusion',
                   'multiscale decoder','residual encoder/adapter','physical projector','A/B/H sampling','all losses and coefficients'],
        new_parameters=0,parameters=342725,trainable_parameters=342725,epochs=300,updates=24900,batch_size=2,
        optimizer=config['optimizer'],scheduler=config['scheduler'],loss=config['loss'],train=config['train'],validation=config['validation'],
        selection='strict minimum mean validation H4 relativeL1 over all300 epochs; no volumetric GT',
        engineering='real CUDA2-step objective/backprop; both seeds paired initialization;2epoch continuous vs1+resume2 for eachseed,4train/2val examples,4 persistent workers; exact integer/RNG and floatingrtol1e-4/atol1e-6',
        formal_tasks=2,DDP=False,parallelism=2,GPU_policy='UUID exclusive lock, two15s idle checks, inherited lockFD,host memory/disk/load gates; no external process signals',
        test_cases=35,test_layouts=17,test_combinations_per_seed=595,new_formal_predictions=1190,new_formal_fields=2380,
        draw_policy='all original17 layouts; ordinary/spread draw0; V3/V6/clustered all5draws, first average draws within case',
        primary=dict(condition='Val-A-V4',stage='C1',metric='relative_l1',delta='ablation minus same-seed full MRGR',sample_unit='case'),
        important_secondary='clustered-V4 C1 relativeL1, five draws averaged per case',
        auxiliary=['C0/C1 relativeL1','relativeL2','Dice50','absolute peak ratio error','centroid distance mm','parameters','partial MACs','inference time'],
        statistics=dict(seed=20261019,resamples=20000,confidence=.95,seed_levels=[20261011,20261019,'two_fixed_seed_mean_descriptive'],
            unit='35 paired cases; average5draws percase, then descriptive same-case mean across2fixedseeds',
            original_multiplicity_family_preserved=14,primary_additional_interval='report original14-denominator Bonferroni-style interval as conservative reference, without adding to or rewriting historical family',
            no_seed_population_uncertainty=True,all_secondary_exploratory=True),
        data_isolation='training/engineering guard forbids test arrays,GT64,GT128,test metrics; all1190new predictions sealed before authorized testGT64 scoring',
        previously_used_test=True,independent_confirmation=False,no_new_controls=True,no_other_factors=True,
        analysis_freeze='evaluation code+metric definitions locked before new predictions and any current GT64 access; no adjustment to training/model design'))
    (BASE/'protocol.json').chmod(0o444);(BASE/'protocol.json.sha256').chmod(0o444)
    write(BASE/'status.json',dict(at=now(),state='registered',engineering_passed=False))
    print('REGISTERED2 fresh paired-seed structural ablations; controls and1190 historical predictions verified; no training yet',flush=True)


if __name__=='__main__':
    try:register()
    except BaseException:
        import traceback
        write(BASE/'registration_failure.json',dict(at=now(),traceback=traceback.format_exc(),formal_training_started=False))
        raise
