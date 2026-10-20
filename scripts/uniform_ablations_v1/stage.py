"""Append-only supplementary ablation stage; reuse frozen numerical engines."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT/'scripts/uniform_final_v1'))
import runtime
from runtime import *
import copy
import argparse
import fcntl
import traceback

PRIOR = runtime.BASE
BASE = PRIOR.parent/'mrgr_uniform_ablations_v1'
SCRIPT = HERE
TASKS = tuple(f'uniform_{arm}_seed{seed}' for arm, seeds in (
    ('no_feedback', (20261011, 20261019)),
    ('zero_residual', (20261011, 20261019)),
    ('no_ray_relations', (20261011,)),
    ('no_consistency', (20261011,)),
    ('single_scale', (20261011,))) for seed in seeds)
ATTACHMENT = Path('/home/djc/.codex/attachments/29463d0e-fb3c-4066-8a03-b5e7f5b82d6d/已粘贴的文本.txt')


def source_identity():
    return runtime.training_identity() | {
        str(p): sha(p) for p in sorted(HERE.glob('*.py'))} | {
        str(p): sha(p) for p in sorted((PROJECT/'scripts/uniform_final_v1').glob('*.py'))}


def run_paths():
    return verified(BASE/'design.json')['runs']


def scope_evidence(base=None):
    return {str(BASE/'stage_registration.json'): sha(BASE/'stage_registration.json'),
            str(PRIOR/'seed_scope_amendment_v1.json'): sha(PRIOR/'seed_scope_amendment_v1.json')}


def configure():
    """Only adapt output/dispatch plumbing in this process; never edit frozen files."""
    runtime.BASE = BASE
    import worker
    worker.BASE = BASE
    worker.engine.OUT = BASE
    # TrainGuard resolves runtime.BASE on construction. Numerical train/model code
    # is byte-identical and each copied protocol also hashes this adapter.
    return worker


def normalized_config(c, *, remove_consistency=False):
    c = copy.deepcopy(c)
    c.pop('experiment', None)
    c['training'].pop('seed', None)
    if remove_consistency:
        c['loss']['consistency'] = .05
    return c


def candidate_inventory():
    roots = [Path('/scratch/djc'), Path('/home/djc/projects/GeoPhy-Set3D/outputs'),
             Path('/data/djc/GeoPhy-Set3D/outputs')]
    roots += [p/'outputs' for p in Path('/home/djc/projects').glob('GeoPhy-Set3D-*')]
    records = []
    seen = set()
    for root in roots:
        if not root.exists():
            records.append(dict(root=str(root),exists=False)); continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in (
                'predictions', '.git', '__pycache__', 'envs', 'node_modules', 'data')]
            if 'protocol.json' not in files: continue
            path = Path(directory)/'protocol.json'
            canonical = path.resolve()
            if canonical in seen: continue
            seen.add(canonical)
            p = read(path)
            arm = p.get('arm', '')
            if not any(x in arm for x in ('uniform', 'feedback', 'residual', 'ray_relations', 'consistency', 'single_scale', 'pruned')):
                continue
            target = path.parent
            ck = target/'checkpoints/best.pt'
            summary = target/'summary.json'
            status = target/'status.json'
            d = dict(experiment_id=target.name,source=str(target),arm=arm,seed=p.get('seed'),
                     protocol_sha256=sha(path),sources=p.get('sources',{}),
                     config_sha256=hashlib.sha256(json.dumps(p.get('config'),sort_keys=True).encode()).hexdigest(),
                     epochs=read(summary).get('epochs') if summary.exists() else None,
                     updates=read(summary).get('updates') if summary.exists() else None,
                     checkpoint=str(ck) if ck.exists() else None,
                     checkpoint_sha256=sha(ck) if ck.exists() else None,
                     last_checkpoint_exists=(target/'checkpoints/last.pt').exists(),
                     status=read(status) if status.exists() else None)
            if p.get('subset_smoke') or 'resume_smoke' in str(path):
                reason='engineering smoke only; wrong budget/seed, cannot serve as formal result'
            elif not arm.startswith('uniform'):
                reason='different backbone; background only'
            elif not ck.exists() and not d['last_checkpoint_exists']:
                reason='registered only; no training checkpoint or recoverable state'
            elif summary.exists() and d['epochs']==300 and d['updates']==24900:
                reason='completed candidate; requires full identity/config/GT/prediction verification below'
            else:
                reason='partial candidate; requires recovery audit, never silently restart'
            d.update(reuse=False,reason=reason,missing=[] if ck.exists() else ['training','checkpoint','predictions','metrics'])
            records.append(d)
    return dict(at=now(),roots=[str(p) for p in roots],records=records,
                policy='metadata/checkpoints/prediction hashes only; no GT64/GT128 opened')


def audit_method(m):
    import torch
    import evaluation
    from uniform_models import model_for
    root=Path(m['prediction_root']); manifest=root/'manifest.jsonl'
    rr=rows(manifest); complete=read(root/'complete.json')
    assert sha(manifest)==complete['manifest_sha256']
    evaluation.validate_rows(verified(HISTORY/'registration.json')['cases']['test'],
                             verified(HISTORY/'layouts.json')['test'],rr)
    assert len(rr)==595 and not complete['audit']['denied']
    assert not any(k.startswith('GT64') for k in complete['audit']['counts'])
    for row in rr: assert sha(root/row['prediction'])==row['sha256']
    metric_rows=[r for r in rows(m['source_metrics']) if r['method']==m['key']]
    assert len(metric_rows)==(1190 if m['kind']=='neural' else 595)
    evidence=dict(experiment_id=m['key'],source=m.get('run',str(root)),
        seed=m.get('seed'),arm=m.get('arm'),model=m.get('arm',m['key']),reuse=True,
        checkpoint_sha256=m.get('checkpoint_sha256'),manifest=str(manifest),manifest_sha256=sha(manifest),
        prediction_hashes_verified=595,metrics=str(m['source_metrics']),metrics_sha256=sha(m['source_metrics']),
        metric_rows=len(metric_rows),GT_free_inference_audit=complete['audit'],
        reusable=['sealed predictions','per-case metrics','historical statistics'],missing=[])
    if m['kind']=='neural':
        if m['arm'] not in ('uniform_fusion','uniform','uniform_pruned_observability','uniform_feedback_oari'):
            assert sha(m['checkpoint'])==m['checkpoint_sha256']
            evidence.update(reason='background reference only; reuse already independently accepted historical prediction/scoring evidence',
                previous_acceptance=str(PRIOR/'acceptance.json'),previous_acceptance_sha256=sha(PRIOR/'acceptance.json'))
            return evidence
        target=Path(m['run']);p=verified(target/'protocol.json');verify_hashes(p['sources'])
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
        last=torch.load(target/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        log=rows(target/'metrics.jsonl')
        assert len(log)==300 and [r['epoch'] for r in log]==list(range(1,301))
        assert last['epoch']==300 and last['updates']==24900 and not last['volumetric_gt_used']
        assert ck['config']==p['config']==m['config']==last['config']
        assert ck['protocol_sha256']==last['protocol_sha256']==sha(target/'protocol.json')
        chosen=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        assert ck['epoch']==chosen['epoch']==m['best_epoch'] and not ck['volumetric_gt_used']
        assert ck['selection_metric']==chosen['val']['measurement']['heldout']['relative_l1']
        arm=m['arm']
        if arm=='uniform_pruned_observability':
            baseline=read(PROJECT/'configs/mrgr_uniform_fusion_v1.json')['training_config']
            assert normalized_config(p['config'])==normalized_config(baseline)
            import importlib.util
            spec=importlib.util.spec_from_file_location('stage_pruned',PROJECT/'scripts/pruning_v1/pruned_models.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.ARMS=('pruned_observability','uniform_pruned_observability')
            model=module.model_for(m['config'],arm,'cpu')
        elif arm in ('uniform_fusion','uniform','uniform_feedback_oari'):
            if arm=='uniform_feedback_oari':
                # Stopped research: checkpoint metadata verified, never instantiate,
                # train or infer this architecture in the new stage.
                model=None
            else: model=model_for(m['config'],'uniform','cpu')
        else: model=None  # backgrounds retain previously accepted model audit
        parameters=None
        if model is not None:
            model.load_state_dict(ck['model'],strict=True)
            parameters=sum(x.numel() for x in model.parameters())
        audits=[target/n for n in ('boundary_audit.json','gt_access_audit.json') if (target/n).exists()]
        assert audits
        for path in audits:
            a=read(path)
            assert not a.get('denied',[])
            assert not any(k.startswith('GT64') for k in a.get('counts',{}))
            for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'):
                assert a.get(k,0)==0
        evidence.update(seed=p['seed'],epochs=300,updates=24900,parameters=parameters,
                        selected_epoch=chosen['epoch'],H4=ck['selection_metric'],
                        protocol_sha256=sha(target/'protocol.json'),
                        config_sha256=hashlib.sha256(json.dumps(p['config'],sort_keys=True).encode()).hexdigest(),
                        training_sources_verified=p['sources'],
                        training_boundary_files={str(a):sha(a) for a in audits},
                        strict_state_load=model is not None,
                        reusable=evidence['reusable']+['complete training','checkpoint','H4 selection'])
        del ck,last,model
    return evidence


def register():
    import torch
    import numpy as np
    from uniform_models import model_for
    from geophy_set3d.data.collate import projection_pool_collate
    backend();BASE.mkdir(exist_ok=True)
    if (BASE/'design.json').exists():
        verify_hashes(verified(BASE/'design.json')['training_sources']);return
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT,text=True).strip().splitlines() or not (BASE/'stage_registration.json').exists()
    for name in ('final_lock.json','acceptance.json','model_identity.json','design.json'):
        verified(PRIOR/name)
    verify_hashes(verified(PRIOR/'final_lock.json')['sources'])
    verify_hashes(verified(PRIOR/'final_lock.json')['data_identity'])
    inventory=candidate_inventory()
    inventory_path=BASE/'candidate_inventory.json'
    if inventory_path.exists():
        # Registration may resume after metadata-format inspection; never replace
        # the original comprehensive search evidence.
        inventory=verified(inventory_path)
    else:freeze(inventory_path,inventory)
    references=verified(PRIOR/'reuse_methods.json')
    for m in verified(PRIOR/'final_lock.json')['methods']:
        references.append(dict(m,prediction_root=str(PRIOR/'evaluation/test'/m['key']),
            source_metrics=str(PRIOR/'evaluation/test/metrics.jsonl'),historical=True))
    assert len(references)==15
    audits=[]
    for m in references:
        audits.append(audit_method(m));print('REUSE VERIFIED',m['key'],flush=True)
    main=next(m for m in references if m['key']=='uniform_fusion_seed20261011')
    assert main['checkpoint_sha256']=='64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc'
    assert main['best_epoch']==285
    baseline=main['config']
    controls=[m for m in references if m.get('arm') in ('uniform_fusion','uniform')]
    assert len(controls)==4
    assert all(normalized_config(m['config'])==normalized_config(baseline) for m in controls)
    # Audit each requested task against every discovered uniform same-factor
    # candidate. Smoke seed20, wrong backbone, and empty registrations cannot pass.
    requested=[]
    for key in TASKS:
        old=PRIOR/'training'/key;p=verified(old/'protocol.json');verify_hashes(p['sources'])
        assert normalized_config(p['config'],remove_consistency=p['arm']=='uniform_no_consistency')==normalized_config(baseline)
        assert not (old/'checkpoints/last.pt').exists() and not (old/'summary.json').exists()
        candidates=[r for r in inventory['records'] if r.get('arm')==p['arm'] and r.get('seed')==p['seed']]
        # Do not overwrite/retrain if an unexpected completed/partial candidate is
        # discovered: preserve it and demand audit integration before registration.
        assert all(not r['checkpoint'] and not r['last_checkpoint_exists'] for r in candidates),candidates
        assert normalized_config(p['config'],remove_consistency=p['arm']=='uniform_no_consistency')==normalized_config(baseline)
        requested.append(dict(experiment_id=key,seed=p['seed'],arm=p['arm'],
            prior_protocol=str(old/'protocol.json'),prior_protocol_sha256=sha(old/'protocol.json'),
            config_sha256=hashlib.sha256(json.dumps(p['config'],sort_keys=True).encode()).hexdigest(),
            sources=p['sources'],checkpoint_sha256=None,epochs=0,updates=0,reuse=False,
            reason='all compatible candidates registered but never trained; no legal recovery state',
            candidates=[r['source'] for r in candidates],reusable=['original one-factor definition'],
            missing=['300epoch training','24900updates','checkpoint','595predictions','1190metric rows','paired statistics']))
    freeze(BASE/'reuse_audit.json',dict(at=now(),passed=True,references=audits,requested=requested,
        inventory_sha256=sha(BASE/'candidate_inventory.json'),new_training_required=list(TASKS),
        history=subprocess.check_output(['git','log','--all','--format=%H %s','-60'],cwd=PROJECT,text=True),
        prior_acceptance_sha256=sha(PRIOR/'acceptance.json'),GT64_read_in_audit=False,GT128_read=False))
    freeze(BASE/'reuse_methods.json',references)
    freeze(BASE/'reuse_prediction_seal.json',dict(at=now(),
        manifests={str(Path(m['prediction_root'])/'manifest.jsonl'):sha(Path(m['prediction_root'])/'manifest.jsonl') for m in references},
        all_prediction_hashes_verified=True,previously_used_test=True,GT64_opened_in_current_phase=False))
    repo=dict(at=now(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        origin_main=subprocess.check_output(['git','rev-parse','origin/main'],cwd=PROJECT,text=True).strip(),
        worktree=subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT,text=True),
        resources=resources(),historical_sources_unmodified=True)
    freeze(BASE/'repository_audit.json',repo)
    freeze(BASE/'stage_registration.json',dict(at=now(),experiment_id=BASE.name,
        user_instruction=str(ATTACHMENT),user_instruction_sha256=sha(ATTACHMENT),
        source_commit=repo['head'],tasks=list(TASKS),previous_test_GT64_access_acknowledged=True,
        result_scope='supplementary paired ablation on same previously used35cases; not independent confirmation',
        main_identity=dict(model='MRGR-UniformFusion-v1',seed=20261011,epoch=285,parameters=342725,sha256=main['checkpoint_sha256']),
        OARI_research='stopped; existing sealed metrics retained solely for original comparison family',
        training_GT64_forbidden=True,GT128_always_forbidden=True,
        evaluation_gate='all included training complete, checkpoint hashes frozen, all7x595new predictions sealed, THEN current GT64 scoring',
        selection='strict minimum validation projection H4 over all300epochs; no test selection',
        statistics=verified(PRIOR/'design.json')['evaluation']['statistics'],
        seed_scope=verified(PRIOR/'seed_scope_amendment_v1.json')['statistics'],
        primary_available=11,original_family_denominator=14,unperformed_two_seed_comparisons=3,
        single_seed_policy='three factors exploratory95% case CI only; no manufactured two-seed primary contrast',
        no_design_conflict='Original pre-test seed-scope amendment explicitly defines this handling; latest user repeats same seven seeds',
        acceptance_conditions=['seven exact tasks complete or verified reuse','all300epochs/24900updates',
            'no training GT/test access','595paired predictions per new model','seal before current GT64',
            'independent arrays and bootstrap verification','all outcomes retained','report/export checks',
            'GitHub sync','main identity unchanged','no feedback OARI restart']))
    # Reuse passed engineering/resume tests only after every numerical source hash
    # is verified. No need to retrain smoke jobs, especially stopped OARI arms.
    for name in ('cpu','gpu'):
        check=verified(PRIOR/f'checks/{name}.json');assert check['passed']
        freeze(BASE/f'checks/{name}.json',dict(passed=True,reused=True,at=now(),
            origin=str(PRIOR/f'checks/{name}.json'),origin_sha256=sha(PRIOR/f'checks/{name}.json'),
            numerical_sources_verified=runtime.training_identity(),
            coverage='same exact uniform dispatcher, FP32 objective/backward, complete epoch resume and RNG states'))
    ck=torch.load(main['checkpoint'],map_location='cpu',weights_only=False)
    with AccessGuard(splits=['val'],output=BASE) as guard:
        batch=projection_pool_collate([dataset('val',verified(HISTORY/'layouts.json')['val'][0])[0]])
        batch={k:v for k,v in batch.items() if not k.startswith('hold_')}
        model=model_for(main['config'],'uniform','cpu').eval();model.load_state_dict(ck['model'],strict=True)
        assert sum(p.numel() for p in model.parameters())==342725
        with torch.no_grad():stages=model(batch,return_stages=True)
        assert all(stages[k].shape==(1,1,64,64,64) and torch.isfinite(stages[k]).all() for k in ('C0','C1'))
        assert not guard.denied
    freeze(BASE/'model_identity.json',dict(at=now(),main=main,parameters=342725,
        config_path=str(PROJECT/'configs/mrgr_uniform_fusion_v1.json'),
        config_sha256=sha(PROJECT/'configs/mrgr_uniform_fusion_v1.json'),strict_state_load=True,
        actual_forward_checked=True,audit=guard.summary(),GT128_read=False))
    runs=[]
    for key in TASKS:
        original=PRIOR/'training'/key/'protocol.json';p=copy.deepcopy(verified(original))
        target=BASE/'training'/key
        p['config']['experiment']['output_root']=str(target)
        p.update(registered_at=now(),sources=source_identity(),phase='formal300 supplementary uniform ablation',
                 original_protocol=str(original),original_protocol_sha256=sha(original),
                 stage_registration_sha256=sha(BASE/'stage_registration.json'))
        model=model_for(p['config'],p['arm'],'cpu')
        assert sum(v.numel() for v in model.parameters())==p['parameters']
        import probe_mrgr_multiscale_v1 as probe
        assert probe.state_hash(model.state_dict())==p['initial_model_sha256']
        freeze(target/'protocol.json',p);runs.append(str(target))
    prior_design=verified(PRIOR/'design.json')
    freeze(BASE/'design.json',dict(at=now(),experiment_id=BASE.name,runs=runs,
        training_sources=source_identity(),main_inference_checkpoint=main['checkpoint_sha256'],
        evaluation=prior_design['evaluation'],resource_policy=prior_design['resource_policy'],
        original_design_sha256=sha(PRIOR/'design.json'),stage_registration_sha256=sha(BASE/'stage_registration.json'),
        training=prior_design['training'],OARI_training_allowed=False))
    freeze(BASE/'registration.json',dict(cases={'test':verified(HISTORY/'registration.json')['cases']['test']}))
    freeze(BASE/'layouts.json',dict(test=verified(HISTORY/'layouts.json')['test']))
    write(BASE/'status.json',dict(at=now(),state='registered',new_training_runs=7))
    print('AUDIT + NEW STAGE REGISTERED: 15 reused methods, 7 missing formal trainings',flush=True)


def worker_run(target,stop=300):
    target=Path(target)
    assert str(target) in run_paths()
    assert target.name in TASKS
    assert 'feedback_oari' not in verified(target/'protocol.json')['arm']
    configure().run(target,stop)


def scheduler():
    import controller as old
    old.BASE=BASE;old.NEW_SEEDS=()
    old.effective_runs=lambda *a:run_paths()
    old.deferred_runs=lambda *a:[]
    old.scope_evidence=scope_evidence
    old.training_jobs=lambda:[dict(key=Path(n).name+'_full300',target=Path(n),stop=300) for n in run_paths()]
    # Execute the actual frozen worker via runpy after output-root adaptation.
    # Its real path remains in /proc/cmdline, so the original scheduler's strict
    # live-worker adoption check remains valid after controller restart.
    bootstrap=(f'import sys,runpy;sys.path.insert(0,{str(HERE)!r});'
               f'from stage import configure;configure();'
               f'runpy.run_path({str(PROJECT/"scripts/uniform_final_v1/worker.py")!r},run_name="__main__")')
    old.args=lambda name,*values:[sys.executable,'-c',bootstrap,*map(str,values)] if name=='worker.py' else args(name,*values)
    old.progress=progress
    return old


def progress():
    report=PROJECT/'docs/reports/mrgr_uniform_ablations_v1_progress_zh.md'
    table=['# 均匀融合 MRGR 正式消融进度','',f'更新时间（北京时间）：{now()}','',
        '均匀融合主模型固定为 seed20261011 / epoch285 / 342725参数。反馈 OARI 已停止。',
        '本阶段为已使用35例的补充配对消融，不是独立确认；GT128始终禁用。',
        '复用15个已封存历史方法；两种子无反馈、零残差及三项单种子消融需要补跑。',
        '原14项校正分母保留；三项单种子仅探索性95%案例区间，不替代原双种子比较。','',
        '| 任务 | 状态 | 轮次 | 更新 | 最佳验证H4 | GPU UUID |','|---|---|---:|---:|---:|---|']
    for name in run_paths():
        t=Path(name);s=read(t/'status.json') if (t/'status.json').exists() else {}
        table.append(f'| {t.name} | {s.get("state","queued")} | {s.get("epoch",0)} / 300 | {s.get("updates",0)} | {s.get("best_val_H", "")} | {s.get("gpu_uuid", "")} |')
    state=read(BASE/'status.json')
    table+=['',f'阶段状态：`{state["state"]}`。',f'运行根目录：`{BASE}`。',
        '训练完成并冻结全部新增预测前，不读取本阶段GT64、不报告三维改善结论。','',
        '执行入口：`/scratch/djc/envs/gas-torch/bin/python scripts/uniform_ablations_v1/stage.py controller`。',
        '调度复用原UUID锁/两次15秒空闲检查/worker继承锁FD/2→最多4并发门槛；失败保留证据并阻断后续评分。']
    report.write_text('\n'.join(table)+'\n')


def controller():
    BASE.joinpath('scheduling').mkdir(exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify_hashes(verified(BASE/'design.json')['training_sources'])
            old=scheduler()
            if not (BASE/'training_complete.json').exists():old.train_all()
            import analysis_stage as analysis
            old.run_gpu_once('benchmark',[sys.executable,str(HERE/'stage.py'),'benchmark'],BASE/'benchmark.json')
            evaluation_register()
            for m in verified(BASE/'final_lock.json')['methods']:
                old.run_gpu_once('predict_'+m['key'],[sys.executable,str(HERE/'stage.py'),'predict','--key',m['key']],
                                 BASE/'evaluation/test'/m['key']/'complete.json')
            old.run_gpu_once('seal_score',[sys.executable,str(HERE/'stage.py'),'score'],BASE/'evaluation/test/gt_access.json')
            if not (BASE/'acceptance.json').exists():
                with (BASE/'scheduling/analysis.log').open('a') as log:
                    subprocess.run([sys.executable,str(HERE/'stage.py'),'finish'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert verified(BASE/'acceptance.json')['completed']
            write(BASE/'status.json',dict(at=now(),state='experiments_accepted',controller_pid=os.getpid()))
            progress()
            analysis.sync()
            write(BASE/'status.json',dict(at=now(),state='complete',controller_pid=os.getpid()))
        except BaseException:
            write(BASE/'status.json',dict(at=now(),state='failed',controller_pid=os.getpid(),traceback=traceback.format_exc()))
            progress();raise


class TestGuard(AccessGuard):
    def __init__(self,**kwargs):
        assert set(kwargs.get('splits',()))<={'test'} and set(kwargs.get('gt_splits',()))<={'test'}
        if kwargs.get('gt_splits'):
            seal=verified(BASE/'evaluation/test/all_predictions_sealed.json')
            assert seal['all_predictions_saved_before_GT']
            verify_hashes(seal['manifests'])
        checkpoints={m['checkpoint'] for m in verified(BASE/'final_lock.json')['methods']}
        assert set(kwargs.get('checkpoints',()))<=checkpoints
        kwargs.update(output=BASE,checkpoints=sorted(checkpoints))
        super().__init__(**kwargs)
        for m in verified(BASE/'reuse_methods.json'):
            root=Path(m['prediction_root'])
            for r in rows(root/'manifest.jsonl'):
                self.allowed[str((root/r['prediction']).resolve())]='historical_sealed_prediction'


def install_evaluation():
    import evaluation
    from uniform_models import model_for
    evaluation.OUT=BASE;evaluation.AccessGuard=TestGuard;evaluation.model_for=model_for;evaluation.SCOPE=None
    return evaluation


def evaluation_register():
    import torch
    from uniform_models import model_for
    if (BASE/'final_lock.json').exists():
        verify_hashes(verified(BASE/'final_lock.json')['sources']);return
    assert verified(BASE/'training_complete.json')['complete']
    assert verified(BASE/'checks/statistics_synthetic.json')['passed']
    methods=[];audits=[]
    for name in run_paths():
        target=Path(name);p=verified(target/'protocol.json');s=read(target/'summary.json')
        verify_hashes(p['sources'])
        assert s['completed'] and s['epochs']==300 and s['updates']==24900
        assert sha(target/'checkpoints/best.pt')==s['best_checkpoint_sha256']
        assert sha(target/'checkpoints/last.pt')==s['last_checkpoint_sha256']
        log=rows(target/'metrics.jsonl');assert len(log)==300 and [r['epoch'] for r in log]==list(range(1,301))
        chosen=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        ck=torch.load(target/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        last=torch.load(target/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==chosen['epoch']==s['best_epoch']
        assert ck['selection_metric']==chosen['val']['measurement']['heldout']['relative_l1']
        assert ck['config']==last['config']==p['config']
        assert ck['protocol_sha256']==last['protocol_sha256']==sha(target/'protocol.json')
        assert not ck['volumetric_gt_used'] and not last['volumetric_gt_used']
        assert last['epoch']==300 and last['updates']==24900
        model=model_for(p['config'],p['arm'],'cpu');model.load_state_dict(ck['model'],strict=True)
        assert sum(v.numel() for v in model.parameters())==p['parameters']
        boundary=read(target/'boundary_audit.json');assert not boundary['denied']
        assert not any(k.startswith('GT64') or k=='projection_test' for k in boundary['counts'])
        methods.append(dict(key=target.name,kind='neural',run=name,checkpoint=str(target/'checkpoints/best.pt'),
            checkpoint_sha256=s['best_checkpoint_sha256'],best_epoch=s['best_epoch'],config=p['config'],
            arm=p['arm'],parameters=p['parameters'],seed=p['seed'],historical=False))
        audits.append(dict(key=target.name,summary=s,protocol_sha256=sha(target/'protocol.json'),
            log_sha256=sha(target/'metrics.jsonl'),boundary_audit=boundary,strict_state_load=True))
        del ck,last,model
    assert len(methods)==7
    freeze(BASE/'training_audit.json',dict(at=now(),passed=True,models=audits))
    prior=verified(PRIOR/'final_lock.json');verify_hashes(prior['data_identity'])
    refs=verified(BASE/'reuse_prediction_seal.json');verify_hashes(refs['manifests'])
    freeze(BASE/'final_lock.json',dict(locked_at=now(),methods=methods,sources=source_identity(),
        evidence={str(BASE/n):sha(BASE/n) for n in ('design.json','stage_registration.json','reuse_audit.json',
            'reuse_prediction_seal.json','training_audit.json','registration.json','layouts.json','benchmark.json','checks/statistics_synthetic.json')},
        data_identity=prior['data_identity'],gt64_stat_only=prior['gt64_stat_only'],
        supplementary=True,previously_used35cases=True,GT128_forbidden=True,
        no_test_checkpoint_selection=True,all_training_complete=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['register','checks','controller','worker','benchmark','predict','score','finish','sync','progress'])
    parser.add_argument('--target');parser.add_argument('--stop-after',type=int,default=300);parser.add_argument('--key')
    o=parser.parse_args()
    if o.action=='register':return register()
    if o.action=='worker':return worker_run(o.target,o.stop_after)
    if o.action=='controller':return controller()
    if o.action=='progress':return progress()
    if o.action in ('predict','score'):
        engine=install_evaluation();lock=verified(BASE/'final_lock.json');verify_hashes(lock['sources'])
        if o.action=='predict':return engine.predict_neural(next(m for m in lock['methods'] if m['key']==o.key),'test')
        if not (BASE/'evaluation/test/all_predictions_sealed.json').exists():engine.seal('test',lock['methods'])
        if not (BASE/'evaluation/test/gt_access.json').exists():engine.score('test')
        return
    import analysis_stage
    return getattr(analysis_stage,{'checks':'checks','benchmark':'benchmark','finish':'finish','sync':'sync'}[o.action])()


if __name__=='__main__':main()
