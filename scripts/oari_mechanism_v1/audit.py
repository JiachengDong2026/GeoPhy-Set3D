"""Read-only identity, reuse and resource audit; freeze the minimal new work."""
from runtime import *


def register():
    backend();assert BASE.parent.is_dir();BASE.mkdir(exist_ok=False)
    accepted=verified(HIST.BASE/'acceptance.json');assert accepted['completed']
    verify_hashes(accepted['evidence'])
    registry=verified(HIST.BASE/'experiment_registry.json');verify_hashes(registry['sources'])
    m=verified(HIST.BASE/'reuse_methods.json')[0]
    final=verified(HIST.PRIOR/'model_identity.json');decision=verified(HIST.PRIOR/'model_decision_after_stage_v1.json')
    assert final['main']==m and m['checkpoint_sha256']==HIST.EXPECTED==decision['checkpoint_sha256']
    assert (m['seed'],m['best_epoch'],m['parameters'])==(20261011,285,342725)
    assert sha(m['checkpoint'])==m['checkpoint_sha256']
    with AccessGuard(checkpoints=[m['checkpoint']],output=BASE) as guard:
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
        assert ck['epoch']==285 and ck['config']==m['config'] and not ck['volumetric_gt_used']
        model=HIST.model_for(m['config'],'uniform','cpu').eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True)
        assert sum(p.numel() for p in model.parameters())==342725 and HIST.state_hash(model)==registry['main_state_sha256']
        assert ck['protocol_sha256']==sha(Path(m['run'])/'protocol.json') and not guard.denied
        state_sha256=HIST.state_hash(model);del ck,model
    cases=verified(HIST.BASE/'cases.json');layouts=verified(HIST.BASE/'layouts.json')['test'];assert len(cases)==35 and len(layouts)==17
    export=read(PROJECT/'docs/reports/artifacts/mrgr_geometry_utilization_v1/export_manifest.json')
    for n in ('paired_metrics.csv','per_case_metrics.csv','case_aggregates.csv','statistics.json'):
        path=PROJECT/'docs/reports/artifacts/mrgr_geometry_utilization_v1'/n
        assert sha(path)==export['files'][str(path.relative_to(PROJECT))]['sha256']
        assert sha(path)==sha(HIST.BASE/n)
    candidates=[str(p.relative_to(HIST.BASE)) for p in HIST.BASE.rglob('*') if p.is_file() and any(k in p.name.lower() for k in ('attention','bias','intermediate','context'))]
    assert not candidates,'Audit saved tensors before introducing duplicate forward work'
    plans=[];historical_records=[];identity_checks=[]
    oldroot=Path(m['prediction_root']);baseline=rows(oldroot/'manifest.jsonl')
    seal=verified(HIST.BASE/'all_predictions_sealed.json')
    for li,setting in enumerate(layouts):
        root=HIST.BASE/'predictions'/f'layout{li:02d}';man=root/'manifest.jsonl'
        assert sha(man)==seal['manifest_hashes'][str(man)]
        rr=rows(man)
        if setting['draw']!=0:continue
        for ci in (0,17,34):
            case=cases[ci];original=next(r for r in baseline if (r['condition'],r['draw'],r['case_id'])==(setting['condition'],0,case))
            pp=next(p for p in verified(HIST.BASE/'permutation_maps.json') if (p['condition'],p['draw'],p['case_id'])==(setting['condition'],0,case))
            plans.append(dict(sample_index=len(plans),layout_index=li,case_index=ci,case_id=case,condition=setting['condition'],draw=0,views=setting['views'],permutation=pp['permutation_maps'][0],input_ids=setting['input_ids'][case],heldout_ids=setting['heldout_ids'][case]))
            for mode,row,path in [('A',original,oldroot/original['prediction'])]+[(mode,next(r for r in rr if r['case_id']==case and r['experiment']==mode and r['permutation_index']==0),None) for mode in ('P1','P2')]:
                if path is None:path=root/row['prediction']
                assert sha(path)==row['sha256']
                if mode!='A':assert row['checkpoint_sha256']==HIST.EXPECTED
                historical_records.append(dict(sample_index=plans[-1]['sample_index'],mode=mode,path=str(path.resolve()),sha256=row['sha256']))
        identity_checks.append(dict(layout_index=li,condition=setting['condition'],draw=setting['draw'],manifest_sha256=sha(man),records=len(rr)))
    assert len(plans)==15 and len(historical_records)==45
    # Also confirm the prior training ablation is officially registered; no rerun.
    ablations=verified(HIST.ABLATIONS/'final_lock.json')['methods']
    ab=next(a for a in ablations if a['key']=='uniform_no_ray_relations_seed20261011')
    assert sha(ab['checkpoint'])==ab['checkpoint_sha256']
    available=dict(B='all35cases and17layouts A/P1/P2 C0/C1 scalar metrics, paired differences, permutations and prediction hashes',
                   A='historical actual_bias_mean_abs only; no content logits, valid-pair bias distribution or no-bias softmax statistics',
                   C='no saved attention matrices or context features; historical45sample outputs available for consistency checks',
                   no_ray_relations='existing trained ablation registered and reused as context; no training or new metrics')
    taskstate=read(TRAIN_BASE/'status.json');resources_before=resources()
    freeze(BASE/'sampled_historical_predictions.json',historical_records)
    freeze(BASE/'dependencies.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)))
    freeze(BASE/'audit.json',dict(at=now(),passed=True,model=m,strict_official_factory_load=True,state_sha256=state_sha256,final_identity_sha256=sha(HIST.PRIOR/'model_identity.json'),
        prior_acceptance_sha256=sha(HIST.BASE/'acceptance.json'),available=available,missing_saved_attention=True,candidates=candidates,
        planned_forwards=45,GT64_needed=False,GT128_needed=False,training_state=taskstate,resources=resources_before,
        original_manifest_identities=identity_checks,training_ablation_identity={k:v for k,v in ab.items() if k!='config'},
        source_metrics={str(HIST.BASE/n):sha(HIST.BASE/n) for n in ('paired_metrics.csv','per_case_metrics.csv','case_aggregates.csv','statistics.json')},
        all_other_existing_predictions_reused=True))
    freeze(BASE/'protocol.json',dict(at=now(),id=BASE.name,model=m,checkpoint_sha256=HIST.EXPECTED,sources=sources(),sample_plans=plans,
        instructions_sha256=sha(Path('/home/djc/.codex/attachments/55ef9df4-549c-4d9a-b8fc-68a000ac2485/已粘贴的文本.txt')),
        fusion='Uniform Fusion',training_allowed=False,backward_allowed=False,optimizer_allowed=False,new_checkpoints=False,model_structure_change=False,
        selection='indices0,17,34 fixed independently of outcomes; all5conditions draw0; first registered P1 permutation only; no outcome filtering',
        new_forwards=45,forward_modes=['A','P1','P2'],engineering='each sampled hooked A vs sealed unhooked historical A, atol1e-5 and relative mean abs1e-4; P1/P2 also vs same historical intervention',
        mask='actual valid=visibility AND view_mask; exclude invalid pairs/all-invalid voxels; softmax over candidate key view; valid query rows only',
        quantiles='signed and absolute P50/P90/P95/P99; exact CPU quantiles per case/head/ordered valid view-pair; no GPU tensor accumulation',
        A=['bias/content mean,std,RMS,meanabs,quantiles','RMS and std ratios','key-centered bias RMS','softmax no-bias vs bias JS/L1/entropy','relation variation vs bias correlation'],
        B='reuse historical metrics only; per identical case/layout/draw/permutation attenuation=deltaC0-deltaC1; average permutations within draw then draws within case; no rescore or new inference',
        C=['attention L1/L2/JS/entropy','per-key mean weight change','bias change','same voxel attention vs context feature change correlation'],
        P1='reuse original Intervention prehook, relation axes only; original local pairmask reapplied',
        P2='reuse original Intervention first-affine replacement Wobs*obs+b, removing Wrel*r; not neutral physical angles',
        statistics=dict(seed=SEED,resamples=20000,unit='case',B_n=35,AC_n=3,all_exploratory=True,no_seed_population_variance=True,no_voxel_pseudoreplication=True),
        GPU='one independently idle UUID with existing exclusive lock and15s stable snapshots; never signal/change existing training;2CPUthreads; fallback wait',
        GT64_read=False,GT128_forbidden=True,previously_used_test=True,independent_confirmation=False))
    for n in ('protocol.json','protocol.json.sha256'):(BASE/n).chmod(0o444)
    lines=['# OARI三项机制诊断：审计计划','',f'模型：正式Uniform Fusion MRGR，seed20261011/epoch285/342725参数，SHA `{HIST.EXPECTED}`。','',
      '| 诊断 | 可复用 | 缺少 | 新前向 | 资源 | GT |','|---|---|---|---:|---|---|',
      '| A偏置有效尺度 | 历史偏置绝对均值摘要、真实输入 | content logits/有效pair分布/softmax差异 | 与C共用45 | 一张空闲GPU、CPU分块汇总 | 不读取 |',
      '| B反馈前后差值 | 全35例×17布局A/P1/P2 C0/C1封存指标 | 只需重新配对统计，无需重评分 | 0 | CPU | 不读取 |',
      '| C注意力关系依赖 | 历史P1/P2定义与对应封存预测 | attention/context未保存 | 与A共用45 | 一张空闲GPU | 不读取 |','',
      '固定案例索引0/17/34，五类布局draw0，A/P1/P2各15次。P1仅第一条既登记排列，不据结果选择。样本结果不推断全35例。',
      'GeometryLifting两正式训练任务保持优先，GPU0/1正在运行；只使用经实时双快照及排他锁确认的空闲GPU。新任务不训练、不反传、不更新模型、不读取GT。',
      '先从历史指标完成B，再补缺失A/C并核验与历史A/P1/P2输出一致。所有历史文件不覆盖。']
    (BASE/'audit_plan_zh.md').write_text('\n'.join(lines)+'\n')
    print('AUDIT PASS; B no new inference; A/C combined45 frozen forwards; no GT access',flush=True)


if __name__=='__main__':register()
