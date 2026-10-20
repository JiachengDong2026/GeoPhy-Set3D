"""Metadata-only data audit and pre-training protocol registration."""
from runtime import *
import copy
import torch
from uniform_models import model_for
import worker
import evaluation as old_eval

def audit_data():
    source = rows(DATA/'source_index.jsonl')
    rawroot = Path('/data/djc/GeoPhy-Set3D/data/raw/Fluent')
    manifests = sorted(rawroot.rglob('manifest.jsonl'))
    raw = [r for p in manifests for r in rows(p)]
    ids = {r['sample_id'] for r in raw}
    assert len(source)==len(raw)==len(ids)==237
    assert ids=={r['sample_id'] for r in source}
    assert all(r['frame_index']==60 and r['physical_time_s']==2. for r in raw)
    processed = sorted(p.name for p in DATA.parent.iterdir() if p.is_dir())
    assert processed==['geophy_projection_pool_v1'], processed
    return dict(at=now(), scope='project-owned raw/processed data roots and registered metadata; no GT array opened',
        raw_manifests={str(p):sha(p) for p in manifests},
        source_index_sha256=sha(DATA/'source_index.jsonl'), total=237,
        cases_by_split={s:[r['case_id'] for r in source if r['split']==s] for s in ('train','val','test')},
        raw_unregistered_cases=sorted(ids-{r['sample_id'] for r in source}),
        available_processed_pools=processed, independent_confirmation_available=False,
        reason='all237available cases already belong to166train/36used-development/35used-test; Pool B changes geometry, not case identity',
        data_roots=[str(rawroot),str(DATA.parent)], GT64_opened=False, GT128_opened=False,
        historical_access_evidence={str(p):sha(p) for p in [
            HISTORY/'evaluation/test/gt_access.json',
            BASE.parent/'mrgr_extension_test35_v1/evaluation/test/gt_access.json']} )

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'design.json').exists():
        verify_hashes(verified(BASE/'design.json')['training_sources'])
        return
    lock=verified(HISTORY/'final_lock.json')
    verify_hashes(lock['data_identity'])
    if not (BASE/'data_audit.json').exists():freeze(BASE/'data_audit.json',audit_data())
    else:verified(BASE/'data_audit.json')
    wanted=('MRGR','LR4E4','multiscale_seed20261019','uniform_fusion_seed20261011',
            'uniform_fusion_seed20261019','NNLS-NL','Tikhonov','TV','SART')
    reused=[]; audits=[]
    for m in lock['methods']:
        if m['key'] not in wanted: continue
        target,rr=old_eval.verify_predictions(m,'test',arrays=False)
        reused.append(dict(m,prediction_root=str(target),source_metrics=str(HISTORY/'evaluation/test/metrics.jsonl'),historical=True))
        if m['kind']=='neural':
            protocol=read(Path(m['run'])/'protocol.json')
            ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
            assert ck['config']==m['config'] and ck['epoch']==m['best_epoch'] and not ck['volumetric_gt_used']
            assert ck['protocol_sha256']==sha(Path(m['run'])/'protocol.json')
            assert sha(m['checkpoint'])==m['checkpoint_sha256']
            if m['arm']=='uniform_fusion':
                verify_hashes(protocol['sources'])
                model=model_for(m['config'],'uniform','cpu')
                model.load_state_dict(ck['model'],strict=True)
                assert sum(p.numel() for p in model.parameters())==342725
                last=torch.load(Path(m['run'])/'checkpoints/last.pt',map_location='cpu',weights_only=False)
                log=rows(Path(m['run'])/'metrics.jsonl')
                assert last['epoch']==300 and last['updates']==24900 and len(log)==300
                chosen=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
                assert chosen['epoch']==m['best_epoch'] and ck['selection_metric']==chosen['val']['measurement']['heldout']['relative_l1']
                audits.append(dict(key=m['key'],parameters=342725,epochs=300,updates=24900,
                    best_epoch=m['best_epoch'],best_H4=ck['selection_metric'],checkpoint_sha256=sha(m['checkpoint']),
                    protocol_sha256=sha(Path(m['run'])/'protocol.json'),strict_state_load=True,
                    initial_architecture='historical multiscale MRGR, only fusion replaced by parameter-free UniformFusion'))
    assert len(reused)==9 and len(audits)==2
    pruning=BASE.parent/'mrgr_extension_test35_v1'
    for m in verified(pruning/'final_lock.json')['methods']:
        if m['arm']!='uniform_pruned_observability':continue
        target=pruning/'evaluation/test'/m['key']
        rr=rows(target/'manifest.jsonl')
        assert sha(target/'manifest.jsonl')==read(target/'complete.json')['manifest_sha256']
        old_eval.validate_rows(verified(HISTORY/'registration.json')['cases']['test'],verified(HISTORY/'layouts.json')['test'],rr)
        for r in rr:assert sha(target/r['prediction'])==r['sha256']
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        reused.append(dict(m,prediction_root=str(target),source_metrics=str(pruning/'evaluation/test/metrics.jsonl'),historical=True))
    assert len(reused)==11
    # Compare actual config, excluding only run path/name and seed.
    configs=[]
    for m in reused:
        if m.get('arm')=='uniform_fusion':
            c=copy.deepcopy(m['config']);c.pop('experiment');c['training'].pop('seed');configs.append(c)
    assert configs[0]==configs[1]
    freeze(BASE/'reuse_methods.json',reused)
    freeze(BASE/'reuse_audit.json',dict(at=now(),uniform_training=audits,
        all11_prediction_manifests_and_arrays_hashed=True,identical_cases_layouts_and_input_ids=True,
        uniform_configs_equal_except_seed_output=True,shared_nonlinear_scoring_operator=True,
        SART_caveat='linear nodal mole-fraction inversion operator; nonlinear mass-fraction P(C) used for common projection scoring',
        sources={str(HISTORY/'final_lock.json'):sha(HISTORY/'final_lock.json'),
                 str(pruning/'final_lock.json'):sha(pruning/'final_lock.json')},
        reuse_policy='existing used-test predictions; no repeat inference or independent-test claim'))
    runs=[]
    for seed in NEW_SEEDS:
        for arm in ('uniform','uniform_feedback_oari'):
            runs.append(str(worker.register(arm,seed)))
    for arm in ARMS[2:]:
        for seed in ABLATION_SEEDS:runs.append(str(worker.register(arm,seed)))
    freeze(BASE/'design.json',dict(at=now(),experiment_id='mrgr_uniform_final_v1',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        side_branch='experiment/uniform-feedback-oari-v1',
        side_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SIDE,text=True).strip(),
        model_id='MRGR-UniformFusion',runs=runs,new_seeds=list(NEW_SEEDS),
        ablation_seeds=list(ABLATION_SEEDS),training_sources=training_identity(),
        main_inference_seed=20261011,
        main_inference_checkpoint='64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc',
        main_checkpoint_policy='first chronologically registered historical uniform seed20261011, epoch285; fixed now, never selected by test ranking; all4 seeds reported without ensemble',
        provenance='uniform direction selected after historical test comparisons; all35-case findings supplementary',
        training='300epochs/24900updates; unchanged original FP32/TF32off/batch2/AdamW2e-4/wd1e-4/cosine300/clip1;4persistentfork workers',
        ablations='only target factor changed; descriptors retained except reused uniform_pruned; no-feedback removes12512parameters; singlescale matched parameters but changes compute/receptive field',
        pilot=dict(epochs=20,updates=1660,seeds=list(NEW_SEEDS),selection_data='validation projections only; no volume GT',
            gate='engineering checks pass; all losses/H4 finite; treatment best H4 <=1.10*matched control best H4; no OOM; treatment epoch seconds<=2*control; peak allocated<=20GiB',
            action='each seed judged separately; prefix retained with full optimizer/scheduler/RNG, continue to300 unchanged; no replacements or tuning; failures archived and not promoted',
            motivation='10% H4 margin/2x cost excludes gross engineering failure, not a volume-accuracy claim'),
        evaluation=dict(cases=35,layouts=17,output='64^3 t2s original XYZ endpoint grid',primary='ordinary fixedV4 C1 volume.relative_l1',
            secondary='clusteredV4 five draws first averaged within case; L1/L2/GTpeak ratio/peak error/centroid mm/Dice50/C0-C1/all layouts/worst cases',
            boundary='no test projections or GT in training; all methods and complete predictions sealed before any current-phase GT64; GT128 always forbidden',
            data='same used35case supplementary evaluation; no new independent confirmation available',
            statistics=dict(seed=20261020,resamples=20000,unit='case, never draw',
                primary_family=14,primary_adjustment='Bonferroni simultaneous paired bootstrap intervals, 14 predeclared comparisons',
                comparisons=['uniform4seed_case_mean-minus-MRGR','uniform4seed_case_mean-minus-LR4E4','uniform4seed_case_mean-minus-NNLS-NL',
                    'uniform4seed_case_mean-minus-Tikhonov','uniform4seed_case_mean-minus-TV','uniform4seed_case_mean-minus-SART',
                    'uniform2seed-minus-matchedGCA2seed','no_feedback-minus-uniform2seed','zero_residual-minus-uniform2seed',
                    'no_ray_relations-minus-uniform2seed','no_consistency-minus-uniform2seed','single_scale-minus-uniform2seed',
                    'pruned_descriptor-minus-uniform2seed','feedbackOARI2seed-minus-matchedUniform2seed'],
                secondary_intervals='unadjusted95%, exploratory; seed-specific and seed-average case differences retained; no training-seed population CI'),
            figures='fixed testcase indices0/17/34; ordinaryV4 and clustereddraw0; peak-zXY slice plus center-yXZ slice; shared GT asinh/linear scales, no per-method normalization'),
        resource_policy=dict(initial_concurrent=2,maximum_after_audit=4,stable_idle_polls=2,poll_seconds=15,
            thresholds=dict(utilization_max=5,used_MiB_below=100,free_MiB_min=20000),locks='shared GPU UUID flock',
            failures='stop affected phase, preserve logs; no automatic precision/budget/model changes')))
    write(BASE/'status.json',dict(at=now(),state='registered',new_training_runs=len(runs)))
    print('Registered14runs; reused11models; data audit and evaluation rules frozen before training',flush=True)

if __name__=='__main__':main()
