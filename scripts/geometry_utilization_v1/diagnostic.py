"""Frozen-weight geometry diagnostics; no optimizer, training or model edits."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'scripts/uniform_final_v1'))
from runtime import *
import argparse
import itertools
import math
import time
import fcntl
import traceback
import numpy as np
import torch
from uniform_models import model_for
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch

PRIOR=BASE
ABLATIONS=BASE.parent/'mrgr_uniform_ablations_v1'
BASE=BASE.parent/'mrgr_geometry_utilization_v1'
ARTIFACT=PROJECT/'docs/reports/artifacts/mrgr_geometry_utilization_v1'
REPORT=PROJECT/'docs/reports/mrgr_geometry_utilization_diagnostic_v1_zh.md'
INSTRUCTION=Path('/home/djc/.codex/attachments/5892c621-4edc-4f61-8240-04b0ead09086/已粘贴的文本.txt')
EXPECTED='64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc'
EPS=1e-6
METRICS=('relative_l1','relative_l2','dice50','absolute_peak_ratio_error','centroid_distance_mm','gt_peak_location_ratio','peak_ratio')


def identities():
    d=training_identity()
    d.update({str(p):sha(p) for p in HERE.glob('*.py')})
    d.update({str(p):sha(p) for p in (PROJECT/'scripts/uniform_final_v1').glob('*.py')})
    return d


def state_hash(model):
    import probe_mrgr_multiscale_v1 as probe
    return probe.state_hash(model.state_dict())


def main_method():return verified(BASE/'reuse_methods.json')[0]


class Guard(AccessGuard):
    def __init__(self,**kwargs):
        assert set(kwargs.get('splits',()))<={'val','test'}
        assert set(kwargs.get('gt_splits',()))<={'test'}
        if kwargs.get('gt_splits'):
            seal=verified(BASE/'all_predictions_sealed.json')
            assert seal['all_new_predictions_saved_before_current_GT64']
            verify_hashes(seal['manifest_hashes'])
        kwargs.update(output=BASE,checkpoints=[main_method()['checkpoint']])
        super().__init__(**kwargs)
        for m in verified(BASE/'reuse_methods.json'):
            root=Path(m['prediction_root'])
            for r in rows(root/'manifest.jsonl'):
                self.allowed[str((root/r['prediction']).resolve())]='historical_sealed_prediction'


def permutations(v):
    pool=[p for p in itertools.permutations(range(v)) if p!=tuple(range(v))]
    rng=np.random.default_rng(20261021+v)
    return [list(pool[i]) for i in rng.choice(len(pool),min(10,len(pool)),replace=False)]


def mapping(batch,perm):
    """Reorder all input view fields, retaining padding slots and all H fields."""
    assert batch['view_mask'].shape[0]==1
    ids=batch['view_mask'][0].nonzero().flatten()
    assert len(ids)==len(perm)
    index=torch.arange(batch['view_mask'].shape[1],device=ids.device)
    index[ids]=ids[torch.as_tensor(perm,device=ids.device)]
    out={}
    for k,v in batch.items():
        if k.startswith('input_') or k=='view_mask':
            assert isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[1]==len(index),k
            out[k]=v[:,index]
        else:out[k]=v
    return out,index


class Intervention:
    """Change only real OARI call inputs/affine activations with removable hooks.

    P2 removes W_rel*r from first affine, preserving W_obs*obs+b. Zero
    contribution is neutral in the learned preactivation space; it is NOT a
    physically neutral zero angle/cosine. No replacement physical geometry is
    fabricated; all observability and other pathways remain byte-identical.
    """
    def __init__(self,model,mode='A',perm=None,expected=None):
        self.mode=mode;self.perm=perm;self.expected=expected
        self.capture={};self.calls=0;self.linear_calls=0
        self.model=model
    def before(self,module,inputs):
        features,geometry,visibility,mask,relations,obs,*rest=inputs
        assert features.shape[0]==1
        valid=visibility & mask[:,:,None,None,None]
        local=valid.permute(0,2,3,4,1).reshape(-1,valid.shape[1])
        pair=local[:,:,None]&local[:,None,:]
        self.calls+=1
        self.capture.update(relations=relations.detach().clone(),observability=obs.detach().clone(),
                            valid=local.detach().clone())
        if self.expected is not None:
            index=torch.as_tensor(self.perm,device=relations.device)
            torch.testing.assert_close(relations,self.expected['relations'][:,index][:,:,index],rtol=0,atol=0)
            torch.testing.assert_close(local,self.expected['valid'][:,index],rtol=0,atol=0)
            torch.testing.assert_close(obs,self.expected['observability'],rtol=2e-5,atol=2e-6)
        if self.mode=='P1':
            index=torch.arange(mask.shape[1],device=mask.device)
            ids=mask[0].nonzero().flatten()
            index[ids]=ids[torch.as_tensor(self.perm,device=mask.device)]
            changed=relations[:,index][:,:,index]
            # Compare actual tensors, not a configuration flag. Original
            # interaction reapplies its unchanged LOCAL visibility pair mask.
            dif=(changed-relations)[pair]
            self.capture['relation_mean_abs_change']=float(dif.abs().mean()) if dif.numel() else 0.
            return (features,geometry,visibility,mask,changed,obs,*rest)
        return None
    def linear(self,module,inputs,output):
        self.linear_calls+=1
        x=inputs[0]
        self.capture['first_affine_input_shape']=list(x.shape)
        if self.mode=='P2':
            replacement=torch.nn.functional.linear(x[...,3:],module.weight[:,3:],module.bias)
            contribution=torch.nn.functional.linear(x[...,:3],module.weight[:,:3],None)
            torch.testing.assert_close(output,replacement+contribution,rtol=2e-5,atol=2e-6)
            self.capture['removed_affine_mean_abs']=float(contribution.abs().mean())
            return replacement
        return None
    def bias(self,module,inputs,output):
        self.capture['actual_bias_mean_abs']=float(output.abs().mean())
        self.capture['actual_bias_finite']=bool(torch.isfinite(output).all())
    def __enter__(self):
        self.hooks=[self.model.interaction.register_forward_pre_hook(self.before),
            self.model.interaction.pairwise_bias[0].register_forward_hook(self.linear),
            self.model.interaction.pairwise_bias.register_forward_hook(self.bias)]
        return self
    def __exit__(self,*args):
        for hook in self.hooks:hook.remove()
        if args[0] is None:assert self.calls==self.linear_calls==1
    def audit(self):
        return dict(interaction_calls=self.calls,first_affine_calls=self.linear_calls,
            **{k:v for k,v in self.capture.items() if not isinstance(v,torch.Tensor)})


def differences(p,q):
    p=p.astype(np.float64);q=q.astype(np.float64);d=np.abs(p-q)
    return dict(mean_abs=float(d.mean()),max_abs=float(d.max()),
        relative_prediction_difference=float(d.mean()/(np.abs(q).mean()+EPS)))


def flatten_old(row):
    return dict(relative_l1=row['volume']['relative_l1'],relative_l2=row['volume']['relative_l2'],
        dice50=row['morphology']['dice_0.5'],absolute_peak_ratio_error=row['morphology']['absolute_peak_ratio_error'],
        centroid_distance_mm=row['morphology']['centroid_distance_mm'],
        gt_peak_location_ratio=row['morphology']['gt_peak_location_ratio'],peak_ratio=row['morphology']['peak_ratio'])


def register():
    backend();BASE.mkdir(exist_ok=True)
    assert not (BASE/'experiment_registry.json').exists(),'Preserve existing stage registration'
    verify_hashes(verified(ABLATIONS/'final_lock.json')['sources'])
    accepted=verified(ABLATIONS/'acceptance.json');assert accepted['completed']
    verify_hashes(accepted['evidence'])
    original=verified(PRIOR/'reuse_methods.json')
    wanted=['uniform_fusion_seed20261011','uniform_no_ray_relations_seed20261011',
            'uniform_pruned_observability_seed20261011','uniform_pruned_observability_seed20261019']
    candidates=original+verified(ABLATIONS/'final_lock.json')['methods']
    reused=[];audits=[]
    import evaluation
    settings=verified(HISTORY/'layouts.json')['test'];cases=verified(HISTORY/'registration.json')['cases']['test']
    for key in wanted:
        m=dict(next(m for m in candidates if m['key']==key))
        if 'prediction_root' not in m:
            m.update(prediction_root=str(ABLATIONS/'evaluation/test'/key),source_metrics=str(ABLATIONS/'evaluation/test/metrics.jsonl'))
        root=Path(m['prediction_root']);rr=rows(root/'manifest.jsonl');completed=read(root/'complete.json')
        assert sha(root/'manifest.jsonl')==completed['manifest_sha256']
        evaluation.validate_rows(cases,settings,rr)
        assert len(rr)==595 and completed['fields']==1190
        assert not completed['audit']['denied']
        for r in rr:assert sha(root/r['prediction'])==r['sha256']
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        scalar=[r for r in rows(m['source_metrics']) if r['method']==key]
        assert len(scalar)==1190
        reused.append(m);audits.append(dict(key=key,checkpoint_sha256=m['checkpoint_sha256'],
            config=m['config'],manifest_sha256=sha(root/'manifest.jsonl'),predictions_hashed=595,
            scalar_rows=1190,metrics_sha256=sha(m['source_metrics']),source_metrics=m['source_metrics'],
            complete_audit=completed['audit'],training_repeated=False,inference_repeated=False))
    freeze(BASE/'reuse_methods.json',reused)
    m=reused[0];assert m['checkpoint_sha256']==EXPECTED and m['best_epoch']==285
    assert read(PROJECT/'configs/mrgr_uniform_fusion_v1.json')['training_config']==m['config']
    ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
    assert ck['epoch']==285 and ck['config']==m['config'] and not ck['volumetric_gt_used']
    model=model_for(m['config'],'uniform','cpu').eval();model.load_state_dict(ck['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==342725
    protocol=verified(Path(m['run'])/'protocol.json');verify_hashes(protocol['sources'])
    assert sha(Path(m['run'])/'protocol.json')==ck['protocol_sha256']
    for p in model.parameters():p.requires_grad_(False)
    initial=state_hash(model)
    freeze(BASE/'reuse_audit.json',dict(at=now(),passed=True,methods=audits,
        main=dict(seed=20261011,epoch=285,parameters=342725,checkpoint_sha256=EXPECTED,state_sha256=initial,strict_load=True),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        source_status=subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT,text=True),
        accepted_stage_sha256=sha(ABLATIONS/'acceptance.json'),GT64_read=False,GT128_read=False))
    freeze(BASE/'layouts.json',dict(test=settings));freeze(BASE/'cases.json',cases)
    plans=[]
    for s in settings:
        for i,case in enumerate(cases):
            maps=permutations(s['views'])
            plans.append(dict(case_id=case,case_index=i,condition=s['condition'],draw=s['draw'],
                views=s['views'],input_ids=s['input_ids'][case],heldout_ids=s['heldout_ids'][case],
                permutation_maps=maps,count=len(maps),nonidentity_available=math.factorial(s['views'])-1,
                insufficient_for10=len(maps)<10,padding_permuted=False))
    freeze(BASE/'permutation_maps.json',plans)
    common=dict(at=now(),seed=20261021,per_view_seed='20261021+V; deterministic sample without replacement',
        permutation_maps_sha256=sha(BASE/'permutation_maps.json'),padding_excluded=True,
        insufficient='V3 only5 nonidentity distinct maps; run all5, never duplicate to claim10')
    freeze(BASE/'permutation_invariance_config.json',dict(common,experiment='P0',
        intervention='permute all input_* and view_mask together; internal lifting/ray/visibility regenerated; Rprime=R[:,P][:,:,P]',
        baseline='sealed A original predictions',epsilon=EPS,
        repeat_baseline_cases=[0,17,34],repeat_count=2,
        repeat_reason='only three fixed cases per each17layouts replayed twice to calibrate repeatability and historical/current numeric drift; no full A prediction rerun',
        absolute_prediction_tolerance=1e-5,relative_prediction_tolerance=1e-4))
    freeze(BASE/'relation_mismatch_config.json',dict(common,experiment='P1',
        intervention='prehook at actual interaction.forward; only relations [BXYZ,V,V,3] axes1/2 permuted',
        local_visibility='unchanged original per-voxel pair masks reapplied; document visibility-mask mismatch as OOD source',
        baseline='sealed A, no rerun'))
    freeze(BASE/'relation_neutralization_config.json',dict(at=now(),experiment='P2',
        actual_affine='z=W_rel*r + W_obs*o + b -> zneutral=W_obs*o+b at pairwise_bias[0] forward output',
        semantic='zero additive explicit relation contribution in learned affine coordinates; NOT zero physical angles/direction differences',
        equal_zero_feature_input='algebraically matches registered no_ray_relations input zero at first3channels; physical-neutrality NOT claimed',
        unchanged=['observability token embedding','observability pair input','all weights','module structure','geometric lifting','measurements','geometry','visibility','uniform fusion','physics projector','residual feedback'],
        numerical_assertion='original affine equals retained part plus removed part',
        limitation='removes all3channels, including self cos1 cue; inference distribution shift possible; not a retrained structural ablation'))
    registry=dict(at=now(),id=BASE.name,main_checkpoint_sha256=EXPECTED,main_state_sha256=initial,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        sources=identities(),instruction_sha256=sha(INSTRUCTION),training_allowed=False,
        test_previously_used=True,independent_confirmation=False,GT128_forbidden=True,
        all_new_predictions_sealed_before_current_GT64=True,case_count=35,layout_count=17,combinations=595,
        permutations_per_input='10 distinct for V4/V6; exhaustive5 for V3',
        new_predictions=sum(2*p['count']+1+(2 if p['case_index'] in (0,17,34) else 0) for p in plans),
        metrics=list(METRICS),primary_layout='Val-A-V4',stages=['C0','C1'],
        statistics=dict(seed=20261020,resamples=20000,unit='case',
            reduce='within case average permutations, then average draws within condition; never treat permutations/draws as independent cases',
            intervals='unadjusted95% paired case bootstrap; ALL diagnostics exploratory, no new primary family and no modification of original Bonferroni14',
            per_permutation='report all fixed maps and mean/std/median/min/max across maps and cases'),
        interpretations=['P0 invariance not geometry benefit','P1 association diagnostic not sufficient proof of generalizable geometry',
            'P2 learned-neutral not physically neutral ray','training ablation and inference intervention separate',
            'other geometry pathway compensation is hypothesis, not identified causal explanation'],
        budget='no training, no optimizer, no backward, no weight edits; frozen eval and torch.no_grad')
    freeze(BASE/'experiment_registry.json',registry)
    write(BASE/'status.json',dict(at=now(),state='registered',new_predictions=registry['new_predictions']))
    print('REGISTERED',registry['new_predictions'],'new diagnostic predictions; A reused; noGT/no training',flush=True)


def cpu_checks():
    backend();r=verified(BASE/'experiment_registry.json');verify_hashes(r['sources'])
    m=main_method();ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
    model=model_for(m['config'],'uniform','cpu').eval();model.load_state_dict(ck['model'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    initial=state_hash(model);checks=[]
    with Guard(splits=['val']) as guard,torch.no_grad():
        sample=dataset('val',verified(HISTORY/'layouts.json')['val'][0])[0]
        batch=projection_pool_collate([sample]);batch={k:v for k,v in batch.items() if not k.startswith('hold_')}
        with Intervention(model) as normal:ref=model(batch,return_stages=True)
        perm=permutations(4)[0]
        permuted,index=mapping(batch,perm)
        with Intervention(model,'P0',perm,normal.capture) as h:other=model(permuted,return_stages=True)
        for stage in ('C0','C1'):torch.testing.assert_close(ref[stage],other[stage],atol=1e-5,rtol=1e-4)
        checks.append(dict(P0=True,Rprime_PRPt_exact=True,observability_invariant=True))
        with Intervention(model,'P1',perm) as h:wrong=model(batch,return_stages=True)
        assert h.capture['relation_mean_abs_change']>0
        checks.append(dict(P1=True,actual_tensor_change=h.capture['relation_mean_abs_change'],changed_output=float((wrong['C1']-ref['C1']).abs().max())))
        with Intervention(model,'P2') as h:neutral=model(batch,return_stages=True)
        # Independent implementation: explicitly zero only3relation features at
        # the original first-affine input, with the rest completely untouched.
        def zero3(module,args):
            x=args[0];return (torch.cat((torch.zeros_like(x[...,:3]),x[...,3:]),-1),)
        hook=model.interaction.pairwise_bias[0].register_forward_pre_hook(zero3)
        try:equivalent=model(batch,return_stages=True)
        finally:hook.remove()
        for stage in ('C0','C1'):torch.testing.assert_close(neutral[stage],equivalent[stage],atol=2e-6,rtol=2e-5)
        checks.append(dict(P2=True,affine_contribution_removed=h.capture['removed_affine_mean_abs'],zero3_algebraic_equivalence=True))
        # Padding with deliberately nonphysical/nonfinite inactive values never
        # enters the real view permutation or the ray pair attention.
        padded={}
        for k,v in batch.items():
            if k.startswith('input_'):
                pad=torch.zeros_like(v[:,:1]);padded[k]=torch.cat((v,pad),1)
            elif k=='view_mask':padded[k]=torch.cat((v,torch.zeros_like(v[:,:1])),1)
            else:padded[k]=v
        reordered,ind=mapping(padded,perm);assert ind[-1]==4 and not reordered['view_mask'][0,-1]
        # The added pad slot is never mixed into any of the4 real slots.
        with Intervention(model,'P1',perm) as h:pw=model(padded,return_stages=True)
        for stage in ('C0','C1'):torch.testing.assert_close(wrong[stage],pw[stage],atol=2e-6,rtol=2e-5)
        with Intervention(model,'P0') as h:pp=model(reordered,return_stages=True)
        for stage in ('C0','C1'):torch.testing.assert_close(ref[stage],pp[stage],atol=1e-5,rtol=1e-4)
        checks.append(dict(padding_fixed=True,masked_pair_mismatch_safe=True))
        assert not guard.denied
    assert state_hash(model)==initial==r['main_state_sha256']
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    # Nonconstant signed case differences; independent indexed vs multinomial
    # bootstrap check using synthetic, completely GT-free arrays.
    delta=np.linspace(-.02,.04,35)**3
    indices=np.random.default_rng(20261020).integers(0,35,(20000,35))
    weights=np.stack([(indices==i).mean(1) for i in range(35)],1)
    np.testing.assert_allclose(np.quantile(weights@delta,[.025,.975]),np.quantile(delta[indices].mean(1),[.025,.975]),atol=1e-14)
    freeze(BASE/'checks/cpu.json',dict(at=now(),passed=True,checks=checks,
        state_before=initial,state_after=state_hash(model),no_gradients=True,audit=guard.summary(),
        real_GT_read=False,synthetic_bootstrap_verified=True))
    print('CPU actual forward/mapping/hooks/neutral equivalence/padding/state checks PASS',flush=True)


def predict(layout_index):
    backend();registry=verified(BASE/'experiment_registry.json');verify_hashes(registry['sources'])
    assert verified(BASE/'checks/cpu.json')['passed']
    setting=verified(BASE/'layouts.json')['test'][layout_index]
    target=BASE/'predictions'/f'layout{layout_index:02d}';target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists():
        verify_hashes(verified(target/'complete.json')['files']);return
    records=rows(target/'manifest.jsonl') if (target/'manifest.jsonl').exists() else []
    for r in records:assert sha(target/r['prediction'])==r['sha256']
    done={(r['case_id'],r['experiment'],r['permutation_index']) for r in records}
    assert len(done)==len(records)
    m=main_method();oldroot=Path(m['prediction_root'])
    old={(r['condition'],r['draw'],r['case_id']):r for r in rows(oldroot/'manifest.jsonl')}
    plans=[p for p in verified(BASE/'permutation_maps.json') if p['condition']==setting['condition'] and p['draw']==setting['draw']]
    started=time.monotonic()
    with Guard(splits=['test']) as guard,torch.no_grad():
        assert sha(m['checkpoint'])==EXPECTED
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
        model=model_for(m['config'],'uniform','cuda:0').eval();model.load_state_dict(ck['model'],strict=True);del ck
        for p in model.parameters():p.requires_grad_(False)
        initial=state_hash(model);assert initial==registry['main_state_sha256']
        ds=dataset('test',setting)
        for plan in plans:
            i=plan['case_index'];sample=ds[i];case=sample['case_id'];assert case==plan['case_id']
            batch=move_batch(projection_pool_collate([sample]),'cuda:0')
            batch={k:v for k,v in batch.items() if not k.startswith('hold_')}
            assert batch['input_station_indices'][0].tolist()==plan['input_ids']
            row=old[setting['condition'],setting['draw'],case]
            with np.load(oldroot/row['prediction'],allow_pickle=False) as z:reference={k:z[k] for k in ('C0','C1')}
            # Only cache geometry tensors, never rerun original A on all inputs.
            valid=model.lifting.angular_grid(batch)[1].reshape(1,setting['views'],32,32,32)
            directions,relations=model.ray_geometry(batch['input_station_origin_m'],model.lifting.voxel_xyz,batch['input_station_azimuth_deg'],valid)
            local=valid.flatten(2).transpose(1,2).reshape(-1,setting['views'])
            count=batch['view_mask'].sum(1)[:,None].expand(-1,32**3).reshape(-1)
            obs=model.observability(directions,local,count)
            expected=dict(relations=relations,valid=local,observability=obs)
            jobs=[('P0',j,p) for j,p in enumerate(plan['permutation_maps'])]+[('P1',j,p) for j,p in enumerate(plan['permutation_maps'])]+[('P2',0,None)]
            if i in (0,17,34):jobs=[('repeat',0,None),('repeat',1,None)]+jobs
            repeat0=None
            for mode,j,perm in jobs:
                if (case,mode,j) in done:continue
                input_batch,index=mapping(batch,perm) if mode=='P0' else (batch,None)
                t=time.perf_counter()
                with Intervention(model,mode,perm,expected if mode=='P0' else None) as hook:
                    stages=model(input_batch,return_stages=True)
                fields={k:stages[k][0,0].cpu().numpy() for k in ('C0','C1')}
                assert all(v.shape==(64,)*3 and np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in fields.values())
                diff={k:differences(fields[k],reference[k]) for k in fields}
                repeat_diff=None
                if mode=='repeat':
                    if j==0:repeat0=fields
                    elif repeat0 is not None:repeat_diff={k:differences(fields[k],repeat0[k]) for k in fields}
                path=target/f'{i:02d}_{mode}_{j:02d}.npz'
                tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**fields);tmp.replace(path)
                record=dict(case_id=case,case_index=i,condition=setting['condition'],draw=setting['draw'],views=setting['views'],
                    experiment=mode,permutation_index=j,permutation=perm,input_ids=plan['input_ids'],heldout_ids=plan['heldout_ids'],
                    prediction=path.name,sha256=sha(path),prediction_difference=diff,repeat_difference=repeat_diff,
                    actual_forward_hook=hook.audit(),checkpoint_sha256=EXPECTED,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    seconds=time.perf_counter()-t)
                records.append(record);append(target/'manifest.jsonl',record)
                del stages,fields,input_batch
            del expected,batch,reference
            print(f'layout{layout_index:02d} {setting["condition"]} draw{setting["draw"]} case{i}: {len(records)} predictions',flush=True)
        assert state_hash(model)==initial
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        assert not guard.denied
    expected_count=sum(2*p['count']+1+(2 if p['case_index'] in (0,17,34) else 0) for p in plans)
    assert len(records)==expected_count
    freeze(target/'complete.json',dict(at=now(),layout_index=layout_index,predictions=len(records),
        files={str(target/'manifest.jsonl'):sha(target/'manifest.jsonl')},
        checkpoint_sha256=EXPECTED,state_before=initial,state_after=state_hash(model),
        no_training=True,no_gradients=True,audit=guard.summary(),seconds=time.monotonic()-started))


def seal():
    if (BASE/'all_predictions_sealed.json').exists():return verified(BASE/'all_predictions_sealed.json')
    registry=verified(BASE/'experiment_registry.json');verify_hashes(registry['sources'])
    manifests={};total=0
    with Guard() as guard:
        for i in range(17):
            target=BASE/'predictions'/f'layout{i:02d}';complete=verified(target/'complete.json');verify_hashes(complete['files'])
            assert not complete['audit']['denied'] and not any(k.startswith('GT64') for k in complete['audit']['counts'])
            assert complete['state_before']==complete['state_after']==registry['main_state_sha256']
            rr=rows(target/'manifest.jsonl');assert len(rr)==complete['predictions']
            for r in rr:
                assert sha(target/r['prediction'])==r['sha256']
                with np.load(target/r['prediction'],allow_pickle=False) as z:
                    assert set(z.files)=={'C0','C1'}
                    assert all(z[k].shape==(64,)*3 and np.isfinite(z[k]).all() and z[k].min()>=0 and z[k].max()<=1 for k in z.files)
            total+=len(rr);manifests[str(target/'manifest.jsonl')]=sha(target/'manifest.jsonl')
        assert total==registry['new_predictions'] and not guard.denied
    freeze(BASE/'all_predictions_sealed.json',dict(at=now(),all_new_predictions_saved_before_current_GT64=True,
        manifest_hashes=manifests,new_predictions=total,fields=2*total,no_training=True,GT128_read=False,audit=guard.summary()))
    print('ALL PREDICTIONS SEALED',total,flush=True)


def scheduler():
    import controller as old
    old.BASE=BASE
    old.progress=progress
    return old


def progress():
    if not (BASE/'status.json').exists():return
    s=read(BASE/'status.json')
    complete=[i for i in range(17) if (BASE/'predictions'/f'layout{i:02d}'/'complete.json').exists()]
    text=['# OARI几何利用与顺序不变性诊断进度','',f'北京时间：{now()}',
        f'状态：`{s["state"]}`；完成布局：{len(complete)}/17。','',
        '冻结UniformFusion seed20261011/epoch285，无训练、无权重更新、反馈OARI未重启。',
        '原始A预测复用。P0/P1每输入10种非恒等置换，V3穷举仅5种并单独记录；P2移除首层仿射中的显式关系加性贡献。',
        '全部新增预测封存后才读GT64；GT128禁用。重复置换不当作独立案例或种子。',
        f'运行目录：`{BASE}`。']
    (PROJECT/'docs/reports/mrgr_geometry_utilization_v1_progress_zh.md').write_text('\n'.join(text)+'\n')


def controller():
    BASE.joinpath('scheduling').mkdir(exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            old=scheduler()
            for i in range(17):
                old.run_gpu_once(f'predict_layout{i:02d}',[sys.executable,str(HERE/'diagnostic.py'),'predict','--layout-index',str(i)],
                    BASE/'predictions'/f'layout{i:02d}'/'complete.json')
            seal()
            import analyze
            if not (BASE/'acceptance.json').exists():
                write(BASE/'status.json',dict(at=now(),state='scoring_and_statistics',controller_pid=os.getpid()));progress()
                with (BASE/'scheduling/analysis.log').open('a') as log:
                    subprocess.run([sys.executable,str(HERE/'diagnostic.py'),'analyze'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert verified(BASE/'acceptance.json')['completed']
            write(BASE/'status.json',dict(at=now(),state='accepted_pending_sync'));progress()
            analyze.sync()
            write(BASE/'status.json',dict(at=now(),state='complete'))
        except BaseException:
            write(BASE/'status.json',dict(at=now(),state='failed',traceback=traceback.format_exc()));progress();raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','checks','predict','seal','controller','analyze','sync'])
    parser.add_argument('--layout-index',type=int);o=parser.parse_args()
    if o.action=='register':return register()
    if o.action=='checks':return cpu_checks()
    if o.action=='predict':return predict(o.layout_index)
    if o.action=='seal':return seal()
    if o.action=='controller':return controller()
    import analyze
    return analyze.main() if o.action=='analyze' else analyze.sync()


if __name__=='__main__':main()
