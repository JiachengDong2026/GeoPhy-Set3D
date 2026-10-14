#!/usr/bin/env python3
"""Registered finite signal diagnosis: 30 complete fields, 6 local autograd graphs."""
import argparse
import gzip
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
import yaml
from geophy_set3d.diagnostics.senseiver_signal_path import (SignalGuard,Capture,intervene,valid_measurements,
    model_hashes,differentiable_input,probe_signs,describe,PROBE_STAGES)
from geophy_set3d.training.senseiver_consistency_probe import validate as validate_source
from geophy_set3d.training.selfsup import make_dataset,move_dual_batch,assert_disjoint
from geophy_set3d.training.ray_senseiver import build_comparison_model,sha256,tensor_identity
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything,write_json
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.data.collate import projection_pool_collate
from run_senseiver_init_probe import now
NAME='ray_senseiver_signal_path_v1'
ART=REPO/'docs/reports/artifacts'/NAME
PRIOR=REPO/'docs/reports/artifacts/ray_senseiver_consistency_probe_v1'


def config(path):
    c=yaml.safe_load(Path(path).read_text())
    assert c['name']==NAME and c['target_indices']==[0,1,2] and c['wrong_source_indices']==[1,2,3]
    assert c['modes']==['normal','zero','wrong_case','gain_plus','gain_minus'] and c['gains']==dict(gain_plus=1.01,gain_minus=.99)
    assert c['sampler_epoch']==0 and c['expected_full_forwards']==30 and c['expected_local_graphs']==6
    assert c['probes']==dict(seed=20261015,count=4,distribution='Rademacher',normalization='sum_sign_x_over_sqrt_numel')
    assert c['relative_rms_epsilon']==1e-12 and c['finite_difference']==dict(gain_step=.01,atol=1e-7,rtol=.05)
    source=validate_source(yaml.safe_load((REPO/c['source_config']).read_text()))
    source['training']['device']=c['device']
    return c,source


def guard_for(c,source):
    root=Path(source['dataset']['root']);records=[json.loads(s) for s in (root/'source_index.jsonl').read_text().splitlines() if json.loads(s)['split']=='train']
    return SignalGuard(root,[v['path'] for v in c['checkpoints'].values()],records,c['output'])


@contextmanager
def audited_guard(c,source):
    guard=guard_for(c,source)
    try:
        with guard:yield guard
    finally:
        session=Path(c['output'])/'access_sessions';session.mkdir(exist_ok=True)
        write_json(session/(str(time.time_ns())+'.json'),dict(at=now(),access=guard.summary()))


def checkpoint(c,source,key):
    info=c['checkpoints'][key];path=Path(info['path']);assert sha256(path)==info['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['epoch']==info['epoch'] and saved['updates']==info['updates'] and saved['arm']=='treatment'
    assert saved['relative_l1']==saved['consistency_weight']==0. and not saved['volumetric_gt_used']
    assert saved['config']==source
    assert saved['registration_sha256']==sha256(PRIOR/'registration.json') and saved['source_manifest_sha256']==sha256(PRIOR/'source_manifest.json')
    if key=='epoch0':assert state_hash(saved['model'])==json.loads((PRIOR/'initialization.json').read_text())['treatment']
    else:assert saved['exit_state']['reason']=='FAILED_TO_LEARN'
    return saved


def samples(ds):
    with gzip.open(PRIOR/'fixed_inputs.json.gz','rt') as f:inventory=json.load(f)
    result=[]
    for i in range(4):
        s=ds[i];assert ds.records[i]['case_id']==inventory[i]['case_id'];assert_disjoint(dual_measurement_collate([s]))
        for k in ['A','B','H']:
            assert tensor_identity(*[s[k][v] for v in sorted(s[k]) if isinstance(s[k][v],torch.Tensor)])==inventory[i]['subsets'][k]['batch_identity']
        result.append(s)
    return result,inventory[:4]


def aligned_source(ds,sample,source_index):
    row,p=ds._projections(source_index);ids=sample['A']['input_station_indices'].numpy()
    aligned=dict(sample['A']);aligned.update(ds._views(p,ids,'input'))
    for k in sample['A']:
        if k.startswith('input_') and k not in ['input_projection_raw','input_projection']:
            assert torch.equal(aligned[k],sample['A'][k]),'Cannot preserve wrong-case ray identity: '+k
    return projection_pool_collate([aligned])


def execution_input(original,registered_cpu):
    """Transfer registered measurement bits; preserve all existing geometry objects."""
    result=dict(original)
    for key in ['input_projection_raw','input_projection']:
        result[key]=registered_cpu[key].to(original['input_projection_raw'].device)
    assert all(result[k] is v for k,v in original.items() if k not in ['input_projection_raw','input_projection'])
    return result


def prepare(c,source,path):
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    if (out/'registration.json').exists():raise FileExistsError('No repeated registration')
    torch.set_num_threads(2)
    with guard_for(c,source) as guard:
        ds=make_dataset(source);fixed,inventory=samples(ds);inputs=[];checkpoints={}
        for key in c['checkpoints']:
            saved=checkpoint(c,source,key)
            checkpoints[key]=dict(**c['checkpoints'][key],model_state_hash=state_hash(saved['model']),config=saved['config'],initial_hash=saved['initial_hash'],reason=saved['exit_state'].get('reason'))
            del saved
        for i in range(3):
            original=dual_measurement_collate([fixed[i]])['A'];wrong=aligned_source(ds,fixed[i],i+1)
            variants={mode:intervene(original,mode,wrong) for mode in c['modes']}
            inputs.append(dict(index=i,case_id=ds.records[i]['case_id'],source_index=i+1,source_case_id=ds.records[i+1]['case_id'],station_ids=original['input_station_indices'].tolist(),
                ray_geometry_exactly_identical=True,valid_rays=int(valid_measurements(original).sum()),valid_zero_measurements=int((original['input_projection_raw'][valid_measurements(original)]==0).sum()),
                original_A_identity=tensor_identity(*[original[k] for k in sorted(original) if isinstance(original[k],torch.Tensor)]),
                modes={mode:tensor_identity(*[v[k] for k in sorted(v) if isinstance(v[k],torch.Tensor)]) for mode,v in variants.items()}))
        access=guard.summary()
        projection_hashes={str(ds.root/r['train_projection_path']):sha256(ds.root/r['train_projection_path']) for r in ds.records[:4]}
        metadata_hashes={str(ds.root/p):sha256(ds.root/p) for p in ['source_index.jsonl','train_statistics.json','geometry/train_geometry.npz']}
    write_json(out/'checkpoint_identity.json',checkpoints);write_json(out/'fixed_inputs.json',inputs)
    with gzip.open(out/'original_fixed_inventory.json.gz','wt') as f:json.dump(inventory,f)
    sources=subprocess.check_output(['git','ls-files','src/**/*.py','third_party/senseiver/*'],cwd=REPO,text=True).splitlines()
    sources+=['src/geophy_set3d/diagnostics/senseiver_signal_path.py','scripts/diagnose_senseiver_signal_path.py','tests/test_senseiver_signal_path.py',str(path),'docs/ray_senseiver_signal_path_v1_protocol_zh.md']
    write_json(out/'source_manifest.json',dict(sha256={p:sha256(REPO/p) for p in sorted(set(sources))}))
    historical=[p for p in PRIOR.rglob('*') if p.is_file()]
    for name in ['ray_senseiver_learning_diagnostic','ray_senseiver_init_probe_v1','ray_senseiver_relative_probe_v1','ray_senseiver_consistency_probe_v1']:
        historical.append(REPO/'docs/reports'/f'{name}_zh.md')
    historical+=[REPO/'docs'/f'{n}_protocol_zh.md' for n in ['ray_senseiver_init_probe_v1','ray_senseiver_relative_probe_v1','ray_senseiver_consistency_probe_v1']]
    plan=[dict(id=f'{key}_case{i}_{mode}',checkpoint=key,index=i,mode=mode) for key in c['checkpoints'] for i in range(3) for mode in c['modes']]
    local_plan=[dict(id=f'local_{key}_case{i}',checkpoint=key,index=i) for key in c['checkpoints'] for i in range(3)]
    reg=dict(at=now(),config=c,source_config=source,checkpoint_identity_sha256=sha256(out/'checkpoint_identity.json'),fixed_inputs_sha256=sha256(out/'fixed_inputs.json'),
        inventory_sha256=sha256(out/'original_fixed_inventory.json.gz'),source_manifest_sha256=sha256(out/'source_manifest.json'),projection_sha256=projection_hashes,metadata_sha256=metadata_hashes,
        historical_sha256={str(p.relative_to(REPO)):sha256(p) for p in historical},full_forward_plan=plan,local_graph_plan=local_plan,
        expected_complete_fields=30,expected_local_sampled_graphs=6,projector_calls_per_full=2,optimizer_created=False,optimizer_steps=0,access=access,
        fixed_259_indices_sha256=hashlib.sha256(torch.linspace(0,262143,259).long().numpy().tobytes()).hexdigest(),
        probe_definition='four fixed Rademacher directions, phi=sum(s*x)/sqrt(N); seed+100*stage+probe; logit/concentration share signs; valid tokens only',
        finite_difference='Use only saved gain+/-1% full-forward scalar projections; compare (phi_plus-phi_minus)/.02 against grad_phi_u dot u; atol1e-7+rtol.05*max(abs(FD),abs(AD)). Retain failures, no tolerance retuning.',
        boundary='No training/optimizer/model or buffer changes, no GT/Val/Pool B/test/other weights. No architecture/gain/loss search; only one later hypothesis.')
    write_json(out/'registration.json',reg)
    for p in out.iterdir():
        if p.is_file():(ART/p.name).write_bytes(p.read_bytes())
    print(json.dumps(dict(checkpoints=checkpoints,inputs=inputs,access=access,full_forwards=30,local_graphs=6),indent=2),flush=True)


def verify(c,source):
    out=Path(c['output']);reg=json.loads((out/'registration.json').read_text());assert reg['config']==c and reg['source_config']==source
    for n in ['registration.json','checkpoint_identity.json','fixed_inputs.json','original_fixed_inventory.json.gz','source_manifest.json']:
        assert subprocess.check_output(['git','show',f'HEAD:docs/reports/artifacts/{NAME}/{n}'],cwd=REPO)==(out/n).read_bytes()
    for p,h in {**reg['metadata_sha256'],**reg['projection_sha256']}.items():assert sha256(p)==h,p
    expected=json.loads((out/'source_manifest.json').read_text())['sha256']
    amendment_path=out/'amendment_v1_1.json'
    if amendment_path.exists():
        amendment=json.loads(amendment_path.read_text())
        assert subprocess.check_output(['git','show',f'HEAD:docs/reports/artifacts/{NAME}/amendment_v1_1.json'],cwd=REPO)==amendment_path.read_bytes()
        assert amendment['original_registration_sha256']==sha256(out/'registration.json') and amendment['original_source_manifest_sha256']==sha256(out/'source_manifest.json')
        assert set(amendment['source_changes'])=={'scripts/diagnose_senseiver_signal_path.py','tests/test_senseiver_signal_path.py'}
        for p,change in amendment['source_changes'].items():
            assert expected[p]==change['old_sha256'];expected[p]=change['new_sha256']
        assert (out/'forward_ledger.jsonl').read_bytes().startswith(amendment['preserved_ledger_prefix'].encode())
        for p,h in amendment['preserved_saved_records'].items():assert sha256(out/p)==h,p
    second_path=out/'amendment_v1_2.json'
    if second_path.exists():
        second=json.loads(second_path.read_text())
        assert subprocess.check_output(['git','show',f'HEAD:docs/reports/artifacts/{NAME}/amendment_v1_2.json'],cwd=REPO)==second_path.read_bytes()
        assert second['prior_amendment_sha256']==sha256(amendment_path)
        assert set(second['source_changes'])=={'scripts/diagnose_senseiver_signal_path.py','tests/test_senseiver_signal_path.py'}
        for p,change in second['source_changes'].items():
            assert expected[p]==change['old_sha256'];expected[p]=change['new_sha256']
        assert (out/'forward_ledger.jsonl').read_bytes().startswith(second['preserved_ledger_prefix'].encode())
        for p,h in second['preserved_saved_records'].items():assert sha256(out/p)==h,p
    for p,h in {**reg['historical_sha256'],**expected}.items():assert sha256(REPO/p)==h,p
    return reg


def ledger(out,identity,event,**extra):
    with (out/'forward_ledger.jsonl').open('a') as f:
        f.write(json.dumps(dict(id=identity,event=event,at=now(),**extra),allow_nan=False)+'\n');f.flush()
        import os;os.fsync(f.fileno())


def check_not_repeated(out,identity):
    rows=[json.loads(s) for s in (out/'forward_ledger.jsonl').read_text().splitlines()] if (out/'forward_ledger.jsonl').exists() else []
    events=[r['event'] for r in rows if r['id']==identity]
    aborted=identity=='local_epoch0_case0' and (out/'amendment_v1_2.json').exists()
    if events:
        if aborted and events==['started','aborted']:return None
        if events==['started','returned','saved'] or (aborted and events==['started','aborted','started','returned','saved']):
            record=json.loads((out/'records'/f'{identity}.json').read_text());last=next(r for r in reversed(rows) if r['id']==identity)
            assert sha256(out/'records'/f'{identity}.json')==last['record_sha256'];return record
        raise RuntimeError('Incomplete prior attempt requires explicit audit; successful forwards never repeated: '+identity)
    return None


def metrics(prediction,batch):
    result={}
    for label,subset in [('input','A'),('H','H')]:
        t=batch[subset];pic=project_input(prediction,t);valid=valid_measurements(t)
        result[label]=relative_projection_metrics(pic[0],t['input_projection_raw'][0],valid[0])
    return result


def local_diagnosis(model,original,full_rows):
    inp,u=differentiable_input(original);captured={};handles=[]
    handles.append(model.encoder.preproc.register_forward_hook(lambda m,a,y:captured.update(preproc=y)))
    handles.append(model.decoder.postproc.register_forward_hook(lambda m,a,y:captured.update(logits=y)))
    try:
        latent=model.encode(inp);indices=torch.linspace(0,262143,259,device=u.device).long()
        concentration=model.decode_queries(latent,model.query_encoding[indices]);z=captured['logits']
        values={'encoder.preproc':captured['preproc'],'encoder.final':latent,'decoder.logits':z,'concentration':concentration}
        vm=valid_measurements(original);rows={};fd={};probe_ledger={}
        for name,x in values.items():
            rows[name]=[];fd[name]=[];probe_ledger[name]=[]
            for i in range(4):
                signs=probe_signs(x.shape,name,i).to(device=x.device,dtype=x.dtype)
                phi=(x*signs).sum()/math.sqrt(x.numel());g=torch.autograd.grad(phi,u,retain_graph=True)[0]
                assert torch.isfinite(g).all() and not torch.count_nonzero(g[~vm])
                ad=float((g*u).detach().double().sum());v=g[vm]
                rows[name].append(dict(probe=i,phi=float(phi.detach()),gradient_u=describe(v),gradient_l2=float(v.detach().double().norm()),gradient_dot_u=ad,invalid_gradient_nonzero_count=int(torch.count_nonzero(g[~vm]))))
                plus=full_rows['gain_plus']['scalar_probes'][name][i];minus=full_rows['gain_minus']['scalar_probes'][name][i]
                numeric=(plus-minus)/.02;absolute=abs(ad-numeric);scale=max(abs(ad),abs(numeric));tolerance=1e-7+.05*scale
                fd[name].append(dict(probe=i,AD=ad,FD=numeric,absolute_error=absolute,relative_error=absolute/scale if scale>1e-12 else None,tolerance=tolerance,passed=absolute<=tolerance,phi_plus=plus,phi_minus=minus,
                    local_vs_full_normal_phi=float(phi.detach())-full_rows['normal']['scalar_probes'][name][i]))
                probe_ledger[name].append(dict(probe=i,shape=list(x.shape),normalization=math.sqrt(x.numel()),signs_sha256=hashlib.sha256(signs.detach().cpu().to(torch.int8).numpy().tobytes()).hexdigest()))
        assert u.grad is None and all(p.grad is None for p in model.parameters())
        derivative=z.sigmoid()*(1-z.sigmoid());relative_activation_gain=1-z.sigmoid()
        ratios=[rows['concentration'][i]['gradient_l2']/rows['decoder.logits'][i]['gradient_l2'] if rows['decoder.logits'][i]['gradient_l2']>1e-12 else None for i in range(4)]
        return dict(stages=rows,finite_difference=fd,probe_identity=probe_ledger,sigmoid_derivative=describe(derivative),dlogC_dlogit=describe(relative_activation_gain),same_probe_concentration_over_logit_gradient_l2=ratios,
                    query_indices=indices.cpu().tolist(),sampled_queries=259,precision='FP32 unchanged; FD scalar reduction in float64 of saved FP32 values',
                    caveat='Full no_grad and local autograd may use different existing attention kernels/query chunk shapes; no attention setting changed; failed FD retained.',optimizer_created=False,optimizer_steps=0)
    finally:
        for h in handles:h.remove()


def run(c,source):
    out=Path(c['output']);torch.set_num_threads(2);seed_everything(20261011);began=time.perf_counter()
    (out/'records').mkdir(exist_ok=True);(out/'arrays').mkdir(exist_ok=True)
    summary={}
    with audited_guard(c,source) as guard:
        reg=verify(c,source);ds=make_dataset(source);fixed,_=samples(ds);registered=json.loads((out/'fixed_inputs.json').read_text())
        for key in c['checkpoints']:
            saved=checkpoint(c,source,key);model=build_comparison_model(source);model.load_state_dict(saved['model']);del saved
            model.eval();assert not model.training;initial=model_hashes(model);model_state=state_hash(model.state_dict());assert model_state==json.loads((out/'checkpoint_identity.json').read_text())[key]['model_state_hash']
            hooks_before=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())
            for i in range(3):
                batch=move_dual_batch(dual_measurement_collate([fixed[i]]),c['device']);original=batch['A']
                cpu_input=dual_measurement_collate([fixed[i]])['A'];wrong_cpu=aligned_source(ds,fixed[i],i+1)
                registered_variants={mode:intervene(cpu_input,mode,wrong_cpu) for mode in c['modes']}
                baseline=None;full_rows={}
                for mode in c['modes']:
                    identity=f'{key}_case{i}_{mode}';previous=check_not_repeated(out,identity)
                    if previous is not None:
                        assert previous['model_identity']==initial
                        full_rows[mode]=previous
                        if mode=='normal':
                            assert sha256(out/previous['baseline_arrays'])==previous['baseline_sha256']
                            with np.load(out/previous['baseline_arrays'],allow_pickle=False) as data:baseline={k:data[k] for k in data.files}
                        continue
                    inp=execution_input(original,registered_variants[mode])
                    assert tensor_identity(*[inp[k] for k in sorted(inp) if isinstance(inp[k],torch.Tensor)])==registered[i]['modes'][mode]
                    ledger(out,identity,'started',kind='full',checkpoint_sha256=c['checkpoints'][key]['sha256'],index=i,mode=mode)
                    t=time.perf_counter();torch.cuda.reset_peak_memory_stats(c['device'])
                    seen=[];handle=model.register_forward_pre_hook(lambda m,a:seen.append(id(a[0])))
                    try:
                        with torch.no_grad(),Capture(model,inp,baseline) as capture:
                            prediction=model(inp);ledger(out,identity,'returned',kind='full')
                            assert seen==[id(inp)] and id(batch['H']) not in seen
                            stages,arrays,small,scalars=capture.finish(prediction)
                            measured=metrics(prediction,batch)
                    finally:handle.remove()
                    if mode=='normal':baseline=arrays
                    if mode!='normal':
                        for stage in ['decoder.query_coordinates_fixed','decoder.learned_query_fixed','decoder.query_preproc_fixed']:
                            assert stages[stage]['tensor_sha256']==full_rows['normal']['stages'][stage]['tensor_sha256']
                    assert model_hashes(model)==initial and all(p.grad is None for p in model.parameters())
                    np.savez(out/'arrays'/f'{identity}.npz',concentration=prediction.detach().cpu().numpy(),logits=small['decoder.logits'])
                    row=dict(id=identity,checkpoint=key,index=i,mode=mode,case_id=ds.records[i]['case_id'],source_case_id=ds.records[i+1]['case_id'] if mode=='wrong_case' else None,
                        stages=stages,sigmoid_derivative=describe((1/(1+np.exp(-small['decoder.logits'].astype(np.float64))))*(1-1/(1+np.exp(-small['decoder.logits'].astype(np.float64))))),encoder_call_order=capture.calls,decomposition=capture.decomposition,scalar_probes=scalars,measurement=measured,
                        original_geometry_path_masks_and_other_fields_same_objects=True,H_excluded_from_network=True,valid_rays=int(valid_measurements(inp).sum()),valid_zero_measurements=int((inp['input_projection_raw'][valid_measurements(inp)]==0).sum()),
                        arrays=str(Path('arrays')/f'{identity}.npz'),arrays_sha256=sha256(out/'arrays'/f'{identity}.npz'),model_identity=initial,parameter_buffers_unchanged=True,optimizer_steps=0,
                        seconds=time.perf_counter()-t,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(c['device']))
                    if mode=='normal':
                        np.savez(out/'arrays'/f'{key}_case{i}_baseline.npz',**baseline)
                        row.update(baseline_arrays=f'arrays/{key}_case{i}_baseline.npz',baseline_sha256=sha256(out/'arrays'/f'{key}_case{i}_baseline.npz'))
                    concentration_pair=stages['concentration']['paired'];concentration_pair['relative_MA_previous_definition']=concentration_pair['MAE']/(concentration_pair['normal_mean_abs']+1e-12)
                    write_json(out/'records'/f'{identity}.json',row);ledger(out,identity,'saved',kind='full',record_sha256=sha256(out/'records'/f'{identity}.json'))
                    full_rows[mode]=row;print(f'saved {identity} seconds={row["seconds"]:.2f}',flush=True)
                    del capture,prediction,small,arrays
                identity=f'local_{key}_case{i}';previous=check_not_repeated(out,identity)
                if previous is None:
                    ledger(out,identity,'started',kind='local_sampled_graph',sampled_queries=259)
                    result=local_diagnosis(model,original,full_rows);ledger(out,identity,'returned',kind='local_sampled_graph')
                    assert model_hashes(model)==initial and all(p.grad is None for p in model.parameters())
                    result.update(id=identity,checkpoint=key,index=i,model_identity=initial,parameter_buffers_unchanged=True,grad_buffers_none=True)
                    write_json(out/'records'/f'{identity}.json',result);ledger(out,identity,'saved',kind='local_sampled_graph',record_sha256=sha256(out/'records'/f'{identity}.json'))
                    print('saved '+identity+' (no optimizer)',flush=True)
                del baseline,full_rows
            assert model_hashes(model)==initial and hooks_before==sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())
            assert sha256(c['checkpoints'][key]['path'])==c['checkpoints'][key]['sha256']
            summary[key]=dict(model_identity_before=initial,model_identity_after=model_hashes(model),hooks_removed=True,grad_buffers_none=all(p.grad is None for p in model.parameters()),checkpoint_unchanged=True)
            del model;torch.cuda.empty_cache()
        verify(c,source)
        rows=[json.loads(s) for s in (out/'forward_ledger.jsonl').read_text().splitlines()]
        for entry in reg['full_forward_plan']+reg['local_graph_plan']:
            expected_events=['started','returned','saved']
            if entry['id']=='local_epoch0_case0' and (out/'amendment_v1_2.json').exists():expected_events=['started','aborted']+expected_events
            assert [r['event'] for r in rows if r['id']==entry['id']]==expected_events
        summary.update(at=now(),full_forward_counts={event:sum(r['kind']=='full' and r['event']==event for r in rows) for event in ['started','returned','saved']},
                       local_graph_counts={event:sum(r['kind']=='local_sampled_graph' and r['event']==event for r in rows) for event in ['started','returned','saved','aborted']},
                       projector_calls=60,optimizer_created=False,optimizer_steps=0,seconds=time.perf_counter()-began,access=guard.summary(),source_history_data_hashes_unchanged=True,execution_version='v1.2' if (out/'amendment_v1_2.json').exists() else 'v1.1' if (out/'amendment_v1_1.json').exists() else 'v1',
                       prior_partial_attempt='Two prior partial attempts retained: CPU/GPU unused normalized-field identity assertion, then missing math import before any gradient. Five complete fields reused; one sampled graph aborted before gradients. Original partial guard counters unavailable; no invented counters. v1.2 records guard exits even on exceptions.')
        assert summary['full_forward_counts']==dict(started=30,returned=30,saved=30) and summary['local_graph_counts']==(dict(started=7,returned=6,saved=6,aborted=1) if (out/'amendment_v1_2.json').exists() else dict(started=6,returned=6,saved=6,aborted=0))
        assert not any(summary['access'][k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
    write_json(out/'complete.json',summary);print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/ray_senseiver_signal_path_v1.yaml');ap.add_argument('--phase',choices=['prepare','run'],required=True);a=ap.parse_args()
    c,source=config(a.config)
    if a.phase=='prepare':prepare(c,source,a.config)
    else:run(c,source)

if __name__=='__main__':main()
