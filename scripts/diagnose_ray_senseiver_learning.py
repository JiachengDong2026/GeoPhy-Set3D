#!/usr/bin/env python3
"""Isolated GT-free learning diagnosis of one atomic own-run checkpoint snapshot.
No optimizer is constructed, no training state/config/selector is modified.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src')]
import numpy as np
import torch
from geophy_set3d.training.ray_senseiver import load_comparison_config,build_comparison_model,sha256,tensor_identity
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.selfsup import make_dataset,objective,move_dual_batch,FROZEN_LOSS
from geophy_set3d.training.utils import write_json,seed_everything
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.models.ray_senseiver_ss import ray_tokens

QUANTILES=[0.,.001,.01,.1,.5,.9,.99,.999,1.]


def moments(x):
    x=x.detach().double()
    return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),std=float(x.std(unbiased=False)),l2=float(x.norm()),quantiles=dict(zip(map(str,QUANTILES),torch.quantile(x.flatten(),torch.tensor(QUANTILES,device=x.device,dtype=torch.float64)).cpu().tolist())))


def gradient_norms(named,grads):
    result={}
    for group,select in [('encoder',lambda n:n.startswith('encoder.') and n!='encoder.latent'),('learned_encoder_latents',lambda n:n=='encoder.latent'),('decoder',lambda n:n.startswith('decoder.')),('learned_output_query',lambda n:n=='decoder.output'),('all',lambda n:True)]:
        values=[g for (n,p),g in zip(named,grads) if select(n) and g is not None]
        result[group]=dict(l2=float(torch.stack([g.detach().double().square().sum() for g in values]).sum().sqrt()),finite=all(bool(torch.isfinite(g).all()) for g in values),parameters_with_gradient=len(values),zero_gradient_parameters=sum(int(torch.count_nonzero(g)==0) for g in values))
    return result


def volume_gradient(g,p):
    q=g.detach().double();p=p.detach().double();chain=q*p*(1-p)
    return dict(l2=float(q.norm()),mean=float(q.mean()),sum=float(q.sum()),positive_fraction=float((q>0).double().mean()),negative_fraction=float((q<0).double().mean()),zero_fraction=float((q==0).double().mean()),logit_chain_l2=float(chain.norm()),logit_chain_attenuation=float(chain.norm()/q.norm().clamp_min(1e-30)))


def spatial(pred):
    x=pred.detach().double()[0,0]
    result=moments(x)
    result['axis_neighbor_mean_abs']={axis:float(torch.diff(x,dim=i).abs().mean()) for i,axis in enumerate('xyz')}
    result['fraction_below_1e_6']=float((x<1e-6).double().mean())
    return result


def geometry(batch):
    m=batch['input_valid_ray'][:,:,0].bool()&batch['view_mask'][:,:,None,None]
    d=batch['input_ray_direction'][m];origin=batch['input_station_origin_m'][:,:,None,None,:].expand_as(batch['input_ray_direction'])[m]
    near=batch['input_t_near'][:,:,0][m,None];length=batch['input_path_length_m'][:,:,0][m,None]
    entry=origin+d*near;exit=entry+d*length
    raw=batch['input_projection_raw'][:,:,0][m];tokens,excluded,counts=ray_tokens(batch)
    return dict(valid_rays=int(m.sum()),packed_real_rays=int((~excluded).sum()),valid_zero_measurements=int((raw==0).sum()),positive_measurement_fraction=float((raw>0).float().mean()),raw_ppm_m=moments(raw),normalized_integral=moments(raw/32284.894580078173),direction_norm=moments(d.norm(dim=-1)),path_length_m=moments(length),entry_min_m=entry.amin(0).cpu().tolist(),entry_max_m=entry.amax(0).cpu().tolist(),exit_min_m=exit.amin(0).cpu().tolist(),exit_max_m=exit.amax(0).cpu().tolist(),entry_exit_in_domain=bool(((entry>=-3e-6)&(entry<=.600003)).all() and ((exit>=-3e-6)&(exit<=.600003)).all()),all_tokens_finite=bool(torch.isfinite(tokens).all()),token_channels=tokens.shape[-1],token_measurement=moments(tokens[...,0]),token_geometry=moments(tokens[...,1:11]),token_fourier=moments(tokens[...,11:]))


def near_zero_probe(batch):
    rows=[]
    for level in [0.,1e-8,1e-7,1e-6,1e-5]:
        value=torch.tensor(level,device=batch['A']['input_projection_raw'].device,requires_grad=True)
        class Constant:
            def __call__(self,b):return value.expand(len(b['case_id']),1,64,64,64)
        loss,parts,volumes,projections=objective(Constant(),batch)
        grad={k:float(torch.autograd.grad(v,value,retain_graph=True)[0]) for k,v in dict(total=loss,**parts).items()}
        rows.append(dict(constant_mass_fraction=level,loss=dict(total=float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()}),d_loss_d_constant_field=grad,interpretation='positive derivative means gradient descent pushes constant field lower; exact-zero absolute-value subgradient differs from small positive output'))
    return rows


def chunk_check(model,batch):
    model.train();latent=model.encode(batch)
    indices=torch.linspace(0,262143,259,device=latent.device).long();coords=model.query_encoding[indices].detach().clone().requires_grad_()
    named=list(model.named_parameters());parameters=tuple(p for n,p in named)+(coords,)
    full=model.decode_queries(latent,coords,259,False);chunk=model.decode_queries(latent,coords,37,True)
    weights=torch.linspace(.1,1,259,device=latent.device).reshape(1,-1,1)
    ga=torch.autograd.grad((full*weights).sum(),parameters,retain_graph=True)
    gb=torch.autograd.grad((chunk*weights).sum(),parameters)
    checks=[]
    for (name,p),a,b in zip(named+[('query_fourier',coords)],ga,gb):
        diff=(a-b).detach().double().norm();scale=a.detach().double().norm().clamp_min(1e-30)
        checks.append(dict(parameter=name,full_gradient_l2=float(scale),difference_l2=float(diff),relative_l2=float(diff/scale),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())))
    prediction_error=float((full-chunk).abs().max());relative_prediction=float((full-chunk).double().norm()/full.detach().double().norm().clamp_min(1e-30))
    return dict(passed=all(r['finite'] and r['relative_l2']<5e-4 for r in checks) and relative_prediction<5e-5,diagnostic_queries=259,full_formal_grid_unchanged=262144,chunk_sizes=[259,37],prediction_max_abs=prediction_error,prediction_relative_l2=relative_prediction,all_query_gradients_nonzero=bool((gb[-1].abs().sum(1)>0).all()),encoder_latent_decoder_and_query_gradients=checks)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/train_ray_senseiver_selfsup_v1.yaml');ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    c=load_comparison_config(a.config);out=Path(c['comparison']['output']);diag=out/'learning_diagnostic';diag.mkdir(exist_ok=True)
    target=diag/'diagnosis.json'
    if target.exists():raise FileExistsError('Independent diagnostic already completed')
    registration=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),case_indices=[0,1,2],train_sampler_epoch=0,own_checkpoint_only='last.pt atomic snapshot',optimizer_created=False,optimizer_steps=0,GT=False,Pool_B=False,test=False,source_sha256=sha256(__file__),config_sha256=sha256(a.config),query_check='259 fixed endpoint grid positions; numerical test only, formal training all 262144 unchanged',near_zero_levels=[0.,1e-8,1e-7,1e-6,1e-5])
    write_json(diag/'registration.json',registration)
    seed_everything(c['training']['seed']);torch.set_num_threads(2)
    ckroot=Path(c['experiment']['output_root'])/'checkpoints'/c['experiment']['name']
    started=time.perf_counter()
    with GTReadGuard(c['dataset']['root'],ckroot) as guard:
        # One opened inode; atomic replacement by the live run cannot mix bytes.
        data=(ckroot/'last.pt').read_bytes();checkpoint_sha=hashlib.sha256(data).hexdigest();ck=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
        assert ck['config']==c and not ck['volumetric_gt_used']
        (diag/'own_run_snapshot.pt').write_bytes(data);del data
        history=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()]
        history=[r for r in history if r['epoch']<=ck['epoch']];assert len(history)==ck['epoch']+1
        best=min(history,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        log_summary=dict(first_logged_epoch=history[0],historical_best_to_snapshot=best,current_checkpoint_epoch=history[-1],notes='First logged train epoch aggregates changing weights across 83 updates; it is not an untrained single-model measurement.',epochs_read=len(history),all_steps_exact=all(r['optimizer_steps']==83*(r['epoch']+1) and r['optimizer_steps_this_epoch']==83 for r in history),all_finite_gradients_and_postclip_max=max(r['post_clip_gradient_norm_max'] for r in history),full_output_trajectory=[dict(epoch=r['epoch']+1,train_measurement=r['train']['measurement'],val_measurement=r['val']['measurement'],train_loss=r['train']['loss'],val_loss=r['val']['loss'],train_output=r['train']['output'],val_output=r['val']['output'],train_gradient_norm_max=r['train']['gradient_norm_max'],post_clip_gradient_norm_max=r['post_clip_gradient_norm_max']) for r in history])
        write_json(diag/'training_log_summary.json',log_summary)
        model=build_comparison_model(c,a.device);model.load_state_dict(ck['model']);before=state_hash(model.state_dict())
        optimizer_steps=sorted({int(s['step']) for s in ck['optimizer']['state'].values()});assert optimizer_steps==[ck['optimizer_steps']]
        fresh_hash=ck['initial_state_sha256'];del ck
        ds=make_dataset(c);ds.set_epoch(0);rows=[];predictions=[];input_measures=[]
        for index in registration['case_indices']:
            b=move_dual_batch(dual_measurement_collate([ds[index]]),a.device);model.eval();logits=[];enc=[]
            h=model.decoder.postproc.register_forward_hook(lambda module,args,output:logits.append(output.detach()))
            e=model.encoder.register_forward_hook(lambda module,args,output:enc.append(output.detach()))
            with torch.no_grad():
                normal=model(b['A']);normal_logits=torch.cat(logits,1);normal_latent=enc[-1];logits.clear();enc.clear()
                zero=dict(b['A']);zero['input_projection_raw']=torch.zeros_like(zero['input_projection_raw']);zero['input_projection']=torch.zeros_like(zero['input_projection'])
                unchanged=[k for k in b['A'] if isinstance(b['A'][k],torch.Tensor) and k not in ['input_projection_raw','input_projection']]
                assert tensor_identity(*[b['A'][k] for k in sorted(unchanged)])==tensor_identity(*[zero[k] for k in sorted(unchanged)])
                zero_prediction=model(zero);zero_latent=enc[-1]
                difference=(normal-zero_prediction).abs();latent_difference=normal_latent-zero_latent
                sigmoid_derivative=normal*(1-normal)
            h.remove();e.remove()
            prediction=spatial(normal);predictions.append(normal.detach().cpu());input_measures.append(b['A']['input_projection_raw'].detach().cpu())
            row=dict(case_index=index,case_id=b['A']['case_id'][0],station_ids={k:b[k]['input_station_indices'][0,b[k]['view_mask'][0]].tolist() for k in ['A','B','H']},prediction=prediction,logits=moments(normal_logits),sigmoid_derivative=moments(sigmoid_derivative),logit_fraction_below_minus10=float((normal_logits < -10).float().mean()),normal_vs_zero_measurement=dict(output_mean_abs=float(difference.mean()),output_max_abs=float(difference.max()),relative_l1=float(difference.sum()/normal.abs().sum().clamp_min(1e-30)),latent_difference_l2=float(latent_difference.norm()),latent_relative_l2=float(latent_difference.norm()/normal_latent.norm().clamp_min(1e-30)),geometry_path_mask_unchanged=True),geometry_and_input_scale=geometry(b['A']))
            model.train();loss,parts,volumes,projections=objective(model,b);named=list(model.named_parameters());parameters=tuple(p for n,p in named)
            branches={}
            for branch,value in dict(total=loss,**parts).items():
                vg=torch.autograd.grad(value,volumes,retain_graph=True,allow_unused=False)
                pg=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                branches[branch]=dict(loss=float(value.detach()),outer_weight=1. if branch=='total' else FROZEN_LOSS[branch],parameter_gradients=gradient_norms(named,pg),d_loss_d_prediction_A=volume_gradient(vg[0],volumes[0]),d_loss_d_prediction_B=volume_gradient(vg[1],volumes[1]))
            row['loss_branch_gradients']=branches
            del loss,parts,volumes,projections,pg,vg
            row['near_zero_constant_field_probe']=near_zero_probe(b)
            if index==0:row['current_weight_chunk_gradient_check']=chunk_check(model,b['A'])
            rows.append(row);write_json(diag/f'case{index}.json',row)
            print(f'Independent no-GT diagnosis case {index}: mean={prediction["mean"]:.8g}, std={prediction["std"]:.8g}, zero response={row["normal_vs_zero_measurement"]["relative_l1"]:.8g}',flush=True)
        pairwise=[dict(left=rows[i]['case_id'],right=rows[j]['case_id'],prediction_mean_abs=float((predictions[i]-predictions[j]).abs().mean()),prediction_max_abs=float((predictions[i]-predictions[j]).abs().max()),relative_l1=float((predictions[i]-predictions[j]).abs().sum()/predictions[i].abs().sum().clamp_min(1e-30))) for i in range(3) for j in range(i+1,3)]
        assert state_hash(model.state_dict())==before and all(p.grad is None for p in model.parameters())
        audit=guard.summary();assert not any(audit[k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
        diagnosis=dict(completed=True,registration=registration,checkpoint_sha256=checkpoint_sha,snapshot_epoch=history[-1]['epoch']+1,snapshot_steps=history[-1]['optimizer_steps'],optimizer_state_steps=optimizer_steps,model_state_sha256=before,initial_state_sha256=fresh_hash,parameters_differ_from_initial=before!=fresh_hash,diagnostic_model_state_unchanged=True,optimizer_created=False,optimizer_steps=0,all_parameter_grad_buffers_none=True,formal_model_optimizer_config_selector_untouched=True,seconds=time.perf_counter()-started,case_results=rows,pairwise_case_outputs=pairwise,access=audit)
        write_json(target,diagnosis)
    print(json.dumps(dict(snapshot_epoch=diagnosis['snapshot_epoch'],snapshot_steps=diagnosis['snapshot_steps'],seconds=diagnosis['seconds'],access=audit),indent=2),flush=True)

if __name__=='__main__':main()
