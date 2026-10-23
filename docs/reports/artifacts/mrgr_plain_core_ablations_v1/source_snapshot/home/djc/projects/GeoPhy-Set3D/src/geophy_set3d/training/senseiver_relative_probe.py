"""Explicit relative-coefficient study; all historical numerical code stays frozen."""
import copy
import math
import time
import hashlib
import json
import torch
from torch.nn import functional as F
from geophy_set3d.data.measurement_only import assert_no_gt
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input,Y_SCALE
from geophy_set3d.training.selfsup import (measurement_discrepancy,consistency_discrepancy,
    assert_disjoint,projection_scores,move_dual_batch)
from geophy_set3d.training.senseiver_init_probe import validate as validate_initial_probe

NAME='ray_senseiver_relative_probe_v1'
INTERVENTION=dict(control_relative_l1=.1,treatment_relative_l1=0.,constant_levels=[0.,1e-8,1e-7,1e-6,1e-5],amplitude_threshold=1e-6)


def validate(c):
    base=copy.deepcopy(c)
    if base.pop('intervention',None)!=INTERVENTION or base['experiment']['name']!=NAME:
        raise ValueError('Only preregistered relative-L1 on/off study allowed')
    base['experiment']['name']='ray_senseiver_init_probe_v1'
    validate_initial_probe(base)
    return c


def components(pic,target):
    """Diagnostics only. Relative is per-case sum/sum, never per-ray division."""
    valid=target['input_valid_ray'].detach().bool()&target['view_mask'][:,:,None,None,None]
    p=torch.where(valid,pic,0)/Y_SCALE
    y=torch.where(valid,target['input_projection_raw'].detach(),0)/Y_SCALE
    smooth=measurement_discrepancy(pic,target,beta=.01,relative_l1=0.,relative_epsilon=1e-6)
    relative=((p-y).abs().flatten(1).sum(1)/y.abs().flatten(1).sum(1).clamp_min(1e-6)).mean()
    return smooth,relative


def from_volumes(volumes,batch,relative_l1):
    if relative_l1 not in (0.,.1):raise ValueError('No coefficient sweep')
    ca,cb=volumes
    projections={label:{k:project_input(c,batch[k]) for k in ('A','B','H')} for label,c in [('C_A',ca),('C_B',cb)]}
    return from_projections(volumes,projections,batch,relative_l1)


def from_projections(volumes,projections,batch,relative_l1):
    ca,cb=volumes
    pairs={'hold':(('C_A','H'),('C_B','H')),'cross':(('C_A','B'),('C_B','A')),'observed':(('C_A','A'),('C_B','B'))}
    parts={k:sum(measurement_discrepancy(projections[label][key],batch[key],beta=.01,relative_l1=relative_l1,relative_epsilon=1e-6) for label,key in ps)*.5 for k,ps in pairs.items()}
    parts['consistency']=consistency_discrepancy(ca,cb)
    total=parts['hold']+.5*parts['cross']+.1*parts['observed']+.05*parts['consistency']
    split={}
    for k,ps in pairs.items():
        values=[components(projections[label][key],batch[key]) for label,key in ps]
        split[k]=dict(smooth=.5*sum(v[0] for v in values),relative_raw=.5*sum(v[1] for v in values))
    smooth=split['hold']['smooth']+.5*split['cross']['smooth']+.1*split['observed']['smooth']
    relative=split['hold']['relative_raw']+.5*split['cross']['relative_raw']+.1*split['observed']['relative_raw']
    diagnostics=dict(measurement_smooth=smooth,measurement_relative_raw=relative,
                     measurement_relative_weighted=.1*relative,consistency_weighted=.05*parts['consistency'],
                     **{f'{k}_{sub}':v for k,values in split.items() for sub,v in values.items()})
    if not all(torch.isfinite(v).all() for v in [total,*parts.values(),*diagnostics.values()]):raise FloatingPointError('Nonfinite loss')
    return total,parts,volumes,projections,diagnostics


def objective(model,batch,relative_l1):
    assert_no_gt(batch);assert_disjoint(batch)
    ca,cb=model(batch['A']),model(batch['B'])
    if ca.shape!=(len(batch['A']['case_id']),1,64,64,64) or cb.shape!=ca.shape or ca.dtype!=torch.float32 or cb.dtype!=torch.float32:
        raise ValueError('Full FP32 XYZ 64³ required')
    if not all(torch.isfinite(c).all() and ((c>=0)&(c<=1)).all() for c in [ca,cb]):raise FloatingPointError('Nonfinite/invalid concentration')
    return from_volumes((ca,cb),batch,relative_l1)


def terms(parts,diagnostics):
    s=diagnostics['measurement_smooth'];r=diagnostics['measurement_relative_weighted'];cons=.05*parts['consistency']
    return dict(measurement_smooth=s,measurement_relative_weighted=r,consistency_weighted=cons,
                hold_control=parts['hold'],cross_control_weighted=.5*parts['cross'],observed_control_weighted=.1*parts['observed'],
                total_control=s+r+cons,total_treatment=s+cons,
                **{k:diagnostics[k] for k in ['hold_smooth','cross_smooth','observed_smooth','hold_relative_raw','cross_relative_raw','observed_relative_raw']})


def exit_event(epoch,history,health,c_init,previous):
    full=health['full_fixed_subset'];collapse=epoch>5 and full['measurement']['heldout']['relative_l1']>=.98 and health['logits']['fraction_below_minus12']>.95 and full['output']['mean']<.01*c_init
    cs=previous.get('collapse_streak',0)+1 if collapse else 0
    windows={}
    for metric in ['relative_l1','relative_l2']:
        recent=sum(r['eval']['measurement']['heldout'][metric] for r in history[-5:])/5 if epoch>=10 else None
        older=sum(r['eval']['measurement']['heldout'][metric] for r in history[-10:-5])/5 if epoch>=10 else None
        windows[metric]=dict(recent5=recent,previous5=older,relative_improvement=(older-recent)/older if older is not None and older>0 else None)
    dependency_count=sum(r['measurement_dependence']['relative_MA']>=1e-3 for r in health['diagnostic_cases'])
    failure=epoch>20 and windows['relative_l1']['recent5']>=.98 and all(w['relative_improvement']<.01 for w in windows.values()) and dependency_count<2
    fs=previous.get('failure_streak',0)+1 if failure else 0
    reason='NEAR_ZERO_COLLAPSE' if cs>=2 else 'FAILED_TO_LEARN' if fs>=2 else 'BUDGET_COMPLETE' if epoch==40 else None
    return dict(epoch=epoch,collapse_condition=collapse,collapse_streak=cs,failure_condition=failure,failure_streak=fs,windows=windows,dependency_case_count=dependency_count,reason=reason)


def verify_resume(saved,c,arm,initial,registration_hash,source_hash):
    if saved['config']!=c or saved['arm']!=arm or saved['relative_l1']!=(.1 if arm=='control' else 0.) or saved['initial_hash']!=initial or saved['registration_sha256']!=registration_hash or saved['source_manifest_sha256']!=source_hash:
        raise ValueError('Resume identity mismatch')
    if saved['exit_state'].get('reason'):raise ValueError('Scientifically stopped arm cannot resume')
    if saved['updates']!=4*saved['epoch'] or saved['scheduler']['last_epoch']!=saved['epoch']:raise ValueError('Incomplete update/scheduler state')


def run_epoch(model,loader,c,relative_l1,optimizer=None):
    """Same sample/accumulation and metrics semantics; explicit independent objective."""
    training=optimizer is not None;model.train(training);started=time.perf_counter()
    totals={k:0. for k in ['total','hold','cross','observed','consistency']};splits={};count=0
    scores={k:[] for k in ['heldout','cross','observed']};moments=[];examples={};pairings=[];gradient_max=0.
    accumulation=c['training']['gradient_accumulation']
    if training:optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step,raw in enumerate(loader):
            batch=move_dual_batch(raw,c['training']['device'])
            loss,parts,volumes,projections,diagnostics=objective(model,batch,relative_l1)
            n=len(volumes[0]);count+=n
            if training:
                start=(step//accumulation)*accumulation;stop=min(start+accumulation,len(loader))
                group_samples=min(stop*loader.batch_size,len(loader.dataset))-start*loader.batch_size
                (loss*n/group_samples).backward()
                if (step+1)%accumulation==0 or step+1==len(loader):
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);gradient_max=max(gradient_max,float(norm));optimizer.step();optimizer.zero_grad(set_to_none=True)
            for k,v in dict(total=loss,**parts).items():totals[k]+=float(v.detach())*n
            for k,v in diagnostics.items():splits[k]=splits.get(k,0.)+float(v.detach())*n
            for k,values in projection_scores(projections,batch).items():scores[k].extend(values)
            for pred in torch.cat(volumes).detach():
                mean=float(pred.mean());std=float(pred.std(unbiased=False));moments.append((float(pred.min()),float(pred.max()),mean,std*std+mean*mean))
            for i,case_id in enumerate(batch['A']['case_id']):
                examples[case_id]=volumes[0][i].detach().cpu()
                pairings.append((case_id,*[batch[k]['input_station_indices'][i,batch[k]['view_mask'][i]].tolist() for k in ['A','B','H']]))
    mean=sum(v[2] for v in moments)/len(moments);ordered=[examples[k] for k in sorted(examples)]
    output=dict(min=min(v[0] for v in moments),max=max(v[1] for v in moments),mean=mean,std=math.sqrt(max(0.,sum(v[3] for v in moments)/len(moments)-mean*mean)),case_to_case_mean_absolute_difference=sum(float((a-b).abs().mean()) for a,b in zip(ordered,ordered[1:]))/7)
    return dict(loss={k:v/count for k,v in totals.items()},loss_diagnostics={k:v/count for k,v in splits.items()},measurement={k:{m:sum(v[m] for v in values)/len(values) for m in ['relative_l1','relative_l2']} for k,values in scores.items()},output=output,gradient_norm_max=gradient_max,case_station_digest=hashlib.sha256(json.dumps(sorted(pairings)).encode()).hexdigest(),sample_count=count,gt_used=False,seconds=time.perf_counter()-started)
