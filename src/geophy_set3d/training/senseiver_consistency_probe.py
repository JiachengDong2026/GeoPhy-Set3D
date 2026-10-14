"""Last bounded loss study: relative fixed zero, consistency explicitly .05 or0."""
import copy
import time
import math
import hashlib
import json
from geophy_set3d.training.selfsup import move_dual_batch,projection_scores
import torch
from geophy_set3d.training.senseiver_relative_probe import (
    validate as validate_relative, INTERVENTION as RELATIVE_INTERVENTION,
    objective as relative_objective, from_volumes as relative_from_volumes,
    from_projections as relative_from_projections, exit_event)

NAME='ray_senseiver_consistency_probe_v1'
INTERVENTION=dict(relative_l1=0.,control_consistency=.05,treatment_consistency=0.,amplitude_alphas=[1.,.1,.01])


def validate(c):
    base=copy.deepcopy(c)
    if base.pop('intervention',None)!=INTERVENTION or base['experiment']['name']!=NAME:
        raise ValueError('Only final registered consistency on/off allowed')
    base['experiment']['name']='ray_senseiver_relative_probe_v1'
    base['intervention']=RELATIVE_INTERVENTION
    # Config loss now describes the control; old validators remain strict/unmodified.
    if base['loss']['relative_l1']!=0.:raise ValueError('Relative is fixed zero')
    base['loss']['relative_l1']=.1
    validate_relative(base)
    return c


def with_consistency(result,consistency_weight):
    if consistency_weight not in (.05,0.):raise ValueError('No consistency coefficient search')
    _,parts,volumes,projections,diagnostics=result
    smooth=parts['hold']+.5*parts['cross']+.1*parts['observed']
    control=smooth+.05*parts['consistency']
    total=control if consistency_weight==.05 else smooth
    diagnostics=dict(diagnostics,consistency_raw=parts['consistency'],
                     total_control=control,total_treatment=smooth,
                     actual_relative_weight=total.new_tensor(0.),
                     actual_consistency_weight=total.new_tensor(consistency_weight))
    if not all(torch.isfinite(v).all() for v in [total,*diagnostics.values()]):raise FloatingPointError('Nonfinite objective')
    return total,parts,volumes,projections,diagnostics


def objective(model,batch,consistency_weight):
    return with_consistency(relative_objective(model,batch,0.),consistency_weight)


def from_volumes(volumes,batch,consistency_weight):
    return with_consistency(relative_from_volumes(volumes,batch,0.),consistency_weight)


def from_projections(volumes,projections,batch,consistency_weight):
    return with_consistency(relative_from_projections(volumes,projections,batch,0.),consistency_weight)


def verify_resume(saved,c,arm,initial,registration_hash,source_hash):
    if saved['config']!=c or saved['arm']!=arm or saved['relative_l1']!=0. or saved['consistency_weight']!=(.05 if arm=='control' else 0.) or saved['initial_hash']!=initial or saved['registration_sha256']!=registration_hash or saved['source_manifest_sha256']!=source_hash:
        raise ValueError('Resume identity mismatch')
    if saved['exit_state'].get('reason'):raise ValueError('Scientifically stopped arm cannot resume')
    if saved['updates']!=4*saved['epoch'] or saved['scheduler']['last_epoch']!=saved['epoch']:
        raise ValueError('Incomplete update/scheduler state')


def run_epoch(model,loader,c,consistency_weight,optimizer=None):
    """Same sample/accumulation and metrics semantics; explicit independent objective."""
    training=optimizer is not None;model.train(training);started=time.perf_counter()
    totals={k:0. for k in ['total','hold','cross','observed','consistency']};splits={};count=0
    scores={k:[] for k in ['heldout','cross','observed']};moments=[];examples={};pairings=[];gradient_max=0.
    accumulation=c['training']['gradient_accumulation']
    if training:optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step,raw in enumerate(loader):
            batch=move_dual_batch(raw,c['training']['device'])
            loss,parts,volumes,projections,diagnostics=objective(model,batch,consistency_weight)
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
