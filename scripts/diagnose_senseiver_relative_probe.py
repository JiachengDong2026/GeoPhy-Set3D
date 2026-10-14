#!/usr/bin/env python3
"""No optimizer: shared-prediction branch gradients and real-projector ray groups."""
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import torch
from torch.nn import functional as F
from geophy_set3d.models.geolift_oari_mrgr_res3d import Y_SCALE
from geophy_set3d.physics.projectors import mass_to_mole
from geophy_set3d.training.selfsup import objective as original_objective
from geophy_set3d.training.senseiver_relative_probe import from_projections,terms
from geophy_set3d.training.senseiver_init_probe import group_norm
from diagnose_ray_senseiver_learning import moments

GROUPS=['exact_zero','positive_le_1e_6','above_1e_6']


def masks(target):
    valid=target['input_valid_ray'].bool()&target['view_mask'][:,:,None,None,None]
    y=target['input_projection_raw'].detach()/Y_SCALE
    return valid,y,dict(exact_zero=valid&(y==0),positive_le_1e_6=valid&(y>0)&(y<=1e-6),above_1e_6=valid&(y>1e-6))


def amplitude_inventory(batch):
    result={}
    for k,target in batch.items():
        valid,y,groups=masks(target)
        result[k]=dict(valid_rays=int(valid.sum()),groups={g:dict(count=int(mask.sum()),normalized_target=moments(y[mask]) if mask.any() else None) for g,mask in groups.items()})
    return result


def vector(values):
    return torch.cat([v.detach().double().flatten() for v in values if v is not None])


def interaction(a,b):
    x=vector(a);y=vector(b);dot=float(torch.dot(x,y));den=float(x.norm()*y.norm())
    return dict(dot=dot,cosine=dot/den if den>0 else None,first_l2=float(x.norm()),second_l2=float(y.norm()))


def field_gradient(g):
    x=g.detach().double()
    return dict(l2=float(x.norm()),min=float(x.min()),max=float(x.max()),mean=float(x.mean()),
                sum=float(x.sum()),positive_fraction=float((x>0).double().mean()),negative_fraction=float((x<0).double().mean()),zero_fraction=float((x==0).double().mean()))


def gradient_split(model,control_output,batch,scalar=None,check_reference=False):
    loss,parts,volumes,projections,diagnostics=control_output
    named=list(model.named_parameters()) if model is not None else []
    params=tuple(p for n,p in named);ca,cb=volumes
    branch=terms(parts,diagnostics);branch['total_control']=loss
    treatment=from_projections(volumes,projections,batch,0.)
    branch['total_treatment']=treatment[0]
    rows={};gradients={}
    for name,value in branch.items():
        allgrads=torch.autograd.grad(value,params+(ca,cb),retain_graph=True,allow_unused=True)
        pg=allgrads[:len(params)];vg=allgrads[len(params):]
        assert all(g is None or torch.isfinite(g).all() for g in allgrads)
        gradients[name]=allgrads
        uniform=sum(float(g.detach().double().sum()) for g in vg if g is not None)
        rows[name]=dict(loss=float(value.detach()),parameter_gradient_l2=group_norm(named,pg) if named else None,
                        dL_dC_A=field_gradient(vg[0]),dL_dC_B=field_gradient(vg[1]),dL_dc_uniform_output_shift=uniform)
        if scalar is not None:rows[name]['dL_dc_constant']=float(torch.autograd.grad(value,scalar,retain_graph=True)[0])
    delta=float((loss-treatment[0]-.1*diagnostics['measurement_relative_raw']).detach())
    actual=gradients['total_control'];removed=gradients['measurement_relative_weighted'];off=gradients['total_treatment']
    diffs=[a-b-r for a,b,r in zip(actual,off,removed)]
    diff_norm=float(vector(diffs).norm());scale_norm=float(vector(actual).norm())
    assert diff_norm<=2e-6+2e-5*scale_norm
    reference={}
    if check_reference:
        class Cached:
            def __call__(self,b):
                assert b is batch['A'] or b is batch['B']
                return ca if b is batch['A'] else cb
        old=original_objective(Cached(),batch)
        torch.testing.assert_close(loss,old[0],rtol=0,atol=0)
        for k in parts:torch.testing.assert_close(parts[k],old[1][k],rtol=0,atol=0)
        grads=torch.autograd.grad(old[0],params+(ca,cb),retain_graph=True)
        for a,b in zip(actual,grads):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        reference=dict(loss_bitwise_equal=True,all_parts_bitwise_equal=True,gradient_difference_l2=float(vector([a-b for a,b in zip(actual,grads)]).norm()))
    return dict(branches=rows,interactions={f'{a}__{b}':dict(parameters=interaction(gradients[a][:len(params)],gradients[b][:len(params)]) if named else None,outputs=interaction(gradients[a][len(params):],gradients[b][len(params):])) for a,b in [('measurement_smooth','measurement_relative_weighted'),('measurement_smooth','consistency_weighted'),('measurement_relative_weighted','consistency_weighted')]},
                equivalence=reference,only_difference=dict(loss_remainder=delta,gradient_remainder_l2=diff_norm,reference_gradient_l2=scale_norm,
                    output_gradient_remainder_l2=float(vector(diffs[len(params):]).norm()),removed_weighted_relative_loss=float((.1*diagnostics['measurement_relative_raw']).detach())))


def grouped_constant_contributions(projections,batch,scalar):
    """Masked contributions retain FULL original valid-count/target-sum denominators.
    Backprop through actual K96 projector to concentration scalar for each group.
    """
    result={}
    pairs={'hold':(('C_A','H'),('C_B','H')),'cross':(('C_A','B'),('C_B','A')),'observed':(('C_A','A'),('C_B','B'))}
    combined={g:dict(smooth=scalar*0,relative_raw=scalar*0) for g in GROUPS}
    for branch,ps in pairs.items():
        result[branch]={}
        weight={'hold':1.,'cross':.5,'observed':.1}[branch]
        for group in GROUPS:
            smooth=scalar*0;relative=scalar*0
            for label,key in ps:
                target=batch[key];valid,y,groups=masks(target);mask=groups[group]
                p=torch.where(valid,projections[label][key],0)/Y_SCALE
                count=valid.flatten(1).sum(1).clamp_min(1)
                denominator=torch.where(valid,y,0).abs().flatten(1).sum(1).clamp_min(1e-6)
                smooth=smooth+.5*((F.smooth_l1_loss(p,y,beta=.01,reduction='none')*mask).flatten(1).sum(1)/count).mean()
                relative=relative+.5*(((p-y).abs()*mask).flatten(1).sum(1)/denominator).mean()
            result[branch][group]=dict(smooth_loss=float(smooth.detach()),relative_raw_loss=float(relative.detach()),
                        smooth_dL_dc=float(torch.autograd.grad(smooth,scalar,retain_graph=True)[0]),
                        relative_raw_dL_dc=float(torch.autograd.grad(relative,scalar,retain_graph=True)[0]))
            combined[group]['smooth']=combined[group]['smooth']+weight*smooth
            combined[group]['relative_raw']=combined[group]['relative_raw']+weight*relative
    result['weighted_measurement_total']={g:dict(smooth_dL_dc=float(torch.autograd.grad(v['smooth'],scalar,retain_graph=True)[0]),relative_raw_dL_dc=float(torch.autograd.grad(v['relative_raw'],scalar,retain_graph=True)[0]),relative_weighted_dL_dc=float(torch.autograd.grad(.1*v['relative_raw'],scalar,retain_graph=True)[0])) for g,v in combined.items()}
    return result


def float64_scalar_probe(batch,level):
    """Numerical check of uniform-field analytic relation, not a training projector."""
    c=torch.tensor(level,dtype=torch.float64,device=batch['A']['input_projection_raw'].device,requires_grad=True)
    smooth={};relative={}
    for key,target in batch.items():
        valid=target['input_valid_ray'].bool()&target['view_mask'][:,:,None,None,None]
        y=torch.where(valid,target['input_projection_raw'].double(),0)/Y_SCALE
        p=torch.where(valid,1e6*target['input_path_length_m'].double()*mass_to_mole(c),0)/Y_SCALE
        smooth[key]=(F.smooth_l1_loss(p,y,beta=.01,reduction='none').flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).mean()
        relative[key]=((p-y).abs().flatten(1).sum(1)/y.abs().flatten(1).sum(1).clamp_min(1e-6)).mean()
    s=smooth['H']+.5*(smooth['A']+smooth['B'])*.5+.1*(smooth['A']+smooth['B'])*.5
    r=relative['H']+.5*(relative['A']+relative['B'])*.5+.1*(relative['A']+relative['B'])*.5
    return dict(smooth_dL_dc=float(torch.autograd.grad(s,c,retain_graph=True)[0]),relative_weighted_dL_dc=float(torch.autograd.grad(.1*r,c,retain_graph=True)[0]),total_control_dL_dc=float(torch.autograd.grad(s+.1*r,c)[0]))


def validate_group_sum(split,contribution):
    for key,branch in [('smooth','measurement_smooth'),('relative_raw','measurement_relative_weighted')]:
        value=sum(r[key+'_dL_dc'] for r in contribution['weighted_measurement_total'].values())
        if key=='relative_raw':value*=.1
        actual=split['branches'][branch]['dL_dc_constant']
        assert abs(value-actual)<=2e-4+5e-5*abs(actual),(key,value,actual)


if __name__=='__main__':
    # Entry is the registered runner, ensuring fixed inputs/source and access guard.
    raise SystemExit('Use run_senseiver_relative_probe.py --phase diagnose')
