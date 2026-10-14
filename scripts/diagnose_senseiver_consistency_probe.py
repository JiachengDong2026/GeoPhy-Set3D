#!/usr/bin/env python3
"""Same-graph gradients and common-amplitude diagnostic; no optimizer here."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import torch
from geophy_set3d.training.senseiver_consistency_probe import from_projections,from_volumes
from geophy_set3d.training.senseiver_relative_probe import objective as prior_objective
from geophy_set3d.training.senseiver_init_probe import group_norm
from diagnose_senseiver_relative_probe import field_gradient,interaction,vector


def group_interactions(named,a,b):
    groups={'encoder':lambda n:n.startswith('encoder.') and n!='encoder.latent',
            'latents':lambda n:n=='encoder.latent','decoder':lambda n:n.startswith('decoder.')}
    result={}
    for group,keep in groups.items():
        x=[v for (n,p),v in zip(named,a) if keep(n)]
        y=[v for (n,p),v in zip(named,b) if keep(n)]
        item=interaction(x,y);vx=vector(x);vy=vector(y)
        item.update(sum_gradient_l2=float((vx+vy).norm()),sum_squared_reconstructed=item['first_l2']**2+item['second_l2']**2+2*item['dot'])
        result[group]=item
    return result


def gradient_split(model,result,batch,check_reference=False):
    loss,parts,volumes,projections,diagnostics=result
    off=from_projections(volumes,projections,batch,0.)
    branches=dict(measurement_smooth=off[0],consistency_weighted=.05*parts['consistency'],
                  total_control=loss,total_treatment=off[0])
    named=list(model.named_parameters()) if model is not None else []
    params=tuple(p for n,p in named);gradients={};rows={}
    for key,value in branches.items():
        grads=torch.autograd.grad(value,params+volumes,retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads)
        gradients[key]=grads
        pg=grads[:len(params)];vg=grads[len(params):]
        rows[key]=dict(loss=float(value.detach()),parameter_gradient_l2=group_norm(named,pg) if named else None,
            dL_dC_A=field_gradient(vg[0]),dL_dC_B=field_gradient(vg[1]),
            uniform_output_shift_derivative=sum(float(g.detach().double().sum()) for g in vg))
    control=gradients['total_control'];treatment=gradients['total_treatment'];cons=gradients['consistency_weighted']
    remainder=[a-b-c for a,b,c in zip(control,treatment,cons)]
    error=float(vector(remainder).norm());reference_norm=float(vector(control).norm())
    assert error<=2e-6+2e-5*reference_norm
    torch.testing.assert_close(loss-off[0],.05*parts['consistency'],atol=2e-8,rtol=2e-4)
    reference={}
    if check_reference:
        class Cached:
            def __call__(self,b):
                assert b is batch['A'] or b is batch['B']
                return volumes[0] if b is batch['A'] else volumes[1]
        old=prior_objective(Cached(),batch,0.)
        torch.testing.assert_close(loss,old[0],atol=0,rtol=0)
        for k in parts:torch.testing.assert_close(parts[k],old[1][k],atol=0,rtol=0)
        oldgrads=torch.autograd.grad(old[0],params+volumes,retain_graph=True)
        for a,b in zip(control,oldgrads):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        reference=dict(loss_parts_bitwise_equal=True,gradient_difference_l2=float(vector([a-b for a,b in zip(control,oldgrads)]).norm()))
    smooth=gradients['measurement_smooth']
    return dict(branches=rows,interaction=dict(parameters=interaction(smooth[:len(params)],cons[:len(params)]) if named else None,
        outputs=interaction(smooth[len(params):],cons[len(params):]),
        parameter_groups=group_interactions(named,smooth[:len(params)],cons[:len(params)]) if named else None),
        equivalence=reference,only_difference=dict(loss_remainder=float((loss-off[0]-.05*parts['consistency']).detach()),
        gradient_remainder_l2=error,parameter_gradient_remainder_l2=float(vector(remainder[:len(params)]).norm()) if named else None,
        output_gradient_remainder_l2=float(vector(remainder[len(params):]).norm()),removed_consistency_loss=float((.05*parts['consistency']).detach())))


def amplitude_probe(volumes,batch,alphas):
    rows=[]
    for alpha in alphas:
        a=torch.tensor(alpha,device=volumes[0].device,requires_grad=True)
        prediction=tuple(a*v.detach() for v in volumes)
        result=from_volumes(prediction,batch,.05);loss,parts,_,_,diagnostics=result
        smooth=parts['hold']+.5*parts['cross']+.1*parts['observed'];cons=.05*parts['consistency']
        values=dict(measurement_smooth=smooth,consistency_weighted=cons,total_control=loss,total_treatment=smooth)
        row=dict(alpha=alpha,branches={k:dict(loss=float(v.detach()),dL_dalpha=float(torch.autograd.grad(v,a,retain_graph=True)[0])) for k,v in values.items()})
        ca,cb=volumes;difference=float((ca.detach()-cb.detach()).abs().double().mean())
        mean=.5*float((ca.detach().abs().double().mean()+cb.detach().abs().double().mean()))
        # Original consistency = alpha*d/(alpha*m+1e-6). Positive d/da is from epsilon.
        row['analytic_consistency_weighted_dL_dalpha']=.05*difference*1e-6/(alpha*mean+1e-6)**2
        row['unscaled_AB_mean_absolute_difference']=difference;row['unscaled_AB_mean_amplitude']=mean
        rows.append(row)
    return rows
