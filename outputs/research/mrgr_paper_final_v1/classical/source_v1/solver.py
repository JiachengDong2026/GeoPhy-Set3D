"""True nonlinear, box-constrained PG-BB baselines with restartable iterates."""
from common import *
import time
import numpy as np
import torch
from torch.nn import functional as F
from geophy_set3d.physics.projectors import mass_to_mole
from geophy_set3d.reconstruction.classical import regularizer

SCALE=32284.894580078173

class CachedProjector:
    """Exact frozen K96 grid_sample expression; cache only geometry, never a dense A."""
    def __init__(self,geometry,n=64,samples=96):
        self.chunks=[]
        for i in range(0,len(geometry['station_origin_m']),2):
            g={k:v[i:i+2] for k,v in geometry.items()}
            origin=g['station_origin_m'][:,None,None,None,:]
            direction=g['ray_direction'][:,:,:,None,:]
            near=g['t_near'][:,:,:,None];length=g['path_length_m'][:,:,:,None]
            fraction=(torch.arange(samples,dtype=origin.dtype,device=origin.device)+.5)/samples
            t=near+length*fraction;points=origin+direction*t[...,None]
            grid=(points[...,[2,1,0]]/.6*2-1).reshape(1,-1,1,1,3)
            self.chunks.append((grid,t.shape,length[...,0],g['valid_ray']))
    def __call__(self,c):
        result=[]
        for grid,shape,length,valid in self.chunks:
            v=F.grid_sample(c[None,None],grid,mode='bilinear',padding_mode='border',align_corners=True).reshape(shape)
            result.append(mass_to_mole(v.clamp(0,1)).mean(-1)*length*1e6*valid)
        return torch.cat(result)


def solve(projector,valid,y,method,config,*,resume=None,save=None,stop_after=None):
    """Last accepted iterate only. Resume includes BB step, gradient, patience & curve."""
    def objective(v):
        data=.5*((projector(v)[valid]-y[valid])/SCALE).square().mean()
        reg=v.new_zeros(()) if method=='NNLS-NL' else config['lambda']*regularizer(v,method,config.get('epsilon',1e-4))
        return data+reg,data
    def vg(v):
        v=v.detach().requires_grad_(True);loss,data=objective(v);g,=torch.autograd.grad(loss,v)
        assert torch.isfinite(g).all() and torch.isfinite(loss)
        return v.detach(),float(loss.detach()),float(data.detach()),g.detach()
    if resume:
        state=torch.load(resume,map_location=y.device,weights_only=False)
        assert state['method']==method and state['config']==config
        c,g=state['c'],state['g'];loss,data=state['loss'],state['data'];alpha=state['alpha'];stable=state['stable'];curve=state['curve'];evals=state['evals']
        # Validate complete solver state against fresh forward/derivative at the resumed iterate.
        _,l,d,gg=vg(c);assert abs(l-loss)<1e-6*max(1.,abs(l));torch.testing.assert_close(g,gg)
    else:
        c=torch.zeros((config.get('resolution',64),)*3,device=y.device,dtype=y.dtype)
        c,loss,data,g=vg(c);alpha=config['initial_step'];stable=0;curve=[];evals=1
    status='budget_exhausted'
    def persist():
        if save:
            path=Path(save);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
            torch.save(dict(method=method,config=config,c=c,g=g,loss=loss,data=data,alpha=alpha,stable=stable,curve=curve,evals=evals),temp);temp.replace(path)
    for step in range(len(curve)+1,config['iterations']+1):
        accepted=False
        for backtrack in range(config['max_backtracks']):
            new=(c-alpha*g).clamp(0,1);delta=new-c
            with torch.no_grad():newloss,newdata=objective(new)
            evals+=1
            if torch.isfinite(newloss) and float(newloss)<=loss+config['armijo']*float((g*delta).sum())+1e-12:accepted=True;break
            alpha*=.5
        if not accepted:status='line_search_failed';break
        oldloss=loss;new,newloss,newdata,newg=vg(new);evals+=1
        s=new-c;dg=newg-g;curvature=float((s*dg).sum())
        nextalpha=float(s.square().sum())/curvature if curvature>0 else alpha
        stable=stable+1 if abs(oldloss-newloss)<=config['tolerance']*max(abs(oldloss),1e-12) else 0
        projected=float((new-(new-newg).clamp(0,1)).abs().max())
        c,loss,data,g=new,newloss,newdata,newg
        curve.append(dict(iteration=step,objective=loss,data_objective=data,regularization=loss-data,step_size=alpha,backtracks=backtrack,projected_gradient_inf=projected))
        alpha=min(max(nextalpha,config['step_bounds'][0]),config['step_bounds'][1])
        converged=step>=config['minimum_iterations'] and stable>=config['patience']
        if save and (step%20==0 or converged or step==config['iterations'] or step==stop_after):persist()
        if converged:status='objective_stable';break
        if step==stop_after:status='interrupted_test';break
    persist()
    return c,dict(iterations=len(curve),status=status,curve=curve,objective=loss,data_objective=data,objective_evaluations=evals)
