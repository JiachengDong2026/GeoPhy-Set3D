"""Independent bounded 3-D baselines. Frozen physics/training code is untouched."""
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from geophy_set3d.physics.projectors import differentiable_project, mass_to_mole

METHANE, AIR = 16.043, 28.965


def mole_to_mass(chi):
    return chi*METHANE/(chi*METHANE+(1-chi)*AIR)


def physical_gradient(c, extent=.6):
    """Forward differences / physical spacing; zero outward derivative (Neumann)."""
    result=[]
    for axis in range(3):
        d=torch.diff(c,dim=axis)/(extent/(c.shape[axis]-1))
        pad=[0]*6;pad[2*(2-axis)+1]=1
        result.append(F.pad(d,pad))
    return torch.stack(result)


def regularizer(c, method, epsilon=1e-4):
    g=physical_gradient(c)
    if method=='Tikhonov':return .5*g.square().sum(0).mean()
    if method=='TV':return (torch.sqrt(g.square().sum(0)+epsilon**2)-epsilon).mean()
    raise ValueError(method)


def geometry_tensor(geometry, device):
    return {k:torch.as_tensor(v,device=device,dtype=torch.bool if k=='valid_ray' else torch.float32)
            for k,v in geometry.items() if k in ('station_origin_m','ray_direction','t_near','path_length_m','valid_ray')}


def project(c, geometry, samples=96):
    # Same two-station chunking and original function as frozen network evaluation.
    return torch.cat([differentiable_project(c,{k:v[i:i+2] for k,v in geometry.items()},samples)
                      for i in range(0,len(geometry['station_origin_m']),2)])


class RayMatrix:
    """Sparse A: XYZ nodal trilinear chi, midpoint ds; units ppm*m. True A.T."""
    def __init__(self, geometry, n=64, samples=96, device='cpu', dtype=torch.float32):
        start=time.perf_counter()
        valid=np.asarray(geometry['valid_ray'],bool).reshape(-1)
        self.valid=torch.as_tensor(valid,device=device)
        direction=np.asarray(geometry['ray_direction'],np.float64).reshape(-1,3)[valid]
        origin=np.broadcast_to(np.asarray(geometry['station_origin_m'])[:,None,None,:],
                               (*geometry['valid_ray'].shape,3)).reshape(-1,3)[valid]
        near=np.asarray(geometry['t_near'],np.float64).reshape(-1)[valid]
        length=np.asarray(geometry['path_length_m'],np.float64).reshape(-1)[valid]
        t=near[:,None]+length[:,None]*(np.arange(samples)+.5)/samples
        p=np.clip((origin[:,None,:]+direction[:,None,:]*t[:,:,None])/.6*(n-1),0,n-1)
        lo=np.minimum(np.floor(p).astype(np.int64),n-2);f=p-lo
        rr=np.broadcast_to(np.arange(len(length))[:,None],t.shape).reshape(-1)
        rows=[];cols=[];values=[]
        for a in (0,1):
            for b in (0,1):
                for c in (0,1):
                    idx=lo+np.array([a,b,c])
                    w=np.prod(np.where(np.array([a,b,c]),f,1-f),axis=-1)*length[:,None]*1e6/samples
                    rows.append(rr);cols.append(((idx[:,:,0]*n+idx[:,:,1])*n+idx[:,:,2]).reshape(-1));values.append(w.reshape(-1))
        matrix=sp.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(length),n**3)).tocsr()
        matrix.eliminate_zeros()
        self.n=n;self.shape=geometry['valid_ray'].shape;self.nnz=matrix.nnz
        def tensor(m):
            return torch.sparse_csr_tensor(torch.as_tensor(m.indptr.astype(np.int64),device=device),
                torch.as_tensor(m.indices.astype(np.int64),device=device),torch.as_tensor(m.data,device=device,dtype=dtype),size=m.shape)
        self.a=tensor(matrix);self.at=tensor(matrix.T.tocsr())
        self.row_sum=torch.as_tensor(np.asarray(matrix.sum(1)).ravel(),device=device,dtype=dtype)
        self.col_sum=torch.as_tensor(np.asarray(matrix.sum(0)).ravel(),device=device,dtype=dtype)
        if (self.row_sum<=0).any():raise ValueError('Valid zero-length ray')
        if device!='cpu':torch.cuda.synchronize(device)
        self.build_seconds=time.perf_counter()-start
        self.bytes=sum(t.numel()*t.element_size() for m in (self.a,self.at) for t in (m.values(),m.col_indices(),m.crow_indices()))

    def forward(self,x):return torch.mv(self.a,x.reshape(-1))
    def adjoint(self,y):return torch.mv(self.at,y)
    def measurement(self,y):return y.reshape(-1)[self.valid]


def sart(blocks, measurements, config, n=64):
    """Each sweep updates all station blocks in supplied order, once each."""
    x=torch.zeros(n**3,device=blocks[0].a.device,dtype=blocks[0].a.dtype)
    ys=[a.measurement(y) for a,y in zip(blocks,measurements)]
    curve=[];status='budget_exhausted';previous=None;stable=0
    for step in range(1,config['iterations']+1):
        for a,y in zip(blocks,ys):
            correction=a.adjoint((y-a.forward(x))/a.row_sum)
            x=(x+config['relaxation']*correction/a.col_sum.clamp_min(1e-12)).clamp(0,1)
        residual=sum((a.forward(x)-y).square().sum() for a,y in zip(blocks,ys))/sum(y.square().sum() for y in ys).clamp_min(1e-12)
        value=float(residual)
        if not np.isfinite(value):status='nonfinite';break
        stable=stable+1 if previous is not None and abs(previous-value)<=config['tolerance']*max(previous,1e-12) else 0
        curve.append(dict(iteration=step,objective=.5*value,solver_relative_l2=value**.5))
        previous=value
        if step>=config['minimum_iterations'] and stable>=config['patience']:status='objective_stable';break
    return x.reshape(n,n,n),dict(iterations=step,status=status,curve=curve,solver_relative_l2=value**.5)


def variational(geometry, y, method, config, scale, n=64):
    """Projected BB gradient with monotone Armijo; nonlinear C data term, no logits."""
    valid=geometry['valid_ray'].bool();c=torch.zeros((n,n,n),device=y.device)
    def objective(v):
        residual=(project(v,geometry)[valid]-y[valid])/scale
        data=.5*residual.square().mean()
        return data+config['lambda']*regularizer(v,method,config.get('epsilon',1e-4)),data
    def value_grad(v):
        v=v.detach().requires_grad_(True);loss,data=objective(v)
        g,=torch.autograd.grad(loss,v)
        return v.detach(),float(loss.detach()),float(data.detach()),g.detach()
    c,loss,data,g=value_grad(c);alpha=config['initial_step'];curve=[];status='budget_exhausted';stable=0;evals=1
    for step in range(1,config['iterations']+1):
        accepted=False
        for backtrack in range(config['max_backtracks']):
            new=(c-alpha*g).clamp(0,1);delta=new-c
            with torch.no_grad():newloss,newdata=objective(new)
            evals+=1
            if torch.isfinite(newloss) and float(newloss)<=loss+config['armijo']*float((g*delta).sum())+1e-12:
                accepted=True;break
            alpha*=.5
        if not accepted:status='line_search_failed';break
        oldloss=loss;new,newloss,newdata,newg=value_grad(new);evals+=1
        s=new-c;dg=newg-g;curvature=float((s*dg).sum())
        nextalpha=float(s.square().sum())/curvature if curvature>0 else alpha
        stable=stable+1 if abs(oldloss-newloss)<=config['tolerance']*max(abs(oldloss),1e-12) else 0
        projected=float((new-(new-newg).clamp(0,1)).abs().max())
        c,loss,data,g=new,newloss,newdata,newg
        curve.append(dict(iteration=step,objective=loss,data_objective=data,regularization=loss-data,
                          step_size=alpha,backtracks=backtrack,projected_gradient_inf=projected))
        alpha=min(max(nextalpha,config['step_bounds'][0]),config['step_bounds'][1])
        if step>=config['minimum_iterations'] and stable>=config['patience']:status='objective_stable';break
    with torch.no_grad():r=project(c,geometry)[valid]-y[valid]
    return c,dict(iterations=len(curve),status=status,curve=curve,objective=loss,data_objective=data,
                 objective_evaluations=evals,solver_relative_l2=float((r.square().sum()/y[valid].square().sum().clamp_min(1e-12)).sqrt()))
