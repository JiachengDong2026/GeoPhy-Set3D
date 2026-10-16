import torch, json
from pathlib import Path
from itertools import product
from geophy_set3d.physics.projectors import differentiable_project

torch.set_num_threads(1)
torch.manual_seed(42)
dtype=torch.float64
n=3; k=7; extent=.6; scale=32284.894580078173
origin=torch.tensor([[.1,.1,.1]],dtype=dtype)
dirs=torch.tensor([[[[1.,0,0],[0,1.,0],[1.,1.,1.]]]],dtype=dtype)
dirs=dirs/dirs.norm(dim=-1,keepdim=True)
length=torch.tensor([[[.15,.2,.3]]],dtype=dtype)
near=torch.zeros_like(length)
geom={'station_origin_m':origin,'ray_direction':dirs,'t_near':near,'path_length_m':length,'valid_ray':torch.ones_like(length,dtype=torch.bool)}
fraction=(torch.arange(k,dtype=dtype)+.5)/k
points=origin[:,None,None,None,:]+dirs[...,None,:]*(near[...,None]+length[...,None]*fraction)[...,None]
u=(points/extent*(n-1)).clamp(0,n-1).reshape(-1,k,3)
lo=u.floor().long().clamp(max=n-2);t=u-lo
indices=[];weights=[]
for corner in product((0,1),repeat=3):
 c=torch.tensor(corner);xyz=lo+c
 indices.append((xyz[...,0]*n+xyz[...,1])*n+xyz[...,2])
 weights.append(torch.where(c.bool(),t,1-t).prod(-1))
idx=torch.stack(indices,-1);w=torch.stack(weights,-1)
a=16.043;b=28.965
coef=1e6*length.reshape(-1)/(scale*k)
def gather(c):return (c.reshape(-1)[idx]*w).sum(-1)
def project(c):
 s=gather(c).clamp(0,1)
 chi=(s/a)/(s/a+(1-s)/b)
 return coef*chi.sum(-1)
def jt(c,v):
 s=gather(c)
 cp=torch.where((s>=0)&(s<=1),a*b/(a+(b-a)*s.clamp(0,1)).square(),0)
 contributions=coef[:,None,None]*cp[...,None]*v[:,None,None]*w
 return c.new_zeros(c.numel()).scatter_add(0,idx.reshape(-1),contributions.reshape(-1)).reshape_as(c)
c=(torch.rand(n,n,n,dtype=dtype)*.5+.02).requires_grad_()
v=torch.tensor([.2,-.3,.4],dtype=dtype)
qref=differentiable_project(c,geom,samples=k).reshape(-1)/scale
q=project(c)
jt_ref=torch.autograd.grad(qref,c,v,retain_graph=True)[0]
jt_explicit=jt(c,v)
d=torch.randn_like(c)
cp=a*b/(a+(b-a)*gather(c)).square()
jd=coef*(cp*gather(d)).sum(-1)
identity_error=((jd*v).sum()-(d*jt_explicit).sum()).abs().item()
try:
 g=torch.autograd.grad((qref.square()).sum(),c,create_graph=True)[0]
 torch.autograd.grad(g.square().sum(),c)
 double_backward={'supported':True}
except RuntimeError as exc:double_backward={'supported':False,'message':str(exc)}
checks={
 'date_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
 'scope':'CPU synthetic 3^3 field, 3 rays, K7, float64; no checkpoint, dataset, optimizer, model forward or training',
 'torch_version':torch.__version__,
 'projection_max_absolute_error':(q-qref).abs().max().item(),
 'explicit_vjp_vs_grid_sample_autograd_max_absolute_error':(jt_explicit-jt_ref).abs().max().item(),
 'jvp_vjp_inner_product_absolute_error':identity_error,
 'fixed_geometry_projection_gradcheck':bool(torch.autograd.gradcheck(project,(c,),eps=1e-6,atol=1e-6,rtol=1e-4)),
 'fixed_geometry_projection_gradgradcheck':bool(torch.autograd.gradgradcheck(project,(c,),eps=1e-6,atol=1e-6,rtol=1e-4)),
 'explicit_vjp_gradcheck':bool(torch.autograd.gradcheck(lambda x,y:jt(x,y),(c,v.requires_grad_()),eps=1e-6,atol=1e-6,rtol=1e-4)),
 'existing_grid_sample_double_backward':double_backward,
 'limitations':['Not a full64/K96 test','Not a CUDA kernel or performance check','All synthetic sample points strictly within box and concentrations within (0,1)','Mask, invalid, boundary and padding cases need implementation-stage tests']}
out=Path('docs/reports/artifacts/jacobian_adjoint_design_v1/synthetic_operator_check.json')
out.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))
