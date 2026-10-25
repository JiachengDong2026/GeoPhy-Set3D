"""Semantic tests for operator definitions and legal input adaptations."""
import copy
import numpy as np
import pytest
import torch
from geophy_set3d.models.baseline_deeponet_v1 import DeepONetV1,ray_maps
from geophy_set3d.models.baseline_fno_v1 import FNOV1,FixedRayBackprojection,SpectralConv3d,Y_SCALE
from geophy_set3d.training.classical_operators import load_config,validate_config

def rays():
    # Four x-parallel rays through different points, one masked poison ray.
    b=dict(input_projection_raw=torch.tensor([[[[[1.,2.],[3.,float('nan')]]]]])*Y_SCALE,
           input_valid_ray=torch.tensor([[[[[1,1],[1,0]]]]],dtype=torch.bool),view_mask=torch.ones(1,1,dtype=torch.bool),
           input_station_origin_m=torch.tensor([[[-.1,.213,.307]]]),input_ray_direction=torch.tensor([1.,0.,0.]).expand(1,1,2,2,3).clone(),
           input_t_near=torch.full((1,1,1,2,2),.1),input_path_length_m=torch.full((1,1,1,2,2),.6))
    return b

def numpy_transpose(b):
    result=np.zeros((64,64,64),dtype=np.float64);coverage=np.zeros_like(result)
    for h,w in [(0,0),(0,1),(1,0)]:
        origin=b['input_station_origin_m'][0,0].numpy().astype('float64')
        d=b['input_ray_direction'][0,0,h,w].numpy().astype('float64')
        near=float(b['input_t_near'][0,0,0,h,w]);length=float(b['input_path_length_m'][0,0,0,h,w])
        value=float(b['input_projection_raw'][0,0,0,h,w])/Y_SCALE
        for q in range(96):
            xyz=(origin+d*(near+length*(q+.5)/96))/.6*63
            base=np.minimum(np.floor(xyz).astype(int),62);frac=xyz-base
            for i in [0,1]:
                for j in [0,1]:
                    for k in [0,1]:
                        weight=np.prod([frac[0] if i else 1-frac[0],frac[1] if j else 1-frac[1],frac[2] if k else 1-frac[2]])
                        idx=tuple(base+np.array([i,j,k]));factor=length/96*1e6/Y_SCALE*weight
                        result[idx]+=value*factor;coverage[idx]+=factor
    return result,coverage

def test_backprojection_numpy_adjoint():
    b=rays();a=FixedRayBackprojection(cache_entries=0);actual=a(b).numpy()[0]
    numerator,coverage=numpy_transpose(b)
    expected=np.divide(numerator,coverage,out=np.zeros_like(numerator),where=coverage>0)
    np.testing.assert_allclose(actual[0],expected,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(actual[1],np.log1p(coverage),rtol=3e-5,atol=2e-6)
    np.testing.assert_array_equal(actual[2],coverage>0)
    # Independent dot identity with a nonconstant analytic grid function.
    grid=np.arange(64**3,dtype=np.float64).reshape(64,64,64)/(64**3)
    op=a.operator;stencil=op.prepare(b,64);v=torch.from_numpy(grid)[None,None].float();zero=torch.zeros_like(v)
    j=op.jvp(zero,v,stencil)/op._chi_prime(torch.tensor(0.))
    valid=b['input_valid_ray'];y=torch.where(valid,b['input_projection_raw']/Y_SCALE,0)
    np.testing.assert_allclose(float((j*y).sum()),float((grid*numerator).sum()),rtol=3e-5,atol=2e-6)
    assert torch.equal(a(b),a(b))
    z=copy.deepcopy(b);z['input_projection_raw'].zero_();assert a(z)[:,0].count_nonzero()==0
    assert torch.equal(a(z)[:,1:],a(b)[:,1:])

def test_spectral_constant_and_gradient():
    layer=SpectralConv3d(1,2)
    with torch.no_grad():
        layer.weights.zero_();layer.weights[0,0,0,0,0,0,0]=1
    x=torch.full((1,1,8,8,8),2.,requires_grad=True);out=layer(x)
    torch.testing.assert_close(out,x)
    out.square().mean().backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

def test_deeponet_inner_product_and_order():
    m=DeepONetV1(features=4,width=8,query_chunk=17)
    coords=m.coordinates
    assert coords.shape==(64**3,3)
    torch.testing.assert_close(coords[1]-coords[0],torch.tensor([0.,0.,2/63]))
    branch=torch.randn(2,4);q=coords[:41]
    torch.testing.assert_close(m.decode(branch,q),branch@m.trunk(q).T+m.bias)
    for c in [17,41]:
        m.query_chunk=c;m.zero_grad();m.decode(branch,q).square().sum().backward()
        if c==17:g=m.trunk[0].weight.grad.clone()
        else:torch.testing.assert_close(g,m.trunk[0].weight.grad,atol=1e-5,rtol=1e-5)

def test_strict_policy_and_gt_rejection():
    for arm in ['deeponet','fno']:
        c=load_config(arm);c['loss']['hold']=0
        with pytest.raises(AssertionError):validate_config(c,arm)
    b=rays();b['gt_volume_64']=torch.zeros(1)
    with pytest.raises(ValueError):FixedRayBackprojection()(b)
    with pytest.raises(ValueError):ray_maps(b)

def test_full_grid_deeponet():
    b=rays()
    # CNN takes complete 64x64 maps, as frozen real data does.
    for k in ['input_projection_raw','input_valid_ray','input_t_near','input_path_length_m']:
        b[k]=b[k][:,:,:,:1,:1].expand(1,1,1,64,64).clone()
    b['input_ray_direction']=b['input_ray_direction'][:,:,:1,:1].expand(1,1,64,64,3).clone()
    b['input_station_indices']=torch.zeros(1,1,dtype=torch.long)
    m=DeepONetV1(features=4,width=8,query_chunk=8192).eval()
    with torch.no_grad():p=m(b)
    assert p.shape==(1,1,64,64,64) and torch.isfinite(p).all() and p.min()>=0 and p.max()<=1
