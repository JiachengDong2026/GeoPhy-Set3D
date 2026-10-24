"""Physical amplitude invariance and finite zero boundary, no data/GT."""
import torch
from geophy_set3d.training.baseline_consistency_repair_v3 import homogeneous_consistency
from geophy_set3d.training.selfsup import consistency_discrepancy,FROZEN_LOSS


def test_removes_common_amplitude_shrink_reward_and_has_correct_gradient():
    a=torch.tensor([[.001,.002,.003,.004]],dtype=torch.double)
    b=a*torch.tensor([[.9,1.1,.8,1.2]],dtype=torch.double)
    expected=homogeneous_consistency(a,b).item()
    for scale in (1.,.001,1e-6,1e-10):
        logscale=torch.tensor(scale,dtype=torch.double).log().requires_grad_()
        x=logscale.exp()
        corrected=homogeneous_consistency(a*x,b*x)
        derivative=torch.autograd.grad(corrected,logscale)[0]
        assert abs(corrected.item()-expected)<1e-14
        assert derivative.abs()<1e-14
        assert torch.autograd.grad(consistency_discrepancy(a*logscale.exp(),b*logscale.exp()),logscale)[0]>0
    # V3 adds no new weights, never changes frozen v1's function/epsilon.
    assert FROZEN_LOSS['consistency']==.05
    assert consistency_discrepancy(a,b)<homogeneous_consistency(a,b)


def test_zero_boundary_symmetry_bounds_and_finite_difference():
    a=torch.zeros(2,1,4,4,4,dtype=torch.double,requires_grad=True)
    b=torch.zeros_like(a)
    z=homogeneous_consistency(a,b);z.backward()
    assert z.item()==0 and torch.isfinite(a.grad).all() and a.grad.abs().max()==0
    a=torch.tensor([[.001,.002,.003]],dtype=torch.double,requires_grad=True)
    b=torch.tensor([[.003,.001,.002]],dtype=torch.double)
    assert 0<=homogeneous_consistency(a,b)<=2
    torch.testing.assert_close(homogeneous_consistency(a,b),homogeneous_consistency(b,a),atol=0,rtol=0)
    assert torch.autograd.gradcheck(homogeneous_consistency,(a,b),eps=1e-7,atol=1e-6,rtol=1e-5)
    assert homogeneous_consistency(a,b*0)==2
