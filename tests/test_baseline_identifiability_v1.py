"""Diagnostic boundary, hook fidelity and independent synthetic quadrature checks."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts/deep_baselines_identifiability_v1'
sys.path.insert(0,str(SCRIPTS))
from diag_context import design
from diagnostics import objective,intervene,valid,Y_SCALE
from trace import Trace,layer_gradients,finite_difference
from synthetic import analytic,reference
from geophy_set3d.models.baseline_measurement_init_v2 import RayDeepSetsMeasurementInitV2,RayGNOTMeasurementInitV2
from test_baseline_deepsets_v1 import synthetic_batch

def test_intervention_preserves_geometry_mask_and_invalid_padding():
    inp=synthetic_batch(1);inp['input_projection']=inp['input_projection_raw']/Y_SCALE
    inp['input_valid_ray'][0,0,0,0,0]=False;inp['input_projection_raw'][0,0,0,0,0]=float('nan')
    for kind in ('normal','zero','half','one_half','permuted'):
        out=intervene(inp,kind)
        for key in inp:
            if key not in ('input_projection_raw','input_projection'):assert torch.equal(inp[key],out[key])
        assert torch.isnan(out['input_projection_raw'][~valid(inp)]).all()
        assert torch.isfinite(out['input_projection_raw'][valid(inp)]).all()
    a=intervene(inp,'permuted')['input_projection_raw'][valid(inp)]
    b=inp['input_projection_raw'][valid(inp)]
    torch.testing.assert_close(a.sort().values,b.sort().values,rtol=0,atol=0)

def test_H_is_rejected_before_any_model_call_and_budget_is_bounded():
    class Never:
        def __call__(self,*args):raise AssertionError('Network must not be called')
    with pytest.raises(ValueError,match='H is forbidden'):objective(Never(),dict(A={},B={},H={}), 'v2')
    d=design();assert d['updates_per_run']==200 and d['total_training_update_limit']==2000<=2500
    assert sum(200 for _ in range(2*(len(d['A_scales'])+len(d['B_conditions']))))==2000

def test_hooks_preserve_full64_forward_bitwise_and_leave_state_unchanged():
    torch.set_num_threads(2);inp=synthetic_batch(1)
    for net in [RayDeepSetsMeasurementInitV2(.004,hidden_dim=8,query_chunk_size=8192),
                RayGNOTMeasurementInitV2(.004,width=8,layers=1,heads=2,experts=1,inner_multiplier=2,query_chunk_size=8192)]:
        net.eval();before=copy.deepcopy(net.state_dict())
        with torch.no_grad():
            expected=net(inp)
            with Trace(net) as tap:actual=tap.forward(inp)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert 'tokens' in tap.values and 'logits_first_chunk' in tap.values
        for k,v in net.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

def test_independent_reference_constant_limit_and_spatially_nontrivial_fields():
    inp=synthetic_batch(1,1)
    sample={k:v[0] for k,v in inp.items() if k!='view_mask'}
    spec=dict(components=[dict(center=[0.,0.,0.],sigma=[1e9,1e9,1e9],angle=0.,peak=.004)])
    got=reference(sample,spec,128)[:,0]
    chi=(.004/16.043)/(.004/16.043+.996/28.965)
    np.testing.assert_allclose(got,.6*chi*1e6,rtol=1e-6,atol=1e-6)
    points=np.stack(np.meshgrid(*(np.linspace(0,.6,16),)*3,indexing='ij'),-1)
    for spec in design()['synthetic_cases']:
        field=analytic(points,spec);assert field.std()/field.mean()>.5 and field.max()<1

def test_full_physics_AB_backward_and_trace_gradients_without_optimizer_updates():
    torch.set_num_threads(2)
    inp=synthetic_batch(1,1)
    for key in ('input_t_near','input_path_length_m','input_valid_ray','input_projection_raw'):
        inp[key]=inp[key].repeat_interleave(32,-1).repeat_interleave(32,-2)
    inp['input_ray_direction']=inp['input_ray_direction'].repeat_interleave(32,2).repeat_interleave(32,3)
    inp['input_valid_ray'][:]=False;inp['input_valid_ray'][...,0,0]=True
    inp['input_projection_raw']=torch.where(inp['input_valid_ray'],inp['input_projection_raw'],0.)
    inp['input_projection']=inp['input_projection_raw']/Y_SCALE
    inp['input_station_indices']=torch.tensor([[0]])
    other=copy.deepcopy(inp);other['input_station_indices'][:]=1
    ab=dict(A=inp,B=other)
    for net in [RayDeepSetsMeasurementInitV2(.004,hidden_dim=8,query_chunk_size=8192),
                RayGNOTMeasurementInitV2(.004,width=8,layers=1,heads=2,experts=1,inner_multiplier=2,query_chunk_size=8192)]:
        before=copy.deepcopy(net.state_dict());net.train()
        total,parts,_,_=objective(net,ab,'v2')
        torch.testing.assert_close(total,.5*parts['cross']+.1*parts['observed']+.05*parts['consistency'],rtol=0,atol=0)
        assert set(parts)=={'cross','observed','consistency'}
        total.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        grads=layer_gradients(net,ab,'v3')
        assert grads['observed']['layers']['token_measurement_channel']['present']
        fd=finite_difference(net,inp);assert len(fd)==4 and all(np.isfinite(r['AD']) and np.isfinite(r['FD']) for r in fd)
        for k,v in net.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
