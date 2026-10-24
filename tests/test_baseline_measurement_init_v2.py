"""Initialization isolation and ray contracts; synthetic measurements, no labels."""
import math
import torch
from geophy_set3d.models.baseline_deepsets_v1 import RayDeepSetsV1
from geophy_set3d.models.baseline_gnot_v1 import RayGNOTV1
from geophy_set3d.models.baseline_measurement_init_v2 import (
    RayDeepSetsMeasurementInitV2,RayGNOTMeasurementInitV2,initialize_readout)
from test_baseline_deepsets_v1 import synthetic_batch


def test_only_final_readout_changes_and_parameter_count_is_unchanged():
    for old,new,kwargs,prefix in [
        (RayDeepSetsV1,RayDeepSetsMeasurementInitV2,dict(hidden_dim=16),'decoder.4'),
        (RayGNOTV1,RayGNOTMeasurementInitV2,dict(width=8,layers=1,heads=2,inner_multiplier=2),'head.2')]:
        torch.manual_seed(20261011);a=old(**kwargs)
        torch.manual_seed(20261011);b=new(mass_fraction=.002,**kwargs)
        assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
        for name,p in a.state_dict().items():
            q=b.state_dict()[name]
            if name==prefix+'.weight':torch.testing.assert_close(q,p*.01,rtol=0,atol=0)
            elif name==prefix+'.bias':torch.testing.assert_close(q,torch.full_like(q,math.log(.002/.998)),rtol=0,atol=0)
            else:torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_measured_scale_contract_mask_permutation_and_gradients():
    torch.set_num_threads(2);batch=synthetic_batch()
    coords=torch.linspace(0,.6,33)[:,None].expand(-1,3)
    models=[RayDeepSetsMeasurementInitV2(.002,hidden_dim=16,query_chunk_size=11),
            RayGNOTMeasurementInitV2(.002,width=8,layers=1,heads=2,inner_multiplier=2,query_chunk_size=11)]
    for model in models:
        def predict(t):
            if hasattr(model,'phi'):return model.decode_queries(model.encode(t),coords)
            from geophy_set3d.models.ray_senseiver_ss import ray_tokens
            tokens,mask,counts=ray_tokens(t)
            return model.forward_tokens(tokens,mask,counts,coords)
        y=predict(batch)
        assert y.dtype==torch.float32 and torch.isfinite(y).all() and y.min()>0 and y.max()<.01
        shuffled={k:v[:,[2,0,1]] for k,v in batch.items()}
        torch.testing.assert_close(y,predict(shuffled),rtol=1e-5,atol=1e-8)
        changed=dict(batch,input_projection_raw=batch['input_projection_raw']*0)
        assert (predict(changed)-y).abs().max()>0
        y.mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_invalid_scale_rejected_without_silent_clipping():
    import pytest
    for value in (0.,1.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):RayDeepSetsMeasurementInitV2(value,hidden_dim=4)
    with pytest.raises(ValueError):initialize_readout(torch.nn.Linear(4,1),.002,.1)
