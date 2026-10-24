"""Tap engineering repair must preserve values and gradients, no optimizer step."""
from pathlib import Path
import sys
import weakref
import gc
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/deep_baselines_identifiability_v1'))
from engineering_memory_v2 import WeakTrace
from trace import Trace
from geophy_set3d.models.baseline_measurement_init_v2 import RayGNOTMeasurementInitV2
from test_baseline_deepsets_v1 import synthetic_batch

def test_weak_taps_release_graph_owner_and_preserve_gradient():
    torch.set_num_threads(2);torch.manual_seed(20261011)
    net=RayGNOTMeasurementInitV2(.004,width=8,layers=1,heads=2,experts=1,inner_multiplier=2,query_chunk_size=8192).train()
    batch=synthetic_batch(1);raw=batch['input_projection_raw'].clone().requires_grad_(True);batch=dict(batch,input_projection_raw=raw)
    expected=net(batch);g1=torch.autograd.grad(expected.mean(),raw)[0]
    with WeakTrace(net) as trace:actual=trace.forward(batch)
    owner=weakref.ref(trace);del trace;gc.collect()
    assert owner() is None,'Output graph retained diagnostic owner'
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    g2=torch.autograd.grad(actual.mean(),raw)[0]
    torch.testing.assert_close(g1,g2,rtol=0,atol=0)
