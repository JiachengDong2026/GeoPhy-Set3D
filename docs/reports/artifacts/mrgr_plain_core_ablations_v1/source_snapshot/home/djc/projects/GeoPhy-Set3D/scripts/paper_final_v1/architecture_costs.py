"""Measured architecture counts at fixed V4; explicitly partial arithmetic accounting."""
from common import *
import numpy as np
import torch
from geophy_set3d.data.collate import projection_pool_collate
from models import model_for

backend();config=verified(OUT/'model_manifest.json')['models'][0]['config'];s=verified(OUT/'layouts.json')['val'][0];records=[]
with AccessGuard(splits=['val']) as guard:
    ds=dataset('val',s);b=projection_pool_collate([ds[0]]);b={k:v for k,v in b.items() if not k.startswith('hold_')}
    for arm in ('multiscale','single_scale','no_feedback','no_consistency'):
        model=model_for(config,arm,'cpu').eval();cost=[0];hooks=[]
        def count(module,inputs,output):
            if isinstance(module,(torch.nn.Conv2d,torch.nn.Conv3d)):cost[0]+=output.numel()*int(np.prod(module.kernel_size))*module.in_channels//module.groups
            elif isinstance(module,torch.nn.Linear):cost[0]+=output.numel()*module.in_features
        for module in model.modules():
            if isinstance(module,(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear)):hooks.append(module.register_forward_hook(count))
        with torch.no_grad():model(b)
        for h in hooks:h.remove()
        records.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),conv_linear_MACs_fixed_V4=cost[0],twice_MACs_partial_FLOPs=2*cost[0]))
write(OUT/'architecture_costs.json',dict(at=now(),records=records,scope='batch1 V4 C1; actual calls include both shared decoder passes when feedback enabled. Conv/linear MACs only; excludes lifting/interpolation/attention matmul/projection/nonlinearities/norm, so not total FLOPs.',case_id=ds.records[0]['case_id'],access=guard.summary()))
print(json.dumps(records,indent=2))
