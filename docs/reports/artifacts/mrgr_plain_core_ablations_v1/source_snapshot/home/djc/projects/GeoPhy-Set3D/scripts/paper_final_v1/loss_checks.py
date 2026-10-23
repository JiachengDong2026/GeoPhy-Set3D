"""Check actual loss coefficients and gradient removal on a real train measurement batch."""
from common import *
import copy
import torch
from models import model_for,install_loss
from geophy_set3d.training import selfsup
from geophy_set3d.data.measurement_only import dual_measurement_collate

backend();config=copy.deepcopy(verified(OUT/'model_manifest.json')['models'][0]['config']);config['training']['device']='cpu';config['loss']['consistency']=0.
with AccessGuard(splits=['train']) as guard:
    ds=selfsup.make_dataset(config);batch=dual_measurement_collate([ds[0]]);m=model_for(config,'no_consistency','cpu').train();install_loss(config)
    total,parts,volumes,projections=selfsup.objective(m,batch)
    expected=parts['hold']+.5*parts['cross']+.1*parts['observed'];torch.testing.assert_close(total,expected,rtol=0,atol=0)
    parameter=m.decoder.context.output.weight
    grad_zero,=torch.autograd.grad(total,parameter,retain_graph=True)
    grad_full,=torch.autograd.grad(total+.05*parts['consistency'],parameter,retain_graph=True)
    grad_cons,=torch.autograd.grad(parts['consistency'],parameter)
    torch.testing.assert_close(grad_full-grad_zero,.05*grad_cons,rtol=1e-4,atol=1e-6)
    assert grad_cons.abs().sum()>0 and grad_zero.abs().sum()>0
write(OUT/'loss_checks.json',dict(at=now(),passed=True,actual_total=float(total.detach()),parts={k:float(v.detach()) for k,v in parts.items()},zero_consistency_coefficient=True,gradient_difference_matches_removed_term=True,case_id=ds.records[0]['case_id'],access=guard.summary()))
print('Actual zero-consistency loss and gradient check PASSED',flush=True)
