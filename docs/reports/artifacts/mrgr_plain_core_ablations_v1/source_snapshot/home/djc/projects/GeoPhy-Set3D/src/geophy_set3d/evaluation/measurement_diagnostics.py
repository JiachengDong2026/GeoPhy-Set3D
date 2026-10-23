"""Numerical/set checks that require no volumetric ground truth."""
import torch
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training.selfsup import measurement_discrepancy
from geophy_set3d.training.utils import permute_views
from geophy_set3d.data.measurement_only import assert_no_gt


def set_properties(model,batch):
    assert_no_gt(batch); model.eval()
    with torch.no_grad():
        original=model(batch,return_stages=True)
        p=torch.arange(batch['view_mask'].shape[1],device=batch['view_mask'].device).flip(0)
        shuffled=model(permute_views(batch,p),return_stages=True)
        padded=dict(batch)
        for key,value in batch.items():
            if (key.startswith('input_') or key=='view_mask') and isinstance(value,torch.Tensor):
                padded[key]=torch.cat((value,torch.zeros_like(value[:,:1])),1)
        padded['input_projection'][:,-1]=float('nan');padded['input_projection_raw'][:,-1]=float('nan')
        pad=model(padded,return_stages=True)
        invalid=dict(batch);invalid['view_mask']=torch.zeros_like(batch['view_mask']);invalid['input_valid_ray']=torch.zeros_like(batch['input_valid_ray'])
        inv=model(invalid,return_stages=True)
        result=dict(permutation_max_diff=max(float((original[k]-shuffled[k]).abs().max()) for k in ('C0','C1')),
                    padding_max_diff=max(float((original[k]-pad[k]).abs().max()) for k in ('C0','C1')),
                    visibility_identical=bool(torch.equal(original['visibility_base'],original['visibility_residual'])),
                    all_invalid_safe=bool(torch.isfinite(inv['C1']).all() and (inv['context1']==0).all()))
    result['passed']=result['permutation_max_diff']<=1e-5 and result['padding_max_diff']<=1e-5 and result['visibility_identical'] and result['all_invalid_safe']
    return result


def feedback_gradients(model,input_batch,heldout_batch):
    assert_no_gt(input_batch);assert_no_gt(heldout_batch)
    model.eval();model.zero_grad(set_to_none=True)
    s=model(input_batch,return_stages=True)
    loss=measurement_discrepancy(project_input(s['C1'],heldout_batch),heldout_batch)
    gy,=torch.autograd.grad(loss,s['measurement_prediction_Y0'],retain_graph=True)
    gc,=torch.autograd.grad(s['measurement_prediction_Y0'],s['C0'],grad_outputs=gy,retain_graph=True)
    loss.backward()
    grads=[p.grad for n,p in model.named_parameters() if n.startswith('residual_')]
    result=dict(residual_gradient_finite=all(g is not None and bool(torch.isfinite(g).all()) for g in grads),
                residual_gradient_abs_sum=sum(float(g.abs().sum()) for g in grads if g is not None),
                projector_feedback_gradient_finite=bool(torch.isfinite(gy).all() and torch.isfinite(gc).all()),
                projector_feedback_gradient_abs_sum=float(gc.abs().sum()),loss_domain='held-out measurements only')
    result['passed']=result['residual_gradient_finite'] and result['projector_feedback_gradient_finite'] and result['residual_gradient_abs_sum']>0 and result['projector_feedback_gradient_abs_sum']>0
    model.zero_grad(set_to_none=True)
    return result
