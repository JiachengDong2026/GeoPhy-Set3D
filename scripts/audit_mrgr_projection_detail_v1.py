#!/usr/bin/env python3
"""Read-only independent late-training intervention-strength audit; no optimization/GT."""
import gc
from pathlib import Path
import torch
import probe_mrgr_projection_detail_v1 as p


def main():
    torch.set_num_threads(2)
    protocol=p.verify(p.OUT);config=protocol['config'];device=config['training']['device']
    dest=p.OUT/'late_intervention_audit.json'
    assert not dest.exists()
    result={}
    with p.base.GTReadGuard(config['dataset']['root'],p.OUT) as guard:
        ds=p.base.selfsup.make_dataset(config);ds.set_epoch(0)
        batch=p.base.selfsup.move_dual_batch(p.base.dual_measurement_collate([ds[0],ds[1]]),device)
        for arm in p.ARMS:
            path=p.OUT/arm/'checkpoints/last.pt'
            saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['epoch']==20
            model=p.ModelProvider.model_for(config,arm,device);model.load_state_dict(saved['model']);model.train()
            before=p.base.state_hash(model.state_dict())
            total,parts,volumes,projections=p.ORIGINAL_OBJECTIVE(model,batch)
            raw=.5*(p.gradient_discrepancy(projections['C_A']['H'],batch['H'])+p.gradient_discrepancy(projections['C_B']['H'],batch['H']))
            parameters=list(model.parameters())
            g0=torch.autograd.grad(total,parameters,retain_graph=True)
            gd=torch.autograd.grad(p.LAMBDA*raw,parameters)
            n0=torch.sqrt(sum((g*g).sum() for g in g0));nd=torch.sqrt(sum((g*g).sum() for g in gd))
            # Also preserve the exact decoder.context.output.weight probe used in initial smoke.
            target_index=next(i for i,q in enumerate(parameters) if q is model.decoder.context.output.weight)
            record=dict(epoch=20,checkpoint_sha256=p.base.digest(path),base_loss=float(total.detach()),gradient_loss=float(raw.detach()),
                weighted_gradient_loss=float((p.LAMBDA*raw).detach()),weighted_to_base_loss=float((p.LAMBDA*raw/total).detach()),
                all_parameter_base_grad_norm=float(n0),all_parameter_added_grad_norm=float(nd),all_parameter_added_to_base_grad_norm=float(nd/n0),
                context_output_base_grad_norm=float(g0[target_index].norm()),context_output_added_grad_norm=float(gd[target_index].norm()),
                context_output_added_to_base_grad_norm=float(gd[target_index].norm()/g0[target_index].norm()),
                finite=all(bool(torch.isfinite(g).all()) for g in [*g0,*gd]),added_gradient_nonzero=bool(nd>0),weights_unchanged=p.base.state_hash(model.state_dict())==before,
                sample_case_ids=batch['A']['case_id'])
            assert record['finite'] and record['added_gradient_nonzero'] and record['weights_unchanged']
            result[arm]=record
            del saved,model,total,parts,volumes,projections,raw,parameters,g0,gd,n0,nd;gc.collect();torch.cuda.empty_cache()
        result['gt_audit']=guard.summary()
    result.update(protocol_sha256=p.base.digest(p.OUT/'protocol.json'),audit_script_sha256=p.base.digest(Path(__file__)),
        selected_fixed_epoch0_train_batch=True,optimization_performed=False,gate_modified=False,
        interpretation='Fixed first two train cases, read-only local gradient sensitivity; no strength threshold or hyperparameter search.')
    p.write(dest,result);print(result)

if __name__=='__main__':main()
