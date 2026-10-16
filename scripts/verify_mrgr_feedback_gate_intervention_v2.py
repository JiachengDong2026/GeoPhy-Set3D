#!/usr/bin/env python3
"""Verify that learned gate changes VISIBLE feedback and final fields, without GT."""
import json
import numpy as np
import torch
import probe_mrgr_feedback_gate_v2 as probe


def main():
    torch.set_num_threads(2)
    p=probe.verify();out=probe.OUT/'effective_intervention_verification.json'
    assert not out.exists()
    config=p['config'];rows=[]
    with probe.base.GTReadGuard(config['dataset']['root'],probe.OUT/'checkpoints') as guard:
        model=probe.model_for(config,'feedback_gate')
        path=probe.OUT/'checkpoints/feedback_gate/last.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['epoch']==p['epochs'] and saved['protocol_sha256']==probe.base.digest(probe.OUT/'protocol.json')
        model.load_state_dict(saved['model']);model.eval()
        ds=probe.base.selfsup.make_dataset(config,validation=True)
        with torch.no_grad():
            for idx in p['val_indices']:
                sample=ds[idx]
                dual=probe.base.selfsup.move_dual_batch(probe.base.dual_measurement_collate([sample]),probe.DEVICE)
                for key in ('A','B'):
                    batch=dual[key]
                    normal=model(batch,return_stages=True)
                    model.residual_adapter.force_identity=True
                    identity=model(batch,return_stages=True)
                    model.residual_adapter.force_identity=False
                    assert torch.equal(normal['C0'],identity['C0'])
                    difference=normal['residual_context']-identity['residual_context']
                    assert bool(torch.isfinite(difference).all())
                    field=normal['C1']-identity['C1']
                    visible=normal['visibility_base']&batch['view_mask'][:,:,None,None,None]
                    masked=difference.masked_select(~visible[:,:,None].expand_as(difference))
                    assert masked.numel()==0 or bool((masked==0).all())
                    rows.append(dict(case_id=sample[key]['case_id'],input=key,
                        visible_delta_intervention_norm=float(difference.norm()),
                        visible_delta_relative_intervention=float(difference.norm()/identity['residual_context'].norm().clamp_min(1e-12)),
                        final_field_mean_abs_change=float(field.abs().mean()),
                        final_field_relative_l1_change=float(field.abs().sum()/identity['C1'].abs().sum().clamp_min(1e-12)),
                        C0_unchanged=True,invalid_context_difference_zero=True))
        audit=guard.summary()
    assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    result=dict(completed=True,script_sha256=probe.base.digest(__file__),
        checkpoint_sha256=probe.base.digest(path),registration_sha256=probe.base.digest(probe.OUT/'protocol.json'),
        records=rows,mean_visible_delta_relative_intervention=float(np.mean([r['visible_delta_relative_intervention'] for r in rows])),
        mean_final_field_relative_l1_change=float(np.mean([r['final_field_relative_l1_change'] for r in rows])),
        nonzero_visible_intervention_records=sum(r['visible_delta_intervention_norm']>0 for r in rows),
        nonzero_field_change_records=sum(r['final_field_mean_abs_change']>0 for r in rows),
        limitation='Post-training gate=1 counterfactual of the SAME learned weights; not a separately trained ablation or evidence of3D superiority',
        gt_access=audit)
    probe.base.write_json(out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
