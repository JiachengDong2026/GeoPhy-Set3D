#!/usr/bin/env python3
"""Registered same-weight one/two-step diagnostic on the fixed first2 validationcases."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import probe_mrgr_two_step_v2 as screen
base=screen.base

def register(out):
    p=screen.verify();assert not out.exists();out.mkdir()
    r=dict(registered_at=base.now(),script_sha256=base.digest(Path(__file__)),protocol_sha256=base.digest(screen.OUT/'protocol.json'),
        cases=p['validation_case_ids'][:2],arm='T_two',checkpoint='fixed20last',no_gt=True,no_test=True,no_training=True,
        intervention='same weights: steps1 vs2; confirm steps1 equals stagedC_feedback1, compute fields/H/observed residual change; no alternate cases or epochs')
    screen.formal.write(out/'registration.json',r);(out/'registration.sha256').write_text(base.digest(out/'registration.json')+'\n')

def run(out):
    p=screen.verify();r=screen.read(out/'registration.json')
    assert base.digest(out/'registration.json')==(out/'registration.sha256').read_text().strip()
    assert base.digest(Path(__file__))==r['script_sha256'] and base.digest(screen.OUT/'protocol.json')==r['protocol_sha256']
    assert not (out/'summary.json').exists()
    config=screen.effective(p,'T_two');path=screen.OUT/'checkpoints/T_two/last.pt'
    assert base.digest(path)==screen.read(screen.OUT/'T_two/summary.json')['checkpoint_sha256']
    with base.GTReadGuard(config['dataset']['root'],screen.OUT/'checkpoints') as guard:
        ckpt=torch.load(path,map_location='cpu',weights_only=False);assert ckpt['epoch']==20 and ckpt['updates']==1660
        model=screen.candidate.model_for(config,'T_two','cuda:0');model.load_state_dict(ckpt['model']);model.eval();before=base.state_hash(model.state_dict())
        ds=base.selfsup.make_dataset(config,validation=True)
        assert [ds.records[i]['case_id'] for i in range(2)]==r['cases']
        b=base.selfsup.move_dual_batch(base.dual_measurement_collate([ds[0],ds[1]]),'cuda:0');rows=[]
        with torch.no_grad():
            for key in ('A','B'):
                model.steps=2;s=model(b[key],return_stages=True);first=s['C_feedback1'];final=s['C1']
                model.steps=1;one=model(b[key]);assert torch.equal(one,first)
                hvalid=b['H']['input_valid_ray'].bool()&b['H']['view_mask'][:,:,None,None,None]
                ivalid=b[key]['input_valid_ray'].bool()&b[key]['view_mask'][:,:,None,None,None]
                h1=base.project_input(first,b['H']);h2=base.project_input(final,b['H'])
                i1=s['measurement_prediction_Y1'];i2=base.project_input(final,b[key])
                for j,case in enumerate(r['cases']):
                    a=base.relative_projection_metrics(h1[j],b['H']['input_projection_raw'][j],hvalid[j]);z=base.relative_projection_metrics(h2[j],b['H']['input_projection_raw'][j],hvalid[j])
                    ia=base.relative_projection_metrics(i1[j],b[key]['input_projection_raw'][j],ivalid[j]);iz=base.relative_projection_metrics(i2[j],b[key]['input_projection_raw'][j],ivalid[j])
                    row=dict(case_id=case,input=key,field_relative_l1_difference=float((final[j]-first[j]).abs().sum()/first[j].abs().sum().clamp_min(1e-6)),
                        first_H=a['relative_l1'],final_H=z['relative_l1'],H_delta_second_minus_first=z['relative_l1']-a['relative_l1'],
                        first_observed_relative_l1=ia['relative_l1'],final_observed_relative_l1=iz['relative_l1'],
                        second_delta_norm=float(s['residual_context_step2'][j].norm()),same_weights_single_equals_feedback1=True)
                    assert all(np.isfinite(x) for k,x in row.items() if k not in ('case_id','input'))
                    rows.append(row)
        model.steps=2;assert before==base.state_hash(model.state_dict())
        audit=guard.summary();assert not any(audit[k] for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    result=dict(completed=True,weights_unchanged=True,checkpoint_sha256=base.digest(path),rows=rows,
        first_H_mean=float(np.mean([x['first_H'] for x in rows])),final_H_mean=float(np.mean([x['final_H'] for x in rows])),
        first_observed_mean=float(np.mean([x['first_observed_relative_l1'] for x in rows])),final_observed_mean=float(np.mean([x['final_observed_relative_l1'] for x in rows])),
        actual_nonzero_second_step=all(x['field_relative_l1_difference']>0 and x['second_delta_norm']>0 for x in rows),GT_access=audit,
        limitations='Fixed first2 developmentcases and same checkpoint; not retrained ablation, not 3D benefit evidence or global monotonicity guarantee')
    screen.formal.write(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','run'));parser.add_argument('--seed',type=int,default=20261018);args=parser.parse_args();torch.set_num_threads(2)
    screen.SEED=args.seed;screen.OUT=screen.ROOT/f'outputs/research/mrgr_two_step_v2_seed{args.seed}';out=screen.OUT/'condition_intervention'
    register(out) if args.action=='register' else run(out)
