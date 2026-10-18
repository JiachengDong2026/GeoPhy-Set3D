#!/usr/bin/env python3
"""Fixed first-two-case same-weight learned O logits intervention, measurement-only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import probe_mrgr_adaptive_geometry_v2 as screen

base=screen.base
OUT=screen.OUT/'condition_intervention'


def register():
    p=screen.verify();assert not OUT.exists();OUT.mkdir()
    record=dict(registered_at=base.now(),case_ids=p['validation_case_ids'][:2],
        source_protocol_sha256=base.digest(screen.OUT/'protocol.json'),script_sha256=base.digest(Path(__file__)),
        arm='D_geometry',checkpoint='fixed epoch20 last.pt; no alternate epoch or cases',
        intervention='Same weights: enabledTrue versusFalse, visibilitymask remains identical; original5Ddescriptor still used byOARI. Disable only extra learned O logits in scale_fusion.',
        measurements='per-input relativeL1 of field differences(normal denominator), alpha absolute difference, meanA/B->H relativeL1',
        no_gt=True,no_final_test=True,no_training=True,no_gate_change=True,device='cuda:0',
        limitations='Same-checkpoint diagnostic, not retrained ablation; first2 developmentcases, not general superiority evidence')
    screen.formal.write(OUT/'registration.json',record)
    (OUT/'registration.sha256').write_text(base.digest(OUT/'registration.json')+'\n')


def run():
    p=screen.verify();r=screen.read(OUT/'registration.json')
    assert base.digest(OUT/'registration.json')==(OUT/'registration.sha256').read_text().strip()
    assert base.digest(Path(__file__))==r['script_sha256']
    assert base.digest(screen.OUT/'protocol.json')==r['source_protocol_sha256']
    assert not (OUT/'summary.json').exists()
    config=screen.effective(p,'D_geometry');config['training']['device']=r['device']
    path=screen.OUT/'checkpoints/D_geometry/last.pt'
    sealed=screen.read(screen.OUT/'D_geometry/summary.json');assert base.digest(path)==sealed['checkpoint_sha256']
    with base.GTReadGuard(config['dataset']['root'],screen.OUT/'checkpoints') as guard:
        ckpt=torch.load(path,map_location='cpu',weights_only=False);assert ckpt['epoch']==20
        m=screen.candidate.model_for(config,'D_geometry',r['device']);m.load_state_dict(ckpt['model']);m.eval()
        initial=base.state_hash(m.state_dict());m.decoder.scale_fusion.record_diagnostics=True
        ds=base.selfsup.make_dataset(config,validation=True)
        assert [ds.records[i]['case_id'] for i in range(2)]==r['case_ids']
        b=base.selfsup.move_dual_batch(base.dual_measurement_collate([ds[i] for i in range(2)]),r['device'])
        results=[]
        with torch.no_grad():
            for key in ('A','B'):
                m.decoder.scale_fusion.enabled=True
                normal=m(b[key]).detach().clone();normal_alpha=m.decoder.scale_fusion.last_diagnostics['alpha'].clone()
                pn=base.project_input(normal,b['H'])
                m.decoder.scale_fusion.enabled=False
                off=m(b[key]).detach().clone();off_alpha=m.decoder.scale_fusion.last_diagnostics['alpha'].clone()
                po=base.project_input(off,b['H'])
                valid=b['H']['input_valid_ray'].bool()&b['H']['view_mask'][:,:,None,None,None]
                for i,case in enumerate(r['case_ids']):
                    n=base.relative_projection_metrics(pn[i],b['H']['input_projection_raw'][i],valid[i])
                    o=base.relative_projection_metrics(po[i],b['H']['input_projection_raw'][i],valid[i])
                    visible=m.decoder.scale_fusion.last_diagnostics['visible'][i].expand(3,-1,-1,-1)
                    row=dict(case_id=case,input=key,
                        field_relative_l1_difference=float((normal[i]-off[i]).abs().sum()/normal[i].abs().sum().clamp_min(1e-6)),
                        alpha_mean_absolute_difference=float((normal_alpha[i]-off_alpha[i]).abs().masked_select(visible).mean()),
                        normal_H=n['relative_l1'],no_extra_O_H=o['relative_l1'],H_delta_off_minus_normal=o['relative_l1']-n['relative_l1'])
                    assert all(np.isfinite(v) for k,v in row.items() if k not in ('case_id','input'))
                    results.append(row)
        m.decoder.scale_fusion.enabled=True
        assert initial==base.state_hash(m.state_dict())
        assert m.decoder._descriptor is None
        audit=guard.summary();assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    cases=[dict(case_id=c,normal_H=float(np.mean([x['normal_H'] for x in results if x['case_id']==c])),
        no_extra_O_H=float(np.mean([x['no_extra_O_H'] for x in results if x['case_id']==c]))) for c in r['case_ids']]
    result=dict(completed=True,registration_sha256=base.digest(OUT/'registration.json'),checkpoint_sha256=base.digest(path),
        weights_unchanged=True,rows=results,case_meanAB=cases,normal_H_mean=float(np.mean([x['normal_H'] for x in cases])),
        no_extra_O_H_mean=float(np.mean([x['no_extra_O_H'] for x in cases])),
        nonzero_conditioning_effect=all(x['field_relative_l1_difference']>0 and x['alpha_mean_absolute_difference']>0 for x in results),
        GT_access=audit,limitations=r['limitations'])
    screen.formal.write(OUT/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','run'));parser.add_argument('--seed',type=int,default=20261018);args=parser.parse_args();screen.SEED=args.seed;screen.OUT=screen.ROOT/f'outputs/research/mrgr_adaptive_geometry_v2_seed{args.seed}';OUT=screen.OUT/'condition_intervention'
    register() if args.action=='register' else run()
