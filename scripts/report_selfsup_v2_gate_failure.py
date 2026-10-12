#!/usr/bin/env python3
"""Inspect the one failed RVTC gate and report NO-GO without GT or retraining."""
import json,sys,subprocess
from pathlib import Path
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'src'))
import torch
from geophy_set3d.training.selfsup_v2 import (V2ReadGuard,initialize_fresh_model,make_dataset,make_loader,
    move_rich_batch,assert_rich_union,teacher_discrepancy,objective,audit_is_clean)
from geophy_set3d.training.utils import load_config,write_json,directories
from geophy_set3d.data.rich_measurement import rich_measurement_collate
from geophy_set3d.evaluation.selfsup_v2_integrity import check_integrity
from geophy_set3d.evaluation.selfsup_v2_report import render_report
from geophy_set3d.evaluation.analysis import write_csv


def main():
    root=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v2')
    cfg=load_config(repo/'configs/train_selfsup_oari_mrgr_v2.yaml');gate=json.loads((root/'gate_summary.json').read_text())
    assert gate['passed'] is False
    paths=directories(cfg,'_gate');torch.set_num_threads(2)
    guard=V2ReadGuard(cfg['dataset']['root'],paths['checkpoints'])
    with guard:
        model=initialize_fresh_model(cfg)
        ckpt=torch.load(paths['checkpoints']/'gate.pt',map_location=cfg['training']['device'],weights_only=False)
        assert ckpt['config']==cfg
        model.load_state_dict(ckpt['model']);model.eval()
        ds=make_dataset(cfg);ds.set_epoch(0)
        checked=0;field_count=0;max_pairing_diff=0.
        for i in range(8):
            sample=ds[i];batch=move_rich_batch(rich_measurement_collate([sample]),cfg['training']['device'])
            assert_rich_union(batch)
            assert {key:sample[key]['input_station_indices'].tolist() for key in ('A','B','H','T')}==gate['station_protocol'][sample['A']['case_id']]
            for key in batch['T']:
                if not key.startswith('input_'):continue
                actual=batch['T'][key][0,batch['T']['view_mask'][0]]
                expected=torch.cat([batch[x][key][0,batch[x]['view_mask'][0]] for x in ('A','B')])
                assert torch.equal(actual,expected),key
                max_pairing_diff=max(max_pairing_diff,float((actual.double()-expected.double()).abs().max()) if actual.numel() else 0.)
                field_count+=1
            for key in ('horizontal_angle_deg','vertical_angle_deg','projection_scale_ppm_m'):
                assert all(torch.equal(batch['T'][key],batch[x][key]) for x in ('A','B','H'))
            checked+=1
        # Actual trained-network outputs: gradient descent in each student output moves toward T.
        _,parts,(a,b,t),_=objective(model,batch)
        ga,gb,gt=torch.autograd.grad(parts['teacher'],(a,b,t),retain_graph=True,allow_unused=True)
        dots=[float((g*(c-t.detach())).sum()) for g,c in ((ga,a),(gb,b))]
        gm,=torch.autograd.grad(parts['hold_teacher'],t)
        assert min(dots)>0 and gt is None and float(gm.abs().sum())>0 and torch.isfinite(gm).all()
        audit=guard.summary();assert audit_is_clean(audit)
        diagnostic=dict(checked_cases=checked,checked_view_fields=field_count,
            T_geometry_construction_correct=True,A_B_union_order_correct=True,view_mask_correct=True,
            T_measurement_pairing_correct=True,max_measurement_geometry_pairing_difference=max_pairing_diff,
            station_protocol_unchanged=True,teacher_stop_gradient_correct=gt is None,
            student_gradient_dot_error=dots,teacher_loss_gradient_direction_correct=min(dots)>0,
            teacher_measurement_gradient_abs_sum=float(gm.abs().sum()),
            permutation_padding_all_invalid_passed=all(p['passed'] for p in gate['set_properties'].values()),
            gt_access=audit,bug_found=False,gate_retrained=False,loss_weights_changed=False,
            conclusion='No construction/mask/measurement/stop-gradient/gradient-direction bug found. Frozen loss-decrease criterion failed; stop without tuning.')
    write_json(root/'gate_failure_diagnostics.json',diagnostic)
    integrity=check_integrity(root,cfg,repo,posthoc=False)
    summary=dict(completed=False,engineering_gate_completed=True,formal_training_executed=False,formal_comparison_executed=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        model='GeoLift-OARI-MRGR-Res3D',parameter_count=332070,architecture_changed=False,
        gate=gate,gate_failure_diagnostics=diagnostic,integrity=integrity,
        volumetric_gt_used_for_training=False,volumetric_gt_used_for_selection=False,
        gt_posthoc_evaluation_executed=False,test_used=False,decision='SelfSup-MRGR v2 NO-GO',
        decision_stage='engineering gate',failed_criteria=[key for key,value in gate['criteria'].items() if not value],
        decision_note='Gate loss-decrease criterion failed without an identified implementation bug. No claim about post-hoc voxel superiority can be made; formal v1/v2 comparison was not executed.',
        retain_baseline='SelfSup v1',automatic_experiments_stopped=True,next_step='Await the next research decision; no v3, EMA, priors or tuning.')
    write_json(root/'summary.json',summary)
    write_json(root/'gt_access_audit.json',dict(memory_smoke=json.loads((root/'memory_smoke.json').read_text())['gt_access'],
        engineering_gate=gate['gt_access'],gate_failure_diagnostics=audit,formal_training=None,
        formal_training_status='not executed: gate failed',posthoc_gt_evaluation=None,
        gt_posthoc_status='not executed: gate failed',all_training_forbidden_attempts_zero=True))
    measurements=[dict(phase=phase,domain=domain,**values) for phase in ('initial','final')
                  for domain,values in gate[phase]['measurement'].items()]
    write_csv(root/'measurement_metrics.csv',measurements)
    for filename,fields in (
        ('voxel_metrics_posthoc.csv',('condition','model','relative_l1','relative_l2','status')),
        ('teacher_diagnostics.csv',('case_id','A_voxel_relative_l1','B_voxel_relative_l1','T_voxel_relative_l1','status')),
        ('v1_vs_v2.csv',('condition','v1_l1','v1_l2','v2_l1','v2_l2','status'))):
        import csv
        with (root/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerow(dict(status='not executed: engineering gate failed; no GT read'))
    (root/'figures').mkdir(exist_ok=True)
    (root/'figures/README.md').write_text('No post-hoc comparison figures were generated: the v2 engineering gate failed, so formal training and GT evaluation were not executed. Measurement-only learning diagnostics are in teacher_training_diagnostics.csv and gate metrics.jsonl.\n')
    render_report(root,repo)
    print(json.dumps(diagnostic,indent=2))


if __name__=='__main__':main()
