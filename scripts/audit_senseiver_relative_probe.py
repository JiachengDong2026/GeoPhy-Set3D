#!/usr/bin/env python3
"""Read-only final own-checkpoint, fixed-input, and historical-integrity audit."""
import gzip
import csv
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import torch
import yaml
from geophy_set3d.training.senseiver_relative_probe import NAME,validate,exit_event
from geophy_set3d.training.senseiver_init_probe import TrainEightGuard
from geophy_set3d.training.selfsup import make_dataset,move_dual_batch
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.training.ray_senseiver import sha256,tensor_identity
from geophy_set3d.training.rvtc_matched import state_hash
from diagnose_ray_senseiver_learning import geometry
from run_senseiver_relative_probe import verify,fixed_samples


def main():
    c=validate(yaml.safe_load((REPO/'configs/ray_senseiver_relative_probe_v1.yaml').read_text()))
    reg,scale,inventory=verify(c);base=Path(c['experiment']['output_root']);out=base/'comparisons'/NAME;ck=base/'checkpoints'/NAME
    torch.set_num_threads(2);ds=make_dataset(c);results={};epoch0_hashes=[]
    for arm in ['control','treatment']:
        complete=json.loads((out/arm/'complete.json').read_text())
        with TrainEightGuard(c['dataset']['root'],ck/arm,ds.records) as guard:
            assert sha256(ck/arm/'last.pt')==complete['last_sha256']
            saved=torch.load(ck/arm/'last.pt',map_location='cpu',weights_only=False)
            initial=torch.load(ck/arm/'epoch0.pt',map_location='cpu',weights_only=False)
            assert state_hash(initial['model'])==reg['initial_hash'];epoch0_hashes.append(state_hash(initial['model']))
            assert sha256(ck/arm/'epoch0.pt')==complete['epoch0_sha256']
            steps=sorted(set(int(v['step']) for v in saved['optimizer']['state'].values()))
            assert steps==[complete['updates']] and saved['updates']==4*saved['epoch']==complete['updates']
            assert saved['scheduler']['last_epoch']==saved['epoch']==complete['completed_epochs']
            assert saved['relative_l1']==complete['relative_l1']==(.1 if arm=='control' else 0.)
            assert not saved['volumetric_gt_used'] and all(torch.isfinite(p).all() for p in saved['model'].values())
            sources={}
            for identity,path in [('original_senseiver','docs/reports/artifacts/ray_senseiver_selfsup_v1/training_source_manifest.json'),('initialization_probe','docs/reports/artifacts/ray_senseiver_init_probe_v1/source_manifest.json'),('relative_probe','docs/reports/artifacts/ray_senseiver_relative_probe_v1/source_manifest.json')]:
                manifest=json.loads((REPO/path).read_text());sources[identity]=all(sha256(REPO/p)==h for p,h in manifest['sha256'].items());assert sources[identity]
            samples=fixed_samples(ds,inventory);geometry_checks=[]
            diagnosis=json.loads((out/'gradient_diagnosis.json').read_text())
            for i,sample in enumerate(samples):
                for subset,t in sample.items():
                    raw=t['input_projection_raw'][t['input_valid_ray'].bool()]
                    assert int((raw==0).sum())==int((raw/32284.894580078173==0).sum())==diagnosis['case_results'][i]['amplitude'][subset]['groups']['exact_zero']['count']
                b=move_dual_batch(dual_measurement_collate([sample]),c['training']['device'])
                for subset in ['A','B']:
                    result=geometry(b[subset]);assert result['valid_rays']==result['packed_real_rays']==sum(r['valid_count'] for r in inventory[i]['subsets'][subset]['stations'])
                    assert result['entry_exit_in_domain'] and result['all_tokens_finite']
                    geometry_checks.append(dict(index=i,subset=subset,**result))
            state={};events=[]
            for h in saved['health'][1:]:
                state=exit_event(h['epoch'],saved['history'][:h['epoch']],h,scale['C_init'],state);events.append(state)
            assert events==saved['events'] and state['reason']==complete['reason']
            observations=[o for r in saved['history'] for o in r['update_observations']]
            assert len(observations)==saved['updates'] and [o['step'] for o in observations]==list(range(1,saved['updates']+1))
            assert all(all(v>0 for v in o['update_l2'].values()) and all(v>0 for v in o['post_clip_gradient_l2'].values()) for o in observations)
            assert all(h['chunk_check']['passed'] for h in saved['health'])
            results[arm]=dict(passed=True,optimizer_state_steps=steps,scheduler_epoch=saved['epoch'],epoch0_state_hash=state_hash(initial['model']),last_hash=complete['last_sha256'],historical_and_probe_sources_unchanged=sources,
                fixed_all8_inputs_exact_identity=True,raw_exact_zero_matches_diagnostic_normalized_zero_no_underflow_group_misclassification=True,full8_AB_ray_geometry_checks=geometry_checks,exit_events_replayed_exactly=True,all_real_updates_and_gradient_groups_positive=True,all_chunk_checks_passed=True,access=guard.summary(),optimizer_steps_audit=0)
    assert len(set(epoch0_hashes))==1
    with (REPO/'docs/reports/artifacts/ray_senseiver_init_probe_v1/epoch_scalars.csv').open() as f:
        prior={int(r['epoch']):r for r in csv.DictReader(f) if r['arm']=='treatment'}
    current=json.loads((out/'control/history.json').read_text())
    results['control']['prior_identical_scientific_condition_scalar_replay']=dict(
        compared_epochs=len(current),max_H_L1_absolute_difference=max(abs(r['eval']['measurement']['heldout']['relative_l1']-float(prior[r['epoch']]['heldout_relative_l1'])) for r in current),
        max_H_L2_absolute_difference=max(abs(r['eval']['measurement']['heldout']['relative_l2']-float(prior[r['epoch']]['heldout_relative_l2'])) for r in current),
        explanation='Independent fresh replay, no old checkpoint opened. FP32 CUDA reductions can differ slightly; original scalar records retained unchanged.')
    target=REPO/'docs/reports/artifacts'/NAME/'final_integrity.json';target.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps({arm:{k:v for k,v in row.items() if k!='full8_AB_ray_geometry_checks'} for arm,row in results.items()},indent=2))


if __name__=='__main__':main()
