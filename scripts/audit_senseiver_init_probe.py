#!/usr/bin/env python3
"""Read-only final verification: own last checkpoints and fixed8 train inputs."""
import gzip
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import torch
import yaml
from geophy_set3d.training.senseiver_init_probe import validate,TrainEightGuard
from geophy_set3d.training.ray_senseiver import sha256,tensor_identity
from geophy_set3d.training.selfsup import make_dataset,move_dual_batch
from geophy_set3d.data.measurement_only import dual_measurement_collate
from diagnose_ray_senseiver_learning import geometry


def main():
    c=validate(yaml.safe_load((REPO/'configs/ray_senseiver_init_probe_v1.yaml').read_text()))
    name=c['experiment']['name'];out=Path(c['experiment']['output_root'])/'comparisons'/name
    ck=Path(c['experiment']['output_root'])/'checkpoints'/name
    torch.set_num_threads(2);ds=make_dataset(c)
    with gzip.open(out/'fixed_inputs.json.gz','rt') as f:inventory=json.load(f)
    final={}
    for arm in ['control','treatment']:
        complete=json.loads((out/arm/'complete.json').read_text())
        with TrainEightGuard(c['dataset']['root'],ck/arm,ds.records) as guard:
            last=ck/arm/'last.pt';assert sha256(last)==complete['last_sha256']
            saved=torch.load(last,map_location='cpu',weights_only=False)
            steps=sorted(set(int(s['step']) for s in saved['optimizer']['state'].values()))
            assert steps==[complete['updates']]
            assert saved['epoch']==complete['completed_epochs']==saved['scheduler']['last_epoch']
            assert saved['updates']==4*saved['epoch']==complete['updates']
            assert saved['initial_hash']==complete['initial_hash'] and not saved['volumetric_gt_used']
            assert all(torch.isfinite(p).all() for p in saved['model'].values())
            original=json.loads((REPO/'docs/reports/artifacts/ray_senseiver_selfsup_v1/training_source_manifest.json').read_text())
            original_unchanged=all(sha256(REPO/p)==h for p,h in original['sha256'].items());assert original_unchanged
            frozen=json.loads((out/'source_manifest.json').read_text())
            assert all(sha256(REPO/p)==h for p,h in frozen['sha256'].items())
            checks=[]
            for i in range(8):
                sample=ds[i]
                assert ds.records[i]['case_id']==inventory[i]['case_id']
                for k in ['A','B','H']:
                    assert tensor_identity(*[sample[k][key] for key in sorted(sample[k]) if isinstance(sample[k][key],torch.Tensor)])==inventory[i]['subsets'][k]['batch_identity']
                b=move_dual_batch(dual_measurement_collate([sample]),c['training']['device'])
                for k in ['A','B']:
                    r=geometry(b[k]);assert r['valid_rays']==r['packed_real_rays'] and r['entry_exit_in_domain'] and r['all_tokens_finite']
                    expected=sum(s['valid_count'] for s in inventory[i]['subsets'][k]['stations']);assert expected==r['valid_rays']
                    checks.append(dict(index=i,subset=k,**r))
            final[arm]=dict(passed=True,optimizer_state_steps=steps,scheduler_epoch=saved['scheduler']['last_epoch'],
                            checkpoint_sha256=sha256(last),original_v1_sources_unchanged=original_unchanged,probe_sources_unchanged=True,
                            all_fixed_case_station_measurement_geometry_identities_equal=True,full8_valid_ray_geometry_checks=checks,
                            no_training_or_optimizer_step=True,access=guard.summary())
    target=REPO/'docs/reports/artifacts'/name/'final_integrity.json'
    target.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{key:value for key,value in row.items() if key!='full8_valid_ray_geometry_checks'} for k,row in final.items()},indent=2))


if __name__=='__main__':main()
