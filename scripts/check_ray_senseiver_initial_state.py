#!/usr/bin/env python3
"""One fixed train-case, hash-verified reconstruction of original fresh initialization."""
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import torch
from diagnose_ray_senseiver_learning import spatial,moments
from geophy_set3d.training.ray_senseiver import load_comparison_config,build_comparison_model,sha256
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything,write_json
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.selfsup import make_dataset,objective,move_dual_batch,projection_scores
from geophy_set3d.data.measurement_only import dual_measurement_collate

c=load_comparison_config('configs/train_ray_senseiver_selfsup_v1.yaml');out=Path(c['comparison']['output']);diag=out/'learning_diagnostic'
expected=json.loads((diag/'diagnosis.json').read_text())['initial_state_sha256']
target=diag/'initial_fixed_train_case.json'
if target.exists():raise FileExistsError(target)
with GTReadGuard(c['dataset']['root'],Path(c['experiment']['output_root'])/'checkpoints'/c['experiment']['name']) as guard:
    seed_everything(c['training']['seed']);torch.set_num_threads(2);model=build_comparison_model(c);model.eval()
    actual=state_hash(model.state_dict());assert actual==expected
    ds=make_dataset(c);b=move_dual_batch(dual_measurement_collate([ds[0]]),c['training']['device'])
    logits=[];hook=model.decoder.postproc.register_forward_hook(lambda m,a,p:logits.append(p.detach()))
    with torch.no_grad():loss,parts,volumes,projections=objective(model,b)
    hook.remove();scores=projection_scores(projections,b)
    result=dict(initial_state_sha256=actual,exact_formal_initial_hash_match=True,weights_loaded=False,optimizer_created=False,optimizer_steps=0,case_index=0,case_id=b['A']['case_id'][0],sampler_epoch=0,scope='single fixed train case; not full initial train/Val aggregate',loss=dict(total=float(loss),**{k:float(v) for k,v in parts.items()}),measurement={k:{m:sum(r[m] for r in rows)/len(rows) for m in ['relative_l1','relative_l2']} for k,rows in scores.items()},output_A=spatial(volumes[0]),output_B=spatial(volumes[1]),logits=moments(torch.cat(logits,1)),access=guard.summary(),source_sha256=sha256(__file__))
    assert state_hash(model.state_dict())==actual
    write_json(target,result)
print(json.dumps(result,indent=2))
