#!/usr/bin/env python3
"""One fixed 16-case readout; no reconstruction training in this entry."""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
import yaml
from geophy_set3d.training.senseiver_readout import ProbeGuard,standardize,ridge,ScalarQueryReadout,errors
from geophy_set3d.training.senseiver_consistency_probe import validate
from geophy_set3d.training.selfsup import make_dataset
from geophy_set3d.training.ray_senseiver import sha256,tensor_identity,build_comparison_model
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything,write_json,atomic_checkpoint
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.diagnostics.senseiver_signal_path import model_hashes,valid_measurements
from run_senseiver_init_probe import now
NAME='ray_senseiver_quick_repair_v1'
OUT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons')/NAME
ART=REPO/'docs/reports/artifacts'/NAME
CONFIG=REPO/'configs'/f'{NAME}.yaml'


def settings():
    c=yaml.safe_load(CONFIG.read_text());s=validate(yaml.safe_load((REPO/c['source_config']).read_text()))
    assert c['name']==NAME and c['seed']==20261011
    assert c['readout']['fit_indices']==list(range(8)) and c['readout']['check_indices']==list(range(8,16))
    assert c['readout']['query_updates']==200 and c['readout']['query_lr']==2e-4 and c['readout']['query_weight_decay']==.01 and c['readout']['ridge_factor']==.001
    layout=next(r for r in json.loads((REPO/c['readout']['geometry_source']).read_text())['layouts'] if r['geometry_id']==c['readout']['geometry_id'])
    s['training']['device']=c['device'];return c,s,layout


def guard(c,s,ds):
    cp=Path(c['readout']['source_checkpoint'])
    return ProbeGuard(s['dataset']['root'],cp.parent,ds.records,16,cp)


def fixed_input(ds,index,ids):
    row,p=ds._projections(index)
    sample=ds._measurement_sample(row,p,np.array(ids,dtype=np.int64),np.empty(0,dtype=np.int64))
    return projection_pool_collate([sample])


def inventory(ds,ids):
    rows=[];geometry=None
    for i in range(16):
        b=fixed_input(ds,i,ids);v=valid_measurements(b)
        fields=[k for k in sorted(b) if k.startswith('input_') and k not in ['input_projection','input_projection_raw']]
        current=tensor_identity(*[b[k] for k in fields]);geometry=geometry or current;assert current==geometry
        assert b['input_station_indices'].tolist()==[ids]
        raw=b['input_projection_raw'][v].double();norm=b['input_projection'][v].double()
        torch.testing.assert_close(norm,raw/32284.894580078173,atol=4e-8,rtol=1e-6)
        rows.append(dict(index=i,case_id=b['case_id'][0],split='fit' if i<8 else 'check',station_ids=ids,
            geometry_identity=current,valid_rays=int(v.sum()),valid_zero=int((raw==0).sum()),
            raw_ppm_m_mean=float(raw.mean()),target=float((raw/32284.894580078173).mean()),
            normalized_field_mean=float(norm.mean()),input_identity=tensor_identity(*[b[k] for k in sorted(b) if isinstance(b[k],torch.Tensor)])))
    return rows


def manifest():
    files=subprocess.check_output(['git','ls-files','src/**/*.py'],cwd=REPO,text=True).splitlines()
    files += ['src/geophy_set3d/training/senseiver_readout.py','scripts/run_senseiver_quick_readout.py',
        'tests/test_senseiver_quick_repair.py',str(CONFIG.relative_to(REPO)),'docs/ray_senseiver_quick_repair_v1_protocol_zh.md']
    return {p:sha256(REPO/p) for p in sorted(set(files))}


def prepare(c,s,layout):
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    target=OUT/'readout_registration.json';assert not target.exists(),'No repeat registration'
    ds=make_dataset(s);g=guard(c,s,ds)
    try:
        with g:
            assert sha256(c['readout']['source_checkpoint'])==c['readout']['checkpoint_sha256']
            inv=inventory(ds,layout['station_ids']);root=Path(s['dataset']['root'])
            hashes={str(root/r['train_projection_path']):sha256(root/r['train_projection_path']) for r in ds.records[:16]}
            hashes.update({str(root/n):sha256(root/n) for n in ['source_index.jsonl','geometry/train_geometry.npz','train_statistics.json']})
            history={str(p.relative_to(REPO)):sha256(p) for name in ['signal_path','consistency_probe','relative_probe','init_probe'] for p in (REPO/'docs/reports').glob(f'ray_senseiver_{name}_v1_zh.md')}
            # signal report naming is covered explicitly as well.
            for p in [REPO/'docs/reports/ray_senseiver_signal_path_v1_zh.md',REPO/'docs/reports/ray_senseiver_learning_diagnostic_zh.md']:
                history[str(p.relative_to(REPO))]=sha256(p)
            reg=dict(at=now(),base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),config=c,
                source_config=s,layout=layout,fixed_inputs=inv,data_sha256=hashes,history_sha256=history,source_sha256=manifest(),
                expected_encoder_calls=16,expected_query_updates=200,check_used_for_selection=False,access=g.summary())
    finally:write_json(OUT/'readout_prepare_access.json',g.summary())
    write_json(target,reg);(ART/target.name).write_bytes(target.read_bytes())
    print(json.dumps(dict(layout=layout,access=g.summary(),registered=16),indent=2))


def verify(c):
    p=OUT/'readout_registration.json';r=json.loads(p.read_text());assert r['config']==c
    assert subprocess.check_output(['git','show',f'HEAD:docs/reports/artifacts/{NAME}/{p.name}'],cwd=REPO)==p.read_bytes()
    for path,h in {**r['source_sha256'],**r['history_sha256']}.items():assert sha256(REPO/path)==h,path
    for path,h in r['data_sha256'].items():assert sha256(path)==h,path
    return r


def run(c,s,layout):
    assert not (OUT/'readout_complete.json').exists(),'Readout already completed'
    assert not (OUT/'latents.npz').exists(),'Latents already extracted; audit before any retry'
    torch.set_num_threads(2);seed_everything(c['seed']);ds=make_dataset(s);g=guard(c,s,ds);started=time.perf_counter()
    try:
        with g:
            reg=verify(c);assert inventory(ds,layout['station_ids'])==reg['fixed_inputs']
            cp=c['readout']['source_checkpoint'];assert sha256(cp)==c['readout']['checkpoint_sha256']
            saved=torch.load(cp,map_location='cpu',weights_only=False)
            assert saved['arm']=='treatment' and saved['epoch']==35 and saved['updates']==140 and saved['relative_l1']==saved['consistency_weight']==0
            assert saved['exit_state']['reason']=='FAILED_TO_LEARN' and saved['config']==s and not saved['volumetric_gt_used']
            model=build_comparison_model(s);model.load_state_dict(saved['model']);del saved
            model.eval().requires_grad_(False);before=model_hashes(model);latents=[]
            with (OUT/'readout_encoder_ledger.jsonl').open('a') as ledger:
                for i in range(16):
                    b={k:v.to(c['device']) if isinstance(v,torch.Tensor) else v for k,v in fixed_input(ds,i,layout['station_ids']).items()}
                    ledger.write(json.dumps(dict(index=i,event='started',at=now()))+'\n');ledger.flush()
                    with torch.no_grad():z=model.encode(b)
                    assert z.shape==(1,64,64) and torch.isfinite(z).all()
                    latents.append(z.cpu());ledger.write(json.dumps(dict(index=i,event='returned',at=now()))+'\n');ledger.flush()
            x=torch.cat(latents);y=torch.tensor([r['target'] for r in reg['fixed_inputs']],dtype=torch.float64)
            assert model_hashes(model)==before and all(p.grad is None for p in model.parameters())
            np.savez(OUT/'latents.npz',latents=x.numpy(),targets=y.numpy())
            z,t,stats=standardize(x,y)
            np.savez(OUT/'readout_stats.npz',**{k:v.numpy() for k,v in stats.items()})
            linear,stability=ridge(z,t)
            seed_everything(c['seed']);head=ScalarQueryReadout().to(c['device']);head.train()
            opt=torch.optim.AdamW(head.parameters(),lr=2e-4,weight_decay=.01)
            fit_z=z[:8].to(c['device'],dtype=torch.float32);fit_y=t[:8].to(c['device'],dtype=torch.float32)
            history=[];best_loss=float('inf');best=None;best_step=None
            for step in range(201):
                if step:
                    opt.zero_grad(set_to_none=True);loss=(head(fit_z)-fit_y).square().mean();loss.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
                    opt.step()
                with torch.no_grad():metric=float((head(fit_z)-fit_y).square().mean())
                assert np.isfinite(metric)
                history.append(dict(step=step,fit_standardized_MSE=metric))
                if metric<best_loss:best_loss=metric;best=copy.deepcopy(head.state_dict());best_step=step
            head.load_state_dict(best);head.eval()
            with torch.no_grad():query=head(z.to(c['device'],dtype=torch.float32)).cpu().double()
            predictions=dict(ridge=linear*stats['target_std']+stats['target_mean'],query=query*stats['target_std']+stats['target_mean'],
                constant=torch.full_like(y,stats['target_mean']),direct_input=y.clone())
            # Independent direct mean from the normalized field, rather than simply reusing target.
            predictions['direct_input']=torch.tensor([r['normalized_field_mean'] for r in reg['fixed_inputs']],dtype=torch.float64)
            metrics={k:errors(v,y) for k,v in predictions.items()}
            readable={k:metrics[k]['check']['MSE']<.5*metrics['constant']['check']['MSE'] for k in ['ridge','query']}
            result=dict(at=now(),config=c,source_checkpoint_sha256=sha256(cp),encoder_identity_before=before,
                encoder_identity_after=model_hashes(model),encoder_grad_buffers_none=all(p.grad is None for p in model.parameters()),
                encoder_calls=16,encoder_updates=0,query_updates=200,best_query_step=best_step,selection='fit MSE only; check used once after selection',
                training_statistics={k:float(v) for k,v in stats.items() if v.numel()==1},statistics_sha256=sha256(OUT/'readout_stats.npz'),
                latents_sha256=sha256(OUT/'latents.npz'),ridge_stability=stability,metrics=metrics,readable=readable,
                predictions=[dict(index=i,split='fit' if i<8 else 'check',target=float(y[i]),**{k:float(v[i]) for k,v in predictions.items()}) for i in range(16)],
                query_history=history,seconds=time.perf_counter()-started,access=g.summary(),no_GT=True)
            atomic_checkpoint(OUT/'readout_query_best.pt',dict(model=best,best_step=best_step,not_reconstruction_warm_start=True))
            write_json(OUT/'readout_complete.json',result)
    finally:write_json(OUT/'readout_run_access.json',g.summary())
    for name in ['readout_complete.json','readout_encoder_ledger.jsonl','readout_prepare_access.json','readout_run_access.json']:
        (ART/name).write_bytes((OUT/name).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ['query_history','predictions']},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','run'],required=True);a=ap.parse_args()
    c,s,l=settings()
    if a.phase=='prepare':prepare(c,s,l)
    else:run(c,s,l)
