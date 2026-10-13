#!/usr/bin/env python3
"""Preregister and execute the single matched explicit augmentation intervention."""
import argparse,copy,json,sys,time,subprocess,shutil,hashlib
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch,yaml
from torch.utils.data import Subset
from geophy_set3d.data.build_utils import digest
from geophy_set3d.evaluation.geometry_stratified_sampler import PlannedABHSampler,SamplerTrainingReadGuard
from geophy_set3d.training.selfsup import make_dataset,make_loader,run_epoch,selection_metric,initialize_fresh_model,validate_selfsup_config
from geophy_set3d.training.geometry_augmentation_formal import run_epoch as observed_epoch
from geophy_set3d.training.utils import seed_everything,write_json,atomic_checkpoint,rng_state,restore_rng
from geophy_set3d.training.rvtc_matched import state_hash
CONFIG=REPO/'configs/geometry_augmentation_formal_v1.yaml'
ARCHIVE=REPO/'docs/reports/artifacts/geometry_augmentation_formal_v1'

def configuration(path=None):
    c=yaml.safe_load(Path(path or CONFIG).read_text());return c,Path(c['output'])

def arm_config(c,arm):
    cfg=copy.deepcopy(yaml.safe_load((REPO/c['base_config']).read_text()))
    cfg['dataset']['root']=c['paths']['dataset'];cfg['experiment']['name']=c['name']+'_'+arm
    cfg['experiment']['output_root']=c['output'];cfg['training']['device']=c['arms'][arm]['device']
    cfg['loader']['num_workers']=0;cfg['loader']['persistent_workers']=False
    validate_selfsup_config(cfg);return cfg

def verify(c,out,committed=True):
    p=json.loads((out/'protocol.json').read_text());assert p['config']==c
    assert digest(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    for k,v in p['source_sha256'].items():assert digest(REPO/k)==v,k
    for k,v in p['input_sha256'].items():assert digest(out/k)==v,k
    for k,v in p['historical_sha256'].items():assert digest(k)==v,k
    if committed:
        assert subprocess.check_output(['git','show','HEAD:docs/reports/artifacts/geometry_augmentation_formal_v1/protocol.json'],cwd=REPO)==(out/'protocol.json').read_bytes()
    return p

def register(c,out):
    if (out/'protocol.json').exists():verify(c,out,False);return
    reg=out/'registration';old=Path(c['paths']['existing_control']);prep=Path(c['paths']['preparation'])
    assert json.loads((reg/'train_mode_smoke.json').read_text())['passed']
    assert json.loads((reg/'control_reuse_audit.json').read_text())['CONTROL_REUSE']=='REJECTED'
    shutil.copy2(old/'Control_station_plan.json',out/'Control_station_plan.json')
    shutil.copy2(prep/'Treatment_station_plan.json',out/'Treatment_station_plan.json')
    for f in ['case_orders.json','selector_station_ids.json']:shutil.copy2(old/f,out/f)
    cfg=arm_config(c,'Control');hashes={}
    with SamplerTrainingReadGuard(cfg['dataset']['root'],reg/'checkpoints',out) as guard:
        for arm in c['arms']:
            seed_everything(c['seed']);model=initialize_fresh_model(arm_config(c,arm));hashes[arm]=state_hash(model.state_dict());del model
        assert len(set(hashes.values()))==1
        write_json(reg/'registration_access_audit.json',guard.summary())
    sourcefiles=sorted((REPO/'src').rglob('*.py'))+[CONFIG,REPO/c['base_config'],REPO/'docs/geometry_augmentation_formal_v1_protocol_zh.md']
    sourcefiles+=list(REPO.glob('scripts/*geometry_augmentation_formal*.py'))+[REPO/'scripts/audit_control_reuse.py',REPO/'scripts/register_confirmatory_geometries.py',REPO/'scripts/smoke_geometry_augmentation_train_mode.py']
    sourcefiles.append(REPO/'tests/test_geometry_augmentation_formal.py')
    # Report text may evolve after results; training, evaluation and statistics code are frozen.
    sourcefiles=[f for f in sourcefiles if f.name!='report_geometry_augmentation_formal.py']
    files=['Control_station_plan.json','Treatment_station_plan.json','case_orders.json','selector_station_ids.json',
        'registration/numeric_tolerance_preregistered.json','registration/preflight_failures.json','registration/control_fresh_numeric_replay.json','registration/train_mode_smoke.json','registration/control_reuse_audit.json','registration/station_marginals.json',
        'confirmatory_geometries/geometry_layouts.json','confirmatory_geometries/evaluation_records.json','confirmatory_geometries/access_audit.json']
    historical={str(f):digest(f) for root in [old,prep,Path(c['paths']['sensitivity']),Path(c['paths']['historical_strata'])] for f in root.rglob('*') if f.is_file() and f.suffix in ('.json','.jsonl','.csv','.yaml','.md','.sha256')}
    # Dataset and GT128 are metadata-only snapshots; no GT contents read.
    from diagnose_clustered_one_swap import snapshot
    write_json(reg/'dataset_before.json',snapshot(c['paths']['dataset']))
    parent=out.parents[1]
    write_json(reg/'history_before.json',{str(f):snapshot(f) for section in ('checkpoints','logs','comparisons','experiments','diagnostics','evaluations') for f in (parent/section).iterdir() if f.is_dir() and f.resolve()!=out.resolve()})
    files.append('registration/history_before.json')
    files.append('registration/dataset_before.json')
    p=dict(status='REGISTERED BEFORE FORMAL TRAINING',config=c,configs={a:arm_config(c,a) for a in c['arms']},initial_state_sha256=hashes,
        source_sha256={str(f.relative_to(REPO)):digest(f) for f in sourcefiles},input_sha256={f:digest(out/f) for f in files},historical_sha256=historical,
        CONTROL_REUSE='REJECTED',optimizer_steps=24900,only_intervention='A/B observation geometry distribution, frozen .50/.25/.25, training hard10deg',
        selector='Original natural Val dual A4/B4 H4 mean heldout measurement relative L1; strict <; no early stop',
        primary='confirmatory G5 case mean relative L1: bootstrap 95% delta CI upper<0',
        noninferiority='confirmatory G1-G4 macro ratio of grand means <=1.03 (point rule); ratio CI additionally reported',
        partial='natural ratio<=1.10 AND (G5 CI upper<0 OR G5 improvement>=10% with CI spanning0); only after GO fails',
        no_go='all remaining outcomes',bootstrap=dict(seed=c['bootstrap_seed'],resamples=c['bootstrap_samples'],units='36 cases after layout averaging'),
        posthoc='both300 complete and best frozen, measurements+selector audited, then independent saved-output GT64; no GT selection',
        runtime_tolerance=c['numeric_tolerance'],limitations='single matched seed, station marginals part of intervention, CUDA grid_sample backward nondeterminism retained')
    write_json(out/'protocol.json',p);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    for f in files+['protocol.json','protocol.sha256','registration/registration_access_audit.json']:
        dst=ARCHIVE/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(out/f,dst)
    print('Formal registration frozen; commit before training',hashes,flush=True)

def train(c,out,arm):
    p=verify(c,out);cfg=arm_config(c,arm);root=out/arm;root.mkdir(exist_ok=True);ck=root/'checkpoints';ck.mkdir(exist_ok=True)
    if (root/'complete.json').exists():assert json.loads((root/'complete.json').read_text())['epochs']==300;print(arm,'complete; no repeated training');return
    with SamplerTrainingReadGuard(cfg['dataset']['root'],ck,root) as audit:
        seed_everything(c['seed']);torch.set_num_threads(2);model=initialize_fresh_model(cfg)
        assert state_hash(model.state_dict())==p['initial_state_sha256'][arm]
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        ds=make_dataset(cfg);val=make_dataset(cfg,validation=True)
        plans=json.loads((out/f'{arm}_station_plan.json').read_text());ds.sampler=PlannedABHSampler(plans)
        orders=json.loads((out/'case_orders.json').read_text());generator=torch.Generator().manual_seed(c['seed'])
        vloader=make_loader(val,cfg);start=0;best=float('inf');count={'steps':0}
        last=ck/'last.pt'
        if last.exists():
            saved=torch.load(last,map_location=cfg['training']['device'],weights_only=False)
            assert saved['config']==cfg and saved['protocol_sha256']==digest(out/'protocol.json')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            restore_rng(saved['rng'],generator);start=saved['epoch']+1;best=saved['best'];count['steps']=saved['optimizer_steps']
            rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines() if json.loads(l)['epoch']<start]
            (root/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        optimizer.register_step_post_hook(lambda *args:count.update(steps=count['steps']+1))
        for epoch in range(start,300):
            ds.set_epoch(epoch);val.set_epoch(0);began=time.perf_counter();steps_before=count['steps'];lr=optimizer.param_groups[0]['lr']
            loader=make_loader(Subset(ds,orders[epoch]),cfg,shuffle=False,generator=generator)
            torch.cuda.reset_peak_memory_stats(cfg['training']['device']);tr=observed_epoch(model,loader,cfg,optimizer);va=run_epoch(model,vloader,cfg)
            assert count['steps']-steps_before==83 and tr['sample_count']==166 and va['sample_count']==36
            expected=[(r['case_id'],r['A'],r['B'],r['H']) for r in plans[epoch]]
            assert tr['case_station_digest']==hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest()
            selector=json.loads((out/'selector_station_ids.json').read_text())
            expected=[(r['case_id'],r['A'],r['B'],r['H']) for r in selector]
            assert va['case_station_digest']==hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest()
            score=selection_metric(va);improved=score<best;best=min(best,score);scheduler.step()
            row=dict(epoch=epoch,arm=arm,train=tr,val=va,learning_rate=lr,seconds=time.perf_counter()-began,optimizer_steps=count['steps'],
                case_order=orders[epoch],selector_metric=score,best_checkpoint_updated=improved,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(cfg['training']['device']))
            payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,
                optimizer_steps=count['steps'],config=cfg,protocol_sha256=digest(out/'protocol.json'),rng=rng_state(generator),volumetric_gt_used=False)
            atomic_checkpoint(last,payload)
            if improved:atomic_checkpoint(ck/'best.pt',payload)
            with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            write_json(root/'access_audit.json',audit.summary())
            print(f'{arm} {epoch+1}/300 steps={count["steps"]} valH_L1={score:.6f} best={best:.6f} seconds={row["seconds"]:.1f}',flush=True)
        assert count['steps']==24900
        write_json(root/'complete.json',dict(epochs=300,optimizer_steps=count['steps'],best_sha256=digest(ck/'best.pt'),last_sha256=digest(last),
            initial_state_sha256=p['initial_state_sha256'][arm],GT_used=False,access=audit.summary(),runtime_flags=__import__('geophy_set3d.training.geometry_augmentation_formal',fromlist=['runtime_flags']).runtime_flags(),selector='original natural held-out mean measurement L1',
            protocol_sha256=digest(out/'protocol.json')))
    verify(c,out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['register','train']);ap.add_argument('--arm',choices=['Control','Treatment']);ap.add_argument('--config');a=ap.parse_args()
    c,out=configuration(a.config)
    if a.phase=='register':register(c,out)
    else:assert a.arm;train(c,out,a.arm)
if __name__=='__main__':main()
