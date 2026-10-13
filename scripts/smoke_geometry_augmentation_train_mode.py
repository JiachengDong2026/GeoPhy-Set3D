#!/usr/bin/env python3
"""Real effective-batch2 train-mode AdamW smoke and fresh first-step replay."""
import argparse,json,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import torch
from torch.utils.data import Subset
from geophy_set3d.evaluation.geometry_stratified_sampler import PlannedABHSampler,SamplerTrainingReadGuard
from geophy_set3d.training.selfsup import make_dataset,make_loader,initialize_fresh_model
from geophy_set3d.training.geometry_augmentation_formal import run_epoch,tree_hash,runtime_flags
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything,write_json
from run_geometry_augmentation_formal import configuration,arm_config

def max_difference(a,b):
    pairs=[]
    def walk(x,y):
        if isinstance(x,torch.Tensor):pairs.append((x.detach().cpu().double(),y.detach().cpu().double()))
        elif isinstance(x,dict):
            for k in x:walk(x[k],y[k])
        elif isinstance(x,(tuple,list)):
            for p,q in zip(x,y):walk(p,q)
        else:assert x==y
    walk(a,b)
    return dict(max_abs=max((float((x-y).abs().max()) for x,y in pairs if x.numel()),default=0),
        max_relative=max((float(((x-y).abs()/x.abs().clamp_min(1e-12)).max()) for x,y in pairs if x.numel()),default=0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config');a=ap.parse_args();c,out=configuration(a.config);reg=out/'registration';reg.mkdir(parents=True,exist_ok=True)
    result=reg/'train_mode_smoke.json'
    if result.exists():assert json.loads(result.read_text())['passed'];print('Smoke completed; no repeat');return
    write_json(reg/'numeric_tolerance_preregistered.json',dict(tolerance=c['numeric_tolerance'],replay_steps=c['numeric_replay_steps'],prior_to_smoke=True))
    old=Path(c['paths']['existing_control']);order=json.loads((old/'case_orders.json').read_text())[0][:2];rows=[];comparisons={};start=time.perf_counter()
    for arm in c['arms']:
        cfg=arm_config(c,arm);path=old/'Control_station_plan.json' if arm=='Control' else Path(c['paths']['preparation'])/'Treatment_station_plan.json'
        plan=json.loads(path.read_text());states=[]
        with SamplerTrainingReadGuard(cfg['dataset']['root'],reg/'checkpoints',out) as guard:
            for repeat in range(2):
                seed_everything(c['seed']);torch.set_num_threads(2);model=initialize_fresh_model(cfg);model.train()
                initial=state_hash(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);opt.formal_trace=[]
                ds=make_dataset(cfg);ds.sampler=PlannedABHSampler(plan);loader=make_loader(Subset(ds,order),cfg,shuffle=False,generator=torch.Generator().manual_seed(c['seed']))
                metrics=run_epoch(model,loader,cfg,opt);assert len(opt.state)>0 and initial!=state_hash(model.state_dict())
                grads=metrics['gradient_norms'];assert grads[0]['preclip']>0 and grads[0]['postclip']>0
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                optim={k:{n:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for n,v in s.items()} for k,s in opt.state_dict()['state'].items()}
                states.append((state,optim));rows.append(dict(arm=arm,repeat=repeat,model_training=model.training,initial_state_sha256=initial,
                    after_state_sha256=state_hash(model.state_dict()),parameter_hash=state_hash(dict(model.named_parameters())),buffer_hash=state_hash(dict(model.named_buffers())),optimizer_state_hash=tree_hash(opt.state_dict()),
                    metrics=metrics,trace=opt.formal_trace,flags=runtime_flags(),case_indices=order,batch_size=2,finite_nonzero_gradient=True,parameters_changed=True,optimizer_state_entries=len(opt.state)))
                del model,opt;torch.cuda.empty_cache()
            d=max_difference(states[0][0],states[1][0]);o=max_difference(states[0][1],states[1][1]);rr=[r for r in rows if r['arm']==arm]
            comparisons[arm]=dict(loss_abs_difference=abs(rr[0]['metrics']['loss']['total']-rr[1]['metrics']['loss']['total']),parameter_difference=d,optimizer_difference=o,
                parameter_hash_equal=rr[0]['after_state_sha256']==rr[1]['after_state_sha256'],optimizer_hash_equal=rr[0]['optimizer_state_hash']==rr[1]['optimizer_state_hash'],audit=guard.summary())
            assert guard.summary()['gt_open_attempts']==guard.summary()['supervised_checkpoint_open_attempts']==guard.summary()['pool_b_open_attempts']==guard.summary()['test_or_unknown_npy_attempts']==0
    write_json(result,dict(passed=True,actual_optimizer_steps=4,seconds=time.perf_counter()-start,rows=rows,repeat_comparisons=comparisons,
        nondeterminism='CUDA grid_sample 2D/3D backward atomic accumulation; historical deterministic flags retained for both arms. Exact hash differences and observed numeric ranges recorded; no GT or seed search.',tolerance=c['numeric_tolerance']))
    print(json.dumps(comparisons,indent=2),flush=True)
if __name__=='__main__':main()
