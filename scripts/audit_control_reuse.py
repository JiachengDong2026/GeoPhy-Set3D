#!/usr/bin/env python3
"""Reject reuse whenever full historical numeric equivalence is unprovable."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import torch
from torch.utils.data import Subset
from geophy_set3d.data.measurement_only import DisjointABHSampler
from geophy_set3d.data.build_utils import digest
from geophy_set3d.evaluation.geometry_stratified_sampler import PlannedABHSampler,SamplerTrainingReadGuard
from geophy_set3d.training.selfsup import make_dataset,make_loader,initialize_fresh_model,run_epoch as original_epoch
from geophy_set3d.training.geometry_augmentation_formal import run_epoch,tree_hash,runtime_flags
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything,write_json
from smoke_geometry_augmentation_train_mode import max_difference
from run_geometry_augmentation_formal import configuration,arm_config

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config');a=ap.parse_args();c,out=configuration(a.config);reg=out/'registration';dest=reg/'control_reuse_audit.json'
    if dest.exists():assert json.loads(dest.read_text())['CONTROL_REUSE']=='REJECTED';print('Existing completed audit, not replayed');return
    assert (reg/'numeric_tolerance_preregistered.json').exists() and json.loads((reg/'train_mode_smoke.json').read_text())['passed']
    old=Path(c['paths']['existing_control']);p=json.loads((old/'protocol.json').read_text());cfg=arm_config(c,'Control');plans=json.loads((old/'Control_station_plan.json').read_text());orders=json.loads((old/'case_orders.json').read_text())
    historical=[json.loads(l) for l in (old/'Control/metrics.jsonl').read_text().splitlines()];assert len(historical)==300
    checks={};trace=[];summaries=[]
    with SamplerTrainingReadGuard(cfg['dataset']['root'],reg/'checkpoints',out) as guard:
        seed_everything(c['seed']);torch.set_num_threads(2);ds=make_dataset(cfg);val=make_dataset(cfg,validation=True);original=DisjointABHSampler(base_seed=c['seed']);gen=torch.Generator().manual_seed(c['seed'])
        for epoch in range(300):
            assert torch.randperm(len(ds),generator=gen).tolist()==orders[epoch]
            assert historical[epoch]['case_order']==orders[epoch]
            for i,r in enumerate(ds.records):
                sets=original.sample(ds.station_count,epoch=epoch,sample_index=i,case_id=r['case_id'])
                assert plans[epoch][i]['case_id']==r['case_id'] and all(s.tolist()==plans[epoch][i][k] for s,k in zip(sets,('A','B','H')))
        checks.update(all300_case_orders=True,all49800_ABH=True,batch_composition_all300=True,dataloader_workers=0,shuffle=False)
        selector=json.loads((old/'selector_station_ids.json').read_text())
        for i,r in enumerate(val.records):
            ss=val.sampler.sample(val.station_count,epoch=0,sample_index=i,case_id=r['case_id']);assert selector[i]['case_id']==r['case_id']
            assert all(s.tolist()==selector[i][k] for s,k in zip(ss,('A','B','H')))
        checks['all36_selector_ABH_exact']=True
        changes={'experiment','loader','training'}
        oldcfg=p['configs']['Control']
        for k in cfg:
            if k not in changes:assert cfg[k]==oldcfg[k],k
        assert {k:v for k,v in cfg['training'].items() if k!='device'}=={k:v for k,v in oldcfg['training'].items() if k!='device'}
        checks['numeric_configuration_exact_except_device_and_output']=True
        # Original versus independent scalar instrumentation: fresh identical 10-step path.
        numeric=[]
        for method in (original_epoch,run_epoch):
            seed_everything(c['seed']);model=initialize_fresh_model(cfg);initial=state_hash(model.state_dict());assert initial==p['initial_state_sha256']['Control']
            opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);ds.sampler=PlannedABHSampler(plans)
            ds.set_epoch(0);loader=make_loader(Subset(ds,orders[0][:20]),cfg,generator=torch.Generator().manual_seed(c['seed']))
            metric=method(model,loader,cfg,opt)
            numeric.append(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},loss=metric['loss'],optimizer_hash=tree_hash(opt.state_dict())))
            del model,opt;torch.cuda.empty_cache()
        difference=max_difference(numeric[0]['state'],numeric[1]['state'])
        # Real fresh 100 optimizer-step replay. No fabricated historical per-step snapshots.
        seed_everything(c['seed']);model=initialize_fresh_model(cfg);initial_hashes=dict(full=state_hash(model.state_dict()),parameters=state_hash(dict(model.named_parameters())),buffers=state_hash(dict(model.named_buffers())))
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);opt.formal_trace=[];sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
        for epoch,n in ((0,166),(1,34)):
            ds.set_epoch(epoch);loader=make_loader(Subset(ds,orders[epoch][:n]),cfg,generator=torch.Generator().manual_seed(c['seed']))
            summaries.append(run_epoch(model,loader,cfg,opt));trace.extend(dict(epoch=epoch,global_step=len(trace)+j+1,**r) for j,r in enumerate(opt.formal_trace));opt.formal_trace=[];sched.step()
        assert len(trace)==c['numeric_replay_steps']==100
        write_json(reg/'control_fresh_numeric_replay.json',dict(first100_steps=trace,epoch_summaries=summaries,initial_hashes=initial_hashes,flags=runtime_flags(),
            original_vs_observed_10_steps=dict(parameter_difference=difference,loss_difference={k:numeric[1]['loss'][k]-numeric[0]['loss'][k] for k in numeric[0]['loss']},optimizer_hash_equal=numeric[0]['optimizer_hash']==numeric[1]['optimizer_hash']),
            historical_epoch0_comparison={k:dict(historical=historical[0]['train']['loss'][k],fresh=summaries[0]['loss'][k],delta=summaries[0]['loss'][k]-historical[0]['train']['loss'][k]) for k in summaries[0]['loss']},
            historical_first1_10_100='UNAVAILABLE: historical logs only epoch aggregates; numeric replay cannot establish historical per-step equivalence'))
        write_json(dest,dict(CONTROL_REUSE='REJECTED',checks=checks,initial_hashes=initial_hashes,initial_full_hash_exact=True,
            missing=['historical initial separate parameter/buffer hashes','historical step1/10/100 parameter and optimizer states','historical clipped gradient norms','complete historical CUDA/cudnn runtime flags'],
            replay_steps=100,replay_sha256=digest(reg/'control_fresh_numeric_replay.json'),tolerance=c['numeric_tolerance'],audit=guard.summary(),
            consequence='Fresh Control and fresh Treatment, both300 epochs. Configuration/plan equality alone insufficient.',instrumentation_comparison=difference))
    print('CONTROL_REUSE = REJECTED; 100 actual fresh replay steps; missing historical numeric snapshots',flush=True)
if __name__=='__main__':main()
