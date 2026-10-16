#!/usr/bin/env python3
"""Small real learning check for execution candidates, never formal acceptance.

Uses the production epoch runner and unchanged loss/sampling, saves no weights.
It deliberately does not create acceptance_result.json or a formal freeze.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from benchmark_plan_a_speed import LoadSampler,sha
from plan_a_budget_v4 import load_budget_config,build_budget_model,configure_backend,budget_provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--train-cases',type=int,choices=(2,16),default=16)
    parser.add_argument('--val-cases',type=int,choices=(2,8),default=8)
    parser.add_argument('--epochs',type=int,choices=(5,20),default=5)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh result path')
    if (args.train_cases,args.val_cases,args.epochs) not in ((16,8,5),(2,2,20)):
        parser.error('Only gate-size 16/8/5 or smoke-size 2/2/20 allowed')
    import torch
    from geophy_set3d.models.set_unrolled3d import build_set_unrolled_model
    from geophy_set3d.training.gt_read_guard import GTReadGuard
    from geophy_set3d.training.research_selfsup import (
        _limited,_provenance,load_research_config,make_research_dataset,
        make_research_loader,measurement_dependence_probe,run_research_epoch)
    from geophy_set3d.training.selfsup import move_dual_batch,selection_metric
    from geophy_set3d.training.utils import seed_everything
    args.output.parent.mkdir(parents=True,exist_ok=True)
    config=load_budget_config(args.config)
    report=dict(protocol='plan_a_budget_learning_probe_v4',status='started',
                formal_acceptance=False,formal_training=False,checkpoints_saved=False,
                runtime=config['execution_runtime'],architecture=config['architecture'],config=str(args.config),device=args.device,
                architecture_script_sha256=sha(ROOT/'scripts/plan_a_budget_v4.py'),
                runtime_script_sha256=sha(ROOT/'scripts/plan_a_runtime_candidates.py'),
                probe_script_sha256=sha(Path(__file__)),epochs=[])
    guard=None
    sampler=LoadSampler()
    began=time.perf_counter()
    code=1
    try:
        config=copy.deepcopy(config)
        config['stage']='learning_probe'
        config['training'].update(device=args.device,max_epochs=args.epochs,batch_size=1,gradient_accumulation=2)
        config['dataset'].update(train_limit=args.train_cases,val_limit=args.val_cases)
        config['scheduler']['T_max']=args.epochs
        report['resolved_config']=config
        torch.set_num_threads(2)
        seed_everything(config['training']['seed'])
        configure_backend(report['runtime'])
        sampler.start()
        guard=GTReadGuard(config['dataset']['root'],args.output.parent/'unused_checkpoints')
        with guard:
            train_base=make_research_dataset(config)
            val_base=make_research_dataset(config,validation=True)
            train=_limited(train_base,args.train_cases)
            val=_limited(val_base,args.val_cases)
            report['provenance']=budget_provenance(config,train,val,_provenance)
            train_loader=make_research_loader(train,config,shuffle=True,generator=torch.Generator().manual_seed(config['training']['seed']))
            val_loader=make_research_loader(val,config,generator=torch.Generator().manual_seed(0))
            model=build_budget_model(config).to(args.device)
            digest=hashlib.sha256()
            for k,v in model.state_dict().items():
                digest.update(k.encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
            report['initial_state_sha256']=digest.hexdigest()
            report['parameter_count']=sum(p.numel() for p in model.parameters())
            optimizer=torch.optim.AdamW(model.parameters(),**{k:v for k,v in config['optimizer'].items() if k!='name'})
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
            torch.cuda.synchronize(args.device)
            report['initial_validation']=run_research_epoch(model,val_loader,config,model.operator)
            for epoch in range(args.epochs):
                train_base.set_epoch(epoch);val_base.set_epoch(0)
                torch.cuda.synchronize(args.device)
                start=time.perf_counter()
                train_result=run_research_epoch(model,train_loader,config,model.operator,optimizer)
                val_result=run_research_epoch(model,val_loader,config,model.operator)
                torch.cuda.synchronize(args.device)
                scheduler.step()
                row=dict(epoch=epoch+1,seconds=time.perf_counter()-start,train=train_result,validation=val_result)
                report['epochs'].append(row)
                print(json.dumps(dict(epoch=epoch+1,seconds=row['seconds'],heldout_relative_l1=selection_metric(val_result))),flush=True)
            report['measurement_dependence']=measurement_dependence_probe(model,move_dual_batch(next(iter(val_loader)),args.device)['A'])
            initial=selection_metric(report['initial_validation'])
            best=min(selection_metric(r['validation']) for r in report['epochs'])
            final=report['epochs'][-1]['validation']
            updates=sum(r['train']['optimizer_updates'] for r in report['epochs'])
            report['criteria']=dict(complete=all(r[p]['complete_epoch'] for r in report['epochs'] for p in ('train','validation')),
                expected_updates=updates==args.epochs*args.train_cases//2,
                heldout_improves_initial=math.isfinite(best) and best<initial,
                heldout_better_than_zero=best<1.,nonzero_gradient=any(r['train']['gradient_norm_max']>0 for r in report['epochs']),
                output_changes_across_cases=final['output']['case_to_case_mean_absolute_difference']>1e-7,
                measurement_dependence=report['measurement_dependence']['passed'],
                finite_losses=all(math.isfinite(r[p]['loss']['total']) for r in report['epochs'] for p in ('train','validation')))
            report.update(initial_heldout_relative_l1=initial,best_heldout_relative_l1=best,optimizer_updates=updates,status='passed')
            code=0
    except Exception as exc:
        report.update(status='failed',error=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc())
    finally:
        sampler.stop()
        report['gt_read_audit']=guard.summary() if guard else {}
        report['gpu_load_samples']=sampler.samples
        report['elapsed_seconds']=time.perf_counter()-began
        audit=report['gt_read_audit']
        report['no_forbidden_reads']=bool(audit) and not any(audit.get(k,1) for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        report['learning_checks_passed']=report['status']=='passed' and report['no_forbidden_reads'] and all(report.get('criteria',{}).values())
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(args.output),status=report['status'],learning_checks_passed=report['learning_checks_passed'])))
    return code


if __name__=='__main__':
    raise SystemExit(main())
