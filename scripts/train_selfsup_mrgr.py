#!/usr/bin/env python3
"""Memory smoke, one GT-free gate, then fresh 300-epoch measurement-only MRGR."""
import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import yaml
from torch.utils.data import Subset
from geophy_set3d.data.measurement_only import DualMeasurementDataset, dual_measurement_collate
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.selfsup import (validate_selfsup_config,initialize_fresh_model,make_dataset,make_loader,run_epoch,selection_metric,zero_volume_baseline,objective,move_dual_batch)
from geophy_set3d.training.utils import (load_config,seed_everything,directories,write_json,move_batch,atomic_checkpoint,environment,rng_state,restore_rng)
from geophy_set3d.evaluation.measurement_diagnostics import set_properties,feedback_gradients

ROOT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v1')


def station_protocol(dataset):
    return {dataset.records[i]['case_id']:{k:dataset[i][k]['input_station_indices'].tolist() for k in ('A','B','H')} for i in range(len(dataset))}


def smoke(config,config_path):
    target=ROOT/'memory_smoke.json'
    if target.exists():raise FileExistsError('Memory smoke already recorded')
    attempts=[]
    while True:
        paths=directories(config,'_smoke');guard=GTReadGuard(config['dataset']['root'],paths['checkpoints'])
        try:
            with guard:
                seed_everything(config['training']['seed']);torch.set_num_threads(2)
                model=initialize_fresh_model(config)
                optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])
                ds=DualMeasurementDataset(config['dataset']['root'],split='train',a_range=(5,5),b_range=(5,5),hold_views=2,
                    base_seed=config['training']['seed'],normalization=config['normalization']['method'])
                loader=make_loader(Subset(ds,range(2)),config)
                torch.cuda.reset_peak_memory_stats(config['training']['device']);began=time.perf_counter()
                result=run_epoch(model,loader,config,optimizer)
                record=dict(passed=True,batch_size=config['training']['batch_size'],gradient_accumulation=config['training']['gradient_accumulation'],
                    worst_case_views=dict(A=5,B=5,H=2),projector_samples=96,all_rays=True,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(config['training']['device']),seconds=time.perf_counter()-began,
                    measurement_only=result,gt_access=guard.summary(),attempts=attempts)
            write_json(target,record);print(json.dumps(record,indent=2),flush=True);return
        except torch.cuda.OutOfMemoryError:
            attempts.append(dict(batch_size=config['training']['batch_size'],gradient_accumulation=config['training']['gradient_accumulation'],oom=True))
            if config['training']['batch_size']!=2:raise
            del model,optimizer,loader;gc.collect();torch.cuda.empty_cache()
            config['training'].update(batch_size=1,gradient_accumulation=2)
            Path(config_path).write_text(yaml.safe_dump(config,sort_keys=False))


def gate(config):
    if not (ROOT.parent/'mrgr_residual_control_v1'/'summary.json').exists():
        raise RuntimeError('Complete Stage A and close supervised architecture experiments first')
    paths=directories(config,'_gate');target=paths['logs']/'gate_summary.json'
    if target.exists():raise FileExistsError('Only one 8-case/80-epoch selfsup gate is authorized')
    if not (ROOT/'memory_smoke.json').exists():raise RuntimeError('Run maximum-view memory smoke first')
    guard=GTReadGuard(config['dataset']['root'],paths['checkpoints'])
    with guard:
        seed_everything(config['training']['seed']);torch.set_num_threads(2)
        model=initialize_fresh_model(config);ds=make_dataset(config);ds.set_epoch(0)
        loader=make_loader(Subset(ds,range(8)),config)
        protocol=station_protocol(SubsetProtocol(ds,8))
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])
        baseline=zero_volume_baseline(loader,config);initial=run_epoch(model,loader,config)
        torch.cuda.reset_peak_memory_stats(config['training']['device']);began=time.perf_counter();rows=[]
        for epoch in range(80):
            tr=run_epoch(model,loader,config,optimizer)
            row=dict(epoch=epoch,train=tr);rows.append(row)
            with (paths['logs']/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(f"selfsup gate {epoch+1}/80: holdL1={tr['measurement']['heldout']['relative_l1']:.6f} loss={tr['loss']['total']:.6f}",flush=True)
        final=run_epoch(model,loader,config)
        batch=move_dual_batch(dual_measurement_collate([ds[0]]),config['training']['device'])
        properties=set_properties(model,batch['A']);gradients=feedback_gradients(model,batch['A'],batch['H'])
        reduction=1-final['measurement']['heldout']['relative_l1']/initial['measurement']['heldout']['relative_l1']
        numerical=final['output'];collapsed=(numerical['max']<=1e-5 or numerical['mean']<=1e-6 or numerical['std']<=1e-6 or numerical['fraction_near_one']>=.01 or numerical['case_to_case_mean_absolute_difference']<=1e-6)
        passed=(reduction>=.10 and final['measurement']['heldout']['relative_l1']<baseline['relative_l1'] and
                not collapsed and properties['passed'] and gradients['passed'] and not guard.summary()['gt_file_opened'])
        summary=dict(passed=passed,epochs=80,case_count=8,station_protocol=protocol,station_epoch=0,
            initialization='fresh random; no checkpoints',initial=initial,final=final,zero_volume_baseline=baseline,
            heldout_l1_relative_reduction=reduction,output_collapsed=collapsed,set_properties=properties,feedback_gradients=gradients,
            gt_access=guard.summary(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(config['training']['device']),
            mean_epoch_seconds=(time.perf_counter()-began)/80,volumetric_metrics_computed=False)
        atomic_checkpoint(paths['checkpoints']/'gate.pt',dict(model=model.state_dict(),config=config,gate=summary))
        (paths['logs']/'effective_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        write_json(target,summary);write_json(ROOT/'gate_summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)


class SubsetProtocol:
    def __init__(self,dataset,count):self.dataset=dataset;self.records=dataset.records[:count];self.count=count
    def __len__(self):return self.count
    def __getitem__(self,i):return self.dataset[i]


def formal(config,resume=None):
    paths=directories(config)
    gate_summary=json.loads((ROOT/'gate_summary.json').read_text())
    if not gate_summary['passed']:raise RuntimeError('Selfsup gate failed; stop without tuning or formal training')
    gate_config=load_config(directories(config,'_gate')['logs']/'effective_config.yaml')
    if gate_config!=config:raise ValueError('Formal protocol differs from the gate')
    if not (ROOT/'training_source_manifest.json').exists():raise RuntimeError('Freeze numerical training sources before formal training')
    if (paths['checkpoints']/'last.pt').exists() and not resume:raise FileExistsError('Formal run already exists; only its exact last.pt may resume')
    if resume and Path(resume).resolve()!=(paths['checkpoints']/'last.pt').resolve():raise ValueError('Resume only this selfsup run last.pt')
    guard=GTReadGuard(config['dataset']['root'],paths['checkpoints'])
    with guard:
        seed_everything(config['training']['seed']);torch.set_num_threads(2)
        model=initialize_fresh_model(config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        generator=torch.Generator().manual_seed(config['training']['seed'])
        train,val=make_dataset(config),make_dataset(config,validation=True)
        assert (len(train),len(val))==(166,36)
        train_loader=make_loader(train,config,True,generator);val_loader=make_loader(val,config)
        protocol=station_protocol(val);station_path=paths['logs']/'validation_station_ids.json'
        if station_path.exists() and json.loads(station_path.read_text())!=protocol:raise ValueError('Frozen validation stations changed')
        write_json(station_path,protocol)
        start=0;best=float('inf');rows=[];env=environment(config)
        if resume:
            ckpt=torch.load(resume,map_location=config['training']['device'],weights_only=False)
            if ckpt['config']!=config or ckpt['training_signal']!='TDLAS measurements only':raise ValueError('Invalid selfsup resume provenance')
            model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler'])
            restore_rng(ckpt['rng'],generator);start=ckpt['epoch']+1;best=ckpt['best_val_metric']
            rows=[json.loads(l) for l in (paths['logs']/'metrics.jsonl').read_text().splitlines() if json.loads(l)['epoch']<start]
            (paths['logs']/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        env.update(initialization='fresh random; no supervised/gate checkpoint',volumetric_gt_used=False)
        write_json(paths['logs']/'environment.json',env)
        (paths['logs']/'effective_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        write_json(paths['logs']/'data_protocol.json',dict(train_cases=166,val_cases=36,train_pool='A',validation_pool='A',
            volumetric_gt_used=False,parameter_count=sum(p.numel() for p in model.parameters()),selection_metric='mean held-out measurement relative L1'))
        for epoch in range(start,300):
            train.set_epoch(epoch);val.set_epoch(0)
            torch.cuda.reset_peak_memory_stats(config['training']['device']);began=time.perf_counter();lr=optimizer.param_groups[0]['lr']
            tr=run_epoch(model,train_loader,config,optimizer);va=run_epoch(model,val_loader,config)
            scheduler.step();score=selection_metric(va);improved=score<best;best=min(best,score)
            row=dict(epoch=epoch,train=tr,val=va,learning_rate=lr,seconds=time.perf_counter()-began,
                     gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(config['training']['device']))
            rows.append(row)
            with (paths['logs']/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,
                best_val_metric=best,config=config,rng=rng_state(generator),git_commit=env['git_commit'],
                training_signal='TDLAS measurements only',volumetric_gt_used=False,selection_metric='mean held-out measurement relative L1',initialization='fresh random')
            atomic_checkpoint(paths['checkpoints']/'last.pt',payload)
            if improved:atomic_checkpoint(paths['checkpoints']/'best.pt',payload)
            write_json(paths['logs']/'gt_access_audit.json',guard.summary())
            print(f"selfsup {epoch+1}/300: train holdL1={tr['measurement']['heldout']['relative_l1']:.6f} val holdL1={score:.6f} best={best:.6f}",flush=True)
        best_row=min(rows,key=lambda r:selection_metric(r['val']))
        summary=dict(completed=True,completed_epochs=len(rows),best_epoch=best_row['epoch'],best_val_heldout_relative_l1=best,
            best_validation=best_row['val'],selection_metric='mean held-out measurement relative L1',
            mean_epoch_seconds=sum(r['seconds'] for r in rows)/len(rows),peak_gpu_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in rows),
            gt_access=guard.summary(),stopping_reason='max_epochs',volumetric_gt_used=False)
        write_json(paths['logs']/'training_summary.json',summary)
        if guard.summary()['gt_file_opened']:raise AssertionError('GT protocol violated')
    # Freeze selection/hash after the measurement-only process has finished.
    from geophy_set3d.data.build_utils import digest
    checkpoint=paths['checkpoints']/'best.pt'
    write_json(ROOT/'frozen_checkpoint.json',dict(checkpoint=str(checkpoint),sha256=digest(checkpoint),best_epoch=summary['best_epoch']+1,
        selection_metric=summary['selection_metric'],completed_epochs=300,volumetric_gt_used=False,
        gt_posthoc_evaluation_started=False,training_source_commit=json.loads((ROOT/'training_source_manifest.json').read_text())['commit']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/train_selfsup_oari_mrgr_v1.yaml')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--smoke',action='store_true');mode.add_argument('--gate',action='store_true')
    parser.add_argument('--resume');args=parser.parse_args()
    config=load_config(args.config);validate_selfsup_config(config)
    if args.smoke:smoke(config,args.config)
    elif args.gate:gate(config)
    else:formal(config,args.resume)


if __name__=='__main__':main()
