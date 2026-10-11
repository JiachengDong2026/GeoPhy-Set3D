#!/usr/bin/env python3
"""Fixed-view, four-train-case FP32 overfit gate before full B0 training."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
import yaml
from torch.utils.data import Subset
from geophy_set3d.models import GeoLiftMeanRes3D
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.engine import make_dataset, make_loader, run_epoch
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.utils import (load_config, seed_everything, directories, environment,
    write_json, atomic_checkpoint, save_slices, save_curves, move_batch)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/train_supervised_b0.yaml')
    parser.add_argument('--cases',type=int,default=4); parser.add_argument('--epochs',type=int,default=300)
    args=parser.parse_args()
    if not 4 <= args.cases <= 8: raise ValueError('Overfit requires 4-8 train cases')
    config=load_config(args.config); config=copy.deepcopy(config)
    config['train'].update(min_input_views=4,max_input_views=4)
    config['loader'].update(num_workers=0,persistent_workers=False)
    config['training'].update(max_epochs=args.epochs,gradient_accumulation=1)
    paths=directories(config,'_overfit'); seed_everything(config['training']['seed']); torch.set_num_threads(2)
    device=config['training']['device']; dataset=make_dataset(config)
    indices=torch.linspace(0,len(dataset)-1,args.cases).round().int().tolist()
    subset=Subset(dataset,indices); generator=torch.Generator().manual_seed(config['training']['seed'])
    loader=make_loader(subset,config,True,generator); eval_loader=make_loader(subset,config)
    model=GeoLiftMeanRes3D().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])
    initial=run_epoch(model,eval_loader,config)
    rows=[dict(epoch=-1,train=initial)]
    (paths['logs']/'effective_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    write_json(paths['logs']/'environment.json',environment(config))
    case_ids=[dataset.records[i]['case_id'] for i in indices]
    write_json(paths['logs']/'case_selection.json',dict(indices=indices,case_ids=case_ids,epoch=0,views=4))
    print('INITIAL '+json.dumps(initial),flush=True)
    with (paths['logs']/'metrics.jsonl').open('w') as log:
        log.write(json.dumps(rows[0])+'\n')
        for epoch in range(args.epochs):
            dataset.set_epoch(0)
            result=run_epoch(model,loader,config,optimizer)
            row=dict(epoch=epoch,train=result); rows.append(row)
            log.write(json.dumps(row)+'\n'); log.flush()
            if epoch%10==0 or epoch==args.epochs-1:
                print(f"epoch={epoch} loss={result['loss']['total']:.6g} relL1={result['metrics']['mean_per_case']['relative_l1']:.6g} relL2={result['metrics']['mean_per_case']['relative_l2']:.6g}",flush=True)
            if epoch%50==0:
                atomic_checkpoint(paths['checkpoints']/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,config=config))
    final=run_epoch(model,eval_loader,config)
    model.eval(); predictions=[]; targets=[]
    with torch.no_grad():
        for i in indices:
            batch=move_batch(projection_pool_collate([dataset[i]]),device); p=model(batch)
            predictions.append(p.cpu()); targets.append(batch['gt_volume_64'].cpu())
            save_slices(paths['figures']/f'case_{i}.png',batch['gt_volume_64'],p,dataset.records[i]['case_id'])
    p=torch.cat(predictions); g=torch.cat(targets)
    matched=VolumeMetrics(); mismatched=VolumeMetrics(); zero=VolumeMetrics(); mean=VolumeMetrics()
    matched.update(p,g); mismatched.update(p.roll(1,0),g); zero.update(torch.zeros_like(g),g); mean.update(g.mean(0,keepdim=True).expand_as(g),g)
    matched_metrics=matched.compute(); mismatched_metrics=mismatched.compute(); zero_metrics=zero.compute(); mean_metrics=mean.compute()
    pairwise=float((p[:,None]-p[None,:]).abs().mean(dim=(2,3,4,5)).max())
    nonzero=bool(p.max() > 1e-3)
    # Independent of initial scale: beat zero AND fixed subset-mean and case-swapped output.
    rel=matched_metrics['mean_per_case']['relative_l1']
    passed=bool(final['loss']['total'] < initial['loss']['total']*.25 and rel < .9
                and rel < mean_metrics['mean_per_case']['relative_l1']*.9
                and rel < mismatched_metrics['mean_per_case']['relative_l1']*.9
                and nonzero and pairwise > 1e-5)
    summary=dict(passed=passed,initial=initial,final=final,matched=matched_metrics,
                 swapped_prediction=mismatched_metrics,zero=zero_metrics,subset_mean=mean_metrics,
                 max_pairwise_prediction_mae=pairwise,prediction_max=float(p.max()),prediction_min=float(p.min()),
                 epochs=args.epochs,case_ids=case_ids,acceptance='loss <25% initial; relL1 <0.9; >10% better than subset-mean and swapped predictions; nonzero and sample-specific',
                 full_training_allowed=passed)
    write_json(paths['logs']/'overfit_summary.json',summary)
    atomic_checkpoint(paths['checkpoints']/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=args.epochs-1,config=config,summary=summary))
    save_curves(paths['figures']/'overfit_curves.png',rows)
    print('SUMMARY '+json.dumps(summary),flush=True)


if __name__=='__main__': main()
