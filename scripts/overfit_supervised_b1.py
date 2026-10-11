#!/usr/bin/env python3
"""Fixed-view, four-train-case FP32 overfit gate before full B1 training."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
import yaml
from torch.utils.data import Subset
from geophy_set3d.models import build_model
from geophy_set3d.training.losses import supervised_loss
from geophy_set3d.models.attention_fusion import normalized_attention_entropy
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.engine import make_dataset, make_loader, run_epoch
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.utils import (load_config, seed_everything, directories, environment,
    write_json, atomic_checkpoint, save_slices, save_curves, move_batch)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/train_supervised_b1_300.yaml')
    parser.add_argument('--cases',type=int,default=4); parser.add_argument('--epochs',type=int,default=120)
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
    model=build_model(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])
    probe=move_batch(projection_pool_collate([dataset[indices[0]]]),device)
    pred,diag=model(probe,return_attention=True)
    supervised_loss(pred,probe['gt_volume_64'],**config['loss'])[0].backward()
    gradient_finite=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.fusion.parameters())
    gradient_nonzero=any(p.grad.abs().sum().item()>0 for p in model.fusion.parameters())
    optimizer.zero_grad(set_to_none=True)
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
    with torch.no_grad():
        _,diag=model(probe,return_attention=True)
    w,m=diag['attention_weights'],diag['visibility']
    attention_finite=bool(torch.isfinite(w).all())
    invalid_zero=bool((w[~m]==0).all())
    weight_sum_error=float((w.sum(1)-m.any(1).float()).abs().max())
    entropy=float(normalized_attention_entropy(w,m))
    # Quick functional gate, without searching attention hyperparameters.
    passed=bool(final['loss']['total'] < initial['loss']['total']*.25 and
                nonzero and pairwise>1e-5 and gradient_finite and gradient_nonzero and
                attention_finite and invalid_zero and weight_sum_error<1e-6)
    summary=dict(passed=passed,gradient_finite=gradient_finite,gradient_nonzero=gradient_nonzero,
                 attention_finite=attention_finite,invalid_weight_zero=invalid_zero,
                 weight_sum_max_error=weight_sum_error,normalized_entropy=entropy,initial=initial,final=final,matched=matched_metrics,
                 swapped_prediction=mismatched_metrics,zero=zero_metrics,subset_mean=mean_metrics,
                 max_pairwise_prediction_mae=pairwise,prediction_max=float(p.max()),prediction_min=float(p.min()),
                 epochs=args.epochs,case_ids=case_ids,acceptance='loss <25% initial; finite nonzero fusion gradients; finite masked attention; nonconstant predictions',
                 full_training_allowed=passed)
    write_json(paths['logs']/'overfit_summary.json',summary)
    atomic_checkpoint(paths['checkpoints']/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=args.epochs-1,config=config,summary=summary))
    save_curves(paths['figures']/'overfit_curves.png',rows)
    print('SUMMARY '+json.dumps(summary),flush=True)


if __name__=='__main__': main()
