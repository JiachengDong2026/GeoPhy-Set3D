#!/usr/bin/env python3
"""Actual FP32 forward/backward B1/B2,V6 plus permutation/padding on frozen train data."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from geophy_set3d.models import GeoLiftMeanRes3D
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.engine import make_dataset
from geophy_set3d.training.losses import supervised_loss
from geophy_set3d.training.utils import load_config, seed_everything, directories, write_json, move_batch, permute_views


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train_supervised_b0.yaml'); args=parser.parse_args()
    config=load_config(args.config); paths=directories(config); seed_everything(config['training']['seed'])
    torch.set_num_threads(2); device=config['training']['device']
    ds=make_dataset(config, fixed_views=6); results=[]
    for b in (1, 2):
        model=GeoLiftMeanRes3D().to(device)
        batch=move_batch(projection_pool_collate([ds[i] for i in range(b)]), device)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        try:
            torch.cuda.synchronize(); t=time.perf_counter(); pred=model(batch); torch.cuda.synchronize(); forward=time.perf_counter()-t
            loss,_=supervised_loss(pred,batch['gt_volume_64'],**config['loss'])
            t=time.perf_counter(); loss.backward(); torch.cuda.synchronize(); backward=time.perf_counter()-t
            grads={name:dict(finite=bool(torch.isfinite(p.grad).all()), abs_sum=float(p.grad.abs().sum())) for name,p in model.named_parameters()}
            row=dict(batch_size=b,views=6,success=True,forward_seconds=forward,backward_seconds=backward,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                     loss=float(loss.detach()), output_min=float(pred.detach().min()),output_max=float(pred.detach().max()),gradients=grads)
            if not all(r['finite'] for r in grads.values()):
                raise FloatingPointError('Nonfinite gradients')
            model.eval(); model.zero_grad(set_to_none=True)
            with torch.no_grad():
                base=model(batch); other=model(permute_views(batch,torch.arange(5,-1,-1,device=device)))
                row['permutation_max_difference']=float((base-other).abs().max())
                row['permutation_mean_difference']=float((base-other).abs().mean())
                masked=dict(batch); masked['view_mask']=batch['view_mask'].clone(); masked['view_mask'][:,-1]=False
                reference=model(masked); changed=dict(masked); changed['input_projection']=batch['input_projection'].clone(); changed['input_projection'][:,-1]=float('nan')
                altered=model(changed); row['padding_max_difference']=float((reference-altered).abs().max())
                print('INVARIANCE '+str({k: v for k,v in row.items() if 'difference' in k}), flush=True)
                if row['permutation_max_difference'] > 1e-5 or row['padding_max_difference'] > 1e-5:
                    raise AssertionError('Permutation/padding invariant failed')
        except torch.cuda.OutOfMemoryError:
            row=dict(batch_size=b,views=6,success=False,reason='CUDA OOM')
        results.append(row); print(row,flush=True)
        del model,batch
        if 'pred' in locals(): del pred,loss
        torch.cuda.empty_cache()
    write_json(paths['logs']/'gpu_smoke.json',dict(parameter_count=sum(p.numel() for p in GeoLiftMeanRes3D().parameters()),runs=results,precision='fp32'))


if __name__=='__main__': main()
