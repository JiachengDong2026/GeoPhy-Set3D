#!/usr/bin/env python3
"""Val-only best-checkpoint metrics, trivial baselines and measurement/geometry shortcuts."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from geophy_set3d.models import GeoLiftMeanRes3D
from geophy_set3d.data.projection_pool_dataset import artifact_path
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.engine import make_dataset
from geophy_set3d.training.losses import supervised_loss
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.utils import (load_config, directories, seed_everything,
    write_json, move_batch, permute_views, save_slices)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/train_supervised_b0.yaml')
    parser.add_argument('--checkpoint'); args=parser.parse_args()
    config=load_config(args.config); paths=directories(config); device=config['training']['device']
    seed_everything(config['training']['seed']); torch.set_num_threads(2)
    path=args.checkpoint or paths['checkpoints']/'best.pt'
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    if checkpoint['config'] != config:
        raise ValueError('Evaluation config must match checkpoint effective config')
    model=GeoLiftMeanRes3D().to(device); model.load_state_dict(checkpoint['model']); model.eval()
    train=make_dataset(config); val=make_dataset(config,validation=True)
    # Read ONLY training GT64 for the prior; never construct a val/test mean.
    mean=np.zeros((64,64,64),dtype=np.float64)
    for row in train.records:
        mean += np.load(artifact_path(train.root,row['gt_volume_64_path']),mmap_mode='r')
    mean /= len(train)
    np.save(paths['logs']/'train_mean_volume.npy',mean.astype(np.float32))
    mean=torch.from_numpy(mean.astype(np.float32))[None,None].to(device)
    names=('normal','zero_volume','train_mean_volume','wrong_geometry','wrong_measurement','zero_measurement')
    metrics={key:VolumeMetrics() for key in names}; losses={key:0. for key in names}
    donors=[]; permutation_max=0.; permutation_sum=0.; permutation_voxels=0
    with torch.no_grad():
        for i in range(len(val)):
            sample=val[i]; batch=move_batch(projection_pool_collate([sample]),device)
            target=batch['gt_volume_64']; donor=val.records[(i+1)%len(val)]
            # Donor measurements use the target's station IDs so Y changes without G changing.
            projections=np.load(artifact_path(val.root,donor['train_projection_path']),mmap_mode='r')
            donor_raw=torch.from_numpy(np.array(projections[sample['input_station_indices'].numpy()],copy=True))[:,None][None].to(device)
            donor_y=donor_raw/val.y_ref
            if val.normalization=='log1p_p99_5': donor_y=torch.log1p(donor_y)
            wrong_y=dict(batch,input_projection=donor_y,input_projection_raw=donor_raw)
            zero_y=dict(batch,input_projection=torch.zeros_like(batch['input_projection']),input_projection_raw=torch.zeros_like(batch['input_projection_raw']))
            # A cyclic shift is a derangement of all four stations, preserving station geometry as a unit.
            permutation=torch.arange(4,device=device).roll(1)
            wrong_g=permute_views(batch,permutation,geometry_only=True)
            predictions=dict(normal=model(batch),zero_volume=torch.zeros_like(target),train_mean_volume=mean,
                             wrong_geometry=model(wrong_g),wrong_measurement=model(wrong_y),zero_measurement=model(zero_y))
            for key,prediction in predictions.items():
                metrics[key].update(prediction,target,batch['case_id'])
                loss,_=supervised_loss(prediction,target,**config['loss']); losses[key]+=float(loss)
            reordered=model(permute_views(batch,torch.tensor([2,0,3,1],device=device)))
            difference=(reordered-predictions['normal']).abs()
            permutation_max=max(permutation_max,float(difference.max()))
            permutation_sum+=float(difference.double().sum()); permutation_voxels+=difference.numel()
            donors.append(dict(case_id=sample['case_id'],donor_case_id=donor['case_id'],station_ids=sample['input_station_indices'].tolist()))
            if i < 4:
                save_slices(paths['figures']/f'val_{i:02d}_best.png',target,predictions['normal'],sample['case_id'])
    results={key:dict(**metric.compute(),supervised_loss=losses[key]/len(val)) for key,metric in metrics.items()}
    summary=dict(checkpoint=str(path),best_epoch=checkpoint['epoch'],split='val',geometry_pool='A',views=4,epoch=0,
                 test_used=False,train_mean_case_count=len(train),results=results,
                 permutation=dict(max_prediction_difference=permutation_max,mean_prediction_difference=permutation_sum/permutation_voxels),
                 wrong_geometry_protocol='Cyclic derangement of all station geometry (including valid/path maps); Y stays in original order',
                 wrong_measurement_protocol='Next val case cyclic donor; use recipient station IDs, keep recipient geometry, masks, path, GT',
                 donors=donors)
    write_json(paths['logs']/'evaluation.json',summary)
    for key,metric in metrics.items(): write_json(paths['logs']/f'evaluation_{key}_per_case.json',metric.rows)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
