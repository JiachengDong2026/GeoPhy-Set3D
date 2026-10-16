#!/usr/bin/env python3
"""Matched frozen historical C0/C1 inference, followed by independent GT analysis."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import analyze_frozen_ab_morphology_v1 as ab
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models import build_model
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.utils import seed_everything

OUT=ROOT/'outputs/research/historical_reconstruction_morphology_v1'
REFERENCE=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v1')


def predict(device):
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'predictions').mkdir()
    freeze=ab.read(REFERENCE/'frozen_checkpoint.json')
    assert ab.sha(Path(freeze['checkpoint']))==freeze['sha256']
    assert freeze['completed_epochs']==300 and not freeze['volumetric_gt_used']
    settings={}
    for cond in ab.CONDITIONS:
        directory=Path(ab.read(ab.RUNS['A']/f'evaluation/frozen_model_test/{cond}_prediction.json')['directory'])
        manifest=directory/'predictions_manifest.jsonl'
        settings[cond]=dict(reference_manifest=str(manifest),sha256=ab.sha(manifest),cases=ab.rows(manifest))
    protocol=dict(checkpoint=freeze,settings=settings,device=device,seed=20261011,
        source_sha256=ab.sha(Path(__file__)),reuse_ab_analysis_sha256=ab.sha(Path(ab.__file__)),
        scope='Historical frozen best model; same cases, ordered stations, XYZ64 as archived A/B; Val only; no training',
        gt_for_inference=False,current_multiscale_checkpoint_read=False,final_test_used=False)
    ab.write(OUT/'protocol.json',protocol)
    seed_everything(20261011);torch.set_num_threads(2)
    began=time.perf_counter();records=[]
    with GTReadGuard(ab.DATA,Path(freeze['checkpoint']).parent) as guard:
        ckpt=torch.load(freeze['checkpoint'],map_location='cpu',weights_only=False)
        config=ckpt['config']
        model=build_model(config).to(device)
        model.load_state_dict(ckpt['model'],strict=True);model.eval()
        assert sum(p.numel() for p in model.parameters())==332070
        for cond,setting in settings.items():
            ds=MeasurementOnlyPoolDataset(ab.DATA,split='val',geometry_pool='train',
                min_input_views=4,max_input_views=4,hold_views=0,base_seed=20261011,
                normalization=config['normalization']['method'],statistics_file=config['normalization']['statistics_file'])
            ids={r['case_id']:r['selected_station_ids'] for r in setting['cases']}
            class Replay:
                def sample(self,station_count,*,epoch,sample_index,case_id):
                    return np.asarray(ids[case_id],dtype=np.int64),np.empty(0,dtype=np.int64)
            ds.sampler=Replay()
            with torch.no_grad():
                for i in range(len(ds)):
                    sample=ds[i];batch=projection_pool_collate([sample])
                    batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                    start=time.perf_counter();stages=model(batch,return_stages=True)
                    fields={k:stages[k][0,0].cpu().numpy() for k in ('C0','C1')}
                    assert all(v.shape==(64,64,64) and np.isfinite(v).all() for v in fields.values())
                    path=OUT/'predictions'/f'{cond}_{i:03d}.npz'
                    np.savez_compressed(path,**fields)
                    records.append(dict(case_id=sample['case_id'],condition=cond,station_ids=sample['input_station_indices'].tolist(),
                        prediction=str(path.relative_to(OUT)),sha256=ab.sha(path),seconds=time.perf_counter()-start))
                    assert records[-1]['station_ids']==ids[sample['case_id']]
                    print(f"{cond} {i+1}/36 seconds={records[-1]['seconds']:.2f}",flush=True)
                    del stages,fields,batch
        (OUT/'prediction_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        audit=guard.summary()
        assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        ab.write(OUT/'predictions_complete.json',dict(fields=144,cases_layouts=72,
            all_predictions_saved_before_gt=True,protocol_sha256=ab.sha(OUT/'protocol.json'),
            manifest_sha256=ab.sha(OUT/'prediction_manifest.jsonl'),audit=audit,seconds=time.perf_counter()-began))


def score():
    from report_historical_reconstruction_v1 import main
    main()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('predict','score'))
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    predict(args.device) if args.action=='predict' else score()
