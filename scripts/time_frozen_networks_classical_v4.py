#!/usr/bin/env python3
"""Actual network-only forward timing; reuse verified historical metrics."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import torch,numpy as np,yaml
from run_classical_v4 import ReadGuard,verify,original_cfg,write
from diagnose_clustered_one_swap import dataset,load_frozen
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch,seed_everything
from geophy_set3d.training.rvtc_matched import state_hash


def main():
    seed_everything(20261011)
    cfg=yaml.safe_load((ROOT/'configs/classical_v4_v1.yaml').read_text());out=Path(cfg['output']);verify(cfg,out)
    if (out/'network_forward_times.json').exists():print('Network timing already complete; no repeat');return
    old=original_cfg();old['device']=cfg['device'];protocol=json.loads((Path(old['output'])/'protocol.json').read_text())
    records=json.loads((out/'station_protocol.json').read_text());history=json.loads((out/'network_historical_rows.json').read_text())
    lookup={(r['model'],r['case_id'],r['draw'],r['condition']):r for r in history};rows=[];torch.set_num_threads(2)
    for name in old['models']:
        with ReadGuard(cfg,out,'register'):
            model=load_frozen(old,name,protocol);before=state_hash(model.state_dict())
        with ReadGuard(cfg,out,'evaluate'):
            ds=dataset(old)
            with torch.inference_mode():
                for record in records:
                    source,p=ds._projections(record['case_index'])
                    for condition in cfg['validation_conditions']:
                        sample=ds._measurement_sample(source,p,np.asarray(record[condition]),np.empty(0,dtype=np.int64))
                        b=move_batch(projection_pool_collate([sample]),cfg['device'])
                        if not rows or (record==records[0] and condition=='Original'):
                            for _ in range(5):model(b)
                        torch.cuda.synchronize(cfg['device']);torch.cuda.reset_peak_memory_stats(cfg['device'])
                        start=time.perf_counter();prediction=model(b);torch.cuda.synchronize(cfg['device']);seconds=time.perf_counter()-start
                        historic=lookup[name,record['case_id'],record['draw'],condition]
                        for field,actual in [('output_mean',float(prediction.mean())),('output_max',float(prediction.max()))]:
                            assert np.isclose(actual,historic[field],atol=1e-8,rtol=1e-5),(name,record['case_id'],field)
                        rows.append(dict(model=name,case_id=record['case_id'],draw=record['draw'],condition=condition,forward_seconds=seconds,
                                         GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(cfg['device']),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(cfg['device']),
                                         input_station_ids=record[condition],checkpoint_sha256=old['models'][name]['sha256'],input_setup_and_projection_evaluation_included=False))
            assert state_hash(model.state_dict())==before and all(p.grad is None for p in model.parameters())
        del model;torch.cuda.empty_cache();print(name,'540 timed forwards',flush=True)
    write(out/'network_forward_times.json',rows)
    write(out/'network_timing_audit.json',dict(actual_forwards=1080,warmup_forwards_per_model=5,parameters_buffers_unchanged=True,GT_read=False,output_mean_max_reproduced=True,device=cfg['device']))
if __name__=='__main__':main()
