#!/usr/bin/env python3
"""Two complete matched epochs per MRGR arm, for runtime rather than accuracy.

Run only on an available GPU. Every arm is fresh; no trained weights are saved
or reused. The complete166/36 objective, loader, physics and updates are timed.
"""
import copy
import gc
import json
from pathlib import Path
import subprocess
import time

import probe_mrgr_multiscale_v1 as probe
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/mrgr_multiscale_runtime_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    protocol=probe.verify()
    config=copy.deepcopy(protocol['config'])
    config['training']['device']='cuda:0'
    torch.set_num_threads(2)
    probe.write_json(OUT/'registration.json',dict(
        stage='performance_only', reference_protocol_sha256=probe.digest(probe.OUT/'protocol.json'),
        script_sha256=probe.digest(Path(__file__)), full_epochs_per_arm=2,
        scientific_budget='166 train cases / 36 val, 83 updates per epoch, original FP32/Q96/full objective',
        weights='fresh same shared initialization in both arms, never saved or exported',
        timing='cold then warm full train+val run_epoch; setup and checkpoint writes excluded',
        extrapolation='300 times warm epoch seconds, no guarantee of future exclusive GPU availability',
        hardware_before=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        gt_allowed=False, final_test_allowed=False))
    arms={}
    with probe.GTReadGuard(config['dataset']['root'],OUT/'unused_checkpoints') as guard:
        for arm in probe.ARMS:
            model=probe.model_for(config,arm,'cuda:0')
            shared={k:v for k,v in model.state_dict().items() if not k.startswith('decoder.context.')}
            assert probe.state_hash(shared)==protocol['base_initial_sha256']
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
            generator=torch.Generator().manual_seed(20261011)
            train=probe.selfsup.make_dataset(config)
            val=probe.selfsup.make_dataset(config,validation=True)
            loader=probe.selfsup.make_loader(train,config,True,generator)
            val_loader=probe.selfsup.make_loader(val,config)
            records=[]
            for epoch in range(2):
                train.set_epoch(epoch);val.set_epoch(0)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize();began=time.perf_counter()
                tr=probe.selfsup.run_epoch(model,loader,config,optimizer)
                torch.cuda.synchronize();train_seconds=time.perf_counter()-began
                began=time.perf_counter()
                va=probe.selfsup.run_epoch(model,val_loader,config)
                torch.cuda.synchronize();val_seconds=time.perf_counter()-began
                scheduler.step()
                assert tr['sample_count']==166 and va['sample_count']==36
                row=dict(epoch=epoch+1,train=tr,val=va,train_seconds=train_seconds,val_seconds=val_seconds,
                    epoch_seconds=train_seconds+val_seconds,seconds_per_update=train_seconds/83,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    loader_rng_sha256=probe.hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
                records.append(row)
                probe.append(OUT/f'{arm}.jsonl',row)
                print(json.dumps(dict(arm=arm,epoch=epoch+1,seconds=row['epoch_seconds'],seconds_per_update=row['seconds_per_update'])),flush=True)
            arms[arm]=dict(records=records,warm_epoch_seconds=records[-1]['epoch_seconds'],
                estimated_300_hours=records[-1]['epoch_seconds']*300/3600,
                parameters=sum(p.numel() for p in model.parameters()))
            del model,optimizer,loader,val_loader;gc.collect();torch.cuda.empty_cache()
        for left,right in zip(arms['reference']['records'],arms['multiscale']['records']):
            assert left['loader_rng_sha256']==right['loader_rng_sha256']
            assert left['train']['case_station_digest']==right['train']['case_station_digest']
            assert left['val']['case_station_digest']==right['val']['case_station_digest']
        result=dict(arms=arms,relative_warm_epoch_cost=arms['multiscale']['warm_epoch_seconds']/arms['reference']['warm_epoch_seconds']-1,
            gt_audit=guard.summary(), checkpoints_saved=False, formal_training=False,
            hardware_after=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
            source_preserved=protocol['sources']==probe.source_hashes())
        assert result['source_preserved']
        assert all(result['gt_audit'][k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        probe.write_json(OUT/'summary.json',result)


if __name__=='__main__':
    main()
