#!/usr/bin/env python3
"""GT-free frozen-epoch20 local-branch intervention; never poll mutable checkpoints."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import mrgr_local_detail_v1 as candidate
import probe_mrgr_multiscale_v1 as base


def diagnose(run, checkpoint, device, max_seconds):
    run=run.resolve();checkpoint=checkpoint.resolve()
    assert checkpoint.is_relative_to(run), 'Snapshot must be under this registered run for GTReadGuard'
    assert checkpoint.name not in ('last.pt','best.pt'), 'Provide an explicit immutable epoch20 snapshot; mutable last/best paths denied'
    out=run/'diagnostics/local_epoch20_intervention_v1'
    assert not out.exists(), 'Refuse overwrite'
    protocol=json.loads((run/'protocol.json').read_text())
    protocol_hash=base.digest(run/'protocol.json')
    assert protocol_hash==(run/'protocol.sha256').read_text().strip()
    assert protocol['sources']==candidate.source_hashes(), 'Registered numerical sources changed'
    snapshot_hash=base.digest(checkpoint)
    out.mkdir(parents=True)
    config=copy.deepcopy(protocol['config']);config['training']['device']=device
    registration=dict(registered_at=base.now(),protocol_sha256=protocol_hash,checkpoint=str(checkpoint),
        checkpoint_sha256=snapshot_hash,expected_epoch=20,script_sha256=base.digest(Path(__file__)),
        device=device,case_ids=protocol['validation_case_ids'][:8],inputs=['A','B'],batch_size=1,
        intervention='same frozen weights; normal vs decoder.local forward output replaced by exact zeros in both C0/C1 calls',
        measurements=['C0/C1 relative field change, mean absolute change','C0/C1 heldout projection relativeL1/L2',
            'local output norm relative to its32channel feature input and output magnitude'],
        selection_changed=False,gt_allowed=False,final_test_allowed=False,max_seconds=max_seconds,
        limitation='Same-weight counterfactual verifies intervention, not independently trained ablation or3D accuracy evidence')
    base.write_json(out/'registration.json',registration)
    started=time.perf_counter();rows=[];branch_records=[]
    guard=base.GTReadGuard(config['dataset']['root'],run)
    with guard:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert base.digest(checkpoint)==snapshot_hash, 'Snapshot changed while loading'
        assert saved['epoch']==20 and saved['updates']==1660 and saved['arm']=='local'
        assert saved['protocol_sha256']==protocol_hash and not saved['volumetric_gt_used']
        model=candidate.model_for(config,'local',device);model.load_state_dict(saved['model'],strict=True);model.eval()
        del saved;gc.collect()
        ds=base.selfsup.make_dataset(config,validation=True)
        assert [r['case_id'] for r in ds.records[:8]]==registration['case_ids']
        torch.cuda.reset_peak_memory_stats(device)
        mode={'zero':False,'case':None,'input':None,'call':0}
        def hook(module,inputs,output):
            # Only scalar records are retained; no graph or volume escapes the call.
            if not mode['zero']:
                x=inputs[0]
                branch_records.append(dict(case_id=mode['case'],input=mode['input'],
                    decoder_call='C0' if mode['call']==0 else 'C1',
                    input_feature_norm=float(x.norm()),local_output_norm=float(output.norm()),
                    local_to_feature_norm_ratio=float(output.norm()/x.norm().clamp_min(1e-12)),
                    local_output_mean_abs=float(output.abs().mean()),
                    input_feature_mean_abs=float(x.abs().mean())))
            mode['call']+=1
            return torch.zeros_like(output) if mode['zero'] else None
        handle=model.decoder.local.register_forward_hook(hook)
        try:
            with torch.no_grad():
                for idx in range(8):
                    assert time.perf_counter()-started<max_seconds, 'Diagnostic wall-time budget exhausted'
                    sample=ds[idx]
                    dual=base.selfsup.move_dual_batch(base.dual_measurement_collate([sample]),device)
                    h=dual['H'];valid=h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]
                    for key in ('A','B'):
                        mode.update(zero=False,case=registration['case_ids'][idx],input=key,call=0)
                        normal=model(dual[key],return_stages=True)
                        assert mode['call']==2
                        mode.update(zero=True,call=0)
                        zero=model(dual[key],return_stages=True)
                        assert mode['call']==2
                        for stage in ('C0','C1'):
                            field=normal[stage];counter=zero[stage];change=field-counter
                            assert bool(torch.isfinite(change).all())
                            measures={}
                            for label,volume in [('normal',field),('zero_local',counter)]:
                                projected=base.project_input(volume,h)
                                measures[label]=base.relative_projection_metrics(projected[0],h['input_projection_raw'][0],valid[0])
                            rows.append(dict(case_id=registration['case_ids'][idx],input=key,stage=stage,
                                mean_abs_field_change=float(change.abs().mean()),
                                relative_l1_field_change=float(change.abs().sum()/counter.abs().sum().clamp_min(1e-12)),
                                relative_l2_field_change=float(change.norm()/counter.norm().clamp_min(1e-12)),
                                heldout=measures,input_station_ids=sample[key]['input_station_indices'].tolist(),
                                H_station_ids=sample['H']['input_station_indices'].tolist()))
                        del normal,zero
        finally:
            handle.remove()
        audit=guard.summary()
        peak=torch.cuda.max_memory_allocated(device)
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==32 and len(branch_records)==32
    assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    assert base.digest(checkpoint)==snapshot_hash and candidate.source_hashes()==protocol['sources']
    summaries={}
    for stage in ('C0','C1'):
        part=[r for r in rows if r['stage']==stage]
        branch=[r for r in branch_records if r['decoder_call']==stage]
        means={label:float(np.mean([r['heldout'][label]['relative_l1'] for r in part])) for label in ('normal','zero_local')}
        summaries[stage]=dict(mean_relative_l1_field_change=float(np.mean([r['relative_l1_field_change'] for r in part])),
            mean_relative_l2_field_change=float(np.mean([r['relative_l2_field_change'] for r in part])),
            nonzero_field_records=sum(r['mean_abs_field_change']>0 for r in part),
            heldout_relative_l1_means=means,zero_minus_normal_H=means['zero_local']-means['normal'],
            mean_local_to_feature_norm_ratio=float(np.mean([r['local_to_feature_norm_ratio'] for r in branch])))
    base.write_json(out/'per_case_records.json',rows)
    base.write_json(out/'branch_magnitude_records.json',branch_records)
    result=dict(completed=True,checkpoint_epoch=20,checkpoint_sha256=snapshot_hash,
        elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=peak,cases=8,inputs=16,
        summary=summaries,gt_access=audit,sources_preserved=True,
        interpretation=registration['limitation'],aggregation='mean over16 matched A/B reconstructions, equivalent to equal8case means')
    base.write_json(out/'summary.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=base.ROOT/'outputs/research/mrgr_local_detail_v1_seed20261011')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--max-seconds',type=float,default=60.)
    args=parser.parse_args();torch.set_num_threads(2)
    diagnose(args.run_dir,args.checkpoint,args.device,args.max_seconds)


if __name__=='__main__':main()
