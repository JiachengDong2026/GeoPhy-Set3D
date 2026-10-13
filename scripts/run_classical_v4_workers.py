#!/usr/bin/env python3
"""Disjoint method workers on identical 4090s; no changes to frozen numerical code."""
import sys,json,argparse,datetime,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import torch,yaml
from run_classical_v4 import (verify,verify_selection,Runtime,ReadGuard,write,digest,original_cfg,inventory,snapshot)


def registration(cfg,out):
    verify(cfg,out);verify_selection(out)
    sha=digest(Path(__file__));path=out/'execution_supplement.json'
    if path.exists():
        result=json.loads(path.read_text());assert result['worker_source_sha256']==sha;return result
    assert not (out/'classical_rows.json').exists()
    supplement=dict(registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker_source_sha256=sha,
                    protocol_sha256=digest(out/'protocol.json'),selected_configs_sha256=digest(out/'selected_configs.json'),
                    numerical_source_sha256=json.loads((out/'selected_configs.json').read_text())['source_sha256'],
                    assignments={'SART':'cuda:1','Tikhonov':'cuda:1','TV':'cuda:0'},hardware='NVIDIA RTX4090 for both, shared server',
                    change='Execution placement only; same selected parameters, precision, inputs and stopping rules. Independent directories; no Val retry/tuning.')
    write(path,supplement);return supplement


def worker(cfg,out,method):
    verify(cfg,out);selected=verify_selection(out)['selected'];supplement=registration(cfg,out)
    cfg=dict(cfg,device=supplement['assignments'][method]);target=out/'workers'/method;target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists():
        complete=json.loads((target/'complete.json').read_text());assert digest(target/'rows.json')==complete['rows_sha256'];print(method,'already complete');return
    records=json.loads((out/'station_protocol.json').read_text());rows=[]
    with ReadGuard(cfg,target,'evaluate'):
        runtime=Runtime(cfg,target,'evaluate')
        for candidate in sorted({v['candidate'] for v in selected[method].values()}):
            identities=[identity for identity,s in selected[method].items() if s['candidate']==candidate];config=selected[method][identities[0]]['config']
            for record in records:
                for condition in cfg['validation_conditions']:
                    entry=dict(case_id=record['case_id'],case_index=record['case_index'],draw=record['draw'],condition=condition,input_station_ids=record[condition],H_station_ids=record['H'])
                    key=f'{method}_c{candidate}_val{record["case_index"]}_d{record["draw"]}_{condition}'
                    row=runtime.reconstruct(entry,method,config,key);rows.append(dict(row,candidate=candidate,selection_identities=identities,compute_device=cfg['device']))
                if record['case_index']==35:print(method,candidate,'draw',record['draw'],'complete',len(rows),flush=True)
        write(target/'rows.json',rows);write(target/'complete.json',dict(reconstructions=len(rows),rows_sha256=digest(target/'rows.json')))


def assemble(cfg,out):
    verify(cfg,out);verify_selection(out);registration(cfg,out);rows=[]
    for method in ['SART','Tikhonov','TV']:
        target=out/'workers'/method;complete=json.loads((target/'complete.json').read_text());assert digest(target/'rows.json')==complete['rows_sha256'];rows+=json.loads((target/'rows.json').read_text())
    keys=set()
    for r in rows:
        key=(r['method'],r['candidate'],r['case_id'],r['draw'],r['condition']);assert key not in keys;keys.add(key)
    selected=verify_selection(out)['selected']
    expected=540*sum(len({v['candidate'] for v in selected[m].values()}) for m in selected);assert len(rows)==expected
    write(out/'classical_rows.json',rows);write(out/'evaluation_complete.json',dict(reconstructions=len(rows),rows_sha256=digest(out/'classical_rows.json'),selected_configs_sha256=digest(out/'selected_configs.json'),workers=True))
    before=json.loads((out/'inventory_before.json').read_text());after=dict(base=inventory(original_cfg()),original_clustered=snapshot(Path(original_cfg()['output'])));assert before==after,'Historical/data metadata changed'
    hashes=json.loads((out/'protected_tracked_hashes.json').read_text());assert all(digest(ROOT/f)==sha for f,sha in hashes.items())
    write(out/'integrity.json',dict(dataset_and_historical_inventory_unchanged=True,protected_tracked_files_unchanged=True,projector_unchanged=True,training_guard_unchanged=True,GT128_PoolB_test_used=False,new_network_training=False,unfinished_old_experiments_resumed=False))
    print('Assembled actual',len(rows),'reconstructions; integrity passed')


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','worker','assemble']);a.add_argument('--method',choices=['SART','Tikhonov','TV']);args=a.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/classical_v4_v1.yaml').read_text());out=Path(cfg['output']);torch.set_num_threads(2)
    if args.stage=='worker':worker(cfg,out,args.method)
    elif args.stage=='register':registration(cfg,out)
    else:assemble(cfg,out)
if __name__=='__main__':main()
