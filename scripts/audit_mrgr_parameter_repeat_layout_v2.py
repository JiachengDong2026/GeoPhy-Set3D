#!/usr/bin/env python3
"""Pre-registered secondary common-layout measurement replay; never replaces gates."""
import argparse,copy,gc,json,time
from pathlib import Path
import numpy as np
import torch
import mrgr_parameter_optimization_v2 as run
OUT=run.OUT/'common_layout_secondary'
SEEDS=(20261018,20261019)
ARMS=('control','lr4e4')
read=run.read
write=run.write

def register():
    assert not OUT.exists()
    protocols={s:run.verify(run.OUT/f'seed{s}') for s in SEEDS}
    p=protocols[SEEDS[0]]
    r=dict(registered_at=run.base.now(),source_sha256=run.base.digest(Path(__file__)),
        protocols={str(s):run.base.digest(run.OUT/f'seed{s}/protocol.json') for s in SEEDS},seeds=SEEDS,arms=ARMS,
        epoch=20,updates=1660,case_ids=p['validation_case_ids'],station_map=p['station_map'],
        source_validation_seed=SEEDS[0],no_gt=True,no_final_test=True,
        purpose='Secondary diagnostic only: common ordinary A/B/H panel across fresh initialization/training seeds. Original gates remain on each seed own pre-registered validation. Cannot rescue failed repeat or determine formal promotion.',
        outcome='Per-case mean A/B->H L1/L2,36 pairedcases; paired mean CI20000seed20261017 perseed, no combined72-case analysis',
        checkpoint_rule='All four fixed epoch20 checkpoint hashes sealed before first common-panel inference; source/config/protocol hashes must match original registrations')
    OUT.mkdir();write(OUT/'registration.json',r);(OUT/'registration.sha256').write_text(run.base.digest(OUT/'registration.json')+'\n');print('Secondary common-layout audit registered')

def verify():
    p=read(OUT/'registration.json');assert run.base.digest(OUT/'registration.json')==(OUT/'registration.sha256').read_text().strip()
    assert run.base.digest(Path(__file__))==p['source_sha256']
    for s in SEEDS:
        run.verify(run.OUT/f'seed{s}');assert run.base.digest(run.OUT/f'seed{s}/protocol.json')==p['protocols'][str(s)]
    return p

def replay():
    p=verify();assert not (OUT/'summary.json').exists();seals={}
    for s in SEEDS:
        for a in ARMS:
            f=run.OUT/f'seed{s}'/a;summary=read(f/'summary.json');assert summary['epochs']==20 and summary['updates']==1660
            path=f/'checkpoints/last.pt';assert run.base.digest(path)==summary['checkpoint_sha256'];seals[f'{s}/{a}']=dict(checkpoint=str(path),sha256=summary['checkpoint_sha256'])
    write(OUT/'endpoints_sealed.json',dict(sealed_at=run.base.now(),checkpoints=seals,registration_sha256=run.base.digest(OUT/'registration.json'),inference_started=False))
    first=read(run.OUT/f'seed{SEEDS[0]}/protocol.json');config=copy.deepcopy(first['config']);results={};all_rows=[];began=time.perf_counter()
    with run.base.GTReadGuard(config['dataset']['root'],run.OUT) as guard:
        for s in SEEDS:
            protocol=read(run.OUT/f'seed{s}/protocol.json');results[str(s)]={}
            for arm in ARMS:
                checkpoint=seals[f'{s}/{arm}'];assert run.base.digest(checkpoint['checkpoint'])==checkpoint['sha256']
                saved=torch.load(checkpoint['checkpoint'],map_location='cpu',weights_only=False);assert saved['epoch']==20
                c=run.config_for(protocol,arm);model=run.base.model_for(c,'multiscale',config['training']['device']);model.load_state_dict(saved['model']);model.eval()
                vd=run.base.selfsup.make_dataset(config,validation=True);vl=run.base.selfsup.make_loader(vd,config)
                cases=run.evaluation.ordinary_cases(model,vl,config);assert [x['case_id'] for x in cases]==p['case_ids']
                for row in cases:assert row['stations']==p['station_map'][row['case_id']]
                values=np.array([x['heldout_relative_l1'] for x in cases]);results[str(s)][arm]=dict(mean=float(values.mean()),p90=float(np.quantile(values,.9)),mean_l2=float(np.mean([x['heldout_relative_l2'] for x in cases])))
                all_rows.extend(dict(seed=s,arm=arm,**row) for row in cases)
                del saved,model,vl;gc.collect();torch.cuda.empty_cache()
            arrays={a:np.array([r['heldout_relative_l1'] for r in all_rows if r['seed']==s and r['arm']==a]) for a in ARMS}
            results[str(s)]['contrast']=run.evaluation.paired(arrays['control'],arrays['lr4e4'])
        audit=guard.summary();assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    assert len(all_rows)==144
    (OUT/'per_case.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
    verify();write(OUT/'summary.json',dict(completed=True,results=results,cases_per_model=36,total_records=144,no_gt=True,no_final_test=True,
        secondary_only=True,changes_original_gate=False,seconds=time.perf_counter()-began,gt_audit=audit,registration_sha256=run.base.digest(OUT/'registration.json'),finished_at=run.base.now()))
    print(json.dumps(results))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','run','verify'));a=parser.parse_args();torch.set_num_threads(2)
    if a.action=='register':register()
    elif a.action=='verify':verify();print('Verified')
    else:replay()
