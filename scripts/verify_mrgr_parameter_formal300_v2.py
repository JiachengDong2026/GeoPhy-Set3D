#!/usr/bin/env python3
"""Independent read-only CPU audit after the LR formal run completes."""
import json,math
from pathlib import Path
import torch
import train_mrgr_optimization_formal_v2 as formal
ROOT=formal.ROOT
OUT=ROOT/'outputs/research/mrgr_optimization_v2_lr4e4_seed20261011'
CONTROL=ROOT/'outputs/research/mrgr_optimization_v2_control_seed20261011'
HISTORY=ROOT/'outputs/research/mrgr_multiscale_formal_v1_seed20261011'
read=formal.read

def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def close(a,b):assert math.isclose(a,b,rel_tol=3e-6,abs_tol=3e-7),(a,b)
def audit():
    p=formal.verify(OUT);formal.old.verify_data(p);s=read(OUT/'summary.json');assert s['completed'] and s['epochs']==300 and s['updates']==24900
    metrics=rows(OUT/'metrics.jsonl');assert len(metrics)==300
    references={name:rows(path/'metrics.jsonl') for name,path in (('fresh_control',CONTROL),('historical_multiscale',HISTORY))}
    last=torch.load(OUT/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    best=torch.load(OUT/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    assert last['epoch']==300 and last['updates']==24900 and last['config']==p['config'] and last['metrics']==metrics
    assert not last['volumetric_gt_used'] and not best['volumetric_gt_used']
    assert last['protocol_sha256']==best['protocol_sha256']==formal.base.digest(OUT/'protocol.json')
    selected=min(metrics,key=lambda r:r['val']['measurement']['heldout']['relative_l1']);score=selected['val']['measurement']['heldout']['relative_l1']
    assert s['best_epoch']==last['best_epoch']==best['epoch']==selected['epoch'];close(s['best_val_H'],score);close(last['best_score'],score);close(best['selection_metric'],score)
    assert sum(x.numel() for x in best['model'].values())==p['parameters']==344678
    assert all(torch.equal(v,last['best_model'][k]) for k,v in best['model'].items())
    assert all(torch.isfinite(v).all() for state in (last['model'],best['model']) for v in state.values())
    assert last['scheduler']['last_epoch']==300;close(last['optimizer']['param_groups'][0]['lr'],0)
    for i,row in enumerate(metrics):
        assert row['epoch']==i+1;close(row['learning_rate'],.0004*.5*(1+math.cos(math.pi*i/300)))
        assert row['seconds']>0 and row['peak_allocated_bytes']>0
        for split in ('train','val'):
            q=row[split];assert q['sample_count']==(166 if split=='train' else 36) and not q['gt_used']
            loss=q['loss'];standard=loss['hold']+.5*loss['cross']+.1*loss['observed']+.05*loss['consistency']
            close(loss['total'],standard);close(q['standard_total'],standard)
            assert q['output']['std']>0 and q['output']['min']>=0 and q['output']['max']<=1
        for name,reference in references.items():
            if i>=len(reference):continue
            other=reference[i];assert other['epoch']==i+1
            assert row['loader_rng_sha256']==other['loader_rng_sha256'],(name,i,'shuffle')
            close(row['learning_rate']/other['learning_rate'],2.)
            for split in ('train','val'):assert row[split]['case_station_digest']==other[split]['case_station_digest'],(name,i,split,'stations')
    assert s['elapsed_seconds']>=sum(r['seconds'] for r in metrics)
    for name in ('best','last'):
        assert formal.base.digest(OUT/f'checkpoints/{name}.pt')==s[f'{name}_checkpoint_sha256']
    for path in sorted((OUT/'checkpoints').glob('epoch_*.pt')):
        state=torch.load(path,map_location='cpu',weights_only=False);assert state['epoch'] in range(25,301,25)
        assert state['protocol_sha256']==last['protocol_sha256'] and not state['volumetric_gt_used']
        assert all(torch.isfinite(v).all() for v in state['model'].values())
    assert len(list((OUT/'checkpoints').glob('epoch_*.pt')))==12
    for value in (s['gt_audit'],read(OUT/'gt_access_audit.json')):
        assert all(value[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    result=dict(passed=True,checked_at=formal.base.now(),epochs=300,updates=24900,parameters=344678,
        best_epoch=selected['epoch'],best_val_H=score,final_val_H=metrics[-1]['val']['measurement']['heldout']['relative_l1'],
        elapsed_seconds=s['elapsed_seconds'],warm_epoch_seconds=sum(r['seconds'] for r in metrics[2:])/298,
        protocol_sha256=formal.base.digest(OUT/'protocol.json'),verifier_sha256=formal.base.digest(Path(__file__)),
        checked=['source/data/evidence/protocol identities','300epochs and24900updates','analytic cosine LR andunchanged objective algebra','all shared epochs shuffle/stations matched vs fresh and historical controls','checkpoint finite tensors/best model identity/strictHselection','last optimizer scheduler RNG commit and12epoch snapshots','zero GT/foreigncheckpoint access'],
        gt_used=False,final_test_used=False,no_3D_evaluation=True)
    formal.old.write(OUT/'independent_training_verification.json',result);print(json.dumps(result))
if __name__=='__main__':audit()
