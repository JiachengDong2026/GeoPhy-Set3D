#!/usr/bin/env python3
"""Read-only same-epoch learning diagnostics for the three registered formal runs."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import mrgr_parameter_optimization_v2 as screen
ROOT=screen.ROOT
SOURCES={
    'historical_multiscale':ROOT/'outputs/research/mrgr_multiscale_formal_v1_seed20261011',
    'fresh_control':ROOT/'outputs/research/mrgr_optimization_v2_control_seed20261011',
    'lr4e4':ROOT/'outputs/research/mrgr_optimization_v2_lr4e4_seed20261011',
}
OUT=screen.OUT/'formal_same_epoch_diagnostics'
NODES=(50,100,200,300)

def gather():
    logs={};protocols={}
    for model,path in SOURCES.items():
        lines=(path/'metrics.jsonl').read_text().splitlines(keepends=True);rows=[json.loads(s) for s in lines]
        assert [r['epoch'] for r in rows]==list(range(1,len(rows)+1))
        logs[model]=(lines,rows);protocols[model]=screen.read(path/'protocol.json')
    assert len({p['initial_model_sha256'] for p in protocols.values()})==1
    OUT.mkdir(exist_ok=True)
    for node in NODES:
        if any(len(rows)<node for _,rows in logs.values()):continue
        records={}
        for model,(lines,rows) in logs.items():
            prefix=rows[:node];best=min(prefix,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
            p=protocols[model];records[model]=dict(run=str(SOURCES[model]),epoch=node,
                heldout_relative_l1=prefix[-1]['val']['measurement']['heldout']['relative_l1'],
                best_so_far_relative_l1=best['val']['measurement']['heldout']['relative_l1'],best_epoch=best['epoch'],
                optimizer_initial_lr=p['config']['optimizer']['lr'],epoch_learning_rate=prefix[-1]['learning_rate'],
                protocol_sha256=screen.base.digest(SOURCES[model]/'protocol.json'),
                prefix_metrics_sha256=hashlib.sha256(''.join(lines[:node]).encode()).hexdigest(),
                cumulative_epoch_seconds=sum(r['seconds'] for r in prefix))
        ratios={}
        for reference in ('historical_multiscale','fresh_control'):
            ratios[reference]=dict(endpoint_ratio=records['lr4e4']['heldout_relative_l1']/records[reference]['heldout_relative_l1'],
                best_so_far_ratio=records['lr4e4']['best_so_far_relative_l1']/records[reference]['best_so_far_relative_l1'])
        path=OUT/f'epoch{node:03d}.json'
        if path.exists():
            old=screen.read(path);assert old['models']==records and old['lr_vs_reference']==ratios;continue
        result=dict(epoch=node,recorded_at=datetime.now(timezone.utc).isoformat(),models=records,lr_vs_reference=ratios,
            purpose='Same-epoch measurement learning dynamics only. No3D conclusion, new promotion decision, LR adjustment, or early stopping.',
            no_gt=True,no_final_test=True,script_sha256=screen.base.digest(Path(__file__)))
        screen.write(path,result);print(json.dumps(result))
if __name__=='__main__':gather()
