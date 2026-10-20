"""Validate completed budgets and seal new checkpoint identities."""
from runtime import *

def main():
    if (BASE/'methods.json').exists():return verified(BASE/'methods.json')
    result=[]
    for name in verified(BASE/'design.json')['runs']:
        run=Path(name);p=verified(run/'protocol.json');s=read(run/'summary.json');log=rows(run/'metrics.jsonl')
        verify_hashes(p['sources']);assert s['completed'] and s['epochs']==300 and s['updates']==24900 and not p['subset_smoke']
        assert len(log)==300 and [r['epoch'] for r in log]==list(range(1,301))
        chosen=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        assert chosen['epoch']==s['best_epoch'] and chosen['val']['measurement']['heldout']['relative_l1']==s['best_val_H']
        assert all(s['gt_audit'][k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        assert not read(run/'development_boundary_audit.json')['denied']
        ck=run/'checkpoints/best.pt';assert sha(ck)==s['best_checkpoint_sha256']
        assert read(run/'environment.json')['initial_model_sha256']==verified(run/'initialization.json')['initial_sha256']
        result.append(dict(key=run.name,kind='neural',run=str(run),checkpoint=str(ck),checkpoint_sha256=sha(ck),best_epoch=s['best_epoch'],arm=p['arm'],config=p['config'],parameters=p['parameters'],seed=p['seed']))
    freeze(BASE/'methods.json',result);return result

if __name__=='__main__':main()
