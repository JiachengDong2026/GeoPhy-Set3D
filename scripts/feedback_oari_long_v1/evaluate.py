"""All17 development layouts using the immutable measurement/GT seal engine."""
from runtime import *
import argparse
import evaluation as engine
from pilot_models import model_for
engine.OUT=BASE;engine.AccessGuard=DevGuard;engine.model_for=model_for;engine.SCOPE=None

def methods():
    if (BASE/'methods.json').exists():return verified(BASE/'methods.json')
    result=[]
    for name in verified(BASE/'design.json')['runs']:
        path=Path(name);p=verified(path/'protocol.json');s=read(path/'summary.json');log=rows(path/'metrics.jsonl')
        assert s['completed'] and s['epochs']==EPOCHS and s['updates']==24900
        assert len(log)==EPOCHS and [r['epoch'] for r in log]==list(range(1,EPOCHS+1))
        best=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        assert best['epoch']==s['best_epoch'] and best['val']['measurement']['heldout']['relative_l1']==s['best_val_H']
        assert not read(path/'boundary_audit.json')['denied']
        ck=path/'checkpoints/best.pt';assert sha(ck)==s['best_checkpoint_sha256'];verify_hashes(p['sources'])
        result.append(dict(key=path.name,kind='neural',run=str(path),checkpoint=str(ck),checkpoint_sha256=sha(ck),best_epoch=s['best_epoch'],arm=p['arm'],config=p['config'],parameters=p['parameters'],seed=SEED))
    freeze(BASE/'methods.json',result);return result

def main(action):
    verify_hashes(verified(BASE/'design.json')['sources']);mm=methods()
    if action=='predict':
        for m in mm:engine.predict_neural(m,'val')
    else:
        seal=BASE/'evaluation/val/all_predictions_sealed.json'
        if not seal.exists():engine.seal('val',mm)
        else:assert verified(seal)['methods']==mm
        if not (BASE/'evaluation/val/metrics.jsonl').exists():engine.score('val')
        else:assert read(BASE/'evaluation/val/gt_access.json')['unique_GT64_files']==36

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['predict','score']);a=p.parse_args();main(a.action)
