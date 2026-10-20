"""Resume all three matched arms to300 without changing their20-epoch prefix."""
from runtime import *
import argparse
import torch
import train as engine
from pilot_models import model_for

engine.OUT=BASE;engine.model_for=model_for;engine.source_identity=identity

def run(target):
    target=Path(target);p=verified(target/'protocol.json')
    verify_hashes(verified(BASE/'design.json')['sources'])
    assert verified(BASE/'gpu_checks.json')['passed'] and verified(BASE/'migration_verification.json')['passed']
    with DevGuard(splits=['train','val']) as guard:
        engine.train(target,EPOCHS)
        assert not guard.denied;write(target/'boundary_audit.json',guard.summary())
    saved=torch.load(target/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert saved['epoch']==300 and saved['updates']==24900 and len(saved['metrics'])==300
    original=torch.load(p['continuation']['checkpoint'],map_location='cpu',weights_only=False)
    assert sha(p['continuation']['checkpoint'])==p['continuation']['checkpoint_sha256']
    assert saved['metrics'][:20]==original['metrics']
    assert not saved['volumetric_gt_used']
    freeze(target/'continuation_complete.json',dict(at=now(),passed=True,prefix_metrics_exact=True,
        epochs=300,updates=24900,prefix_epochs=20,added_epochs=280,
        last_checkpoint_sha256=sha(target/'checkpoints/last.pt'),summary_sha256=sha(target/'summary.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',required=True);a=p.parse_args();run(a.target)
