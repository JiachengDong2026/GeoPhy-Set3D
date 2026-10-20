"""Process-local adapter to the immutable full-budget training engine."""
from runtime import *
import argparse
import train as engine
from pruned_models import model_for

engine.original_source_identity=engine.source_identity
engine.OUT=BASE
engine.model_for=model_for
engine.source_identity=source_identity

def run(target,stop=300):
    p=verified(Path(target)/'protocol.json')
    if not p['subset_smoke']:
        assert verified(BASE/'checks/passed.json')['passed']
        assert verified(BASE/'reuse_audit.json')['passed']
        assert verified(BASE/'design.json')['epochs']==300
    with DevGuard(splits=['train','val']) as guard:
        engine.train(target,stop)
        assert not guard.denied
        write(Path(target)/'development_boundary_audit.json',guard.summary())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--stop-after',type=int,default=300);a=p.parse_args()
    run(a.target,a.stop_after)
