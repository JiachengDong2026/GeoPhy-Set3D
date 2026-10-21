"""Reuse the canonical formal engine, optimizer, sampler, selection and recovery."""
from runtime import *
import argparse
import train as engine

engine.model_for=model_for


def run(target,stop):
    target=Path(target);p=verified(target/'protocol.json')
    if not p['subset_smoke']:
        assert verified(BASE/'engineering_checks.json')['passed']
        assert str(target) in verified(BASE/'protocol.json')['runs']
    with TrainGuard() as guard:
        engine.train(target,stop)
        assert not guard.denied
        write(target/'boundary_audit.json',guard.summary())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',required=True)
    parser.add_argument('--stop-after',type=int,default=300);a=parser.parse_args()
    run(a.target,a.stop_after)
