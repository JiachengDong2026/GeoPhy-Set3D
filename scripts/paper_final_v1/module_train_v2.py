"""Reuse the verified training/recovery engine without editing frozen train.py."""
from common import *
import argparse
import train as engine
from module_models_v2 import model_for,ARMS

BASE=OUT/'module_ablations_v2'
original_identity=engine.source_identity


def identity():
    mapping=original_identity()
    for name in ('module_models_v2.py','module_train_v2.py'):
        p=ROOT/'scripts/paper_final_v1'/name;mapping[str(p)]=sha(p)
    return mapping


engine.model_for=model_for
engine.source_identity=identity


def register(arm,seed,smoke=None):
    assert arm in ARMS
    target=engine.register(arm,seed,smoke=smoke)
    verify_hashes(verified(target/'protocol.json')['sources'])
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','register']);p.add_argument('--target');p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int);p.add_argument('--stop-after',type=int,default=300);a=p.parse_args()
    if a.action=='register':print(register(a.arm,a.seed))
    else:
        protocol=verified(Path(a.target)/'protocol.json')
        if not protocol['subset_smoke']:
            assert verified(BASE/'checks.json')['passed']
            reg=verified(BASE/'registration.json');assert str(Path(a.target)) in reg['runs']
        engine.train(a.target,a.stop_after)
