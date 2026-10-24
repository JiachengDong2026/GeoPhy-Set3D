"""Exact-path dispatch avoids historical generic train/evaluate module collisions."""
from diag_context import *
import argparse
import importlib.util
import engineering_memory_v2 as memory

def load_training():
    # Historical context imports its own train module. Never use that cached
    # name to dispatch this separate, already sealed diagnostic worker.
    spec=importlib.util.spec_from_file_location('identifiability_eval_exact_v1',HERE/'evaluate.py')
    evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
    sys.modules['evaluate']=evaluation
    spec=importlib.util.spec_from_file_location('identifiability_train_exact_v1',HERE/'train.py')
    training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
    assert Path(training.__file__).resolve()==HERE/'train.py'
    return training

def main():
    amendment=verified('engineering_dispatch_v3.json');c.verify_hashes(amendment['sources'])
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',choices=['A','B'],required=True)
    ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--version',choices=['v2','v3'],required=True)
    ap.add_argument('--n',choices=[1,2,4],type=int,required=True);a=ap.parse_args()
    memory.install();worker=load_training();worker.run(a.experiment,a.arm,a.version,a.n)

if __name__=='__main__':main()
