"""Use the original35-case predictor, seal and metric implementation unchanged."""
from runtime import *
import argparse
import evaluation as engine
engine.OUT=BASE;engine.AccessGuard=TestGuard;engine.model_for=model_for;engine.SCOPE=None

def main(action,key=None):
    lock=verified(BASE/'final_lock.json');verify_hashes(lock['sources']);mm=lock['methods']
    if action=='predict':return engine.predict_neural(next(m for m in mm if m['key']==key),'test')
    seal=BASE/'evaluation/test/all_predictions_sealed.json'
    if not seal.exists():engine.seal('test',mm)
    else:assert verified(seal)['methods']==mm
    if action=='score':
        if not (BASE/'evaluation/test/metrics.jsonl').exists():engine.score('test')
        else:assert read(BASE/'evaluation/test/gt_access.json')['unique_GT64_files']==35

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['predict','seal','score']);p.add_argument('--key');a=p.parse_args();main(a.action,a.key)
