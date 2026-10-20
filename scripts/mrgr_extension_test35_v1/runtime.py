"""Isolated, explicitly authorized supplementary35-case evaluation."""
import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
SCRIPT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'scripts/paper_final_v1'))
from common import *
HISTORY=OUT
BASE=Path('/scratch/djc/GeoPhy-Set3D/mrgr_extension_test35_v1')
REPORT=PROJECT/'docs/reports/mrgr_extension_test35_v1_zh.md'
PRUNING=BASE.parent/'mrgr_gca_uniform_observability_pruning_v1'
FEEDBACK=BASE.parent/'mrgr_feedback_oari_long_v1'
ARMS=('pruned_observability','uniform_pruned_observability','baseline','feedback_oari','feedback_oari_zero_residual')
GPU_LOCKS=BASE.parent/'gpu_locks'

def identity():
    result=sources()
    for folder in (SCRIPT,PROJECT/'scripts/pruning_v1',PROJECT/'scripts/feedback_oari_long_v1'):
        result.update({str(p):sha(p) for p in folder.glob('*.py')})
    return result

def methods():return verified(BASE/'final_lock.json')['methods']
def args(name,*values):return [sys.executable,str(SCRIPT/name),*map(str,values)]

class TestGuard(AccessGuard):
    def __init__(self,**kwargs):
        assert set(kwargs.get('splits',()))<={'test'}
        assert set(kwargs.get('gt_splits',()))<={'test'}
        if kwargs.get('gt_splits'):
            seal=verified(BASE/'evaluation/test/all_predictions_sealed.json')
            assert seal['methods']==methods() and seal['all_predictions_saved_before_GT']
            verify_hashes(seal['manifests'])
        registered={m['checkpoint'] for m in methods()} if (BASE/'final_lock.json').exists() else set()
        assert set(kwargs.get('checkpoints',()))<=registered
        kwargs['checkpoints']=sorted(registered)
        kwargs['output']=BASE
        super().__init__(**kwargs)


def load_module(name,path):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

# Source files copied byte-for-byte from the already evaluated development implementations.
def model_for(config,arm,device):
    if arm in ('pruned_observability','uniform_pruned_observability'):
        return load_module('test35_pruned',PROJECT/'scripts/pruning_v1/pruned_models.py').model_for(config,arm,device)
    return load_module('test35_feedback',PROJECT/'scripts/feedback_oari_long_v1/pilot_models.py').model_for(config,arm,device)
