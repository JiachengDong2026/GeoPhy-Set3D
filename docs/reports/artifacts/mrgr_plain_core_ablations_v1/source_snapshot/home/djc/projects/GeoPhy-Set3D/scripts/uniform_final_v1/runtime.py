"""Isolated MRGR-UniformFusion freeze; no historical numerical code is edited."""
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT/'scripts/paper_final_v1'))
from common import *
HISTORY = OUT
BASE = Path('/scratch/djc/GeoPhy-Set3D/mrgr_uniform_final_v1')
SIDE = Path('/home/djc/projects/GeoPhy-Set3D-uniform-oari-v1')
SIDE_SOURCE = SIDE/'scripts/uniform_feedback_oari_v1/feedback_model.py'
GPU_LOCKS = BASE.parent/'gpu_locks'
NEW_SEEDS = (20261020, 20261021)
ABLATION_SEEDS = (20261011, 20261019)
ARMS = ('uniform', 'uniform_feedback_oari', 'uniform_no_feedback',
        'uniform_zero_residual', 'uniform_no_ray_relations',
        'uniform_no_consistency', 'uniform_single_scale')
TRAIN_FILES = ('runtime.py', 'uniform_models.py', 'worker.py')

def training_identity():
    import train
    result = train.source_identity()
    result.update({str(p): sha(p) for p in (PROJECT/'scripts/paper_final_v1').glob('*.py')})
    result.update({str(SCRIPT/n): sha(SCRIPT/n) for n in TRAIN_FILES})
    result[str(SIDE_SOURCE)] = sha(SIDE_SOURCE)
    return result

def args(name, *values):
    return [sys.executable, str(SCRIPT/name), *map(str, values)]

class TrainGuard(AccessGuard):
    def __init__(self):
        super().__init__(splits=['train', 'val'], output=BASE)

def load_side():
    import importlib.util
    spec = importlib.util.spec_from_file_location('uniform_feedback_research', SIDE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
