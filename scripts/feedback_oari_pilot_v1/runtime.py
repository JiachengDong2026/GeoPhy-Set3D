"""Independent short-budget development pilot, with immutable historical imports."""
import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
HISTORICAL=Path('/home/djc/projects/GeoPhy-Set3D')
sys.path.insert(0,str(HISTORICAL/'scripts/paper_final_v1'))
from common import *
HISTORY=OUT
BASE=Path('/scratch/djc/GeoPhy-Set3D/mrgr_feedback_oari_pilot_v1')
SCRIPT=PROJECT/'scripts/feedback_oari_pilot_v1'
REPORT=PROJECT/'docs/reports/mrgr_feedback_oari_pilot_v1_zh.md'
ARMS=('baseline','feedback_oari','feedback_oari_zero_residual')
SEED=20261011
EPOCHS=20
GPU_LOCKS=BASE.parent/'gpu_locks'

def identity():
    import train
    mapping=train.original_identity()
    mapping.update({str(p):sha(p) for p in SCRIPT.glob('*.py')})
    mapping.update({str(p):sha(p) for p in (HISTORICAL/'scripts/paper_final_v1').glob('*.py')})
    return mapping

class DevGuard(AccessGuard):
    def __init__(self,**kwargs):
        assert set(kwargs.get('splits',()))<={'train','val'}
        assert set(kwargs.get('gt_splits',()))<={'val'}
        kwargs['output']=BASE
        super().__init__(**kwargs)

def args(name,*values):return [sys.executable,str(SCRIPT/name),*map(str,values)]
