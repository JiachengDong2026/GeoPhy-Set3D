"""Isolated development-only experiment; immutable historical engine imports."""
import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
HISTORICAL=Path('/home/djc/projects/GeoPhy-Set3D')
sys.path.insert(0,str(HISTORICAL/'scripts/paper_final_v1'))
from common import *
HISTORY=OUT
BASE=Path('/scratch/djc/GeoPhy-Set3D/mrgr_gca_uniform_observability_pruning_v1')
SCRIPT=PROJECT/'scripts/pruning_v1'
ARMS=('pruned_observability','uniform_pruned_observability')
SEEDS=(20261011,20261019)
REPORT=PROJECT/'docs/reports/mrgr_gca_uniform_observability_pruning_v1_zh.md'


def source_identity():
    import train
    mapping=train.original_source_identity() if hasattr(train,'original_source_identity') else train.source_identity()
    mapping.update({str(p):sha(p) for p in (HISTORICAL/'scripts/paper_final_v1').glob('*.py')})
    mapping.update({str(p):sha(p) for p in SCRIPT.glob('*.py')})
    return mapping


class DevGuard(AccessGuard):
    def __init__(self,**kwargs):
        assert set(kwargs.get('splits',()))<= {'train','val'}
        assert set(kwargs.get('gt_splits',()))<= {'val'}
        kwargs['output']=BASE
        super().__init__(**kwargs)


def job_args(name,*args):return [str(SCRIPT/name),*map(str,args)]
