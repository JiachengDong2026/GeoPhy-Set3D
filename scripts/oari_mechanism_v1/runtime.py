"""Frozen OARI diagnostics, isolated from ongoing GeometryLifting training."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'scripts/paper_final_v1'))
from common import *
import importlib.util
import numpy as np
import torch
import csv


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def historical():
    previous=sys.modules.get('runtime')
    try:
        sys.modules['runtime']=load('oari_actual_uniform_runtime',PROJECT/'scripts/uniform_final_v1/runtime.py')
        return load('oari_historical_diagnostic',PROJECT/'scripts/geometry_utilization_v1/diagnostic.py')
    finally:
        if previous is None:sys.modules.pop('runtime',None)
        else:sys.modules['runtime']=previous

HIST=historical()
BASE=HIST.BASE.parent/'mrgr_oari_mechanism_diagnostics_v1'
ARTIFACT=PROJECT/'docs/reports/artifacts'/BASE.name
REPORT=PROJECT/'docs/reports/mrgr_oari_mechanism_diagnostics_v1_zh.md'
TRAIN_BASE=BASE.parent/'mrgr_no_spatial_lifting_v1'
CONDITIONS=('Val-A-V4','spread-V4','V3','V6','clustered-V4')
METRICS=('relative_l1','relative_l2','dice50','absolute_peak_ratio_error','centroid_distance_mm','gt_peak_location_ratio','peak_ratio')
SEED=20261023


def csv_read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def csv_write(path,records):
    assert not path.exists()
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)),lineterminator='\n');writer.writeheader();writer.writerows(records)


def sources():
    result={str(p):sha(p) for p in HERE.glob('*.py')}
    result.update({str(PROJECT/k):sha(PROJECT/k) for k in ['scripts/geometry_utilization_v1/diagnostic.py','scripts/uniform_final_v1/uniform_models.py','scripts/uniform_final_v1/runtime.py','scripts/paper_final_v1/common.py','scripts/paper_final_v1/module_models_v2.py','scripts/paper_final_v1/models.py','scripts/probe_mrgr_multiscale_v1.py']})
    result.update({str(p):sha(p) for p in (PROJECT/'src').rglob('*.py')})
    return result


def method():return verified(BASE/'protocol.json')['model']


class Guard(AccessGuard):
    def __init__(self):
        m=method();super().__init__(splits=['test'],checkpoints=[m['checkpoint']],output=BASE)
        for r in verified(BASE/'sampled_historical_predictions.json'):self.allowed[r['path']]='sealed_historical_prediction'


def scheduler():
    # Existing scheduler is loaded with its actual runtime, then only task paths
    # and progress callback are scoped to this independent diagnostic.
    previous=sys.modules.get('runtime')
    folder=PROJECT/'scripts/uniform_final_v1';sys.path.insert(0,str(folder))
    try:
        sys.modules['runtime']=load('oari_scheduler_runtime',folder/'runtime.py')
        s=load('oari_uuid_scheduler',folder/'controller.py')
    finally:sys.modules['runtime']=previous
    s.BASE=BASE;s.PROJECT=PROJECT;s.GPU_LOCKS=BASE.parent/'gpu_locks';s.progress=lambda:None
    return s
