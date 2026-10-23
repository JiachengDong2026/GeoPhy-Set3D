"""Read-only reuse of audited plain model, formal operators, metrics and GPU locks."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'scripts/plain_core_ablations_v1'))
import common_stage as core
from common_stage import *
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
CORE=core.BASE
TARGETED=core.PRIOR
BASE=CORE.parent/'mrgr_plain_final_v1'
ART=PROJECT/'docs/reports/artifacts'/BASE.name
MAIN=f'{PLAIN}_seed20261011'
SECOND=f'{PLAIN}_seed20261019'
CLASSICS=('NNLS-NL','Tikhonov','TV','SART')
CONDITIONS=('Val-A-V4','spread-V4','V3','V6','clustered-V4')
import csv
import shutil
import platform
import importlib.metadata
import time
import fcntl
import evaluation as official
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
from geophy_set3d.reconstruction.classical import geometry_tensor,project
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from solver import CachedProjector


def model_for(config,arm,device):
    assert arm==PLAIN,'This stage cannot construct or train any alternative model'
    return core.prior.model_for(config,PLAIN,device)


def command(name,*values):return [sys.executable,str(HERE/name),*map(str,values)]


def sources():
    d=core.prior.training_sources()
    d.update({str(p):sha(p) for p in HERE.glob('*.py')})
    for relative in ('scripts/paper_final_v1/evaluation.py','scripts/paper_final_v1/solver.py','scripts/paper_final_v1/sart.py',
        'scripts/report_historical_reconstruction_v1.py','scripts/analyze_frozen_ab_morphology_v1.py','docs/mrgr_plain_final_v1_protocol_zh.md',
        'configs/mrgr_uniform_plain_set_interaction_v1.json'):
        d[str(PROJECT/relative)]=sha(PROJECT/relative)
    return d


def io_guard(*,splits=(),gt=False,predictions=(),checkpoints=()):
    g=AccessGuard(splits=splits,gt_splits=['test'] if gt else [],checkpoints=checkpoints,output=BASE)
    for r in predictions:g.allowed[str(Path(r['prediction']).resolve())]='audited_prediction'
    if gt:
        seal=verified(BASE/'all_predictions_sealed.json');assert seal['all_predictions_saved_before_GT64']
        assert sha(BASE/'prediction_manifest.json')==seal['prediction_manifest_sha256']
        allowed_cases=set(verified(BASE/'cases.json'))
        allowed={str((DATA/r['gt_volume_64_path']).resolve()) for r in rows(DATA/'source_index.jsonl') if r['case_id'] in allowed_cases and r['split']=='test'}
        assert len(allowed)==35
        for p in list(g.allowed):
            if g.allowed[p]=='GT64_test' and p not in allowed:del g.allowed[p]
    return g


def batch_for(ds,i,device='cpu'):
    batch=move_batch(projection_pool_collate([ds[i]]),device)
    return {k:v for k,v in batch.items() if not k.startswith('hold_')}


def check_fields(path,expected):
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)==set(expected),(path,z.files,expected)
        for k in z.files:
            x=z[k];assert x.shape==(64,)*3 and x.dtype==np.float32 and np.isfinite(x).all() and x.min()>=0 and x.max()<=1,path


def csv_write(path,records):
    assert records and not path.exists(),path
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)),lineterminator='\n')
        w.writeheader();w.writerows(records)


def flatten(volume,morphology,measurement=None):
    d=official.scalar_metrics(dict(volume=volume,morphology=morphology,measurement=measurement or {},reconstruct_seconds=None,peak_allocated_bytes=None))
    return {k:v for k,v in d.items() if k.startswith(('volume.','morphology.','measurement.'))}


def scheduler():
    s=core.scheduler();s.BASE=BASE;s.GPU_LOCKS=BASE.parent/'gpu_locks';s.progress=lambda:None
    return s


sys.path.insert(0,str(HERE))
