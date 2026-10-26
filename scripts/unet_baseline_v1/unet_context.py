"""Self-contained U-Net execution context; historical module names unchanged."""
import sys
from unet_paths import *
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'scripts')]
import numpy as np
import torch
from geophy_set3d.training.unet_baseline_v1 import load_config,build_model as initialized_model
from geophy_set3d.training.utils import seed_everything,atomic_checkpoint as original_atomic_checkpoint,rng_state,move_batch,environment
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.selfsup import make_dataset,make_loader,objective,run_epoch,selection_metric,move_dual_batch,zero_volume_baseline
from geophy_set3d.training.deep_comparison import check_prediction,tensor_identity
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics

def setup():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;seed_everything(20261011)
def no_forbidden(a):
 assert not any(a[k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
def build_model(c,device=None):
 scale=read(ART/'initialization.json')
 return initialized_model(c,scale['c0'],scale['alpha'],device or c['training']['device'])
def verify():
 r=read(ART/'registration.json')
 for p,h in r['source_sha256'].items():assert sha(PROJECT/p)==h,p
 for p,h in r['data_sha256'].items():assert sha(p)==h,p
 return r

def atomic_checkpoint(path,payload):
 original_atomic_checkpoint(path,payload)
 with Path(path).open('rb') as f:os.fsync(f.fileno())
 fd=os.open(str(Path(path).parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
