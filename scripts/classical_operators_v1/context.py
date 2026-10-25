"""Shared paths and registered source verification for this experiment only."""
import sys,json,subprocess
from pathlib import Path
from audit import PROJECT,ROOT,ART,sha,read,write
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'scripts')]
import numpy as np
import torch
from geophy_set3d.training.classical_operators import load_config,build_model
from geophy_set3d.training.utils import seed_everything,atomic_checkpoint,rng_state,restore_rng,move_batch
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.selfsup import make_dataset,make_loader,objective,run_epoch,selection_metric,move_dual_batch,zero_volume_baseline
from geophy_set3d.training.deep_comparison import check_prediction,tensor_identity
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics

def register(revision=False):
    paths=list((PROJECT/'src').rglob('*.py'))+list((PROJECT/'scripts/classical_operators_v1').glob('*.py'))
    paths+=list((PROJECT/'configs/classical_operators_v1').glob('*.yaml'))
    paths+=[PROJECT/'docs/classical_operators_v1_protocol_zh.md',PROJECT/'tests/test_classical_operators_v1.py']
    m=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),sha256={str(p.relative_to(PROJECT)):sha(p) for p in sorted(paths)})
    target=ART/'source_manifest.json'
    if revision:
        assert not (ROOT/'deeponet/pilot').exists() and not (ROOT/'fno/pilot').exists()
        assert not (ART/'source_manifest_smoke.json').exists()
        (ART/'source_manifest_smoke.json').write_bytes(target.read_bytes())
        (ROOT/'registration_smoke.json').write_bytes((ROOT/'registration.json').read_bytes())
    else:assert not target.exists()
    write(target,m)
    write(ROOT/'registration.json',dict(source_manifest_sha256=sha(target),protocol_sha256=sha(PROJECT/'docs/classical_operators_v1_protocol_zh.md'),
        models={arm:load_config(arm) for arm in ['deeponet','fno']},pilot_train_indices=list(range(32)),pilot_val_count=36,epochs=30,updates=480))

def verify():
    m=read(ART/'source_manifest.json')
    assert sha(ART/'source_manifest.json')==read(ROOT/'registration.json')['source_manifest_sha256']
    for p,h in m['sha256'].items():assert sha(PROJECT/p)==h,p
    return m

def setup():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;seed_everything(20261011)

def no_forbidden(a):
    assert not any(a[k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])

if __name__=='__main__':register('--prepilot-revision' in sys.argv)
