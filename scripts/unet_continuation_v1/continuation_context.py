"""New continuation only; historical experiments stay sealed."""
import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'scripts/voxel_probe_v1'))
from probe_context import *
from geophy_set3d.training.utils import restore_rng,move_batch
OLD_ART=ART
OLD_ROOT=ROOT
ART=PROJECT/'docs/reports/artifacts/unet_continuation_v1'
ROOT=Path('/scratch/djc/GeoPhy-Set3D/unet_continuation_v1')
SOURCE=OLD_ROOT/'case0_original_refinement/last.pt'


def verify_all():
    verify()
    p=read(ART/'protocol.json')
    for name,digest in p['sources'].items():assert sha(PROJECT/name)==digest,name
    assert sha(SOURCE)==p['source_checkpoint_sha256']
    for name,digest in read(OLD_ART/'delivery_manifest.json').items():assert sha(PROJECT/name)==digest,name
    for name,digest in read(PROJECT/'docs/reports/artifacts/unet_baseline_v1/history_audit.json')['frozen_sha256'].items():assert sha(name)==digest,name
    return p
