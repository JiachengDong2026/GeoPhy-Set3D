"""Minimal comparison contract, strict independent baseline policy and identity."""
from pathlib import Path
import hashlib
import json
import torch
import yaml
from geophy_set3d.models import build_model
from geophy_set3d.data.measurement_only import assert_no_gt
from .selfsup import FROZEN_LOSS

MODEL = dict(name='geolift_deepsets_sum', input_channels=3, feature_channels=32,
             lift_resolution=32, output_resolution=64, output_activation='sigmoid', v_ref=4)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def validate_comparison_config(c):
    if c['model'] != MODEL or c['loss'] != FROZEN_LOSS:
        raise ValueError('DeepSets v1 freezes model and exact historical loss')
    for key, expected in [('train',dict(split='train',geometry_pool='train',a_views=[3,5],b_views=[3,5],heldout_views=2)),
                          ('validation',dict(split='val',geometry_pool='train',a_views=[4,4],b_views=[4,4],heldout_views=4))]:
        if c[key] != expected: raise ValueError('Frozen disjoint natural station protocol')
    t=c['training']
    if any(t[k]!=v for k,v in dict(seed=20261011,max_epochs=300,precision='fp32',gradient_clip_norm=1.,early_stopping_patience=None).items()):
        raise ValueError('Full 300 epochs, original seed, FP32 and clipping required')
    if (t['batch_size'],t['gradient_accumulation']) not in [(2,1),(1,2)]:
        raise ValueError('Effective batch2 required')
    if c['optimizer']!=dict(name='adamw',lr=.0002,weight_decay=.0001) or c['scheduler']!=dict(name='cosine',T_max=300):
        raise ValueError('Original optimizer/scheduler required')
    if c['normalization'] != dict(method='linear_p99_5',statistics_file='train_statistics.json'):
        raise ValueError('Original normalization required')
    if c['experiment']['name']!='deepsets_selfsup_v1' or Path(c['comparison']['output']).name!='deepsets_selfsup_v1':
        raise ValueError('Independent run directory required')
    if c['comparison']['bootstrap_samples']!=10000 or c['comparison']['bootstrap_seed']!=2026101801:
        raise ValueError('Frozen statistics required')
    if set(c) != {'dataset','experiment','loader','loss','model','normalization','optimizer','train','training','validation','scheduler','comparison'}:
        raise ValueError('Unsupported new settings')
    return c


def load_comparison_config(path):
    return validate_comparison_config(yaml.safe_load(Path(path).read_text()))


def build_comparison_model(c, device=None):
    return build_model(c).to(device or c['training']['device'])


def check_prediction(prediction, batch):
    assert_no_gt(batch)
    if prediction.shape!=(len(batch['case_id']),1,64,64,64) or prediction.dtype!=torch.float32:
        raise ValueError('Comparison contract: FP32 [B,1,64,64,64]')
    if not torch.isfinite(prediction).all() or not ((prediction>=0)&(prediction<=1)).all():
        raise FloatingPointError('Finite mass fraction in [0,1] required')
    return prediction


def tensor_identity(*values):
    h=hashlib.sha256()
    for t in values:
        a=t.detach().cpu().contiguous().numpy()
        h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def model_identity(model,c):
    return dict(name=c['model']['name'],model_config=c['model'],parameter_count=sum(p.numel() for p in model.parameters()),
                fp32_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                information='full station measurements/valid rays/path/geometry, H excluded from network',
                labels='no volumetric labels',selector='mean held-out measurement relative L1')


def verify_registration(c):
    out=Path(c['comparison']['output']);p=json.loads((out/'protocol.json').read_text())
    if p['config']!=c or sha256(out/'protocol.json')!=(out/'protocol.sha256').read_text().strip():
        raise ValueError('Registration/config mismatch')
    for path,h in {**p['reference_hashes'],**p['metadata_sha256']}.items():
        if sha256(path)!=h:raise ValueError('Changed frozen source: '+path)
    return p
