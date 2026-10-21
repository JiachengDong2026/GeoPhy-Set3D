"""Frozen model and shared immutable I/O for mask-controlled OARI diagnostics."""
import sys
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts/paper_final_v1'))
from common import *
import csv
import numpy as np
import torch


def load_unique(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIOR = load_unique('registered_oari_v1_runtime', PROJECT / 'scripts/oari_mechanism_v1/runtime.py')
HIST = PRIOR.HIST
OLD = PRIOR.BASE
BASE = OLD.parent / 'mrgr_oari_mechanism_v2'
ARTIFACT = PROJECT / 'docs/reports/artifacts' / BASE.name
REPORT = PROJECT / 'docs/reports/mrgr_oari_mechanism_v2_zh.md'
SEED = 20261024
EPS = 1e-6
CHANGE = 1e-5
CONDITIONS = ('Val-A-V4', 'spread-V4', 'clustered-V4', 'V3', 'V6')
CHANNELS = dict(drop_cos=[0], drop_sin=[1], drop_angle=[0, 1], drop_azimuth=[2],
                obs_bias=[3, 4, 5, 6, 7], obs_both=[3, 4, 5, 6, 7])
GROUPS = ['A0', 'A1_P2', 'A2_P1'] + [f'A3_{i}' for i in range(8)] + list(CHANNELS) + ['obs_token']


def csv_write(path, records):
    path = Path(path)
    assert not path.exists()
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in records for k in row)), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def tensor_hash(t):
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def input_hash(batch):
    h = hashlib.sha256()
    for key, value in sorted(batch.items()):
        h.update(key.encode())
        h.update(tensor_hash(value).encode() if isinstance(value, torch.Tensor) else repr(value).encode())
    return h.hexdigest()


def sources():
    result = {str(p): sha(p) for p in HERE.glob('*.py')}
    result.update({str(p): sha(p) for p in (PROJECT / 'src').rglob('*.py')})
    for relative in ('scripts/oari_mechanism_v1/runtime.py', 'scripts/geometry_utilization_v1/diagnostic.py',
                     'scripts/uniform_final_v1/uniform_models.py', 'scripts/uniform_final_v1/runtime.py',
                     'scripts/uniform_final_v1/controller.py', 'scripts/paper_final_v1/common.py',
                     'scripts/paper_final_v1/models.py', 'scripts/paper_final_v1/module_models_v2.py',
                     'scripts/probe_mrgr_multiscale_v1.py'):
        result[str(PROJECT / relative)] = sha(PROJECT / relative)
    return result


def validate():
    p = verified(BASE / 'protocol.json')
    verify_hashes(p['sources'])
    assert sha(p['model']['checkpoint']) == HIST.EXPECTED
    return p


def model_load(device):
    p = validate()
    m = p['model']
    ck = torch.load(m['checkpoint'], map_location='cpu', weights_only=False)
    model = HIST.model_for(m['config'], 'uniform', device).eval().requires_grad_(False)
    model.load_state_dict(ck['model'], strict=True)
    assert sum(t.numel() for t in model.parameters()) == 342725
    assert HIST.state_hash(model) == p['state_sha256']
    return model


class Guard(AccessGuard):
    def __init__(self, projections=True):
        p = verified(BASE / 'protocol.json')
        super().__init__(splits=['test'] if projections else [], checkpoints=[p['model']['checkpoint']], output=BASE)
        for item in verified(BASE / 'reuse_manifest.json'):
            self.allowed[str(Path(item['path']).resolve())] = 'sealed_historical_prediction'


def scheduler():
    s = PRIOR.scheduler()
    s.BASE = BASE
    s.GPU_LOCKS = BASE.parent / 'gpu_locks'
    s.progress = lambda: None
    return s


def geometry_batch(ds, ids):
    out = {'view_mask': torch.ones(1, len(ids), dtype=torch.bool)}
    for key in ('station_origin_m', 'center_direction', 'right_direction', 'up_direction', 'station_azimuth_deg'):
        out['input_' + key] = torch.from_numpy(np.array(ds.geometry[key][ids], copy=True)).float()[None]
    for key in ('horizontal_angle_deg', 'vertical_angle_deg'):
        out[key] = torch.from_numpy(ds.geometry[key].copy()).float()[None]
    return out
