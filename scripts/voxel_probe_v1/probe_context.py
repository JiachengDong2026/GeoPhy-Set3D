"""Isolated bounded mechanism experiment; historical experiment files read-only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
import numpy as np
import torch
from geophy_set3d.training.unet_baseline_v1 import load_config, build_model
from geophy_set3d.training.selfsup import make_dataset, objective, projection_scores
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything, rng_state, atomic_checkpoint
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics

ART = PROJECT / 'docs/reports/artifacts/voxel_probe_v1'
ROOT = Path('/scratch/djc/GeoPhy-Set3D/voxel_probe_v1')
DATA = Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
UNET_ART = PROJECT / 'docs/reports/artifacts/unet_baseline_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def setup():
    torch.set_num_threads(2)
    seed_everything(20261011)


def verify():
    r = read(ART / 'registration.json')
    for name, digest in r['source_sha256'].items():
        assert sha(PROJECT / name) == digest, name
    frozen = read(UNET_ART / 'registration.json')
    for name, digest in frozen['source_sha256'].items():
        assert sha(PROJECT / name) == digest, name
    for name, digest in frozen['data_sha256'].items():
        assert sha(name) == digest, name
    return r


class DirectField(torch.nn.Module):
    def __init__(self, initial, c0, arm):
        super().__init__()
        self.c0 = c0
        self.arm = arm
        value = torch.logit(initial) if arm == 'voxel_sigmoid' else initial / c0
        self.value = torch.nn.Parameter(value.clone())

    def forward(self, batch):
        return self.value.sigmoid() if self.arm == 'voxel_sigmoid' else self.value * self.c0


def fresh():
    seed_everything(20261011)
    initial = read(UNET_ART / 'initialization.json')
    model = build_model(load_config(), initial['c0'], initial['alpha'], 'cpu')
    assert state_hash(model.state_dict()) == initial['initialized_state_sha256']
    return model


@torch.no_grad()
def evaluate(model, batch, extra, directory, step, save_arrays):
    model.eval()
    total, parts, volumes, projections = objective(model, batch)
    scores = projection_scores(projections, batch)
    row = dict(step=step, loss=float(total), parts={k: float(v) for k, v in parts.items()},
               measurement={k: float(np.mean([r['relative_l1'] for r in values])) for k, values in scores.items()},
               fields=[], extra_E_H=[])
    arrays = {}
    for label, volume in zip(['A', 'B'], volumes):
        p = project_input(volume, extra)
        mask = extra['input_valid_ray'].bool() & extra['view_mask'][:, :, None, None, None]
        row['extra_E_H'].append(relative_projection_metrics(p, extra['input_projection_raw'], mask)['relative_l1'])
        v = volume.double()
        row['fields'].append(dict(branch=label, mean=float(v.mean()), std=float(v.std(correction=0)),
                                  min=float(v.min()), max=float(v.max()), nearzero=float((v < 1e-10).double().mean())))
        if save_arrays:
            arrays[label + '_volume'] = volume.numpy()
            for target in ['A', 'B', 'H']:
                arrays[label + '_' + target + '_projection'] = projections['C_' + label][target].numpy()
            arrays[label + '_E_projection'] = p.numpy()
    row['extra_E_H'] = float(np.mean(row['extra_E_H']))
    if save_arrays:
        for target in ['A', 'B', 'H', 'E']:
            b = extra if target == 'E' else batch[target]
            for key in ['input_station_origin_m', 'input_ray_direction', 'input_t_near', 'input_path_length_m', 'input_projection_raw']:
                arrays[target + '_' + key] = b[key].numpy()
            arrays[target + '_mask'] = (b['input_valid_ray'].bool() & b['view_mask'][:, :, None, None, None]).numpy()
        path = directory / f'fields_step{step}.npz'
        np.savez_compressed(path, **arrays)
        row['arrays'] = dict(path=str(path), sha256=sha(path))
    return row
