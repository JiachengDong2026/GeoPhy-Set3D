"""Read-only, manifest-driven training/evaluation access to frozen Projection Pools."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from geophy_set3d.data.view_sampling import ViewSampler


def load_source_index(root):
    """Membership comes exclusively from source_index.jsonl, never directory scans."""
    rows = [json.loads(line) for line in (Path(root) / 'source_index.jsonl').read_text().splitlines()
            if line.strip()]
    if not rows:
        raise ValueError('Empty source index')
    for key in ('sample_id', 'case_id'):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f'Duplicate {key} in source index')
    for row in rows:
        if row['split'] not in ('train', 'val', 'test') or row['generation_status'] != 'complete':
            raise ValueError('Source index must contain complete, explicitly split records')
        for key in ('train_projection_path', 'ood_projection_path', 'gt_volume_64_path',
                    'source_target_path', 'condition_meta'):
            if key not in row:
                raise ValueError(f'Missing source-index field {key}')
    return rows


def artifact_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Artifact escapes dataset root: {relative}')
    return path


class ProjectionPoolDataset(Dataset):
    """XYZ CH4 GT and separate physical/normalized, randomly selected station views.

    Sampling is deterministic for (seed, epoch, index, case_id, worker_id).
    Call set_epoch BEFORE starting each epoch's iterator. Shared epoch state also
    reaches persistent workers. Repeated access in the same worker/epoch replays
    the same views; worker-count or scheduling changes can change the views.
    Evaluation uses epoch zero unless explicitly changed by the caller.
    """

    def __init__(self, dataset_root, split='train', geometry_pool='train', *,
                 min_input_views=3, max_input_views=6, hold_views=1,
                 base_seed=20261011, normalization='linear_p99_5',
                 statistics_file='train_statistics.json', load_gt128=False,
                 allow_train_ood_debug=False):
        if split not in ('train', 'val', 'test') or geometry_pool not in ('train', 'ood'):
            raise ValueError('split must be train/val/test; geometry_pool must be train/ood')
        if split == 'train' and geometry_pool == 'ood' and not allow_train_ood_debug:
            raise ValueError('Training is restricted to Pool A; Pool B is evaluation only')
        if normalization not in ('linear_p99_5', 'log1p_p99_5'):
            raise ValueError('Unknown fixed training-scale normalization')
        self.root = Path(dataset_root).resolve()
        self.split, self.geometry_pool = split, geometry_pool
        rows = load_source_index(self.root)
        self.records = [row for row in rows if row['split'] == split]
        if not self.records:
            raise ValueError(f'No records for split {split}')
        statistics = json.loads(artifact_path(self.root, statistics_file).read_text())
        train_ids = {row['case_id'] for row in rows if row['split'] == 'train'}
        if (statistics['split'] != 'train' or set(statistics['case_ids']) != train_ids
                or statistics['case_count'] != len(train_ids)
                or statistics['geometry_pool'] != 'A only; Pool B excluded'
                or statistics['projection_unit'] != 'ppm_m'):
            raise ValueError('Normalization statistics must describe train cases, Pool A, ppm_m')
        self.y_ref = float(statistics['projection_valid_rays']['p99.5'])
        if not np.isfinite(self.y_ref) or self.y_ref <= 0:
            raise ValueError('Training valid-ray p99.5 must be finite and positive')
        self.normalization, self.load_gt128 = normalization, load_gt128
        self.statistics_path = artifact_path(self.root, statistics_file)
        # Load ONLY the requested shared geometry, once; no geometry recomputation.
        self.geometry_path = self.root / 'geometry' / f'{geometry_pool}_geometry.npz'
        with np.load(self.geometry_path, allow_pickle=False) as arrays:
            self.geometry = {key: arrays[key] for key in arrays.files}
        self.station_count = len(self.geometry['station_origin_m'])
        self.projection_shape = self.geometry['valid_ray'].shape
        if (self.projection_shape != (self.station_count, 64, 64)
                or self.geometry['valid_ray'].dtype != np.bool_):
            raise ValueError('Expected frozen boolean validity geometry [stations,64,64]')
        self.sampler = ViewSampler(min_input_views, max_input_views, hold_views, base_seed)
        if self.station_count < max_input_views + hold_views:
            raise ValueError('Insufficient candidate stations')
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError('epoch must be a nonnegative integer')
        self._epoch.fill_(epoch)

    def _views(self, projections, indices, prefix):
        # Advanced indexing copies only selected station maps from the read-only mmap.
        raw = torch.from_numpy(np.array(projections[indices], copy=True))[:, None]
        if not torch.isfinite(raw).all() or (raw < 0).any():
            raise ValueError('Projection must be finite, nonnegative ppm_m')
        normalized = raw / self.y_ref
        if self.normalization == 'log1p_p99_5':
            normalized = torch.log1p(normalized)
        fields = {f'{prefix}_station_indices': torch.from_numpy(indices),
                  f'{prefix}_projection_raw': raw,
                  f'{prefix}_projection': normalized}
        for key, values in self.geometry.items():
            if key in ('horizontal_angle_deg', 'vertical_angle_deg'):
                continue
            array = np.array(values[indices], copy=True)
            value = torch.from_numpy(array)
            value = value.bool() if key == 'valid_ray' else value.float()
            if key in ('valid_ray', 'path_length_m', 't_near', 't_far'):
                value = value[:, None]
            fields[f'{prefix}_{key}'] = value
        if torch.count_nonzero(raw[~fields[f'{prefix}_valid_ray']]):
            raise ValueError('Invalid rays must have zero frozen measurement')
        return fields

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.records[index]
        worker = get_worker_info()
        inputs, holds = self.sampler.sample(
            self.station_count, epoch=int(self._epoch.item()), sample_index=index,
            case_id=row['case_id'], worker_id=worker.id if worker is not None else 0)
        projections = np.load(artifact_path(self.root, row[f'{self.geometry_pool}_projection_path']),
                              mmap_mode='r', allow_pickle=False)
        if projections.shape != self.projection_shape or projections.dtype != np.float32:
            raise ValueError('Frozen projection shape/dtype does not match selected geometry')
        gt = np.load(artifact_path(self.root, row['gt_volume_64_path']), mmap_mode='r', allow_pickle=False)
        if (gt.shape != (64, 64, 64) or gt.dtype != np.float32 or not np.isfinite(gt).all()
                or np.any((gt < 0) | (gt > 1))):
            raise ValueError('Expected finite XYZ GT64 mass fraction in [0,1]')
        result = dict(case_id=row['case_id'], sample_id=row['sample_id'], split=self.split,
                      geometry_pool=self.geometry_pool, condition_meta=copy.deepcopy(row['condition_meta']),
                      projection_scale_ppm_m=torch.tensor(self.y_ref, dtype=torch.float64),
                      gt_volume_64=torch.from_numpy(np.array(gt, copy=True))[None])
        result.update(self._views(projections, inputs, 'input'))
        result.update(self._views(projections, holds, 'hold'))
        for key in ('horizontal_angle_deg', 'vertical_angle_deg'):
            result[key] = torch.from_numpy(self.geometry[key].copy()).float()
        if self.load_gt128:
            # The index contains the explicit source GT path; no filename inference.
            gt128 = np.load(row['source_target_path'], mmap_mode='r', allow_pickle=False)
            if gt128.shape != (128, 128, 128) or gt128.dtype != np.float32:
                raise ValueError('Expected source XYZ float32 GT128')
            if not np.isfinite(gt128).all() or np.any((gt128 < 0) | (gt128 > 1)):
                raise ValueError('Invalid source GT128 mass fraction')
            result['gt_volume_128'] = torch.from_numpy(np.array(gt128, copy=True))[None]
        return result
