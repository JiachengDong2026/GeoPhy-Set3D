"""Small frozen-schema fixture, with paths deliberately unrelated to case IDs."""
import json
from pathlib import Path
import tempfile

import numpy as np

from geophy_set3d.geometry.tdlas import make_geometry


def make_pool():
    temporary = tempfile.TemporaryDirectory(prefix='geophy-pool-test-')
    root = Path(temporary.name) / 'pool'
    root.mkdir()
    (root / 'geometry').mkdir()
    config = dict(azimuth_deg=[0, 180], radius_m=[5.5, 6.5], height_m=[.25, .35],
                  target_m=[.3]*3, projection_hw=[64, 64],
                  horizontal_fan_deg=10, vertical_scan_deg=8)
    geometries = {name: make_geometry(config, count, seed)
                  for name, count, seed in [('train', 12, 1), ('ood', 10, 2)]}
    for name, geometry in geometries.items():
        np.savez(root / 'geometry' / f'{name}_geometry.npz', **geometry)
    rows = []
    for i, split in enumerate(['train']*3 + ['val']*2 + ['test']*2):
        folder = root / 'stored' / f'volume_{i}'
        folder.mkdir(parents=True)
        paths = {}
        for name, geometry in geometries.items():
            value = np.broadcast_to((np.arange(len(geometry['valid_ray']), dtype=np.float32)
                                     + i + 1)[:, None, None] * 2000,
                                    geometry['valid_ray'].shape).copy()
            value[~geometry['valid_ray']] = 0
            # A VALID zero ray must remain valid in returned masks.
            station, y, z = np.argwhere(geometry['valid_ray'])[0]
            value[station, y, z] = 0
            path = folder / f'observations_{name}.npy'
            np.save(path, value)
            paths[f'{name}_projection_path'] = str(path.relative_to(root))
        path = folder / 'supervised_target.npy'
        np.save(path, np.full((64,)*3, (i+1)*.05, np.float32))
        source = Path(temporary.name) / f'source_{i}.npy'
        np.save(source, np.full((128,)*3, (i+1)*.05, np.float32))
        rows.append(dict(case_id=f'case-{i}', sample_id=f'frame-{i}', split=split,
                         **paths, gt_volume_64_path=str(path.relative_to(root)),
                         source_target_path=str(source), generation_status='complete',
                         condition_meta={'source_mode': None, 'pressure_pa': i*100}))
    (root / 'source_index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    statistics = dict(split='train', geometry_pool='A only; Pool B excluded',
                      projection_unit='ppm_m', case_count=3,
                      case_ids=[row['case_id'] for row in rows if row['split'] == 'train'],
                      projection_valid_rays={'p99.5': 1000.0})
    (root / 'train_statistics.json').write_text(json.dumps(statistics))
    return temporary, root, rows, geometries
