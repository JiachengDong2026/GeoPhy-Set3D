#!/usr/bin/env python3
"""Rerun existing generation QA into external outputs, without refreshing frozen files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import yaml

from geophy_set3d.data.build_utils import digest, canonical_hash, write_json, source_snapshot
from geophy_set3d.data.generation import (load_records, choose_debug_records, leakage_audit,
                                        check_array, SOURCE, OUTPUT)
from geophy_set3d.data.projection_pool_dataset import load_source_index
from geophy_set3d.data.validation import geometry_qa, projector_validation
from geophy_set3d.physics.projectors import ReferenceProjector


def validate(output):
    output = Path(output).resolve()
    if output.is_relative_to(OUTPUT) or output.is_relative_to(SOURCE):
        raise ValueError('QA output must be outside frozen/source datasets')
    build = json.loads((OUTPUT / 'dataset_build_manifest.json').read_text())
    config = yaml.safe_load((OUTPUT / 'generation_config.yaml').read_text())
    assert canonical_hash(config) == build['config_sha256']
    assert digest(SOURCE / 'manifest.jsonl') == build['source_manifest_sha256']
    assert digest(OUTPUT / 'source_index.jsonl') == build['source_index_sha256']
    assert digest(OUTPUT / 'train_statistics.json') == build['statistics_sha256']
    # Read-only verification of EVERY frozen artifact, source fingerprint and split.
    artifacts = json.loads((OUTPUT / 'audits/artifact_hashes.json').read_text())
    assert canonical_hash(artifacts) == build['artifact_inventory_sha256']
    for relative, expected in artifacts.items():
        assert digest(OUTPUT / relative) == expected, relative
    for relative, expected in build['split_hashes'].items():
        assert digest(OUTPUT / 'splits' / relative) == expected, relative
    fingerprint = source_snapshot(SOURCE)
    assert fingerprint == json.loads((OUTPUT / 'audits/source_fingerprint_before.json').read_text())
    rows = load_source_index(OUTPUT)
    by_sample = {r['sample_id']: r for r in rows}
    records, source_qa = load_records(config)
    assert len(records) == len(rows)
    for record in records:
        row = by_sample[record['sample_id']]
        record.update(split=row['split'], gt_sha256=row['source_gt_sha256'])
        assert digest(SOURCE / record['target_path']) == record['gt_sha256']
    with np.load(OUTPUT / 'geometry/train_geometry.npz', allow_pickle=False) as f: train = dict(f)
    with np.load(OUTPUT / 'geometry/ood_geometry.npz', allow_pickle=False) as f: ood = dict(f)
    geometry_qa(train, ood, output / 'geometry_qa.json')
    leakage_audit(records, output / 'split_leakage_qa.json')
    for row in rows:
        for pool, geometry in [('train', train), ('ood', ood)]:
            check_array(np.load(OUTPUT / row[f'{pool}_projection_path'], mmap_mode='r', allow_pickle=False),
                        geometry, len(geometry['station_origin_m']))
        gt = np.load(OUTPUT / row['gt_volume_64_path'], mmap_mode='r', allow_pickle=False)
        assert gt.shape == (64,)*3 and gt.dtype == np.float32
        assert np.isfinite(gt).all() and np.all((gt >= 0) & (gt <= 1))
    projector = ReferenceProjector(config['projector'], output / 'cache')
    # Existing analytical, convergence, SciPy and autograd validation uses ONLY train GT.
    projector_validation(projector, config, choose_debug_records(records), SOURCE,
                         output / 'projector_validation.json')
    write_json(output / 'generation_regression_summary.json',
               dict(passed=True, frozen_artifacts_checked=len(artifacts),
                    source_files_checked=len(fingerprint), source_gt_qa=source_qa['accepted_count'],
                    projection_arrays_checked=len(rows)*2, split_unchanged=True,
                    projector_validation_cases='train only', frozen_dataset_writes=0))
    print('Existing generation QA passed; frozen dataset was read only.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='/data/djc/GeoPhy-Set3D/outputs/audits/training_interface_v1/generation_regression')
    validate(parser.parse_args().output_root)
