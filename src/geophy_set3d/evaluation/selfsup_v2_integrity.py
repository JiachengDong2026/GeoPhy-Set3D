"""RVTC metadata integrity before freeze; optional GT hashes only after completion."""
import json
from pathlib import Path
from geophy_set3d.data.build_utils import digest
from geophy_set3d.training.utils import write_json


def metadata_snapshot(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,inode=p.stat().st_ino)
            for p in sorted(root.rglob('*')) if p.is_file()}


def check_integrity(root,config,repo,*,posthoc=False):
    root,repo=Path(root),Path(repo)
    before=json.loads((root/'inventory_before.json').read_text())
    base=Path(config['experiment']['output_root'])
    history=all((base/kind/name).is_file() and metadata_snapshot_entry(base/kind/name)==value
                for kind,rows in before['existing_artifacts'].items() for name,value in rows.items())
    raw=all(Path(p).is_file() and metadata_snapshot_entry(p)==value for p,value in before['source_gt128_metadata'].items())
    unchanged=lambda rows:all((repo/p).is_file() and digest(repo/p)==sha for p,sha in rows.items())
    protected=before['protected_code']
    result=dict(dataset_unchanged=metadata_snapshot(config['dataset']['root'])==before['dataset'],
        all_historical_artifacts_unchanged=history,source_gt128_metadata_unchanged=raw,
        projector_unchanged=unchanged({p:s for p,s in protected.items() if p.startswith('src/geophy_set3d/physics/')}),
        network_architecture_unchanged=unchanged({p:s for p,s in protected.items() if p.startswith('src/geophy_set3d/models/')}),
        v1_training_unchanged=unchanged({p:s for p,s in protected.items() if p in ('src/geophy_set3d/training/selfsup.py',
            'src/geophy_set3d/data/measurement_only.py','scripts/train_selfsup_mrgr.py','configs/train_selfsup_oari_mrgr_v1.yaml')}),
        numerical_training_sources_unchanged=unchanged(json.loads((root/'training_source_manifest.json').read_text())['sha256']),
        test_used=False,pool_b_training_unused=True,gt_hashes_checked=posthoc)
    if posthoc:
        freeze=json.loads((root/'frozen_checkpoint.json').read_text())
        if freeze['completed_epochs']!=300:raise RuntimeError('No GT hash read before full training')
        frozen=json.loads((root.parent/'selfsup_mrgr_v1/inventory_before.json').read_text())['dataset']
        data=Path(config['dataset']['root'])
        result['dataset_hashes_unchanged']=all(digest(data/p)==v['sha256'] for p,v in frozen.items() if 'sha256' in v)
        rows=[json.loads(line) for line in (data/'source_index.jsonl').read_text().splitlines() if line.strip()]
        result['source_gt128_hash_unchanged']=all(digest(row['source_target_path'])==row['source_gt_sha256'] for row in rows)
    result['passed']=all(v for k,v in result.items() if k.endswith('_unchanged'))
    write_json(root/'integrity_summary.json',result)
    if not result['passed']:raise AssertionError('Frozen architecture/dataset/artifacts/protocol changed')
    return result


def metadata_snapshot_entry(path):
    stat=Path(path).stat()
    return dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino)
