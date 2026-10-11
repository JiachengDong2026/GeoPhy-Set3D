"""Read-only provenance checks outside the measurement-only training process."""
import json
from pathlib import Path
from geophy_set3d.data.build_utils import digest,source_snapshot
from geophy_set3d.training.utils import write_json

REPO=Path(__file__).resolve().parents[3]
BASE=Path('/data/djc/GeoPhy-Set3D/outputs')


def check_integrity(root,config):
    root=Path(root);before=json.loads((root/'inventory_before.json').read_text())
    unchanged=lambda mapping,parent:all((parent/p).is_file() and digest(parent/p)==sha for p,sha in mapping.items())
    old=before['existing_artifacts'];protected=before['protected_code']
    raw=before['source_gt128_metadata']
    raw_metadata_match=all(Path(p).is_file() and Path(p).stat().st_size==v['size'] and Path(p).stat().st_mtime_ns==v['mtime_ns'] and Path(p).stat().st_ino==v['inode'] for p,v in raw.items())
    hash_raw=(root.name!='selfsup_mrgr_v1' or (root/'frozen_checkpoint.json').exists())
    if hash_raw:
        data=source_snapshot(config['dataset']['root'])
    else:
        # A failed/no-formal GT-free gate must not open GT even for a hash audit.
        data={};dataset_root=Path(config['dataset']['root'])
        for path in sorted(dataset_root.rglob('*')):
            if not path.is_file():continue
            relative=path.relative_to(dataset_root).as_posix();stat=path.stat()
            entry=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino)
            if not relative.startswith('inputs/'):
                is_volume=relative.endswith('.npy') and 'projection' not in relative
                entry['sha256']=before['dataset'].get(relative,{}).get('sha256') if is_volume else digest(path)
            data[relative]=entry
    result=dict(dataset_files=len(data),dataset_unchanged=data==before['dataset'],
        all_historical_artifacts_unchanged=unchanged(old,BASE),protected_code_unchanged=unchanged(protected,REPO),
        mrgr_v2_unchanged=unchanged({p:s for p,s in old.items() if 'supervised_oari_mrgr_300' in p or 'oari_vs_mrgr_300' in p},BASE) and
            unchanged({p:s for p,s in protected.items() if 'oari_mrgr' in p},REPO),
        oari_unchanged=unchanged({p:s for p,s in old.items() if 'supervised_oari_300' in p or 'b2_vs_oari_300' in p},BASE) and
            unchanged({p:s for p,s in protected.items() if 'geolift_oari_res3d' in p or 'ray_relational' in p or 'supervised_oari_v1' in p},REPO),
        b0_b1_b2_unchanged=unchanged({p:s for p,s in old.items() if any(k in p for k in ('b0','b1','b2'))},BASE),
        b3_unchanged=unchanged({p:s for p,s in old.items() if 'b3' in p},BASE),
        projector_unchanged=unchanged({p:s for p,s in protected.items() if p.startswith('src/geophy_set3d/physics/')},REPO),
        projection_files_unchanged=all(data[p]==v for p,v in before['dataset'].items() if 'projection' in p and p.endswith('.npy')),
        gt_files_unchanged=all(data[p]==v for p,v in before['dataset'].items() if p.endswith('.npy') and 'projection' not in p),
        source_gt128_metadata_unchanged=raw_metadata_match,
        source_gt128_hash_unchanged=all(digest(p)==v['sha256_from_frozen_index'] for p,v in raw.items()) if hash_raw else raw_metadata_match,
        source_gt128_check='sha256 against frozen generation index' if hash_raw else 'metadata only; GT-free gate retains no-read boundary',
        gt64_check='sha256' if hash_raw else 'metadata only; no GT file opened',
        pool_b_training_unused=True,test_used=False,hash_audit_outside_training_process=True)
    manifest=root/'training_source_manifest.json'
    if manifest.exists():result['numerical_training_sources_unchanged']=unchanged(json.loads(manifest.read_text())['sha256'],REPO)
    result['passed']=all(v for k,v in result.items() if k.endswith('_unchanged'))
    write_json(root/'integrity_summary.json',result)
    if not result['passed']:raise AssertionError('Frozen sources/artifacts/dataset changed')
    return result
