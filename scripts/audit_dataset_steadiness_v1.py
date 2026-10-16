"""Inventory local metadata for steady-state evidence, without opening arrays."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1')
POOL = Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
OUT = ROOT / 'outputs/research/dataset_steadiness_audit_v1'


def main():
    opened = []
    sources = {}

    def read_text(path):
        path = Path(path).resolve()
        assert path.suffix in {'.md', '.json', '.jsonl'}, path
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        opened.append(str(path))
        return raw.decode('utf-8-sig')

    def read_json(path):
        return json.loads(read_text(path))

    def read_jsonl(path):
        return [json.loads(s) for s in read_text(path).splitlines() if s.strip()]

    raw_files = sorted(p for p in RAW.rglob('*') if p.is_file())
    manifest = read_jsonl(RAW / 'manifest.jsonl')
    summary = read_json(RAW / 'dataset_summary.json')
    read_text(RAW / 'README.md')
    index = read_jsonl(POOL / 'source_index.jsonl')
    by_id = {r['sample_id']: r for r in index}
    rows = []
    metadata_keys = Counter()
    field_frames = defaultdict(set)
    mismatches = []
    for row in manifest:
        sid = row['sample_id']
        meta = read_json(RAW / row['metadata_path'])
        frames = read_jsonl(POOL / by_id[sid]['train_projection_path'].split('/projections_train.npy')[0] / 'frame_manifest.jsonl')
        metadata_keys.update(meta.keys())
        for field in ('frame_index', 'physical_time_s', 'sample_id'):
            if row.get(field) != meta.get(field):
                mismatches.append({'case_id': sid, 'field': field})
        for frame in frames:
            field_frames[sid].add((frame['frame_index'], frame['physical_time_s']))
        if field_frames[sid] != {(row['frame_index'], row['physical_time_s'])}:
            mismatches.append({'case_id': sid, 'field': 'processed_frame_manifest'})
        rows.append({'case_id': sid, 'split': by_id[sid]['split'],
                     'frame_index': row['frame_index'], 'physical_time_s': row['physical_time_s'],
                     'wind_speed_m_s': row['wind_speed_m_s'], 'pressure_pa': row['pressure_pa'],
                     'active_hole_count': row['active_hole_count'],
                     'geometry_id': row['geometry_id'], 'role': row['role'],
                     'source_run_dir': row['source_run_dir'],
                     'source_validation_file': row['source_validation_file'],
                     'source_run_path_available_locally': Path(row['source_run_dir']).exists(),
                     'source_audit_path_available_locally': Path(row['source_validation_file']).exists()})

    search_roots = [Path(p) for p in ['/data/djc', '/home/djc/projects',
                    '/scratch/djc/checkpoint', '/scratch/djc/experiments',
                    '/scratch/djc/logs', '/scratch/djc/lmdb', '/scratch/djc/tmp']]
    prune = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'envs',
             'conda_pkgs', 'pip_cache', 'Senseiver-official'}
    suffixes = ('.cas.h5', '.dat.h5', '.cas', '.dat', '.trn', '.jou', '.out',
                '.res', '.cgns', '.vtk', '.vtu', '.msh.h5')
    names = ('completeness_audit', 'residual_history', 'mass_flow',
             'monitor', 'time_history', 'transient_history')
    cfd_candidates = []
    search_file_counts = {}
    for root in search_roots:
        count = 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in prune]
            for filename in files:
                count += 1
                lower = filename.lower()
                if lower.endswith(suffixes) or any(term in lower for term in names):
                    cfd_candidates.append(str(Path(directory) / filename))
        search_file_counts[str(root)] = count

    counts = lambda key: dict(sorted(Counter(str(row[key]) for row in rows).items()))
    result = {
        'audit_time_asia_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'scope': 'Local metadata and filename inventory; not a temporal/CFD residual test',
        'raw_root': str(RAW), 'processed_root': str(POOL),
        'raw_file_count': len(raw_files),
        'raw_suffix_counts': dict(Counter(p.suffix for p in raw_files)),
        'raw_directories': sorted(str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_dir()),
        'raw_summary': summary,
        'manifest_records': len(manifest), 'source_index_records': len(index),
        'unique_manifest_ids': len({r['sample_id'] for r in manifest}),
        'metadata_field_counts': dict(metadata_keys),
        'frame_index_counts': counts('frame_index'),
        'physical_time_s_counts': counts('physical_time_s'),
        'wind_speed_m_s_counts': counts('wind_speed_m_s'),
        'pressure_pa_counts': counts('pressure_pa'),
        'active_hole_count_counts': counts('active_hole_count'),
        'split_counts': counts('split'),
        'role_counts': counts('role'),
        'frames_per_case_counts': dict(Counter(str(len(v)) for v in field_frames.values())),
        'metadata_mismatches': mismatches,
        'locally_available_source_run_paths': sum(r['source_run_path_available_locally'] for r in rows),
        'locally_available_source_audit_paths': sum(r['source_audit_path_available_locally'] for r in rows),
        'search_roots': [str(p) for p in search_roots],
        'search_pruned_directory_names': sorted(prune),
        'search_file_counts': search_file_counts,
        'cfd_candidate_filenames': cfd_candidates,
        'conclusion': 'insufficient_evidence_to_classify_t2_as_steady_or_unsteady',
        'steady_state_PDE_ready': False,
        'array_content_opened': False, 'model_checkpoint_opened': False,
        'test_array_content_opened': False,
        'metadata_includes_all_splits': True,
        'sources_sha256': sources,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'case_inventory.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (OUT / 'metadata_reads.json').write_text(json.dumps(opened, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in
                     {'sources_sha256', 'raw_summary', 'metadata_field_counts'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
