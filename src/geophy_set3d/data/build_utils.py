"""Atomic outputs and frozen source fingerprints (never write source files)."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_jsonl(path, values):
    atomic_text(path, ''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))


def save_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, value, allow_pickle=False)
    os.replace(tmp, path)


def save_npz(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **value)
    os.replace(tmp, path)


def source_snapshot(source):
    result = {}
    for p in sorted(Path(source).rglob('*')):
        if p.is_file():
            s = p.stat()
            relative = p.relative_to(source).as_posix()
            entry = dict(size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)
            # Historical inputs are ignored: inventory only, no data read.
            if not relative.startswith('inputs/'):
                entry['sha256'] = digest(p)
            result[relative] = entry
    return result
