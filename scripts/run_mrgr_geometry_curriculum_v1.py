#!/usr/bin/env python3
"""Serialization-only adapter for registered geometry curriculum (no numeric changes)."""
import functools
import json
from pathlib import Path
import numpy as np
import probe_mrgr_geometry_curriculum_v1 as probe

def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Cannot JSON encode {type(value)}')

if __name__=='__main__':
    path=probe.OUT/'serialization_amendment.json'
    payload=dict(registered_at=probe.base.now(),reason='Smoke completed numerical checks but JSON serialization rejected numpy.bool_. Convert numpy scalar types to Python scalars only; no changes to numerical code or registered protocol.',
        before_any_optimizer_updates=True,runner_sha256=probe.base.digest(Path(__file__)),
        protocol_sha256=probe.base.digest(probe.OUT/'protocol.json'),numerical_script_unchanged=True)
    if not path.exists():
        assert not any((probe.OUT/a/'metrics.jsonl').exists() for a in probe.ARMS)
        probe.base.write_json(path,payload)
    else:
        saved=json.loads(path.read_text())
        assert saved['runner_sha256']==payload['runner_sha256'] and saved['protocol_sha256']==payload['protocol_sha256']
    json.dumps=functools.partial(json.dumps,default=scalar)
    probe.main()
