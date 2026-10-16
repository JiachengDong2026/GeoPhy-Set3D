#!/usr/bin/env python3
"""Persistently run B qualification and authorized formal training in sequence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', required=True, type=Path)
    args = parser.parse_args()
    path = args.launch.resolve()
    state = json.loads(path.read_text())
    repo = Path(state['repo'])

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(path)

    def sha(file):
        return hashlib.sha256(Path(file).read_bytes()).hexdigest()

    def frozen_sources():
        manifest = json.loads((repo/state['source_manifest']).read_text())
        if manifest['source_sha256'] != state['source_sha256']:
            raise ValueError('Frozen source manifest changed')
        for name, expected in manifest['files'].items():
            if sha(repo/name) != expected:
                raise ValueError(f'Frozen source changed: {name}')
        if {str(p.relative_to(repo)) for p in (repo/'src').rglob('*.py')} != {
                p for p in manifest['files'] if p.startswith('src/')}:
            raise ValueError('Frozen src membership changed')

    try:
        state.update(status='running', supervisor_pid=os.getpid())
        save()
        for phase in state['phases']:
            frozen_sources()
            if sha(repo/phase['config']) != phase['config_sha256']:
                raise ValueError('Phase configuration changed')
            # Bind physical GPU1 explicitly; every child uses the physical cuda:1 index.
            gpu = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid,memory.free',
                                           '--format=csv,noheader,nounits'], text=True).strip().split(',')
            if gpu[0].strip() != state['gpu_uuid']:
                raise ValueError('Physical GPU1 identity changed')
            if int(gpu[1].strip()) < 15*1024:
                raise RuntimeError('GPU1 has less than 15 GiB free for the validated runtime')
            state['current_phase'] = phase['name']
            phase.update(status='starting', started_at=now())
            save()
            env = os.environ.copy()
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            with (path.parent/phase['log']).open('ab', buffering=0) as log:
                child = subprocess.Popen(phase['command'], cwd=repo, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                phase.update(status='running', trainer_pid=child.pid)
                state['trainer_pid'] = child.pid
                save()
                returncode = child.wait()
            phase.update(returncode=returncode, ended_at=now())
            if returncode:
                phase['status'] = 'failed'
                raise RuntimeError(f"{phase['name']} exited {returncode}; see {phase['log']}")
            run = repo/phase['run_dir']
            summary = json.loads((run/'training_summary.json').read_text())
            if not summary.get('completed') or summary.get('completed_epochs') != phase['epochs']:
                phase['status'] = 'failed'
                raise RuntimeError(f"{phase['name']} did not complete its configured epochs")
            phase.update(completed_epochs=summary['completed_epochs'],
                         optimizer_updates=summary['optimizer_updates'])
            if phase['name'] in ('smoke', 'gate'):
                qualified = json.loads((run/'qualification_result.json').read_text())
                phase.update(qualification_passed=qualified['passed'],
                             strict_acceptance_passed=qualified['strict_acceptance_passed'])
                if not qualified['passed']:
                    phase['status'] = 'failed'
                    raise RuntimeError(f"{phase['name']} guide qualification failed; formal run not started")
            phase['status'] = 'completed'
            save()
        frozen_sources()
        state.update(status='completed', ended_at=now())
        save()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), ended_at=now())
        save()
        raise


if __name__ == '__main__':
    main()
