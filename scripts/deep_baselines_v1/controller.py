"""One-shot locked GPU jobs; double idle snapshots, inherited UUID lock, conflict stop."""
from baseline_context import *
import argparse
import fcntl
import os
import signal
import subprocess
import time


def launch(s, key, command, uuid, lock):
    old_path = s.task_path(key)
    old = c.read(old_path) if old_path.exists() else None
    attempt = 1 if old is None else old['attempt'] + 1
    log = ROOT / 'scheduling' / f'{key}_attempt{attempt}.log'
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=uuid)
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   pass_fds=(lock.fileno(),), start_new_session=True)
    record = dict(key=key, state='running', attempt=attempt, command=command, started_at=c.now(),
                  gpu_uuid=uuid, pid=process.pid, process_group=process.pid, log=str(log), controller_pid=os.getpid())
    c.write(old_path, record); c.append(ROOT / 'scheduling/events.jsonl', record)
    return dict(record=record, process=process, lock=lock)


def terminate_owned_group(process):
    # The child starts a fresh session: this group contains this task and its
    # loader/nested-check children only, never another user's process.
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait()
    # Leader exit is insufficient if a nested child ignores TERM. The group
    # belongs exclusively to this task, so clear any remaining descendants.
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass


def run(key, command, recover_reason=None):
    ROOT.joinpath('scheduling').mkdir(exist_ok=True)
    task_lock = (ROOT / 'scheduling' / (key + '.controller.lock')).open('a')
    fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    s = c.scheduler(); s.BASE = ROOT; s.progress = lambda: None
    prior_path = ROOT / 'scheduling' / (key + '.json')
    if prior_path.exists():
        prior = c.read(prior_path)
        assert prior['state'] != 'complete', 'Completed job must not be repeated'
        assert not s.alive(prior['pid']), 'Existing task process is still alive'
        assert recover_reason, 'Inspect failed task and provide --recover-reason before resuming'
        c.append(ROOT / 'scheduling/recovery.jsonl', dict(at=c.now(), key=key, prior=prior,
            inspected=True, reason=recover_reason, action='resume authoritative last.pt; no config/budget change'))
    # Existing global UUID lock namespace and 15-second double snapshots.
    uuid, lock = s.wait_gpu()
    item = launch(s, key, command, uuid, lock)
    process = item['process']
    conflict = None
    def stop_signal(signum, frame):
        raise KeyboardInterrupt('Controller received signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop_signal)
    try:
        while process.poll() is None:
            snapshot = s.snapshot()
            foreign = []
            for line in snapshot['gpu_processes'].splitlines():
                fields = [x.strip() for x in line.split(',')]
                if fields[0] != uuid or len(fields) < 2:
                    continue
                pid = int(fields[1])
                try: own_group = os.getpgid(pid) == process.pid
                except ProcessLookupError: continue
                if not own_group:
                    foreign.append(dict(pid=pid, process=line))
            if foreign:
                conflict = foreign
                c.append(ROOT / 'scheduling/events.jsonl', dict(at=c.now(), event='foreign_gpu_conflict', key=key, foreign=foreign,
                    action='terminate only own child; preserve checkpoint and replay interrupted epoch'))
                terminate_owned_group(process)
                break
            time.sleep(15)
        s.finish(item, valid=process.returncode == 0 and conflict is None)
    finally:
        if process.poll() is None:
            terminate_owned_group(process)
            record = dict(item['record'], state='interrupted', finished_at=c.now(), exit_code=process.returncode)
            c.write(s.task_path(key), record)
            c.append(ROOT / 'scheduling/events.jsonl', record)
        lock.close()
        task_lock.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True)
    ap.add_argument('--recover-reason')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    assert a.command
    run(a.key, a.command[1:] if a.command[0] == '--' else a.command, a.recover_reason)
