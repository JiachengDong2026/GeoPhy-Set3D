"""Inspected resource-conflict recovery; GPU0 only, unchanged frozen train workers."""
from baseline_context import *
import argparse
import fcntl
import signal
import subprocess

ALLOWED_UUID = 'GPU-37424437-e208-62d1-e617-53c421c94a49'
REASON = ('Inspected 2026-10-24 GPU3 gap between yl jobs: foreign PID1377578 resumed after double idle snapshots; '
          'our controller stopped only its own process group. Queue fail-stop interrupted seed11; '
          'resume authoritative epoch commit with RNG/AdamW/cosine and exact original300/24900 budget. '
          'Restrict scheduling to GPU0; other UUIDs are reserved for ongoing external multi-job workflows.')


def worker(target):
    import controller
    original_scheduler = c.scheduler
    def restricted_scheduler():
        s = original_scheduler()
        original_devices = s.devices
        s.devices = lambda snapshot: [d for d in original_devices(snapshot) if d['uuid'] == ALLOWED_UUID]
        return s
    # Process-local scheduling filter only; no numerical/shared source changes.
    c.scheduler = restricted_scheduler
    key = target.name + '_formal300'
    command = [c.sys.executable, str(HERE / 'run.py'), 'train', '--target', str(target)]
    controller.run(key, command, recover_reason=REASON)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--target')
    args = ap.parse_args()
    if args.target:
        worker(Path(args.target)); return
    lock = (ROOT / 'inspected_recovery_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = c.verified(ROOT / 'formal_queue_plan_v2.json')
    targets = [Path(j['target']) for j in plan['jobs']]
    # Continue the committed run first, then the two independent DeepSets runs.
    targets.sort(key=lambda p: (p.name != 'gnot_seed20261011', p.name))
    active = None
    def terminate(signum, frame): raise KeyboardInterrupt('Recovery queue signal ' + str(signum))
    signal.signal(signal.SIGTERM, terminate)
    try:
        for target in targets:
            p = c.verified(target / 'protocol.json'); c.verify_hashes(p['sources'])
            assert p['epochs'] == 300 and p['planned_updates'] == 24900 and not p['subset_smoke']
            if (target / 'summary.json').exists():
                summary = c.read(target / 'summary.json')
                assert summary['completed'] and summary['updates'] == 24900
                continue
            command = [c.sys.executable, __file__, '--target', str(target)]
            log = ROOT / 'scheduling' / (target.name + '_inspected_recovery_controller.log')
            with log.open('x') as stream:
                active = subprocess.Popen(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
            c.write(ROOT / 'queue_status.json', dict(at=c.now(), state='active_inspected_recovery', controller_pid=c.os.getpid(),
                active_controller_pid=active.pid, active_target=str(target), allowed_gpu_uuid=ALLOWED_UUID,
                all_targets=list(map(str, targets)), reason=REASON, source_sha256=sha(__file__), plan_sha256=sha(ROOT / 'formal_queue_plan_v2.json')))
            code = active.wait()
            assert code == 0, 'Recovery failed; inspect before retry: ' + target.name
            assert c.read(target / 'summary.json')['updates'] == 24900
            active = None
    finally:
        if active is not None and active.poll() is None:
            active.terminate(); active.wait(timeout=45)
    c.freeze(ROOT / 'inspected_recovery_queue_complete.json', dict(at=c.now(), completed=True, source_sha256=sha(__file__)))
    c.write(ROOT / 'queue_status.json', dict(at=c.now(), state='complete', plan_sha256=sha(ROOT / 'formal_queue_plan_v2.json')))


if __name__ == '__main__': main()
