"""Registered continuation from inspected epoch checkpoints after process exit.

Original training/evaluation sources stay frozen. Only the original controller's
fresh-start-only scheduling gate is replaced for this explicitly recorded resume.
"""
from runtime import *
import controller as original
import time


def resume_all(s):
    registration = verified(BASE / 'recovery_execution_v1.json')
    assert registration['allow_resume']
    verify_hashes(registration['recovery_sources'])
    pending = list(SEEDS)
    active = {}
    stable = {}
    failures = []
    snapshots = registration['runs']
    while pending or active:
        for seed, item in list(active.items()):
            if not s.done(item):
                continue
            try:
                s.finish(item)
                summary = read(run_protocol(seed) / 'summary.json')
                assert summary['completed'] and summary['epochs'] == 300 and summary['updates'] == 24900
            except Exception:
                failures.append(seed)
            del active[seed]
        if failures:
            pending = []
        snap = s.snapshot()
        occupied = {item['record']['gpu_uuid'] for item in active.values()}
        for device in s.devices(snap):
            uuid = device['uuid']
            stable[uuid] = stable.get(uuid, 0) + 1 if device['idle'] and uuid not in occupied else 0
            candidates = [seed for seed in pending if snapshots[str(seed)]['original_gpu_uuid'] == uuid]
            if not candidates or len(active) >= 2 or stable[uuid] < 2:
                continue
            lock = s.reserve(uuid)
            if lock is None:
                continue
            seed = candidates[0]
            pending.remove(seed)
            target = run_protocol(seed)
            assert sha(target / 'checkpoints/last.pt') == snapshots[str(seed)]['last_checkpoint_sha256']
            verify_hashes(verified(target / 'protocol.json')['sources'])
            active[seed] = s.launch(f'train_seed{seed}', [sys.executable, str(HERE / 'worker.py'), '--target', str(target)], uuid, lock)
            occupied.add(uuid)
        original.progress('training_recovery_v1', pending=pending,
                          active={str(seed): item['record'] for seed, item in active.items()}, failures=failures)
        if pending or active:
            time.sleep(15)
    assert not failures, failures
    evidence = {str(run_protocol(seed) / name): sha(run_protocol(seed) / name)
                for seed in SEEDS for name in ('summary.json', 'metrics.jsonl', 'checkpoints/best.pt', 'checkpoints/last.pt')}
    freeze(BASE / 'training_complete.json', dict(at=now(), passed=True, seeds=list(SEEDS), evidence=evidence,
                                               recovery_execution_sha256=sha(BASE / 'recovery_execution_v1.json')))


if __name__ == '__main__':
    original.train_all = resume_all
    original.main()
