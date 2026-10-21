"""Registered post-training continuation; fix duplicate seed in status only."""
from runtime import *
import fcntl
import traceback


def progress(state, **extra):
    runs = [{**read(run_protocol(seed) / 'status.json'), 'seed': seed} for seed in SEEDS]
    write(BASE / 'status.json', dict(at=now(), state=state, controller_pid=os.getpid(), runs=runs, **extra))


def main():
    registration = verified(BASE / 'post_training_recovery_v2.json')
    verify_hashes(registration['sources'])
    verify_hashes(verified(BASE / 'protocol.json')['sources'])
    verify_hashes(verified(BASE / 'analysis_source_prelock.json')['sources'])
    with (BASE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for seed in SEEDS:
                summary = read(run_protocol(seed) / 'summary.json')
                assert summary['completed'] and summary['epochs'] == 300 and summary['updates'] == 24900
                for key in ('best', 'last'):
                    assert sha(run_protocol(seed) / f'checkpoints/{key}.pt') == summary[f'{key}_checkpoint_sha256']
            evidence = {str(run_protocol(seed) / name): sha(run_protocol(seed) / name)
                        for seed in SEEDS for name in ('summary.json', 'metrics.jsonl', 'checkpoints/best.pt', 'checkpoints/last.pt')}
            freeze(BASE / 'training_complete.json', dict(at=now(), passed=True, seeds=list(SEEDS), evidence=evidence,
                   recovery_execution_sha256=sha(BASE / 'recovery_execution_v1.json'),
                   post_training_recovery_sha256=sha(BASE / 'post_training_recovery_v2.json')))
            s = scheduler()
            progress('training_complete_preparing_evaluation')
            with (BASE / 'scheduling/evaluation.log').open('x') as log:
                subprocess.run([sys.executable, str(HERE / 'evaluate.py'), 'register'], cwd=PROJECT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            for seed in SEEDS:
                s.run_gpu_once(f'predict_seed{seed}', [sys.executable, str(HERE / 'evaluate.py'), 'predict', '--seed', str(seed)],
                               BASE / 'evaluation' / str(seed) / 'complete.json')
            s.run_gpu_once('benchmark', [sys.executable, str(HERE / 'evaluate.py'), 'benchmark'], BASE / 'benchmark.json')
            progress('sealing_and_scoring')
            with (BASE / 'scheduling/scoring.log').open('x') as log:
                subprocess.run([sys.executable, str(HERE / 'evaluate.py'), 'score'], cwd=PROJECT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            assert verified(BASE / 'acceptance.json')['completed']
            progress('complete_pending_execution_review')
        except BaseException:
            progress('failed', traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
