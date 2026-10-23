"""Use the unmodified formal training and epoch-commit recovery engine."""
from common_stage import *


def run(target,stop=300):
    target=Path(target);p=verified(target/'protocol.json')
    assert (p['arm'],p['seed']) in TASKS
    if not p['subset_smoke']:
        assert stop==300
        assert verified(BASE/'checks/cpu.json')['passed'] and verified(BASE/'checks/gpu.json')['passed']
        assert verified(BASE/'checks/resume.json')['passed']
    engine=canonical_worker.engine;engine.OUT=BASE;engine.model_for=model_for
    with TrainGuard() as guard:
        engine.train(target,stop)
        assert not guard.denied
        write(target/'boundary_audit.json',guard.summary())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--stop',type=int,default=300);a=p.parse_args();run(a.target,a.stop)
