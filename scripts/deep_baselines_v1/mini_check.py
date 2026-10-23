"""Four-method nonformal integration only; all predictions sealed before Val GT."""
from baseline_context import *
import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['infer', 'score'])
    args = ap.parse_args()
    arms = ('plain', *ARMS)
    for arm in arms:
        target = ROOT / 'evaluation_mini' / arm
        assert c.read(target / 'protocol.json')['mini']
    if args.phase == 'infer':
        for arm in arms:
            subprocess.run([c.sys.executable, str(HERE / 'evaluate.py'), 'infer', '--run', str(ROOT / 'evaluation_mini' / arm)], check=True)
        for arm in arms:
            assert c.read(ROOT / 'evaluation_mini' / arm / 'predictions_sealed.json')['all_predictions_saved_before_GT']
        c.freeze(ROOT / 'checks/four_method_mini_predictions.json', dict(at=c.now(), passed=True, formal=False, methods=list(arms),
            all40_predictions_saved_before_GT=True,
            seals={arm: sha(ROOT / 'evaluation_mini' / arm / 'predictions_sealed.json') for arm in arms}))
    else:
        seal = c.verified(ROOT / 'checks/four_method_mini_predictions.json')
        for arm in arms:
            assert sha(ROOT / 'evaluation_mini' / arm / 'predictions_sealed.json') == seal['seals'][arm]
        for arm in arms:
            for phase in ('gt', 'stats'):
                subprocess.run([c.sys.executable, str(HERE / 'evaluate.py'), phase, '--run', str(ROOT / 'evaluation_mini' / arm)], check=True)
            if arm != 'plain':
                subprocess.run([c.sys.executable, str(HERE / 'evaluate.py'), 'compare', '--run', str(ROOT / 'evaluation_mini' / arm),
                    '--reference-run', str(ROOT / 'evaluation_mini/plain')], check=True)
        c.freeze(ROOT / 'checks/four_method_mini_evaluation.json', dict(at=c.now(), passed=True, formal=False, cases=2, groups=5, draws=1,
            metrics='relative L1/L2, MAE, RMSE, mass/peak amplitude, observed/heldout projection; draw/case aggregation; paired bootstrap',
            no_test=True, no_GT128=True, no_Pool_B=True, no_model_changes_from_GT=True))


if __name__ == '__main__': main()
