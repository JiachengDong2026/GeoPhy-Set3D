#!/usr/bin/env python3
"""Exactly three screening candidates, then one independently initialized B3 run."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
import yaml
from geophy_set3d.training.utils import load_config, write_json, directories
from geophy_set3d.training.measurement_runs import train_run
from geophy_set3d.evaluation.protocols import load_model
from geophy_set3d.evaluation.measurement_consistency import evaluate_measurements

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/b2_vs_b3_phys_v1')
B2 = Path('/data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_b2_gcsi_300/best.pt')
CANDIDATES = dict(weak=(.05, .025), medium=(.1, .05), strong=(.2, .1))


def config_for(name=None):
    config = copy.deepcopy(load_config(REPO/'configs/train_supervised_b2_300.yaml'))
    config['train']['hold_views'] = config['validation']['hold_views'] = 1
    config['physics'] = dict(lambda_obs=0., lambda_hold=0., scale_ppm_m=32284.894580078173,
                             beta=.01, projector_samples=96, schedule='warmup_ramp')
    config['experiment']['name'] = 'supervised_b3_gcsi_phys_300'
    if name:
        wo, wh = CANDIDATES[name]
        config['experiment']['name'] = 'b3_lambda_gate_'+name
        config['training']['max_epochs'] = 20
        config['optimizer']['lr'] = 5e-5
        config['scheduler'] = dict(name='constant')
        config['physics'].update(lambda_obs=wo, lambda_hold=wh, schedule='constant')
    return config


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    for d in ('lambda_gate', 'formal_run', 'figures'):
        (ROOT/d).mkdir(exist_ok=True)
    if (ROOT/'inventory_before.json').exists():
        return
    cfg = load_config(REPO/'configs/train_supervised_b2_300.yaml')
    dataset = Path(cfg['dataset']['root'])
    paths = list(p for p in dataset.rglob('*') if p.is_file())
    # Only the pre-B3 tracked repository tree belongs to the frozen reference.
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '12fb245'], cwd=REPO, text=True).splitlines()
    paths += [REPO/n for n in names]
    out = Path(cfg['experiment']['output_root'])
    for directory in [out/'comparisons/b0_b1_b2_300', out/'audits/representation_capacity_64_v1',
                      out/'checkpoints/supervised_b2_gcsi_300', out/'logs/supervised_b2_gcsi_300',
                      out/'figures/supervised_b2_gcsi_300']:
        paths += [p for p in directory.rglob('*') if p.is_file()]
    write_json(ROOT/'inventory_before.json', {str(p): dict(sha256=digest(p), size=p.stat().st_size,
                                                         mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(paths))})
    protocol = dict(seed=20261011, candidates=CANDIDATES, gate_epochs=20,
        gate_initial_checkpoint=str(B2), gate_lr=5e-5, gate_checkpoint='final_epoch_20',
        gate_voxel_worsening_budget_absolute=.005, gate_close_heldout_l1_absolute=.001,
        gate_physics_gradient_ratio_limit=2., gate_requires_heldout_improvement=True,
        gate_requires_observed_improvement=True,
        formal_random_initialization=True, formal_epochs=300, model_parameters=319042,
        formal_checkpoint_selection='minimum fixed Val-A-V4 supervised loss',
        formal_nonmaterial_voxel_worsening_absolute=.005,
        final_meaningful_voxel_improvement_absolute=.002,
        final_meaningful_heldout_improvement_relative=.02,
        final_shortcut_worsening_ratio=2., final_wrong_hold_worsening_ratio=1.5,
        hold_sampling='independent complement; legacy input station IDs unchanged',
        projection_aggregation='mean per-case valid-ray relative L1/L2',
        projection_training_loss='per-case valid-ray SmoothL1(raw/32284.894580078173, beta=.01), then batch mean',
        schedule='1-20 zero; 21=zero,40=target, linear in between;41+ target',
        samples=96, pool_b_training=False, test_inference=False,
        device='cuda:3', precision='fp32', batch_size=2, ray_subsampling=False)
    (ROOT/'effective_protocol.yaml').write_text(yaml.safe_dump(protocol, sort_keys=False))
    for name in CANDIDATES:
        (REPO/f'configs/train_b3_gate_{name}.yaml').write_text(yaml.safe_dump(config_for(name), sort_keys=False))
    (REPO/'configs/train_supervised_b3_phys_300.yaml').write_text(yaml.safe_dump(config_for(), sort_keys=False))


def save_evaluation(model, config, path, **options):
    result, rows = evaluate_measurements(model, config, **options)
    path.mkdir(parents=True, exist_ok=True)
    write_json(path/'metrics.json', result)
    (path/'per_case_metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    return result


def gate(resume=False):
    config = load_config(REPO/'configs/train_supervised_b2_300.yaml')
    baseline_path = ROOT/'lambda_gate/b2_baseline/metrics.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
    else:
        model, _ = load_model(B2, config['training']['device'])
        baseline = save_evaluation(model, config, baseline_path.parent, wrong_hold=True)
        del model; torch.cuda.empty_cache()
    results = {}
    for name, (wo, wh) in CANDIDATES.items():
        candidate_config = load_config(REPO/f'configs/train_b3_gate_{name}.yaml')
        paths = train_run(candidate_config, initial_checkpoint=B2, resume=resume)
        model, checkpoint = load_model(paths['checkpoints']/'last.pt', config['training']['device'])
        assert checkpoint['epoch'] == 19
        metrics = save_evaluation(model, config, ROOT/'lambda_gate'/name)
        gradient = json.loads((paths['logs']/'gradient_diagnostics.json').read_text())
        worsening = metrics['voxel']['mean_per_case']['relative_l1']-baseline['voxel']['mean_per_case']['relative_l1']
        improvement = baseline['heldout']['relative_l1']-metrics['heldout']['relative_l1']
        eligible = (worsening <= .005 and improvement > 0 and
                    metrics['observed']['relative_l1'] < baseline['observed']['relative_l1'] and
                    gradient['physics_to_supervised_ratio'] <= 2.)
        results[name] = dict(lambda_obs=wo, lambda_hold=wh, metrics=metrics, gradient=gradient,
                             voxel_worsening_absolute=worsening, heldout_improvement_absolute=improvement,
                             eligible=eligible, checkpoint=str(paths['checkpoints']/'last.pt'))
        write_json(ROOT/'lambda_gate/progress.json', results)
        del model; torch.cuda.empty_cache()
    eligible = [n for n in CANDIDATES if results[n]['eligible']]
    selected = None
    if eligible:
        largest = max(results[n]['heldout_improvement_absolute'] for n in eligible)
        selected = next(n for n in eligible if largest-results[n]['heldout_improvement_absolute'] <= .001)
    summary = dict(baseline=baseline, candidates=results, selected=selected, gate_passed=selected is not None)
    write_json(ROOT/'lambda_gate/summary.json', summary)
    if selected:
        config = config_for()
        config['physics'].update(lambda_obs=CANDIDATES[selected][0], lambda_hold=CANDIDATES[selected][1])
        (REPO/'configs/train_supervised_b3_phys_300.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(summary), flush=True)


def formal(resume=False):
    gate_summary = json.loads((ROOT/'lambda_gate/summary.json').read_text())
    if not gate_summary['gate_passed']:
        raise RuntimeError('NO-GO: all lambda candidates failed; full training forbidden')
    config = load_config(REPO/'configs/train_supervised_b3_phys_300.yaml')
    chosen = CANDIDATES[gate_summary['selected']]
    if (config['physics']['lambda_obs'], config['physics']['lambda_hold']) != chosen:
        raise ValueError('Formal lambda differs from frozen gate selection')
    train_run(config, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'gate', 'formal', 'evaluate', 'report'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'prepare':
        freeze()
    elif args.stage == 'gate':
        gate(args.resume)
    elif args.stage == 'formal':
        formal(args.resume)
    else:
        from geophy_set3d.evaluation.b3_report import final_evaluation, render_report
        (final_evaluation if args.stage == 'evaluate' else render_report)(ROOT, REPO)


if __name__ == '__main__':
    main()
