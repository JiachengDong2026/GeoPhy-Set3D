#!/usr/bin/env python3
"""Train the independently registered v4 budget variant; no implicit formal run."""
import argparse
import copy
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from plan_a_budget_v4 import (PROTOCOL, budget_provenance, build_budget_model,
                              configure_backend, load_budget_config)


def install_budget(config):
    import geophy_set3d.models.set_unrolled3d as models
    import geophy_set3d.training.research_selfsup as training
    original_seed = training.seed_everything
    original_provenance = training._provenance
    original_acceptance = training._validate_acceptance
    original_write = training.write_json
    runtime = copy.deepcopy(config['execution_runtime'])
    architecture = copy.deepcopy(config['architecture'])
    training.PROTOCOL = PROTOCOL

    def seed(value):
        original_seed(value)
        configure_backend(runtime)

    def build(current):
        if current['architecture'] != architecture or current['execution_runtime'] != runtime:
            raise ValueError('Budget architecture/runtime changed within a process')
        return build_budget_model(current)

    def provenance(current, train, val):
        return budget_provenance(current, train, val, original_provenance)

    def acceptance(current, prov):
        if current['stage'] == 'formal':
            for kind in ('smoke', 'gate'):
                directory = Path(current.get('acceptance', {}).get(kind + '_run', ''))
                prior = yaml.safe_load((directory / 'config_resolved.yaml').read_text())
                previous = json.loads((directory / 'provenance.json').read_text())
                if any(prior.get(key) != current.get(key) for key in ('protocol', 'architecture', 'execution_runtime', 'geometry_cache')):
                    raise ValueError(f'{kind} used a different budget architecture/runtime/protocol')
                if previous.get('protocol') != PROTOCOL or previous.get('execution_runtime') != runtime or previous.get('architecture') != architecture:
                    raise ValueError(f'{kind} lacks matching budget provenance')
        original_acceptance(current, prov)

    def write(path, value):
        if Path(path).name == 'environment.json':
            value = dict(value, precision='fp32_parameters_physics_' + runtime['network_precision'] + '_network',
                         architecture=architecture, execution_runtime=runtime)
        original_write(path, value)

    models.build_set_unrolled_model = build
    training.seed_everything = seed
    training._provenance = provenance
    training._validate_acceptance = acceptance
    training.write_json = write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    config = load_budget_config(args.config)
    if args.device:
        config['training']['device'] = args.device
    if args.validate_only:
        print('Valid independent budget config:', args.config)
        return
    install_budget(config)
    from geophy_set3d.training import research_selfsup as training
    if args.preflight_only:
        from geophy_set3d.training.gt_read_guard import GTReadGuard
        with GTReadGuard(config['dataset']['root'], Path('/tmp/plan_a_budget_unused_checkpoints')) as guard:
            train = training.make_research_dataset(config)
            val = training.make_research_dataset(config, validation=True)
            prov = training._provenance(config, train, val)
            training._validate_acceptance(config, prov)
            print(json.dumps(dict(passed=True, protocol=PROTOCOL, source_sha256=prov['source_sha256'],
                                  gt_read_audit=guard.summary(), optimizer_updates=0)))
        return
    print(training.train_research(config, resume=args.resume))


if __name__ == '__main__':
    main()
