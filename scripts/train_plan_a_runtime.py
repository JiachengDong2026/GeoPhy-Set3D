#!/usr/bin/env python3
"""Opt-in FP32 execution launcher with separate provenance and acceptance.

v2 training sources/configs remain frozen. Only the tested compiled-prior plus
cuDNN autotune candidate is eligible here; reduced precision stays experimental.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import yaml
from plan_a_runtime_candidates import configure_backend,configure_model

RUNTIME=dict(network_layout='contiguous',cudnn_autotune=True,compile_prior=True,network_precision='fp32')


def load_runtime_config(path):
    from geophy_set3d.training.research_selfsup import validate_research_config
    config=yaml.safe_load(Path(path).read_text())
    if config.get('status')!='runtime_executable_v3' or config.get('protocol')!='scheme_a_runtime_fp32_v3':
        raise ValueError('Use a separately registered v3 runtime config')
    if config.get('execution_runtime')!=RUNTIME:
        raise ValueError('Only the qualified FP32 compile/autotune execution is supported')
    config=copy.deepcopy(config)
    config.update(status='executable',protocol='scheme_a_measurement_only_v1',entrypoint_required='scripts/train_plan_a_runtime.py')
    if config['model']['variant']!='A2' or config['model']['checkpoint_prior']:
        raise ValueError('Runtime v3 requires A2 without prior checkpointing')
    return validate_research_config(config)


def install_runtime():
    """Process-local installation; additional source hashes prevent v2 reuse."""
    import geophy_set3d.models.set_unrolled3d as models
    import geophy_set3d.training.research_selfsup as training
    original_build=models.build_set_unrolled_model
    original_seed=training.seed_everything
    original_provenance=training._provenance
    original_acceptance=training._validate_acceptance

    def build(config):
        if config.get('execution_runtime')!=RUNTIME:
            raise ValueError('Missing or changed runtime configuration')
        return configure_model(original_build(config),RUNTIME)

    def seed(value):
        original_seed(value)
        configure_backend(RUNTIME)

    def provenance(config,train,val):
        value=original_provenance(config,train,val)
        for name in ('scripts/train_plan_a_runtime.py','scripts/plan_a_runtime_candidates.py'):
            value['code'][name]=training._sha(ROOT/name)
        value['source_sha256']=training._canonical(value['code'])
        value['execution_runtime']=copy.deepcopy(RUNTIME)
        return value

    def acceptance(config,prov):
        if config['stage']=='formal':
            for kind in ('smoke','gate'):
                path=Path(config.get('acceptance',{}).get(kind+'_run',''))
                prior=yaml.safe_load((path/'config_resolved.yaml').read_text())
                if prior.get('execution_runtime')!=config['execution_runtime']:
                    raise ValueError(f'{kind} used a different execution runtime')
                previous=json.loads((path/'provenance.json').read_text())
                if previous.get('execution_runtime')!=RUNTIME:
                    raise ValueError(f'{kind} lacks runtime provenance')
        original_acceptance(config,prov)

    models.build_set_unrolled_model=build
    training.seed_everything=seed
    training._provenance=provenance
    training._validate_acceptance=acceptance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--device')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    config=load_runtime_config(args.config)
    if args.device:
        config['training']['device']=args.device
    if args.validate_only:
        print('Valid separate FP32 runtime config:',args.config)
        return
    install_runtime()
    from geophy_set3d.training import research_selfsup as training
    if args.preflight_only:
        from geophy_set3d.training.gt_read_guard import GTReadGuard
        with GTReadGuard(config['dataset']['root'],Path('/tmp/plan_a_runtime_unused_checkpoints')) as guard:
            train=training.make_research_dataset(config)
            val=training.make_research_dataset(config,validation=True)
            prov=training._provenance(config,train,val)
            training._validate_acceptance(config,prov)
            print(json.dumps(dict(passed=True,source_sha256=prov['source_sha256'],gt_read_audit=guard.summary(),optimizer_updates=0)))
        return
    print(training.train_research(config,resume=args.resume))


if __name__=='__main__':
    main()
