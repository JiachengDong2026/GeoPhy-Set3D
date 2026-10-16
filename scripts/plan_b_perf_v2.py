"""Isolated FP32 runtime for B: identical network/loss/Q96, larger work blocks.

Load the reference trainer in its own module namespace so the frozen A/B
implementations and their Python globals remain untouched. Every new runtime
run records both the reference sources and this adapter in its provenance.
"""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'scheme_b_continuous_measurement_only_perf_v2'
POLICY_PATH = ROOT/'protocols/plan_b_perf_v2/qualification_policy.json'


def qualification(run):
    """Derive guide qualification without changing the strict historical result."""
    run = Path(run)
    policy = json.loads(POLICY_PATH.read_text())
    names = ('acceptance_result.json', 'config_resolved.yaml', 'provenance.json',
             'gt_read_audit.json', 'training_summary.json')
    result = json.loads((run/names[0]).read_text())
    config = yaml.safe_load((run/names[1]).read_text())
    provenance = json.loads((run/names[2]).read_text())
    audit = json.loads((run/names[3]).read_text())
    summary = json.loads((run/names[4]).read_text())
    stage = config['stage']
    if stage not in ('smoke', 'gate'):
        raise ValueError('Guide qualification applies only to real smoke/gate')
    required = policy[stage+'_required']
    checks = {key: result.get('criteria', {}).get(key) is True for key in required}
    expected_cases = 2 if stage == 'smoke' else 16
    valid = (result.get('completed') is True and result.get('synthetic') is False
             and result.get('stage') == stage and config['dataset']['kind'] == 'projection_pool'
             and config['protocol'] == provenance['protocol'] == PROTOCOL
             and len(provenance['train_case_ids']) == config['dataset']['train_limit'] == expected_cases
             and summary.get('completed') is True
             and summary.get('completed_epochs') == config['training']['max_epochs']
             and all(audit.get(key, 1) == 0 for key in
                     ('gt_open_attempts', 'forbidden_npy_attempts', 'supervised_checkpoint_open_attempts')))
    if stage == 'smoke':
        valid = valid and summary.get('optimizer_updates') == config['training']['max_epochs'] == 20
    else:
        valid = valid and result.get('passed') is True
    engine = runtime_engine()
    return dict(policy=policy['policy'], policy_sha256=engine._sha(POLICY_PATH),
        stage=stage, passed=bool(valid and all(checks.values())),
        completed_real_run=bool(valid), required_criteria=checks,
        strict_acceptance_passed=result.get('passed'),
        diagnostic_only={key: result.get('criteria', {}).get(key) for key in policy.get(stage+'_diagnostic_only', [])},
        source_sha256=provenance['source_sha256'], data_sha256=provenance['data_sha256'],
        evidence_sha256={name: engine._sha(run/name) for name in names})


def write_qualification(run):
    value = qualification(run)
    runtime_engine().write_json(Path(run)/'qualification_result.json', value)
    return value


@lru_cache(maxsize=1)
def runtime_engine():
    spec = importlib.util.spec_from_file_location(
        '_geophy_plan_b_perf_v2_engine', ROOT/'research/plan_b/training.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.PROTOCOL = PROTOCOL
    engine.MODEL_SPEC = dict(engine.MODEL_SPEC, query_chunk=196608,
                             ray_chunk=2048, checkpoint_query=False)
    reference_provenance = engine._provenance

    def provenance(config, train, val):
        value = reference_provenance(config, train, val)
        for name in ('scripts/plan_b_perf_v2.py', 'scripts/train_plan_b_perf.py',
                     'protocols/plan_b_perf_v2/qualification_policy.json'):
            value['code'][name] = engine._sha(ROOT/name)
        value['source_sha256'] = engine._canonical(value['code'])
        value['execution_runtime'] = dict(precision='fp32', query_chunk=196608,
            ray_chunk=2048, checkpoint_query=False, reference_sources_preserved=True)
        value['qualification_policy_sha256'] = engine._sha(POLICY_PATH)
        return value

    def acceptance(config, current):
        if config['stage'] != 'formal':
            return
        for kind in ('smoke', 'gate'):
            directory = Path(config.get('acceptance', {}).get(kind+'_run', ''))
            recorded = json.loads((directory/'qualification_result.json').read_text())
            expected = qualification(directory)
            if recorded != expected or not expected['passed'] or expected['stage'] != kind:
                raise ValueError(f'Formal B requires completed, qualified real-data {kind}')
            prior_config = yaml.safe_load((directory/'config_resolved.yaml').read_text())
            prior = json.loads((directory/'provenance.json').read_text())
            if prior_config['model'] != config['model'] or prior_config['dataset']['root'] != config['dataset']['root']:
                raise ValueError(f'Formal {kind} architecture/data differs')
            if any(prior.get(k) != current.get(k) for k in
                   ('protocol', 'source_sha256', 'git_commit', 'qualification_policy_sha256')):
                raise ValueError(f'Formal {kind} source/protocol/policy differs')
            if any(current['data']['files'].get(path) != sha for path, sha in prior['data']['files'].items()):
                raise ValueError(f'Formal {kind} data/statistics/geometry differs')

    engine._provenance = provenance
    engine._validate_acceptance = acceptance
    return engine


def load_perf_config(path):
    return runtime_engine().validate_config(yaml.safe_load(Path(path).read_text()))
