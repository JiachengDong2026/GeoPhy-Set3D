"""Runtime isolation and provenance guard against invalidating frozen A/B runs."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

import research.plan_b.training as reference

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('plan_b_perf_adapter_tests', ROOT/'scripts/plan_b_perf_v2.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_isolated_runtime_keeps_reference_validation_and_parameter_schema():
    before = copy.deepcopy(reference.MODEL_SPEC)
    engine = adapter.runtime_engine()
    assert reference.PROTOCOL == 'scheme_b_continuous_measurement_only_v1'
    assert engine is not reference
    assert reference.MODEL_SPEC == before
    config = adapter.load_perf_config(ROOT/'configs/research/scheme_b_perf_v2_gate.yaml')
    assert engine.MODEL_SPEC['ray_chunk'] == 2048
    assert not engine.MODEL_SPEC['checkpoint_query']
    with pytest.raises(ValueError):
        reference.validate_config(config)
    old = yaml.safe_load((ROOT/'configs/research/scheme_b_gate_gpu1_extended30.yaml').read_text())
    reference.validate_config(old)
    a, b = reference.build_model(old), engine.build_model(config)
    assert {k: v.shape for k,v in a.state_dict().items()} == {k: v.shape for k,v in b.state_dict().items()}
    b.load_state_dict(a.state_dict(), strict=True)
    changed = copy.deepcopy(config)
    changed['model']['samples'] = 48
    with pytest.raises(ValueError):
        engine.validate_config(changed)


def test_runtime_provenance_includes_adapter_and_policy():
    engine = adapter.runtime_engine()
    config = yaml.safe_load((ROOT/'configs/research/scheme_b_synthetic_smoke.yaml').read_text())
    config.update(protocol=adapter.PROTOCOL, model=engine.MODEL_SPEC.copy())
    engine.validate_config(config)
    train = engine.make_research_dataset(config)
    val = engine.make_research_dataset(config, True)
    provenance = engine._provenance(config, train, val)
    for name in ('research/plan_b/model.py', 'research/plan_b/training.py',
                 'scripts/plan_b_perf_v2.py', 'scripts/train_plan_b_perf.py'):
        assert provenance['code'][name] == engine._sha(ROOT/name)
    assert provenance['protocol'] == adapter.PROTOCOL
    assert provenance['source_sha256'] == engine._canonical(provenance['code'])
    assert provenance['qualification_policy_sha256'] == engine._sha(adapter.POLICY_PATH)


def evidence(tmp_path, stage):
    engine = adapter.runtime_engine()
    policy = json.loads(adapter.POLICY_PATH.read_text())
    directory = tmp_path/stage
    directory.mkdir()
    config = yaml.safe_load((ROOT/f'configs/research/scheme_b_perf_v2_{stage}.yaml').read_text())
    criteria = {key: True for key in policy['gate_required']}
    criteria['heldout_better_than_zero'] = stage == 'gate'
    result = dict(passed=stage=='gate', completed=True, synthetic=False, stage=stage, criteria=criteria)
    provenance = dict(protocol=adapter.PROTOCOL, source_sha256='test-source', data_sha256='test-data',
        git_commit='test-commit', qualification_policy_sha256=engine._sha(adapter.POLICY_PATH),
        train_case_ids=list(range(2 if stage=='smoke' else 16)), data=dict(files={}))
    (directory/'config_resolved.yaml').write_text(yaml.safe_dump(config))
    for name, value in [('acceptance_result',result), ('provenance',provenance),
        ('gt_read_audit',dict(gt_open_attempts=0, forbidden_npy_attempts=0, supervised_checkpoint_open_attempts=0)),
        ('training_summary',dict(completed=True, completed_epochs=config['training']['max_epochs'], optimizer_updates=20 if stage=='smoke' else 240))]:
        (directory/f'{name}.json').write_text(json.dumps(value))
    return directory, result, provenance


def test_guide_smoke_keeps_strict_failure_and_requires_learning(tmp_path):
    run, strict, _ = evidence(tmp_path, 'smoke')
    value = adapter.write_qualification(run)
    assert value['passed'] and not value['strict_acceptance_passed']
    assert value['diagnostic_only'] == dict(heldout_better_than_zero=False)
    assert json.loads((run/'acceptance_result.json').read_text()) == strict
    strict['criteria']['nonzero_gradient'] = False
    (run/'acceptance_result.json').write_text(json.dumps(strict))
    assert not adapter.qualification(run)['passed']


def test_guide_gate_still_requires_better_than_zero(tmp_path):
    run, result, _ = evidence(tmp_path, 'gate')
    assert adapter.qualification(run)['passed']
    result['criteria']['heldout_better_than_zero'] = False
    (run/'acceptance_result.json').write_text(json.dumps(result))
    assert not adapter.qualification(run)['passed']


def test_formal_requires_untampered_same_source_qualified_evidence(tmp_path):
    smoke, _, current = evidence(tmp_path, 'smoke')
    gate, _, prior = evidence(tmp_path, 'gate')
    for run in (smoke,gate):
        adapter.write_qualification(run)
    config = yaml.safe_load((ROOT/'configs/research/scheme_b_perf_v2_formal_seed20261011.yaml').read_text())
    config['acceptance'] = dict(smoke_run=str(smoke), gate_run=str(gate))
    engine = adapter.runtime_engine()
    engine._validate_acceptance(config,current)
    audit_path = gate/'gt_read_audit.json'
    audit = json.loads(audit_path.read_text())
    audit['gt_open_attempts'] = 1
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='qualified real-data gate'):
        engine._validate_acceptance(config,current)
    audit['gt_open_attempts'] = 0
    audit_path.write_text(json.dumps(audit))
    prior['source_sha256'] = 'other-source'
    (gate/'provenance.json').write_text(json.dumps(prior))
    adapter.write_qualification(gate)
    with pytest.raises(ValueError, match='source/protocol/policy differs'):
        engine._validate_acceptance(config,current)
