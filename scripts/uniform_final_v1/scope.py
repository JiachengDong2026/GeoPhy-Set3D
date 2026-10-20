"""Auditable scheduling-only amendment; original training identities stay frozen."""
from runtime import BASE, Path, verified, sha

def scope_evidence(base=None):
    root=BASE if base is None else Path(base)
    paths=[root/n for n in ('seed_scope_amendment_v1.json','evaluation_stage_amendment_v1.json')]
    return {str(path):sha(path) for path in paths if path.exists()}

def evaluation_stage(base=None):
    root=BASE if base is None else Path(base)
    path=root/'evaluation_stage_amendment_v1.json'
    if not path.exists():return None
    stage=verified(path)
    assert stage['original_design_sha256']==sha(root/'design.json')
    assert stage['previous_seed_scope_sha256']==sha(root/'seed_scope_amendment_v1.json')
    assert stage['before_current_test_prediction_and_GT'] and stage['training_budget_unchanged']
    assert stage['automatic_promotion_to_final_model'] is False
    return stage

def deferred_runs(base=None):
    stage=evaluation_stage(base)
    return [] if stage is None else stage['deferred_runs']

def effective_runs(base=None):
    root=BASE if base is None else Path(base)
    design=verified(root/'design.json')
    path=root/'seed_scope_amendment_v1.json'
    runs=design['runs']
    if path.exists():
        amendment=verified(path)
        assert amendment['original_design_sha256']==sha(root/'design.json')
        assert amendment['before_current_test_prediction_and_GT']
        assert amendment['training_budget_unchanged']
        excluded=set(amendment['excluded_runs'])
        assert excluded <= set(design['runs'])
        runs=[r for r in design['runs'] if r not in excluded]
        assert runs==amendment['included_runs']
    stage=evaluation_stage(root)
    if stage is not None:
        deferred=set(stage['deferred_runs'])
        assert deferred <= set(runs)
        runs=[r for r in runs if r not in deferred]
        assert runs==stage['included_runs']
        assert len(runs)==4
    return runs
