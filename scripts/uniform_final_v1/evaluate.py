"""Freeze completed models, reuse matched historical predictions, then seal/score."""
from runtime import *
from scope import effective_runs, scope_evidence, deferred_runs, evaluation_stage
import argparse
import torch
import evaluation as engine
from uniform_models import model_for

def reused():return verified(BASE/'reuse_methods.json')

class TestGuard(AccessGuard):
    def __init__(self,**kwargs):
        assert set(kwargs.get('splits',()))<={'test'}
        assert set(kwargs.get('gt_splits',()))<={'test'}
        if kwargs.get('gt_splits'):
            seal=verified(BASE/'evaluation/test/all_predictions_sealed.json')
            assert seal['all_predictions_saved_before_GT']
            references=verified(BASE/'reuse_prediction_seal.json')
            verify_hashes(references['manifests'])
        checkpoints={m['checkpoint'] for m in verified(BASE/'final_lock.json')['methods']}
        checkpoints.update(m['checkpoint'] for m in reused() if m['kind']=='neural')
        assert set(kwargs.get('checkpoints',()))<=checkpoints
        kwargs.update(output=BASE,checkpoints=sorted(checkpoints))
        super().__init__(**kwargs)
        for m in reused():
            root=Path(m['prediction_root'])
            for r in rows(root/'manifest.jsonl'):
                self.allowed[str((root/r['prediction']).resolve())]='historical_sealed_prediction'

def register():
    if (BASE/'final_lock.json').exists():
        verify_hashes(verified(BASE/'final_lock.json')['sources']);return
    complete=verified(BASE/'training_complete.json')
    assert complete['complete']
    from statistics_checks import statistics_digest, report_digest
    check=verified(BASE/'checks/statistics_synthetic.json')
    assert check['passed']  # Original audit is immutable; require amended-current audit below.
    report_checks=[p for p in (BASE/'checks').glob('statistics_and_reports_synthetic*.json')
        if verified(p).get('checks_source_sha256')==sha(SCRIPT/'statistics_checks.py')
        and verified(p).get('report_source_sha256')==report_digest()
        and verified(p).get('statistics_source_sha256')==statistics_digest()
        and verified(p).get('scope_evidence',{})==scope_evidence()]
    assert report_checks,'Current statistical/report checks must pass before evaluation'
    report_check_path=report_checks[-1];report_check=verified(report_check_path)
    assert report_check['passed'] and report_check['five_report_templates']
    design=verified(BASE/'design.json');verify_hashes(design['training_sources'])
    allowed=set(effective_runs())
    assert set(complete['completed_runs'])<=allowed
    assert complete.get('scope_evidence',{})==scope_evidence()
    assert complete.get('deferred',[])==[Path(n).name+'_full300' for n in deferred_runs()]
    for name in allowed-set(complete['completed_runs']):
        p=verified(Path(name)/'protocol.json')
        assert p['arm']=='uniform_feedback_oari' and not verified(BASE/'gates'/f'seed{p["seed"]}.json')['promote']
    identity_record=verified(BASE/'model_identity.json')
    assert sha(identity_record['config_path'])==identity_record['config_sha256']
    assert sha(BASE/'parameter_inventory.json')==identity_record['trainable_inventory_sha256']
    assert read(identity_record['config_path'])['training_config']==identity_record['main']['config']
    new=[];audits=[]
    for name in complete['completed_runs']:
        target=Path(name);p=verified(target/'protocol.json');verify_hashes(p['sources'])
        s=read(target/'summary.json');log=rows(target/'metrics.jsonl')
        assert s['completed'] and s['updates']==24900 and s['epochs']==300
        assert len(log)==300 and [r['epoch'] for r in log]==list(range(1,301))
        chosen=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
        assert chosen['epoch']==s['best_epoch']
        path=target/'checkpoints/best.pt';assert sha(path)==s['best_checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['epoch']==s['best_epoch'] and ck['config']==p['config']
        assert ck['protocol_sha256']==sha(target/'protocol.json') and not ck['volumetric_gt_used']
        assert ck['selection_metric']==chosen['val']['measurement']['heldout']['relative_l1']
        last=torch.load(target/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        assert last['epoch']==300 and last['updates']==24900
        assert last['config']==p['config'] and last['protocol_sha256']==sha(target/'protocol.json')
        model=model_for(p['config'],p['arm'],'cpu');model.load_state_dict(ck['model'],strict=True)
        assert sum(x.numel() for x in model.parameters())==p['parameters']
        new.append(dict(key=target.name,kind='neural',run=str(target),checkpoint=str(path),checkpoint_sha256=sha(path),
            best_epoch=s['best_epoch'],config=p['config'],arm=p['arm'],parameters=p['parameters'],seed=p['seed'],historical=False))
        audits.append(dict(key=target.name,epochs=300,updates=24900,best_epoch=s['best_epoch'],best_H4=ck['selection_metric'],
            strict_state_load=True,boundary_audit=read(target/'boundary_audit.json'),
            prefix=read(target/'prefix_complete.json') if (target/'prefix_complete.json').exists() else None))
    assert sum(m['arm']=='uniform' for m in new)==2
    freeze(BASE/'training_audit.json',dict(at=now(),passed=True,models=audits))
    old=verified(HISTORY/'final_lock.json');verify_hashes(old['data_identity'])
    freeze(BASE/'registration.json',dict(cases={'test':verified(HISTORY/'registration.json')['cases']['test']},statistics=design['evaluation']['statistics']))
    freeze(BASE/'layouts.json',dict(test=verified(HISTORY/'layouts.json')['test']))
    cases=verified(BASE/'registration.json')['cases']['test'];settings=verified(BASE/'layouts.json')['test']
    manifests={};reference_audit=[]
    for m in reused():
        root=Path(m['prediction_root']);rr=rows(root/'manifest.jsonl')
        engine.validate_rows(cases,settings,rr)
        assert sha(root/'manifest.jsonl')==read(root/'complete.json')['manifest_sha256']
        for r in rr:assert sha(root/r['prediction'])==r['sha256']
        if m['kind']=='neural':assert sha(m['checkpoint'])==m['checkpoint_sha256']
        manifests[str(root/'manifest.jsonl')]=sha(root/'manifest.jsonl')
        reference_audit.append(dict(key=m['key'],count=len(rr),source_metrics=m['source_metrics'],metrics_sha256=sha(m['source_metrics'])))
    freeze(BASE/'reuse_prediction_seal.json',dict(at=now(),manifests=manifests,methods=reference_audit,
        inputs_and_layouts_match=True,all_prediction_hashes_verified=True,previously_used_test=True,GT64_opened_in_current_phase=False))
    identity=training_identity()
    identity.update({str(p):sha(p) for p in SCRIPT.glob('*.py')})
    identity[str(PROJECT/'scripts/pruning_v1/pruned_models.py')]=sha(PROJECT/'scripts/pruning_v1/pruned_models.py')
    for p in (BASE/'design.json',BASE/'reuse_methods.json',BASE/'reuse_audit.json',BASE/'training_audit.json',BASE/'data_audit.json'):
        assert p.exists()
    freeze(BASE/'final_lock.json',dict(locked_at=now(),methods=new,sources=identity,
        evidence={str(p):sha(p) for p in (BASE/'design.json',BASE/'reuse_methods.json',BASE/'reuse_prediction_seal.json',
            BASE/'training_audit.json',BASE/'registration.json',BASE/'layouts.json',BASE/'benchmark.json',
            BASE/'checks/statistics_synthetic.json',report_check_path,BASE/'model_identity.json')} | scope_evidence(),
        data_identity=old['data_identity'],gt64_stat_only=old['gt64_stat_only'],
        primary=design['evaluation']['primary'],statistics=design['evaluation']['statistics'],
        result_scope='four_run_priority_3D_evaluation' if evaluation_stage() else 'full_registered_evaluation',
        deferred_runs=deferred_runs(),OARI_final_model_decision='pending_user_review',
        supplementary=True,previously_used35cases=True,GT128_forbidden=True,
        no_test_checkpoint_selection=True,all_training_complete=True))
    print('Final evaluation lock complete:',len(new),'new models +11historical references',flush=True)

def install():
    engine.OUT=BASE;engine.AccessGuard=TestGuard;engine.model_for=model_for;engine.SCOPE=None

def main(action,key=None):
    if action=='register':return register()
    install();lock=verified(BASE/'final_lock.json');verify_hashes(lock['sources'])
    if action=='predict':return engine.predict_neural(next(m for m in lock['methods'] if m['key']==key),'test')
    if not (BASE/'evaluation/test/all_predictions_sealed.json').exists():engine.seal('test',lock['methods'])
    if action=='score' and not (BASE/'evaluation/test/metrics.jsonl').exists():engine.score('test')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','predict','seal','score'])
    parser.add_argument('--key');o=parser.parse_args();main(o.action,o.key)
