"""Final delivery checks and metadata-only scope/recovery inventory (no model/GT)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'plain_final_v1'))
from context import *
from collections import Counter


def main():
    acceptance=verified(BASE/'acceptance.json');verify_hashes(acceptance['reports']);verify_hashes(acceptance['exports'])
    verify_hashes(verified(BASE/'protocol.json')['sources'])
    snapshots=verified(BASE/'source_snapshot_manifest.json')
    for path,r in snapshots.items():assert sha(r['snapshot'])==r['sha256']==sha(path)
    for m in verified(BASE/'model_manifest.json')['models']:assert sha(m['checkpoint'])==m['checkpoint_sha256']
    gt=verified(BASE/'GT64_access.json');assert sha(BASE/'per_case_metrics.csv')==gt['csv_sha256'] and sha(BASE/'metrics.jsonl')==gt['metrics_sha256']
    metrics=list(csv.DictReader((BASE/'per_case_metrics.csv').open()));assert len(metrics)==3570
    assert Counter((r['method'],r['stage']) for r in metrics)=={(MAIN,'C0'):595,(MAIN,'C1'):595,**{(m,'C1'):595 for m in CLASSICS}}
    source=rows(DATA/'source_index.jsonl');debug=rows(DATA/'debug/source_index.jsonl');caseids={r['case_id'] for r in source};assert {r['case_id'] for r in debug}<=caseids
    root=DATA.parents[1];inventory=[]
    for folder in [root,*sorted(p for p in root.rglob('*') if p.is_dir() and len(p.relative_to(root).parts)<=3)]:
        children=list(folder.iterdir());inventory.append(dict(path=str(folder),directories=sorted(p.name for p in children if p.is_dir()),
            files_by_suffix=dict(Counter(p.suffix for p in children if p.is_file()))))
    # Existing known dataset roots are the only immediate raw/processed projects.
    assert {p.name for p in (root/'processed').iterdir()}=={'geophy_projection_pool_v1'}
    assert {p.name for p in (root/'raw').iterdir()}=={'Fluent'}
    assert {p.name for p in (root/'raw/Fluent').iterdir()}=={'static_final_frame_partial_v1'}
    recovery=[]
    for i in range(1,4):
        old=BASE.with_name(BASE.name+f'_registration_attempt{i}');audit=verified(old/'failure_audit.json');assert not (old/'inference').exists()
        item=dict(path=str(old),audit=audit,failure_log_sha256=sha(old/'registration_failure.log'))
        recovery.append(item)
        for name in ['failure_audit.json','failure_audit.json.sha256','registration_failure.log']:
            dst=ART/'recovery'/f'registration_attempt{i}'/name;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copyfile(old/name,dst)
    resource=rows(BASE/'scheduling/resources.jsonl');compact=[{k:r[k] for k in ('at','gpus','gpu_processes','memory','load')} for r in resource]
    jsonl(ART/'resource_snapshots.jsonl',compact)
    copy_names=['source_snapshot_manifest.json','source_snapshot_manifest.json.sha256','configs/'+MAIN+'.json','configs/'+MAIN+'.json.sha256',
        'configs/'+SECOND+'.json','configs/'+SECOND+'.json.sha256','scheduling/inference.json','scheduling/inference_attempt1.log','scheduling/score.py.log','scheduling/analysis.py.log','scheduling/reports.py.log','status.json']
    for name in copy_names:
        p=BASE/name;dst=ART/name;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copyfile(p,dst)
    freeze(BASE/'completion_review.json',dict(at=now(),passed=True,script_sha256=sha(__file__),acceptance_sha256=sha(BASE/'acceptance.json'),
        source_snapshots_verified=len(snapshots),debug_case_ids=sorted(r['case_id'] for r in debug),debug_cases_all_historical=True,
        data_directory_metadata_inventory=inventory,registration_recovery=recovery,
        numerical_training_or_inference_failures=0,scoring_or_statistics_failures=0,
        resources_sha256=sha(BASE/'scheduling/resources.jsonl'),resource_snapshots=len(resource),
        inference_task=read(BASE/'scheduling/inference.json'),independent_test_status='BLOCKED_NO_INDEPENDENT_DATA',GT128_read=False,Pool_B_arrays_read=False,
        git_precommit=dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())))
    shutil.copyfile(BASE/'completion_review.json',ART/'completion_review.json');shutil.copyfile(BASE/'completion_review.json.sha256',ART/'completion_review.json.sha256')
    exported={str(p):sha(p) for p in ART.rglob('*') if p.is_file()}
    freeze(BASE/'delivery_manifest.json',dict(at=now(),files=exported,large_binary_files_in_git=False))
    shutil.copyfile(BASE/'delivery_manifest.json',ART/'delivery_manifest.json');shutil.copyfile(BASE/'delivery_manifest.json.sha256',ART/'delivery_manifest.json.sha256')
    print('DELIVERY_AUDIT_PASSED',len(exported),'files')


if __name__=='__main__':main()
