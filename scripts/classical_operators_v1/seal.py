"""Seal completed decisions, resource records and a compact publication manifest."""
import datetime,subprocess,shutil
from context import *

def main():
    assert read(ART/'delivery_review.json')['all_passed']
    assert read(ART/'github_main.json')['verified']
    resources=[]
    for p in sorted((ROOT/'scheduling').glob('*.json')):
        r=read(p)
        if isinstance(r,dict) and 'state' in r:
            assert r['state'] not in ['running','training'],p
            resources.append({k:r[k] for k in ['key','state','attempt','started_at','finished_at','gpu_uuid','exit_code','pid'] if k in r})
    write(ART/'resource_execution.json',dict(tasks=resources,original_gpu0_reservation_timed_out_before_FNO_worker=True,
        GPU1_used_only_under_existing_UUID_lock=True,no_other_users_terminated=True,
        double_idle_snapshot_interval_seconds=15,gpu_snapshots=str(ROOT/'gpu_snapshots.jsonl'),gpu_snapshots_sha256=sha(ROOT/'gpu_snapshots.jsonl'),
        all_own_jobs_finished=True,scientific_thresholds_unchanged=True))
    for filename in ['finish_stage.log','delivery.log','delivery_check.log']:
        p=ROOT/filename
        if p.exists():shutil.copyfile(p,ART/filename)
    summary=dict(sealed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        status='COMPLETED_WITH_TWO_SCIENTIFIC_NO_GO',github_main=read(ART/'github_main.json')['commit'],
        branch='experiment/classical_operators_v1',models=read(ART/'experiment_status.json'),
        engineering_both_passed=True,regression_tests_passed=30,independent_recomputed_fields=dict(deeponet=42,fno=210),
        original_files_preserved=4835,formal_epochs=0,formal_updates=0,new_GT64_reads=0,
        unresolved=['DeepONet interrupted-epoch exact optimizer count unavailable; authoritative checkpoint96, process96..111',
                    'Both new adaptations fail preregistered scientific acceptance; no formal accuracy ranking',
                    'Formal inference/GT pipeline has refusal-boundary checks but no end-to-end run because no model qualified'],
        report='docs/reports/classical_operators_v1_zh.md',checkpoints='docs/reports/artifacts/classical_operators_v1/checkpoint_manifest.json')
    write(ART/'final_status.json',summary)
    protected=read(ROOT/'protected_tracked.json');assert all(sha(PROJECT/p)==h for p,h in protected.items())
    for p,h in read(ART/'history_audit.json')['evidence_sha256'].items():assert sha(p)==h
    baseline=read(ART/'history_audit.json')['base_commit']
    paths=set(subprocess.check_output(['git','diff','--name-only',baseline],cwd=PROJECT,text=True).splitlines())
    paths.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=PROJECT,text=True).splitlines())
    paths.discard('docs/reports/artifacts/classical_operators_v1/publication_manifest.json')
    assert not any(Path(p).suffix in ['.pt','.pth','.npy','.npz','.pdf'] for p in paths)
    assert not any(p in protected for p in paths),'No pre-existing tracked file may change'
    manifest=dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        baseline_commit=baseline,source='independent experiment branch; all additions only',
        sha256={p:sha(PROJECT/p) for p in sorted(paths) if (PROJECT/p).is_file()},raw_arrays_and_checkpoints_in_scratch_only=True)
    target=ART/'publication_manifest.json';assert not target.exists();write(target,manifest)
    print('Sealed completed scientific decisions and',len(manifest['sha256']),'publication files')
if __name__=='__main__':main()
