"""Export small traceable evidence; keep weights/arrays and full process snapshots local."""
from baseline_context import *
import csv
import hashlib
import json
import shutil


def main():
    art = PROJECT / 'docs/reports/artifacts/deep_baselines_v1'
    art.mkdir(parents=True, exist_ok=True)
    def export(relative, dest=None):
        src = ROOT / relative
        if not src.exists(): return
        target = art / (dest or relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    anchor = c.verified(ROOT / 'main_anchor.json')
    anchor.pop('resource_snapshot')
    anchor['export'] = dict(original_path=str(ROOT / 'main_anchor.json'), original_sha256=sha(ROOT / 'main_anchor.json'),
        omitted='full GPU/process/RAM/disk snapshot remains local; model/config/source/protocol/input/output identities preserved')
    c.write(art / 'main_anchor.json', anchor)
    for name in ('cpu_model_tests.log', 'cpu_classic_regression.log', 'cpu_contract_statistics_tests.log',
                 'checks/main_regression.json', 'checks/main_regression_after_integration.json', 'checks/resume.json', 'checks/four_method_mini_predictions.json',
                 'checks/four_method_mini_evaluation.json', 'formal_queue_plan_v2.json', 'formal_queue_launch.json',
                 'inspected_resource_recovery.json', 'inspected_recovery_launch.json', 'queue_status.json',
                 'formal_registration_v1_superseded.json', 'engineering_issues.json', 'scheduling/events.jsonl', 'scheduling/recovery.jsonl'):
        export(name)
    for arm in ARMS:
        for filename in ('result.json',): export('smoke/' + arm + '/' + filename)
        for filename in ('protocol.json', 'metrics.jsonl', 'pilot_complete.json', 'boundary_audit.json', 'gt_access_audit.json', 'environment.json'):
            export(f'pilots/{arm}_seed20261011/{filename}')
        export(f'gates/{arm}.json')
    plan = c.verified(ROOT / 'formal_queue_plan_v2.json')
    queue = c.read(ROOT / 'queue_status.json') if (ROOT / 'queue_status.json').exists() else {}
    jobs = []
    for job in plan['jobs']:
        target = Path(job['target']); protocol = c.verified(target / 'protocol.json')
        for filename in ('protocol.json', 'status.json', 'summary.json', 'boundary_audit.json', 'gt_access_audit.json'):
            export(str((target / filename).relative_to(ROOT)))
        status = c.read(target / 'status.json') if (target / 'status.json').exists() else dict(state='waiting_for_idle_gpu', epoch=0, updates=0)
        record = ROOT / 'scheduling' / (target.name + '_formal300.json')
        if record.exists():
            export(str(record.relative_to(ROOT)))
            record_value = c.read(record)
            if not (target / 'status.json').exists():
                status.update(state=record_value['state'], pid=record_value['pid'], gpu_uuid=record_value['gpu_uuid'])
            elif record_value['state'] == 'running':
                status.update(pid=record_value['pid'], gpu_uuid=record_value['gpu_uuid'], state='training')
        if queue.get('state') == 'active_inspected_recovery' and str(target) != queue['active_target'] and not (target / 'summary.json').exists():
            status.update(previous_attempt_state=status['state'], state='waiting_for_GPU0_after_inspected_recovery')
        jobs.append(dict(arm=protocol['arm'], seed=protocol['seed'], target=str(target), protocol_sha256=sha(target / 'protocol.json'),
                         status_at_export=status, completed=(target / 'summary.json').exists()))
    miniature = []
    for arm in ('plain', *ARMS):
        folder = ROOT / 'evaluation_mini_v2' / arm
        if not (folder / 'summary.json').exists(): continue
        for filename in ('protocol.json', 'protocol.sha256', 'predictions_sealed.json', 'first_GT_boundary.json', 'GT_access.json',
                         'GT_complete.json', 'per_draw_metrics.csv', 'case_group_metrics.csv', 'summary.json', 'group_summary.csv',
                         'geometry_contrasts.csv', 'paired_to_plain.json', 'paired_to_plain.csv'):
            export(str((folder / filename).relative_to(ROOT)))
        for path in folder.glob('inference_access_*.json'): export(str(path.relative_to(ROOT)))
        rows = [c.read(folder / 'rows' / p.name) for p in sorted((folder / 'rows').glob('*.json'))]
        c.write(art / 'evaluation_mini_v2' / arm / 'prediction_manifest.json', rows)
        summary = c.read(folder / 'summary.json')
        groups = {r['group']: r['mean_per_case'] for r in summary['summary'] if r['metric'] == 'relative_l1'}
        miniature.append(dict(arm=arm, scope='NONFORMAL_2case_1draw_10updates_baselines', **groups,
                               checkpoint_sha256=summary['checkpoint_sha256']))
    c.write(art / 'mini_integration_summary.json', miniature)
    with (art / 'mini_integration_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(miniature[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(miniature)
    snapshot = c.resources()
    # Publish GPU occupancy and ownership only; full process commandlines stay local.
    gpu = dict(at=snapshot['at'], gpus=snapshot['gpus'], gpu_processes=snapshot['gpu_processes'],
               ownership=c.subprocess.check_output(['ps', '-eo', 'user,pid,ppid'], text=True))
    pids = {line.split(',')[1].strip() for line in snapshot['gpu_processes'].splitlines() if line.strip()}
    gpu['ownership'] = '\n'.join(line for line in gpu['ownership'].splitlines() if len(line.split()) > 1 and line.split()[1] in pids)
    c.write(art / 'gpu_snapshot.json', gpu)
    c.write(art / 'delivery_status.json', dict(at=c.now(), jobs=jobs, formal_deep_comparison_complete=False,
        senseiver='HISTORICAL_SCIENTIFIC_NO_GO; engineering pilot completed, no duplicate formal300',
        formal_test_executed=False, independent_test='BLOCKED_NO_INDEPENDENT_DATA', miniature_formal=False,
        full_validation_900_executed=False, main_retrained=False, classic_results_overwritten=False))
    for path in (ROOT / 'independent_review').glob('*'):
        if path.is_file(): export(str(path.relative_to(ROOT)))
    for path in (ROOT / 'source_archives').rglob('*.py'):
        export(str(path.relative_to(ROOT)), str(path.relative_to(ROOT)) + '.txt')
    formal_rows = []
    blank = dict(G1_relative_l1='', G2_relative_l1='', G3_relative_l1='', G4_relative_l1='', G5_relative_l1='',
                 G5_minus_G1='', G5_div_G1='', evaluation='not_run_900_val; no_new_test')
    formal_rows.append(dict(method='Plain_Set_Interaction', seed=SEEDS[0], parameters=342209, supervision='measurement_only',
        status='frozen_reused', committed_epochs=300, committed_updates=24900, max_epochs=300, max_updates=24900, **blank))
    for job in jobs:
        st = job['status_at_export']
        formal_rows.append(dict(method=job['arm'], seed=job['seed'], parameters=c.verified(Path(job['target']) / 'protocol.json')['parameters'],
            supervision='measurement_only', status=st['state'], committed_epochs=st.get('epoch', 0), committed_updates=st.get('updates', 0),
            max_epochs=300, max_updates=24900, **blank))
    for seed in SEEDS:
        formal_rows.append(dict(method='senseiver', seed=seed, parameters=254913, supervision='measurement_only',
            status='blocked_historical_scientific_NO_GO', committed_epochs=0, committed_updates=0, max_epochs=300, max_updates=24900, **blank))
    with (art / 'formal_comparison_status.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(formal_rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(formal_rows)
    identity = {str(p.relative_to(art)): sha(p) for p in art.rglob('*') if p.is_file() and p.name != 'export_manifest.json'}
    c.write(art / 'export_manifest.json', dict(at=c.now(), files=identity,
        predictions_and_checkpoints='scratch only; per-prediction/checkpoint hashes published; no GT arrays exported'))
    print(json.dumps(dict(at=c.now(), artifacts=str(art), jobs=jobs), ensure_ascii=False))


if __name__ == '__main__': main()
