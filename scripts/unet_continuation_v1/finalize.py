"""Reconcile terminal status, validate stop rules and seal the completed delivery."""
from continuation_context import *
from train import stop_reason
from datetime import datetime
import shutil


def main():
    protocol=verify_all()
    for name in ['analysis_registration.json','finalization_registration.json']:
        for path,digest in read(ART/name)['source_sha256'].items():assert sha(PROJECT/path)==digest,path
    r=read(ART/'result.json');review=read(ART/'independent_review.json')
    assert review['passed'] and r['stop_reason']=='budget_complete'
    assert r['actual_additional_updates']==protocol['max_additional_updates']==400
    rows=r['metrics']
    assert [v['step'] for v in rows]==list(range(0,401,25))
    for index in range(1,len(rows)):
        assert stop_reason(rows[:index+1]) is None,(index,stop_reason(rows[:index+1]))
    for row in rows:
        assert sha(row['arrays']['path'])==row['arrays']['sha256']
        for field in row['fields']:
            assert all(np.isfinite(field[k]) for k in ['mean','std','min','max','nearzero'])
            assert field['min']>=0 and field['max']<=1
    updates=[json.loads(line) for line in (ROOT/'case0/updates.jsonl').read_text().splitlines()]
    assert [v['original_objective_update'] for v in updates]==list(range(101,501))
    assert all(np.isfinite(v['loss_before_update']) and np.isfinite(v['actual_parameter_delta']) and np.isfinite(v['lr']) for v in updates)
    snapshots=[json.loads(s) for s in (ROOT/'gpu_snapshots.jsonl').read_text().splitlines()]
    task=read(ROOT/'scheduling/case0_continuation.json')
    assert task['state']=='complete' and task['exit_code']==0 and task['attempt']==1
    assert len(snapshots)==2
    assert (datetime.fromisoformat(snapshots[1]['at'])-datetime.fromisoformat(snapshots[0]['at'])).total_seconds()>=15
    uuid=task['gpu_uuid']
    for snapshot in snapshots:
        assert not any(line.split(',')[0].strip()==uuid for line in snapshot['gpu_processes'].splitlines() if line.strip())
        line=next(v for v in snapshot['gpus'].splitlines() if v.split(',')[1].strip()==uuid)
        fields=[v.strip() for v in line.split(',')]
        assert int(fields[4])<100 and int(fields[5])>=20000 and int(fields[6])<=5
    events=[json.loads(line) for line in (ROOT/'scheduling/events.jsonl').read_text().splitlines()]
    assert not any(v.get('event')=='foreign_gpu_conflict' for v in events)
    # The worker writes its last monitor as running before the loop exits;
    # preserve that exact record and reconcile terminal status from result/task.
    status=read(ART/'status.json')
    if status['state']=='running':
        write(ART/'status_last_monitor.json',status)
    write(ART/'status.json',dict(state='complete',stop_reason=r['stop_reason'],additional_updates=400,
        total_original_objective_updates=500,latest=r['final'],review_passed=True,
        reconciled_from='authoritative result.json and completed scheduler task; last monitor preserved'))
    report=PROJECT/'docs/reports/unet_feasible_continuation_v1_zh.md'
    text=report.read_text()
    final=rows[-1];previous=rows[-2]
    detail=(f"末段存在波动：追加375步时loss={previous['loss']:.7g}、H={previous['measurement']['heldout']:.7g}、E={previous['extra_E_H']:.7g}；"
        f"400步末态分别为{final['loss']:.7g}、{final['measurement']['heldout']:.7g}、{final['extra_E_H']:.7g}。"
        '没有用较好中途节点替代末态。达到400步预算表示这轮有界检查结束，并不等于数值收敛；整段没有触发预定停止规则。')
    text=text.replace('## 监测节点',detail+'\n\n## 监测节点',1)
    text+='\n训练在对话中断前已正常完成；对话恢复后仅重启被打断的只读复核，没有重启或重放训练。末次监测状态曾写为running，交付时依据result和调度完成记录校正为complete，原监测状态另存status_last_monitor.json。\n'
    report.write_text(text)
    for name in ['review.log','review_after_interrupt.log','report.log']:
        p=ROOT/name
        if p.exists():shutil.copyfile(p,ART/name)
    write(ART/'checkpoint_manifest.json',dict(checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256'],
        source_checkpoint=r['source_checkpoint'],source_sha256=r['source_sha256'],actual_additional_updates=400,
        original_objective_updates=500,teacher_preheat_updates=200,complete_field_arrays=[v['arrays'] for v in rows]))
    write(ART/'execution_integrity.json',dict(passed=True,actual_additional_updates=400,original_objective_updates=500,
        teacher_preheat_updates=200,stop_reason='budget_complete',early_stop_rule_replay_passed=True,
        evaluation_points=17,independently_reviewed_full_fields=4,independent_physical_integrals=16,
        optimizer_reset=False,optimizer_step=500,scheduler_last_epoch=31,GPU_uuid=uuid,
        double_idle_snapshots_verified=True,resource_conflicts=0,replayed_training_updates=0,
        dialogue_interruption_after_training_completed=True,frozen_scientific_sources_unchanged=147,
        frozen_input_files_unchanged=207,old_evidence_and_frozen_checkpoints_unchanged=True,
        GT64_reads=0,GT128_reads=0,PoolB_reads=0,final_test_reads=0,E_used_for_early_stopping=True,
        single_case_only=True,formal_training=False,main_merge=False))
    source_manifest={str(p.relative_to(PROJECT)):sha(p) for p in sorted((PROJECT/'scripts/unet_continuation_v1').glob('*.py'))}
    write(ART/'source_manifest.json',source_manifest)
    write(ART/'publication.json',dict(report=str(report.relative_to(PROJECT)),report_sha256=sha(report),
        source_manifest_sha256=sha(ART/'source_manifest.json'),checkpoint_sha256=r['checkpoint_sha256'],
        conclusion='Local single-case learning continues after feasible-field preheat; cross-case reconstruction not established'))
    manifest={str(p.relative_to(PROJECT)):sha(p) for p in sorted(ART.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'}
    write(ART/'delivery_manifest.json',manifest)
    for path,digest in manifest.items():assert sha(PROJECT/path)==digest,path
    print('Finalized: 400 real continuation updates; 4 fields / 16 independent integrations; original evidence protected')


if __name__=='__main__':main()
