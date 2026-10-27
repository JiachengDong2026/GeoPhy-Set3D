"""Read-only final integrity audit; hashes large artifacts without adding them to Git."""
from feasible_context import *
from datetime import datetime,timezone


def main():
    verify()
    active_art=ART/'sparse_guard_v2';active_root=ROOT/'sparse_guard_v2'
    from sparse_guard_v2_context import verify as verify_sparse
    verify_sparse()
    original=read(ART/'unet_summary.json')
    assert original['decision']=='STOP_PERSISTENT_COLLAPSE' and original['epochs']==175 and not original['formal_complete']
    assert read(ART/'pipeline_status.json')['state']=='stopped_for_analysis'
    assert read(active_art/'pipeline_status.json')['state']=='formal_training_and_test_complete'
    assert read(ART/'other_algorithms_status.json')['state']=='bounded_multicase_checks_complete'
    assert read(ART/'completion_observer_status.json')['state']=='all_training_decisions_and_reviews_complete'
    states={}
    for key in ['unet_workers','unet_sparse_guard_v2','unet_sparse_guard_v2_test','pilot_deeponet','pilot_fno_workers','pilot_senseiver','pilot_senseiver_sparse_guard_v2','pilot_gnot','pilot_deepsets','pilot_deeponet_lr2e5']:
        path=ROOT/'scheduling'/(key+'.json')
        states[key]=read(path);assert states[key]['state']=='complete',(key,states[key])
    for name in ['deeponet_multicase_review.json','fno_multicase_review.json','senseiver_multicase_review.json',
        'gnot_multicase_review.json','deepsets_multicase_review.json','deeponet_lr2e5_review.json']:
        assert read(ART/name)['passed'],name
    for name in ['unet_validation_review.json','selected_unet_causal_review.json','unet_main_comparison.json','senseiver_complete_independent_review.json']:
        assert read(active_art/name)['passed'],name
    score=read(active_art/'unet_test_score.json')
    assert score['GT64_files']==35 and score['GT128_reads']==0 and score['predictions']==595
    assert score['GT64_after_prediction_seal'] and not score['denied']
    assert sha(score['metrics_path'])==score['metrics_sha256']
    seal=read(active_art/'unet_test_prediction_seal.json');lock=read(active_art/'unet_test_lock.json')
    assert seal['all_predictions_saved_before_GT64'] and not seal['denied']
    assert seal['checkpoint_sha256']==lock['sha256']==sha(lock['checkpoint'])
    assert len(seal['predictions'])==len({(r['case_id'],r['condition'],r['draw']) for r in seal['predictions']})==595
    large={}
    for row in seal['predictions']:
        assert sha(row['prediction'])==row['sha256'];large[row['prediction']]=row['sha256']
    reviews=[ART/name for name in ['deeponet_multicase_review.json','fno_multicase_review.json','senseiver_multicase_review.json',
        'gnot_multicase_review.json','deepsets_multicase_review.json','deeponet_lr2e5_review.json','health_guard_audit.json']]
    reviews += [active_art/name for name in ['unet_validation_review.json','selected_unet_causal_review.json','senseiver_multicase_review.json']]
    for path in reviews:
        for row in read(path)['records']:
            assert sha(row['path'])==row['sha256'];large[row['path']]=row['sha256']
    for row in read(ART/'teachers.json')['cases']:
        assert sha(row['path'])==row['sha256'];large[row['path']]=row['sha256']
    directories=[ROOT/('unet' if arm=='unet' else 'pilot_'+arm) for arm in ['unet','deeponet','fno','senseiver','gnot','deepsets','deeponet_lr2e5']]
    directories += [active_root/'unet',active_root/'pilot_senseiver']
    for directory in directories:
        summary=read(directory/'summary.json')
        checkpoint=Path(summary['best_checkpoint'])
        assert sha(checkpoint)==summary['best_checkpoint_sha256']
        assert sha(directory/'last.pt')==summary['last_checkpoint_sha256']
        for filename in ['best.pt','last.pt','metrics.jsonl','preheat_updates.jsonl','physical_updates.jsonl','summary.json']:
            path=directory/filename
            if path.exists():large[str(path)]=sha(path)
    small={}
    for directory in [ART,PROJECT/'scripts/classical_feasible_v1']:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.name!='final_delivery_manifest.json' and not path.name.startswith('git_'):
                small[str(path.relative_to(PROJECT))]=sha(path)
    for name in ['docs/classical_feasible_formal_v1_protocol_zh.md','docs/classical_sparse_guard_v2_protocol_zh.md','docs/reports/classical_feasible_v1_training_test_zh.md']:
        small[name]=sha(PROJECT/name)
    write(ART/'final_delivery_manifest.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        final_audit_passed=True,small_files=small,large_files=large,controller_states=states,
        frozen_history_verified=True,GT128_reads=0,test_predictions=595,large_files_in_Git=False,
        original_U_Net_stop_preserved=True,sparse_guard_revision_disclosed=True,
        scope='U-Net sparseguard-v2 cumulative300 and sealed35case17layout test; original175 stop and all pilot decisions preserved'))
    print('FINAL_AUDIT_PASSED',len(small),'small files',len(large),'large artifacts',flush=True)


if __name__=='__main__':main()
