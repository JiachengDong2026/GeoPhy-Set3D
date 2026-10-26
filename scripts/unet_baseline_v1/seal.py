"""Seal completed evidence without model execution or optimizer updates.

Run after deliver.py, report.py and all evidence log writers have finished.
Historical byte verification is recorded by deliver.py; this step verifies its
protected inventory identities and rechecks registered sources, inputs and
saved checkpoint bytes. The delivery manifest excludes only itself.
"""
import gzip

from unet_paths import ART, PROJECT, ROOT, read, sha, write


def main():
    registration = read(ART / 'registration.json')
    status = read(ART / 'final_status.json')
    assert status['status'] == 'NO_GO'
    for path, digest in registration['source_sha256'].items():
        assert sha(PROJECT / path) == digest, path
    for path, digest in registration['data_sha256'].items():
        assert sha(path) == digest, path
    supplementary = read(ART / 'full_validation_review_registration.json')
    for path, digest in supplementary['source_sha256'].items():
        assert sha(PROJECT / path) == digest, path

    script_paths = sorted((PROJECT / 'scripts/unet_baseline_v1').glob('*.py'))
    source_groups = {'registered_before_updates': {},
                     'supplementary_read_only_review': {},
                     'post_analysis_and_delivery': {}}
    for path in script_paths:
        relative = str(path.relative_to(PROJECT))
        if relative in registration['source_sha256']:
            group = 'registered_before_updates'
        elif relative in supplementary['source_sha256']:
            group = 'supplementary_read_only_review'
        else:
            group = 'post_analysis_and_delivery'
        source_groups[group][relative] = sha(path)
    write(ART / 'analysis_source_manifest.json', source_groups)

    for name, record in status['histories_protected'].items():
        assert sha(ROOT / name) == record['sha256'], name
        assert len(read(ROOT / name)) == record['count'], name
    checkpoints = read(ART / 'checkpoint_manifest.json')
    assert len(checkpoints) == 31
    assert sha(ART / 'checkpoint_manifest.json') == status['checkpoint_manifest_sha256']
    for epoch, record in checkpoints.items():
        assert record['updates'] == 16 * int(epoch)
        assert sha(record['path']) == record['sha256'], record['path']
    with gzip.open(ART / 'actual_updates.jsonl.gz', 'rt') as stream:
        import json
        updates = [json.loads(line) for line in stream]
    assert [entry['update'] for entry in updates] == list(range(1, 481))
    assert all(entry['postclip_gradient_norm'] > 0 and
               entry['actual_parameter_update_l2'] > 0 for entry in updates)
    independent = read(ART / 'unet_independent_review.json')
    full = read(ART / 'full_validation_independent_review.json')
    assert independent['all_passed'] and independent['status'] == status['status']
    assert full['passed'] and full['gate_A_matches']
    assert read(ART / 'unet_numpy_projected_interventions.json')['all_intervention_gates_match']
    assert read(ART / 'primary_regression.json')['bitwise_equal']
    assert status['CPU_passed'] and status['actual_pilot_updates'] == 480
    assert status['formal_updates'] == status['new_formal_predictions'] == 0
    assert not status['merged_into_main']
    report = PROJECT / 'docs/reports/unet_baseline_v1_zh.md'
    integrity = {
        'status': status['status'],
        'registered_sources_reverified': len(registration['source_sha256']),
        'registered_inputs_reverified': len(registration['data_sha256']),
        'supplementary_review_sources_reverified': len(supplementary['source_sha256']),
        'historical_byte_verification_evidence': {
            'final_status_sha256': sha(ART / 'final_status.json'),
            'delivery_log_sha256': sha(ART / 'delivery.log'),
            'protected_inventory_identities_reverified': status['histories_protected'],
            'original_main_files_unchanged': status['original_main_files_unchanged'],
            'v2_branch_unchanged': status['v2_branch_unchanged'],
        },
        'checkpoint_bytes_reverified': len(checkpoints),
        'actual_updates_reverified': len(updates),
        'independent_full_fields': independent['recomputed_full_fields'] + full['full_fields'],
        'primary_C0_C1_bitwise': True,
        'report': str(report.relative_to(PROJECT)),
        'report_sha256': sha(report),
        'analysis_source_manifest_sha256': sha(ART / 'analysis_source_manifest.json'),
        'formal_updates': 0,
        'new_formal_predictions': 0,
        'merged_into_main': False,
    }
    assert integrity['independent_full_fields'] == 426
    write(ART / 'integrity.json', integrity)
    manifest_path = ART / 'delivery_manifest.json'
    manifest = {str(path.relative_to(PROJECT)): sha(path)
                for path in sorted(ART.rglob('*'))
                if path.is_file() and path != manifest_path}
    write(manifest_path, manifest)
    for path, digest in read(manifest_path).items():
        assert sha(PROJECT / path) == digest, path
    assert sha(report) == integrity['report_sha256']
    print(f"Sealed {len(manifest)} evidence files; 31 checkpoints; 480 updates; "
          "426 independently reviewed fields; NO_GO")


if __name__ == '__main__':
    main()
