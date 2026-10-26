"""Final provenance and byte checks; never loads a model or optimizer."""
from diagnose import *


def main():
    registration = json.loads((ART / 'registration.json').read_text())
    for name, digest in registration['frozen_sources'].items():
        assert sha(PROJECT / name) == digest, name
    for name, digest in registration['frozen_inputs'].items():
        assert sha(name) == digest, name
    for name, digest in registration['diagnostic_source_sha256'].items():
        assert sha(PROJECT / name) == digest, name
    supplementary = json.loads((ART / 'parameter_probe_registration.json').read_text())
    for name, digest in supplementary['source_sha256'].items():
        assert sha(PROJECT / name) == digest, name
    old_art = PROJECT / 'docs/reports/artifacts/unet_baseline_v1'
    assert sha(old_art / 'delivery_manifest.json') == registration['frozen_delivery_manifest_sha256']
    old_manifest = json.loads((old_art / 'delivery_manifest.json').read_text())
    for name, digest in old_manifest.items():
        assert sha(PROJECT / name) == digest, name
    history = json.loads((old_art / 'history_audit.json').read_text())
    for name, digest in history['frozen_sha256'].items():
        assert sha(name) == digest, name
    old_integrity = json.loads((old_art / 'integrity.json').read_text())
    assert sha(PROJECT / old_integrity['report']) == old_integrity['report_sha256']
    checkpoints = json.loads((old_art / 'checkpoint_manifest.json').read_text())
    parameter = json.loads((ART / 'parameter_probe.json').read_text())
    fields = json.loads((ART / 'saved_field_gradients.json').read_text())
    for row in parameter['rows']:
        ck = checkpoints[str(row['epoch'])]
        assert sha(ck['path']) == row['checkpoint_sha256'] == ck['sha256']
        field = next(r for r in fields['rows'] if r['epoch'] == row['epoch'] and r['index'] == 0)
        np.testing.assert_allclose(row['readout_bias_gradient']['total'],
            field['gradient']['total']['joint_logit_bias_derivative'], rtol=.001, atol=1e-6)
    review = json.loads((ART / 'independent_review.json').read_text())
    assert review['passed'] and review['comparisons'] == 2490
    for name in ['execution.json', 'independent_review.json', 'parameter_probe.json']:
        record = json.loads((ART / name).read_text())
        assert record['optimizer_updates'] == 0
        assert not any(record['access'][k] for k in ['gt_open_attempts', 'forbidden_npy_attempts',
                                                    'supervised_checkpoint_open_attempts'])
    sources = {str(p.relative_to(PROJECT)): sha(p) for p in sorted((PROJECT / 'scripts/shared_failure_v1').glob('*.py'))}
    write('analysis_source_manifest.json', sources)
    report = 'docs/reports/shared_selfsup_failure_diagnosis_v1_zh.md'
    write('integrity.json', dict(registered_sources_unchanged=len(registration['frozen_sources']),
        registered_inputs_unchanged=len(registration['frozen_inputs']),
        old_unet_evidence_unchanged=len(old_manifest), old_unet_report_unchanged=True,
        old_frozen_checkpoint_identities_verified=len(history['frozen_sha256']),
        read_only_unet_checkpoint_count=2, original_objective_crosscheck_conditions=2490,
        independent_saved_field_and_parameter_bias_gradients_match=True,
        optimizer_updates=0, model_parameters_modified=False, historical_conclusions_modified=False,
        report=report, report_sha256=sha(PROJECT / report),
        source_manifest_sha256=sha(ART / 'analysis_source_manifest.json'),
        main_merge=False))
    manifest = {str(p.relative_to(PROJECT)): sha(p) for p in sorted(ART.rglob('*'))
                if p.is_file() and p.name != 'delivery_manifest.json'}
    write('delivery_manifest.json', manifest)
    for name, digest in manifest.items():
        assert sha(PROJECT / name) == digest, name
    print('Final diagnostic integrity passed; zero optimizer updates;', len(manifest), 'evidence files')


if __name__ == '__main__':
    main()
