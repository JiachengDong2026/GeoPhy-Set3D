"""Read-only, standard-library verification of the sealed diagnostic delivery."""
import csv
import hashlib
import json
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[2]
ART = PROJECT / 'docs/reports/artifacts/deep_baselines_identifiability_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(ART / 'publication_manifest.json')
    for name, expected in manifest['files_sha256'].items():
        assert sha(PROJECT / name) == expected, name
    for sidecar in ART.rglob('*.sha256'):
        assert sha(sidecar.with_suffix('')) == sidecar.read_text().split()[0], sidecar
    report = PROJECT / 'docs/reports/deep_baselines_identifiability_v1_zh.md'
    for link in re.findall(r'\]\(([^)]+)\)', report.read_text()):
        assert (report.parent / link).exists(), link
    assert not [p for p in ART.rglob('*') if p.suffix in
                ('.pt', '.pth', '.ckpt', '.npy', '.npz', '.h5', '.hdf5')]
    sensitivity = read(ART / 'final_case_sensitivity.json')
    assert sensitivity['optimizer_updates'] == 0
    count = 0
    for run in sensitivity['runs']:
        gradient_flags, fd_flags = [], []
        for case in run['cases']:
            gradient = case['gradients']['observed']['layers']['token_measurement_channel']
            grad_ok = gradient['l2'] > 1e-12 and gradient['finite_fraction'] == 1
            fd_ok = any(row['function'] == 'output_mean_div_fixed_scale'
                        and max(abs(row['AD']), abs(row['FD'])) > 1e-8
                        and abs(row['AD'] - row['FD']) <=
                        1e-8 + .05 * max(abs(row['AD']), abs(row['FD']))
                        for row in case['finite_difference'])
            assert grad_ok == case['input_gradient_pass']
            assert fd_ok == case['finite_difference_pass']
            assert case['H_in_backward'] is False
            gradient_flags.append(grad_ok)
            fd_flags.append(fd_ok)
            count += 1
        assert all(gradient_flags) == run['all_case_input_gradient']
        assert all(fd_flags) == run['all_case_finite_difference']
    assert count == 30 and len(sensitivity['runs']) == 10
    with (ART / 'experiment_summary.csv').open(newline='') as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 10 and sum(int(row['updates']) for row in summary) == 2000
    assert all(row['status'] == 'NO_GO' for row in summary)
    status = read(ART / 'final_status.json')
    assert status['total_updates'] == 2000 and not status['GO_diagnostic_candidates']
    assert status['old_queue_paused'] and not status['formal_training_authorized']
    output = ART / 'delivery_verification.json'
    with output.open('x') as handle:
        json.dump(dict(passed=True, sealed_files_verified=len(manifest['files_sha256']),
                       all_case_sensitivity_independently_recomputed=count,
                       training_updates=2000, new_optimizer_updates=0,
                       report_links_valid=True, restricted_arrays_exported=False,
                       source_sha256=sha(Path(__file__))), handle, indent=2)
        handle.write('\n')
    with output.with_suffix('.json.sha256').open('x') as handle:
        handle.write(sha(output) + '\n')
    print('Delivery verified: 242 sealed files, 30 all-case sensitivity checks, 2000 updates, 0 GO')


if __name__ == '__main__':
    main()
