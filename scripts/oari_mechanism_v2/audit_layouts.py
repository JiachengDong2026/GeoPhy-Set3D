"""Offline completion checks per layout; never a model call or GT read."""
from runtime import *
import time
import traceback


def main(watch=False):
    validate()
    receipt = BASE / 'layout_audit_registration.json'
    if not receipt.exists():
        freeze(receipt, dict(at=now(), source_sha256=sha(__file__), parent_protocol_sha256=sha(BASE / 'protocol.json'),
                             scope='completion counts/hash audit and preregistered case-summary preview only; no intervention/statistic changes'))
    assert verified(receipt)['source_sha256'] == sha(__file__)
    while True:
        done = 0
        for li, condition in enumerate(CONDITIONS):
            dest = BASE / f'layout_completion_{li}.json'
            if dest.exists():
                verified(dest); done += 1; continue
            folders = [BASE / 'units' / f'p{pid:03d}' for pid in range(li * 35, (li + 1) * 35)]
            if not all((folder / 'complete.json.sha256').exists() for folder in folders):
                continue
            files, case_values = {}, []
            new = reused = 0
            with Guard(projections=False) as guard:
                for folder in folders:
                    verify_hashes(verified(folder / 'complete.json')['files'])
                    controlled = []
                    for group in GROUPS:
                        record = verified(folder / f'{group}.json')
                        verify_hashes(record['files'])
                        assert record['eval'] and record['no_grad'] and record['checkpoint_sha256'] == HIST.EXPECTED
                        assert record['numerical_qc']['all_invalid_context_zero']
                        if group.startswith('A3_'):
                            assert record['intervention']['valid_pairs_receiving_invalid_source'] == 0
                            controlled.append(next(r['relative_l1'] for r in record['output_rows'] if r['stage'] == 'C1'))
                        new += not record['prediction_reused']; reused += record['prediction_reused']
                        files[str(folder / f'{group}.json')] = sha(folder / f'{group}.json')
                    assert len(controlled) == 8
                    case_values.append(float(np.mean(controlled)))
                assert new == 525 and reused == 105 and not guard.denied
                freeze(dest, dict(at=now(), passed=True, condition=condition, inputs=35, group_records=630,
                                  new_predictions=new, reused_predictions=reused, identity_and_all_file_hashes_verified=True,
                                  all_worker_numerical_checks_passed=True, group_records_sha256=files,
                                  preregistered_P1star_C1_response_case_median=float(np.median(case_values)),
                                  no_GT=True, no_new_forward=True, guard=guard.summary()))
            print(f'LAYOUT VERIFIED {condition}:35 inputs,630 groups,525 new+105 reused predictions', flush=True)
            done += 1
        if done == 5 or not watch:
            return
        time.sleep(45)


if __name__ == '__main__':
    try:
        main('--watch' in sys.argv)
    except BaseException:
        freeze(BASE / f'failure_layout_audit_{time.time_ns()}.json', dict(at=now(), traceback=traceback.format_exc()))
        raise
