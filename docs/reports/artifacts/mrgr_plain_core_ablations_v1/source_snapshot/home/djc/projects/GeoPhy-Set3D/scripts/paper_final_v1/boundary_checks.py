"""Fail-closed tests use real registered layout identities without opening labels."""
from common import *
import copy
import evaluation as ev

reg=verified(OUT/'registration.json');settings=verified(OUT/'layouts.json')['val']
rr=rows(OLD_EVAL/'prediction_manifest.jsonl');ev.validate_rows(reg['cases']['val'],settings,rr)
mutations=[rr[:-1],rr[:-1]+[rr[0]]]
wrong=copy.deepcopy(rr);wrong[0]['input_ids']=list(reversed(wrong[0]['input_ids']));mutations.append(wrong)
wrong=copy.deepcopy(rr);wrong[0]['heldout_ids']=wrong[0]['input_ids'];mutations.append(wrong)
for bad in mutations:
    try:ev.validate_rows(reg['cases']['val'],settings,bad)
    except AssertionError:pass
    else:raise AssertionError('Corrupt manifest accepted')
with AccessGuard() as guard:
    try:ev.score('test')
    except FileNotFoundError:pass
    else:raise AssertionError('Scored without final seal')
assert not guard.counts and not guard.denied
write(OUT/'boundary_checks.json',dict(at=now(),passed=True,rejected=['missing_case','duplicate_key','wrong_station_order','overlapping_H4','test_score_without_global_seal'],actual_GT_opens=0))
print('Five fail-closed boundary checks PASSED')
