"""Independent NumPy causal and signal-health review of the single Senseiver continuation."""
from sparse_guard_v2_context import *
from pilot_sparse_guard_v2 import verify_stage
import review_multicase


def numpy_health(field):
    a=field.astype('float64');saturated=field*(1-field)<4.54e-5
    return dict(mean=float(a.mean()),std=float(a.std()),nearzero=float((a<1e-10).mean()),
        saturated=float(saturated.mean()),saturated_mass_fraction=float(a[saturated].sum()/max(a.sum(),1e-30)))


def main():
    setup();verify_stage()
    review_multicase.ART=ART;review_multicase.ROOT=ROOT;review_multicase.verify=verify_stage
    sys.argv=[sys.argv[0],'senseiver'];review_multicase.main()
    review=read(ART/'senseiver_multicase_review.json');assert review['passed']
    records=[]
    for row in review['records']:
        if row['index']>=(4 if row['split']=='val' else 3):continue
        with np.load(row['path'],allow_pickle=False) as arrays:
            for branch in ['A','B']:
                stats=numpy_health(arrays[branch+'_normal']);records.append(dict(split=row['split'],index=row['index'],branch=branch,**stats))
    val=[r for r in review['records'] if r['split']=='val' and r['index']<4]
    train=[r for r in review['records'] if r['split']=='train']
    summary=read(ART/'senseiver_pilot_summary.json');score=review['independent_val_H']
    checks=dict(healthy=healthy(records),val_H=score<=.99,improvement=score<=.95*summary['initial_val_H'] and summary['initial_val_H']-score>=.01,
        val_benefit=sum(r['zero_benefit']>=.001 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in val)>=2,
        train_dependency=sum(r['wrong_increase']>=1e-6 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in train)>=2)
    expected=read(ROOT/'pilot_senseiver/gate_epoch30.json');assert checks==expected
    write(ART/'senseiver_complete_independent_review.json',dict(passed=True,scientific_passed=all(checks.values()),checks=checks,normal_field_health=records,
        projection_review_sha256=sha(ART/'senseiver_multicase_review.json'),boundary29_review_sha256=sha(ART/'senseiver_guard_boundary_audit.json'),
        no_test_or_GT=True,new_updates=0,training_decision_preserved=summary['decision']))
    verify_stage();print('SENSEIVER_V2_INDEPENDENT_REVIEW_PASSED scientific_go=',all(checks.values()),flush=True)


if __name__=='__main__':main()
