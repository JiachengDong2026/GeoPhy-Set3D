"""Correction-only FP64 pair audit; preserve original evaluation and failed review."""
from latent_context import *
from evaluate_latent import decision

def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

def main():
    verify_seal();assert not (ROOT/'pair_precision_amendment.json').exists()
    freeze(ROOT/'pair_precision_amendment.json',dict(at=now(),reason='Independent review found original-output pair cosine used FP32 norm; full FP64 differences and norms now required. No tolerance relaxation.',
        original_pair_sha256=sha(ROOT/'pair_metrics.csv'),original_evaluation_sha256=sha(ROOT/'evaluation.json'),failed_review_log_sha256=sha(ROOT/'review.log'),failed_access_sha256=sha(ROOT/'access_review.json'),
        original_reviewer_sha256=sha(HERE/'review_latent.py'),adapter_sha256=sha(__file__),original_metric_source_sha256=sha(HERE/'evaluate_latent.py'),
        thresholds_changed=False,predictions_changed=False,coefficients_changed=False,neural_updates=0,original_files_retained=True))
    pairs=[]
    with Access('evaluate_fp64'):
        truth=[np.load(OLD/f'inputs/truth_{i}.npy') for i in range(8)]
        for a in EXPECTED:
            for b in ('A','B'):
                for i,j in old.design()['pairs']:
                    dy=(truth[i]-truth[j]).ravel()
                    for m in ('original','normal_ridge','zero_ridge','zero_field'):
                        path=ROOT/f'predictions/{a}/{b}/{m}'
                        dp=(np.load(path/f'{i}.npy').astype(np.float64)-np.load(path/f'{j}.npy').astype(np.float64)).ravel();den=np.linalg.norm(dp)*np.linalg.norm(dy)
                        pairs.append(dict(arm=a,branch=b,i=i,j=j,group='seen_seen' if j<6 else 'unseen_unseen' if i>=6 else 'seen_unseen',method=m,difference_cosine=float(dp@dy/den) if den>0 else None,difference_relative_l1=float(np.abs(dp-dy).sum()/np.abs(dy).sum()),prediction_rms=float(np.sqrt(np.mean(dp*dp))),truth_rms=float(np.sqrt(np.mean(dy*dy)))))
        csvwrite(ROOT/'pair_metrics_fp64.csv',pairs)
        case=rows(ROOT/'case_metrics.csv');reps=rows(ROOT/'representation_differences.csv')
        for r in case:
            r['case']=int(r['case']);r['relative_l1']=float(r['relative_l1'])
        for r in reps:r['normalized_l2']=float(r['normalized_l2'])
        ev=read(ROOT/'evaluation.json');decisions=decision(case,pairs,reps,read(ROOT/'probe_seal.json'));assert decisions==ev['decisions']
        ev.update(at=now(),pair_metrics_authoritative='pair_metrics_fp64.csv',pair_precision_correction=True,original_decisions_unchanged=True)
        freeze(ROOT/'evaluation_fp64.json',ev)
    print('FP64 pair correction complete, all original classifications unchanged',flush=True)
if __name__=='__main__':main()
