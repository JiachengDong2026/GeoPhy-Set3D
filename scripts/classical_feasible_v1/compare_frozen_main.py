"""Paired35case comparison from sealed scores; never reruns the frozen main model."""
from feasible_context import *
import csv


def main():
    setup();verify();registration=read(ART/'frozen_main_comparison_registration.json')
    for path,digest in registration['sources'].items():assert sha(PROJECT/path)==digest,path
    for path,digest in registration['inputs'].items():assert sha(path)==digest,path
    score=read(ART/'unet_test_score.json');assert score['GT64_after_prediction_seal'] and score['predictions']==595
    assert sha(score['metrics_path'])==score['metrics_sha256']
    method='uniform_plain_set_interaction_seed20261011'
    old_path=Path(registration['main_metrics'])
    old=[r for r in csv.DictReader(old_path.open()) if r['method']==method and r['stage']=='C1']
    old_keys={(r['case_id'],r['condition'],int(r['draw'])):r for r in old}
    assert len(old_keys)==210
    seal=read(ART/'unet_test_prediction_seal.json')
    prediction_keys={(r['case_id'],r['condition'],r['draw']):r for r in seal['predictions']}
    for r in [json.loads(line) for line in Path(registration['main_manifest']).read_text().splitlines()]:
        key=(r['case_id'],r['condition'],r['draw']);assert key in prediction_keys and key in old_keys
        assert r['input_ids']==prediction_keys[key]['input_ids']
        assert r['checkpoint_sha256']==registration['main_checkpoint_sha256']
    new={}
    for line in Path(score['metrics_path']).read_text().splitlines():
        r=json.loads(line);new[r['case_id'],r['condition'],r['draw']]=r
    assert set(old_keys)<=set(new)
    cases=read(ART/'test_cases.json');paired=[];summary=[]
    for condition in registration['conditions']:
        for metric in ['relative_l1','relative_l2']:
            for case in cases:
                keys=sorted(k for k in old_keys if k[0]==case and k[1]==condition)
                assert len(keys)==(1 if condition=='Val-A-V4' else 5)
                baseline=float(np.mean([float(old_keys[k][metric]) for k in keys]))
                treatment=float(np.mean([new[k]['volume'][metric] for k in keys]))
                paired.append(dict(condition=condition,case_id=case,metric=metric,draws=len(keys),
                    frozen_main=baseline,unet=treatment,delta_unet_minus_main=treatment-baseline))
            selected=[r for r in paired if r['condition']==condition and r['metric']==metric]
            delta=np.asarray([r['delta_unet_minus_main'] for r in selected])
            rng=np.random.default_rng(2026102703);bootstrap=delta[rng.integers(0,35,size=(10000,35))].mean(1)
            summary.append(dict(condition=condition,metric=metric,cases=35,
                main_mean=float(np.mean([r['frozen_main'] for r in selected])),
                unet_mean=float(np.mean([r['unet'] for r in selected])),
                delta_unet_minus_main=float(delta.mean()),paired_ci95_low=float(np.quantile(bootstrap,.025)),
                paired_ci95_high=float(np.quantile(bootstrap,.975)),unet_lower_error_cases=int((delta<0).sum())))
    for name,rows in [('unet_main_paired_cases.csv',paired),('unet_main_paired_summary.csv',summary)]:
        with (ART/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write(ART/'unet_main_comparison.json',dict(passed=True,main_rerun=False,new_GT_reads=0,
        paired_cases=35,matched_predictions=210,conditions=registration['conditions'],summary=summary,
        main_checkpoint_sha256=registration['main_checkpoint_sha256'],registration_sha256=sha(ART/'frozen_main_comparison_registration.json'),
        comparison_scope='same35cases and identical input station IDs; official Val-A-V4 and five clustered-V4 draws only; extra preheat budget disclosed'))
    verify();print('Paired frozen-main comparison completed from sealed scores only',flush=True)


if __name__=='__main__':main()
