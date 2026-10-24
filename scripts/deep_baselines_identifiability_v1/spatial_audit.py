"""Independent saved-field diversity and intermediate logit reconstruction."""
from diag_context import *
import csv

def main():
    rows=[]
    with TrainGuard() as access:
        for result in sorted((ROOT/'runs').glob('*/result.json')):
            p=c.verified(result.parent/'protocol.json');history=[c.verified(result.parent/'initial.json'),*c.rows(result.parent/'evaluations.jsonl')]
            for ev in history:
                for branch in ('A','B'):
                    fields=[]
                    for case in ev['cases']:
                        path=result.parent/f"raw/update{ev['update']:03d}_case{case['case_index']}.npz"
                        with np.load(path,allow_pickle=False) as a:fields.append(a['field_'+branch][0,0].astype(np.float64))
                    field=np.stack(fields);flat=field.reshape(-1);safe=np.clip(flat,1e-30,1-1e-15)
                    logits=np.log(safe)-np.log1p(-safe)
                    pairwise=[float(np.mean(np.abs(fields[i]-fields[j]))) for i in range(len(fields)) for j in range(i)]
                    quantiles=np.quantile(logits,[0,.01,.5,.99,1])
                    rows.append(dict(run=result.parent.name,update=ev['update'],branch=branch,cases=len(fields),
                        between_case_pairwise_MAE=None if not pairwise else float(np.mean(pairwise)),
                        between_case_fixed_scale_response=None if not pairwise else float(np.mean(pairwise)/design()['initial_mass_fraction']),
                        between_case_voxel_std_mean=None if len(fields)==1 else float(field.std(axis=0).mean()),
                        output_mean=float(flat.mean()),spatial_std_mean=float(np.mean([x.std() for x in fields])),
                        sigmoid_derivative_mean=float(np.mean(flat*(1-flat))),
                        logit_min=float(quantiles[0]),logit_p01=float(quantiles[1]),logit_median=float(quantiles[2]),
                        logit_p99=float(quantiles[3]),logit_max=float(quantiles[4]),
                        exact_zero_fraction=float(np.mean(flat==0)),exact_one_fraction=float(np.mean(flat==1)),
                        logits_source='inverted saved FP32 sigmoid output; endpoint clamps reported; actual hook logits at initial/final in evaluation JSON'))
    assert len(rows)==10*7*2
    with (ROOT/'spatial_and_logit_statistics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    freeze('spatial_audit.json',dict(at=c.now(),passed=True,rows=len(rows),optimizer_updates=0,
        real_GT_read=False,access=access.summary(),method='independent NumPy on saved fields; case1 diversity undefined'))
    print('SPATIAL AND LOGIT AUDIT',len(rows),'rows',flush=True)

if __name__=='__main__':main()
