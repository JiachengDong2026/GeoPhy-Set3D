"""Independent complete-field review of the single self-generated field teacher probe."""
from review import *


def main():
    r=read(ART/'case0_teacher_probe.json')
    records=[]
    max_error=0.
    for key in ['initial','final']:
        metric=r[key]
        assert sha(metric['arrays']['path'])==metric['arrays']['sha256']
        with np.load(metric['arrays']['path'],allow_pickle=False) as arrays:
            scores={};losses={}
            for branch in ['A','B']:
                volume=arrays[branch+'_volume']
                assert np.isfinite(volume).all() and volume.min()>=0 and volume.max()<=1
                for target in ['A','B','H','E']:
                    mask=arrays[target+'_mask'].astype(bool)
                    p=numpy_project(volume,arrays[target+'_input_station_origin_m'],arrays[target+'_input_ray_direction'],
                        arrays[target+'_input_t_near'],arrays[target+'_input_path_length_m'],mask)
                    y=arrays[target+'_input_projection_raw']
                    stored=arrays[branch+'_'+target+'_projection']
                    error=float(np.abs(p[mask]-stored[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6))
                    max_error=max(max_error,error)
                    assert error<2e-5,(key,branch,target,error)
                    scores[branch,target]=float(np.abs(p[mask]-y[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6))
                    losses[branch,target]=discrepancy(p,y,mask)
            a,b=arrays['A_volume'].astype(np.float64),arrays['B_volume'].astype(np.float64)
            consistency=float(np.abs(a-b).mean()/(.5*(np.abs(a).mean()+np.abs(b).mean())+1e-6))
            fitted=dict(observed=.5*(scores['A','A']+scores['B','B']),cross=.5*(scores['A','B']+scores['B','A']),heldout=.5*(scores['A','H']+scores['B','H']))
            extra=.5*(scores['A','E']+scores['B','E'])
            loss=.5*(losses['A','H']+losses['B','H'])+.25*(losses['A','B']+losses['B','A'])+.05*(losses['A','A']+losses['B','B'])+.05*consistency
            for name,value in fitted.items():np.testing.assert_allclose(value,metric['measurement'][name],rtol=2e-5,atol=2e-6)
            np.testing.assert_allclose(extra,metric['extra_E_H'],rtol=2e-5,atol=2e-6)
            np.testing.assert_allclose(loss,metric['loss'],rtol=2e-5,atol=2e-7)
            records.append(dict(point=key,measurement=fitted,E=extra,loss=loss,
                fit=all(v<=.9 for v in fitted.values()) and loss<r['initial']['loss'],E_below099=extra<=.99))
        print('Independent network solver',key,fitted,'E',extra,flush=True)
    directory=ROOT/'case0_teacher_probe'
    journal=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
    assert len(journal)==r['actual_updates']==200
    assert [v['update'] for v in journal]==list(range(1,201))
    assert all(v['actual_parameter_delta']>0 and v['gradient_norm']>0 for v in journal)
    assert sha(directory/'last.pt')==r['checkpoint_sha256']
    with np.load(r['teacher_source'],allow_pickle=False) as teacher_arrays:
        teacher=teacher_arrays['A_volume'].astype(np.float64)
    assert sha(r['teacher_source'])==r['teacher_sha256']
    for key in ['initial','final']:
        with np.load(r[key]['arrays']['path'],allow_pickle=False) as arrays:
            predictions=[arrays[b+'_volume'].astype(np.float64) for b in ['A','B']]
            mse=float(np.mean([np.mean((p-teacher)**2) for p in predictions])/np.mean(teacher**2))
            l1=float(np.mean([np.sum(np.abs(p-teacher)) for p in predictions])/np.sum(np.abs(teacher)))
            np.testing.assert_allclose(mse,r[key]['teacher_normalized_MSE'],rtol=2e-5,atol=1e-7)
            np.testing.assert_allclose(l1,r[key]['teacher_relative_L1'],rtol=2e-5,atol=1e-7)
    write(ART/'teacher_independent_review.json',dict(passed=True,full_fields=4,physical_integrals=16,
        records=records,max_projection_relative_to_target=max_error,actual_teacher_updates=200,
        teacher_is_GT=False,new_optimizer_updates=0))


if __name__=='__main__':main()
