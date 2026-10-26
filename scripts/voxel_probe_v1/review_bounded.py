"""Independent all-target physical recomputation for bounded solver fields."""
from review import *


def main():
    protocol=read(ART/'bounded_solver_registration.json')
    assert sha(PROJECT/protocol['source'])==protocol['source_sha256']
    rows=[]
    fields=0
    max_error=0.
    for index in protocol['indices']:
        result=read(ART/f'case{index}_bounded_solver.json')
        for key in ['initial','final']:
            metric=result[key]
            entry=metric['arrays']
            assert sha(entry['path'])==entry['sha256']
            scores={};losses={}
            with np.load(entry['path'],allow_pickle=False) as arrays:
                for branch in ['A','B']:
                    volume=arrays[branch+'_volume']
                    assert np.isfinite(volume).all() and volume.min()>=0 and volume.max()<=1
                    fields+=1
                    for target in ['A','B','H','E']:
                        mask=arrays[target+'_mask'].astype(bool)
                        p=numpy_project(volume,arrays[target+'_input_station_origin_m'],arrays[target+'_input_ray_direction'],
                            arrays[target+'_input_t_near'],arrays[target+'_input_path_length_m'],mask)
                        y=arrays[target+'_input_projection_raw']
                        stored=arrays[branch+'_'+target+'_projection']
                        error=float(np.abs(p[mask]-stored[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6))
                        max_error=max(max_error,error)
                        assert error<2e-5,(index,key,branch,target,error)
                        scores[branch,target]=float(np.abs(p[mask]-y[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6))
                        losses[branch,target]=discrepancy(p,y,mask)
                fitted=dict(observed=.5*(scores['A','A']+scores['B','B']),cross=.5*(scores['A','B']+scores['B','A']),heldout=.5*(scores['A','H']+scores['B','H']))
                extra=.5*(scores['A','E']+scores['B','E'])
                loss=.5*(losses['A','H']+losses['B','H'])+.25*(losses['A','B']+losses['B','A'])+.05*(losses['A','A']+losses['B','B'])
                assert np.array_equal(arrays['A_volume'],arrays['B_volume'])
                for name,value in fitted.items():np.testing.assert_allclose(value,metric['measurement'][name],rtol=2e-5,atol=2e-6)
                np.testing.assert_allclose(extra,metric['extra_E_H'],rtol=2e-5,atol=2e-6)
                np.testing.assert_allclose(loss,metric['loss'],rtol=2e-5,atol=2e-7)
                fit=all(v<=.9 for v in fitted.values()) and loss<result['initial']['loss']
                rows.append(dict(index=index,point=key,loss=loss,measurement=fitted,E=extra,fit=fit,E_below099=extra<=.99))
            print('Independent bounded solver',index,key,fitted,'E',extra,flush=True)
        assert rows[-1]['fit']==result['fitted_ABH'] and rows[-1]['E_below099']==result['E_below099']
        directory=ROOT/f'case{index}_bounded_solver'
        evaluations=[json.loads(s) for s in (directory/'function_evaluations.jsonl').read_text().splitlines()]
        accepted=[json.loads(s) for s in (directory/'accepted_iterations.jsonl').read_text().splitlines()]
        assert len(evaluations)==result['function_evaluations']<=120
        assert len(accepted)==result['accepted_iterations']<=60
        assert all(v['delta_l2']>0 for v in accepted)
        assert [v['evaluation'] for v in evaluations]==list(range(1,len(evaluations)+1))
        assert [v['iteration'] for v in accepted]==list(range(1,len(accepted)+1))
        assert sha(directory/'final_field.pt')==result['final_checkpoint_sha256']
    write(ART/'bounded_independent_review.json',dict(passed=True,full_fields=fields,physical_integrals=fields*4,rows=rows,
        max_projection_relative_to_target=max_error,network_updates=0,new_optimizer_iterations=0))


if __name__=='__main__':
    main()
