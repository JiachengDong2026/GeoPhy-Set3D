"""Review all train targets and independently integrate six preregistered fields."""
from feasible_context import *
sys.path.insert(0,str(PROJECT/'scripts/unet_baseline_v1'))
from numpy_projector import project as numpy_project


def main():
    setup();verify();r=read(ART/'teachers.json');assert len(r['cases'])==166
    ds=make_dataset(config());records=[];maximum=0.;independent=0
    for index,row in enumerate(r['cases']):
        assert row['index']==index and row['case_id']==ds.records[index]['case_id']
        assert row['accepted_iterations']<=60 and row['function_evaluations']<=120
        assert sha(row['path'])==row['sha256']
        data=torch.load(row['path'],map_location='cpu',weights_only=False);field=data['concentration']
        assert field.shape==(1,1,64,64,64) and torch.isfinite(field).all() and field.min()>=0 and field.max()<=1
        assert not row['GT'] and row['sampling_epoch']==0
        if index in [0,1,2,82,83,165]:
            batch=dual_measurement_collate([ds[index]]);scores={}
            for key in ['A','B','H']:
                b=batch[key];mask=(b['input_valid_ray'].bool()&b['view_mask'][:,:,None,None,None]).numpy()
                p=numpy_project(field.numpy(),b['input_station_origin_m'].numpy(),b['input_ray_direction'].numpy(),
                    b['input_t_near'].numpy(),b['input_path_length_m'].numpy(),mask)
                y=b['input_projection_raw'].numpy()
                with torch.no_grad():original=project_input(field,b).numpy()
                error=float(np.abs(p[mask]-original[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6))
                maximum=max(maximum,error);assert error<2e-5,(index,key,error)
                scores[key]=float(np.abs(p[mask]-y[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6));independent+=1
            for key,score in [('heldout',scores['H']),('observed',.5*(scores['A']+scores['B'])),('cross',.5*(scores['A']+scores['B']))]:
                np.testing.assert_allclose(score,row['measurement'][key],rtol=2e-5,atol=2e-6)
            records.append(dict(index=index,scores=scores))
            print('INDEPENDENT_TEACHER',index,scores,flush=True)
    assert not any(r['access'][k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
    fit=[all(v<=.9 for v in row['measurement'].values()) and row['final_loss']<row['initial_loss'] for row in r['cases']]
    write(ART/'teacher_review.json',dict(passed=True,fields_checked=166,independently_integrated_fields=6,
        independent_integrations=independent,maximum_projection_error_relative_to_target=maximum,
        nontrivial_fit_cases=sum(fit),H_mean=float(np.mean([v['measurement']['heldout'] for v in r['cases']])),
        cross_mean=float(np.mean([v['measurement']['cross'] for v in r['cases']])),independent_records=records,new_optimizer_updates=0))


if __name__=='__main__':main()
