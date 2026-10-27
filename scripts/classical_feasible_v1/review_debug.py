"""Independent full-field NumPy integration and dense-target checks for debug runs."""
from feasible_context import *
sys.path.insert(0,str(PROJECT/'scripts/voxel_probe_v1'))
from review import numpy_project,discrepancy


def main():
    setup();verify();summary=[];maximum=0.;fields=0
    teacher=torch.load(ROOT/'teachers/000.pt',map_location='cpu',weights_only=False)['concentration'].numpy().astype('float64')
    for arm in ['deeponet','fno','senseiver','gnot','deepsets']:
        r=read(ART/f'debug_{arm}.json');assert r['status']=='bounded_complete'
        assert sha(ROOT/'teachers/000.pt')==r['teacher_sha256']
        records=[]
        for key in ['initial','dense_final','final']:
            row=r[key];assert sha(row['arrays']['path'])==row['arrays']['sha256']
            scores={};losses={}
            with np.load(row['arrays']['path'],allow_pickle=False) as arrays:
                for branch in ['A','B']:
                    volume=arrays[branch+'_volume'];fields+=1
                    for target in ['A','B','H','E']:
                        mask=arrays[target+'_mask'].astype(bool);y=arrays[target+'_input_projection_raw']
                        p=numpy_project(volume,arrays[target+'_input_station_origin_m'],arrays[target+'_input_ray_direction'],
                            arrays[target+'_input_t_near'],arrays[target+'_input_path_length_m'],mask)
                        error=float(np.abs(p[mask]-arrays[branch+'_'+target+'_projection'][mask]).sum()/max(np.abs(y[mask]).sum(),1e-6))
                        maximum=max(maximum,error);assert error<2e-5,(arm,key,branch,target,error)
                        scores[branch,target]=float(np.abs(p[mask]-y[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6))
                        losses[branch,target]=discrepancy(p,y,mask)
                a,b=arrays['A_volume'].astype('float64'),arrays['B_volume'].astype('float64')
                consistency=np.abs(a-b).mean()/(.5*(np.abs(a).mean()+np.abs(b).mean())+1e-6)
                loss=.5*(losses['A','H']+losses['B','H'])+.25*(losses['A','B']+losses['B','A'])+.05*(losses['A','A']+losses['B','B'])+.05*consistency
                for name,pairs in [('observed',[('A','A'),('B','B')]),('cross',[('A','B'),('B','A')]),('heldout',[('A','H'),('B','H')])]:
                    np.testing.assert_allclose(np.mean([scores[x] for x in pairs]),row['measurement'][name],rtol=2e-5,atol=2e-6)
                extra=.5*(scores['A','E']+scores['B','E']);np.testing.assert_allclose(extra,row['extra_E_H'],rtol=2e-5,atol=2e-6)
                np.testing.assert_allclose(loss,row['loss'],rtol=2e-5,atol=2e-7)
                mse=.5*((a-teacher)**2+(b-teacher)**2).mean()/(teacher**2).mean()
                np.testing.assert_allclose(mse,row['teacher_normalized_MSE'],rtol=2e-5,atol=2e-6)
                records.append(dict(point=key,loss=float(loss),E=float(extra),teacher_MSE=float(mse)))
        directory=ROOT/'debug'/arm
        for phase in ['dense','physical']:
            journal=[json.loads(line) for line in (directory/(phase+'_updates.jsonl')).read_text().splitlines()]
            assert len(journal)==200 and [v['update'] for v in journal]==list(range(1,201))
            assert all(v['gradient_norm_after_clip']>0 and v['parameter_delta']>0 for v in journal)
        ck=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
        assert ck['optimizer_steps']==200 and ck['phase']=='physical' and ck['scheduler']['last_epoch']==12
        assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={200}
        assert sha(directory/'last.pt')==r['checkpoint_sha256']
        passed=r['dense_MSE_half'] and r['fitted_ABH'] and r['E_below099']
        summary.append(dict(arm=arm,review_passed=True,local_fit_passed=passed,records=records))
        print('REVIEW_DEBUG',arm,'local_fit',passed,'H',r['final']['measurement']['heldout'],'E',r['final']['extra_E_H'],flush=True)
    write(ART/'debug_independent_review.json',dict(passed=True,full_fields=fields,physical_integrals=fields*4,
        maximum_projection_error_relative_to_target=maximum,models=summary,new_optimizer_updates=0))


if __name__=='__main__':main()
