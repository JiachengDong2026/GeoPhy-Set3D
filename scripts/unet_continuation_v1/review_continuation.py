"""Independent complete-field review of the single optimizer-only network probe."""
from continuation_context import *
from review import numpy_project,discrepancy


def main():
    r=read(ART/'result.json')
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
        print('Independent continuation',key,fitted,'E',extra,flush=True)
    directory=ROOT/'case0'
    updates=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
    n=r['actual_additional_updates']
    assert len(updates)==n<=400 and [v['update'] for v in updates]==list(range(1,n+1))
    assert all(v['actual_parameter_delta']>0 and np.isfinite(v['gradient_norm']) and v['gradient_norm']>0 for v in updates)
    assert sha(directory/'last.pt')==r['checkpoint_sha256']
    ck=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
    assert ck['steps']==100+n and ck['added_updates']==n and ck['rng']
    assert ck['scheduler']['last_epoch']==(100+n)//16
    assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={100+n}
    assert all(torch.isfinite(v).all() for v in ck['model'].values())
    verify_all()
    assert not r['optimizer_reset'] and r['E_used_for_early_stopping']
    assert not any(r['access'][k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
    assert read(ART/'protocol.json')['engineering_amendment']['attempt1_updates']==0
    write(ART/'independent_review.json',dict(passed=True,full_fields=4,physical_integrals=16,
        records=records,max_projection_relative_to_target=max_error,actual_additional_updates=n,
        cumulative_original_optimizer_step=100+n,scheduler_last_epoch=(100+n)//16,
        frozen_sources_inputs_evidence_and_checkpoints_unchanged=True,new_optimizer_updates=0))
    print('Verified continuation updates, optimizer/scheduler, protected historical evidence',flush=True)


if __name__=='__main__':main()
