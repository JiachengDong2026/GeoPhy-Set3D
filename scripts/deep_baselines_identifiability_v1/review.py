"""Independent NumPy reconstruction from saved arrays, not training score helpers."""
from diag_context import *
import csv

def metrics(pred,y,mask):
    p=np.asarray(pred,dtype=np.float64)[mask];t=np.asarray(y,dtype=np.float64)[mask];err=p-t
    return float(np.abs(err).sum()/max(np.abs(t).sum(),1e-6)),float(np.sqrt(np.square(err).sum()/max(np.square(t).sum(),1e-12)))

def check(a,b):
    assert abs(a-b)<=1e-10+1e-8*max(abs(a),abs(b)),(a,b)

def main():
    registration=verified('registration.json');pre=verified('preregistration.json')
    c.verify_hashes(registration['sources']);c.verify_hashes(pre['historical_hashes'])
    details=[];count=0;total=0;paired={}
    with TrainGuard(synthetic_truth=True) as access:
        for path in sorted((ROOT/'runs').glob('*/result.json')):
            run=c.verified(path);target=path.parent;protocol=c.verified(target/'protocol.json')
            history=[c.verified(target/'initial.json'),*c.rows(target/'evaluations.jsonl')]
            assert len(history)==7 and [r['update'] for r in history]==[0,10,25,50,100,150,200]
            logs=c.rows(target/'updates.jsonl');assert len(logs)==200 and [r['update'] for r in logs]==list(range(1,201))
            assert not any(r['H_in_training'] or r['truth_in_training'] or r['forbidden_accesses'] for r in logs)
            ck=torch.load(run['checkpoint'],map_location='cpu',weights_only=False)
            assert ck['updates']==200 and all(int(v['step'])==200 for v in ck['optimizer']['state'].values())
            assert c.sha(run['checkpoint'])==run['checkpoint_sha256'];total+=200
            assert protocol['initial_state_sha256']==registration['initial_states'][protocol['arm']]
            recomputed=[]
            for evaluation in history:
                c.verify_hashes(evaluation['raw_sha256']);rowcases=[]
                for case in evaluation['cases']:
                    idx=case['case_index'];p=target/f"raw/update{evaluation['update']:03d}_case{idx}.npz"
                    with np.load(p,allow_pickle=False) as a:
                        trains=[];held=[];healthy=[];response=[]
                        for branch in ('A','B'):
                            field=a['field_'+branch].astype(float);mean=field.mean();std=field.std()
                            check(mean,case['branches'][branch]['field']['mean']);check(std,case['branches'][branch]['field']['std'])
                            healthy.append(mean>=1e-8 and std/max(mean,1e-15)>=.05)
                            changes=[]
                            for kind in ('zero','permuted'):
                                d=a['field_'+branch+'_'+kind].astype(float)-field;mae=np.abs(d).mean()
                                check(mae,case['branches'][branch]['interventions'][kind]['mae'])
                                changes.append(mae/design()['initial_mass_fraction']>=.001)
                            response.append(any(changes))
                            for group in ('A','B','H'):
                                l1,l2=metrics(a['p_'+branch+'_'+group],a['y_'+group],a['mask_'+group])
                                z1,z2=metrics(np.zeros_like(a['y_'+group]),a['y_'+group],a['mask_'+group])
                                check(z1,1.);check(z2,1.)
                                stored=next(r for r in case['measurement'] if r['prediction']==branch and r['target']==group)
                                check(l1,stored['relative_l1']);check(l2,stored['relative_l2']);count+=2
                                (held if group=='H' else trains).append((l1,l2))
                            if protocol['experiment']=='B':
                                truth=np.load(ROOT/f'inputs/synthetic_truth_{idx}.npy',allow_pickle=False).astype(float)
                                err=field[0,0]-truth;volume=case['branches'][branch]['synthetic_volume']
                                check(float(np.abs(err).sum()/np.abs(truth).sum()),volume['relative_l1'])
                                check(float(np.abs(err).mean()),volume['mae']);check(float(np.sqrt(np.square(err).mean())),volume['rmse'])
                        ab=float(np.mean([v[0] for v in trains]));h=float(np.mean([v[0] for v in held]))
                        check(ab,case['train_relative_l1']);check(h,case['H_relative_l1'])
                        rowcases.append(dict(AB=ab,H=h,healthy=all(healthy),response=all(response)))
                        details.append(dict(run=target.name,experiment=protocol['experiment'],arm=protocol['arm'],version=protocol['version'],
                            n=protocol['n'],update=evaluation['update'],case=case['case'],case_index=idx,train_relative_l1=ab,H_relative_l1=h,
                            train_relative_l2=float(np.mean([v[1] for v in trains])),H_relative_l2=float(np.mean([v[1] for v in held])),
                            output_mean_A=case['branches']['A']['field']['mean'],spatial_cv_A=case['branches']['A']['field']['spatial_cv'],
                            intervention_zero_MAE_A=case['branches']['A']['interventions']['zero']['mae'],
                            synthetic_relative_l1_A=case['branches']['A'].get('synthetic_volume',{}).get('relative_l1')))
                check(np.mean([r['AB'] for r in rowcases]),evaluation['train_relative_l1'])
                check(np.mean([r['H'] for r in rowcases]),evaluation['H_relative_l1']);recomputed.append(rowcases)
            train=all(r['AB']<=.99 and recomputed[0][i]['AB']-r['AB']>=.01 for i,r in enumerate(recomputed[-1]))
            held=all(r['H']<=.99 for ev in recomputed[-2:] for r in ev)
            healthy=all(r['healthy'] for r in recomputed[-1]);response=all(r['response'] for r in recomputed[-1])
            grad=history[-1]['layer_gradients']['observed']['layers']['token_measurement_channel']
            grad_ok=grad['l2']>1e-12 and grad['finite_fraction']==1
            fd_ok=False
            for r in history[-1]['finite_difference']:
                err=abs(r['AD']-r['FD']);check(err,r['abs_error'])
                if r['function']=='output_mean_div_fixed_scale' and max(abs(r['AD']),abs(r['FD']))>1e-8 and err<=1e-8+.05*max(abs(r['AD']),abs(r['FD'])):fd_ok=True
            checks=dict(train_improvement=train,independent_H=held,nontrivial_structure=healthy,
                meaningful_repeated_response=response,input_gradient=grad_ok,finite_difference=fd_ok)
            assert checks==run['checks'],(target.name,checks,run['checks'])
            assert all(checks.values())==(run['status']=='GO_DIAGNOSTIC_ONLY')
            paired.setdefault((protocol['experiment'],protocol['arm'],protocol['n']),{})[protocol['version']]=protocol
        assert total==2000 and len(list((ROOT/'runs').glob('*/result.json')))==10
        for (exp,arm,n),versions in paired.items():
            if exp=='B':assert versions['v2']['initial_state_sha256']==versions['v3']['initial_state_sha256'] and versions['v2']['inputs_sha256']==versions['v3']['inputs_sha256']
        c_states=0;c_hashes=set()
        for arm in ARMS:
            cc=c.verified(ROOT/f'C/{arm}/result.json');assert cc['optimizer_updates']==0 and not cc['audit']['denied']
            for s in cc['states']:
                c_states+=1
                for case in s['cases']:
                    assert case['H_in_backward'] is False
                    if case['raw'] in c_hashes:continue
                    c_hashes.add(case['raw']);assert c.sha(case['raw'])==case['raw_sha256']
                    with np.load(case['raw'],allow_pickle=False) as arrays:
                        for response in case['responses']:
                            key=response['branch']+'_'+response['intervention']
                            l1,l2=metrics(arrays[key+'_pH'],arrays['y_H'],arrays['mask_H'])
                            check(l1,response['H']['relative_l1']);check(l2,response['H']['relative_l2']);count+=2
        assert c_states==14
        assert c.read(legacy.ROOT/'queue_status.json')['state']=='paused_by_user'
    out=dict(at=c.now(),passed=True,independent_numpy=True,recomputed_measurement_fields=count,
        training_updates=total,C_states=c_states,C_updates=0,H_in_training=False,
        frozen_hashes_unchanged=True,paired_B_initialization_and_inputs_equal=True,access=access.summary())
    freeze('independent_review.json',out)
    path=ROOT/'case_metrics.csv'
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    print('INDEPENDENT REVIEW',json.dumps(out),flush=True)

if __name__=='__main__':main()
