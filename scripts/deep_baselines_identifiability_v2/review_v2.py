"""Independent NumPy reconstruction; no runtime metric/decision imports."""
from ctx_v2 import *
import csv

def check(a,b):
    if a is None or b is None:assert a is None and b is None,(a,b)
    else:assert abs(float(a)-float(b))<=1e-10+1e-8*max(abs(float(a)),abs(float(b))),(a,b)
def csv_write(p,items):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in items for k in r)),lineterminator='\n');w.writeheader();w.writerows(items)
def l1(x,t):return float(np.sum(np.abs(x-t))/max(float(np.sum(np.abs(t))),1e-6))
def l2(x,t):return float(np.sqrt(np.sum((x-t)**2))/max(float(np.sqrt(np.sum(t*t))),1e-6))
def diff(a,b):
    a=a.astype(np.float64).reshape(-1);b=b.astype(np.float64).reshape(-1);error=a-b
    u=a-a.mean();v=b-b.mean();den=np.sqrt((u*u).sum()*(v*v).sum())
    return dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error*error).mean())),fixed_scale_response=float(np.abs(error).mean()/design()['initial_mass_fraction']),
        spatial_correlation=None if den<1e-30 else float(np.sum(u*v)/den),
        peak_displacement_m=float(np.sqrt(np.sum((np.array(np.unravel_index(a.argmax(),(64,)*3))-np.array(np.unravel_index(b.argmax(),(64,)*3)))**2))*.6/63))

def main():
    reg=verified(ROOT/'registration.json');verify(reg['sources']);verify(reg['inputs'])
    pre=verified(ROOT/'preregistration.json');verify(pre['historical_tracked']);verify(pre['frozen_main_checkpoints'])
    geometry=verified(ROOT/'input_registration.json');assert geometry['all_case_geometry_equal'] and geometry['all_measurements_and_truth_distinct']
    real_reads=[p for p in geometry['audit']['reads'] if Path(p).is_relative_to(DATA)]
    assert real_reads==[str(GEOMETRY.resolve())] and not geometry['audit']['denied']
    cases_csv=[];pairs_csv=[];swaps_csv=[];summaries=[];count=0;total=0
    with Boundary('audit') as access:
        truth=[np.load(ROOT/f'inputs/truth_{i}.npy',allow_pickle=False) for i in range(8)]
        ab=torch.load(ROOT/'inputs/train_AB.pt',map_location='cpu',weights_only=False)+torch.load(ROOT/'inputs/unseen_AB.pt',map_location='cpu',weights_only=False)
        for i,sample in enumerate(ab):
            for b in ('A','B'):
                assert value_hash(nonmeasurement(sample[b]))==geometry['identity'][i]['geometry'][b]
                assert tensor_hash(sample[b]['input_projection_raw'])==geometry['identity'][i]['measurements'][b]
        for arm in design()['arms']:
            path=ROOT/'runs'/arm;train=verified(path/'training_result.json');result=verified(path/'evaluation_result.json')
            assert train['updates']==200 and train['evaluation_in_this_process'] is False and not train['access']['denied']
            assert list(train['access']['reads'])==[str(ROOT/'inputs/train_AB.pt')]
            logs=rows(path/'updates.jsonl');assert [v['update'] for v in logs]==list(range(1,201));total+=len(logs)
            assert all(not r['H_in_training'] and not r['truth_in_training'] and r['learning_rate']==2e-4 for r in logs)
            assert all(r['case_indices']==[((r['update']-1)*2+i)%6 for i in range(2)] for r in logs)
            history=[]
            for update in design()['evaluation_updates']:
                ev=verified(path/f'evaluation_{update:03d}.json');verify(ev['raw_sha256'])
                ckpath=path/f'checkpoint_{update:03d}.pt';assert sha(ckpath)==ev['checkpoint_sha256']
                ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert ck['updates']==update
                if update:assert all(int(v['step'])==update for v in ck['optimizer']['state'].values())
                else:assert state_hash(ck['model'])==reg['initial_states'][arm] and not ck['optimizer']['state']
                del ck
                fields={};recomputed_cases=[]
                for case in ev['cases']:
                    i=case['index'];details={};m=[]
                    with np.load(path/f'raw/update{update:03d}_case{i}.npz',allow_pickle=False) as arrays:
                        for b in ('A','B'):
                            value=arrays['field_'+b].astype(np.float64).reshape(64,64,64);fields[i,b]=value
                            assert np.isfinite(value).all() and value.min()>=0 and value.max()<=1
                            vol=case['branches'][b]['volume'];mean=float(value.mean());std=float(value.std())
                            vl1=l1(value,truth[i]);check(vl1,vol['relative_l1']);check(np.abs(value-truth[i]).mean(),vol['mae']);check(np.sqrt(np.mean((value-truth[i])**2)),vol['rmse'])
                            for k,x in dict(mean=mean,std=std,spatial_cv=std/max(mean,1e-15),min=value.min(),max=value.max(),nearzero_fraction=(value<1e-10).mean()).items():check(x,vol[k])
                            assert np.allclose(vol['peak_xyz_m'],np.asarray(np.unravel_index(value.argmax(),value.shape))*.6/63,rtol=0,atol=1e-15)
                            grad=case['branches'][b]['gradient'];assert grad['optimizer_updates']==0
                            assert all(grad[k]['finite_fraction']==1 for k in ('measurement_channel','geometry_channels','readout_weight','readout_bias'))
                            for kind in ('zero','permuted'):
                                delta=diff(arrays['field_'+b+'_'+kind],value)
                                stored=case['branches'][b]['interventions'][kind]
                                for k,v in delta.items():check(v,stored[k])
                                assert stored['repeat_bitwise']
                            perbranch=[]
                            for target in ('A','B','H'):
                                mask=arrays['mask_'+target];y=arrays['y_'+target][mask].astype(np.float64);pred=arrays['p_'+b+'_'+target][mask].astype(np.float64)
                                h1=l1(pred,y);h2=l2(pred,y)
                                r=next(v for v in case['measurement'] if v['branch']==b and v['target']==target)
                                check(h1,r['relative_l1']);check(h2,r['relative_l2']);check(l1(np.zeros_like(y),y),1.);check(l2(np.zeros_like(y),y),1.)
                                perbranch.append(dict(target=target,relative_l1=h1,relative_l2=h2));m.append(dict(branch=b,**perbranch[-1]));count+=2
                            details[b]=dict(volume_L1=vl1,mean=mean,cv=std/max(mean,1e-15),nearzero=vol['nearzero_fraction'],gradient=grad)
                            cases_csv.append(dict(arm=arm,update=update,index=i,role=case['role'],branch=b,name=case['name'],
                                AB_L1=float(np.mean([x['relative_l1'] for x in perbranch[:2]])),AB_L2=float(np.mean([x['relative_l2'] for x in perbranch[:2]])),
                                H_L1=perbranch[-1]['relative_l1'],H_L2=perbranch[-1]['relative_l2'],volume_L1=vl1,volume_MAE=vol['mae'],volume_RMSE=vol['rmse'],
                                mean=mean,std=std,spatial_cv=std/max(mean,1e-15),nearzero_fraction=vol['nearzero_fraction'],min=vol['min'],max=vol['max'],
                                measurement_gradient_l2=grad['measurement_channel']['l2'],readout_gradient_l2=grad['readout_weight']['l2'],
                                zero_MAE=case['branches'][b]['interventions']['zero']['mae'],perm_MAE=case['branches'][b]['interventions']['permuted']['mae'],
                                checkpoint_sha256=ev['checkpoint_sha256']))
                    abmean=float(np.mean([r['relative_l1'] for r in m if r['target']!='H']));check(abmean,case['AB_L1'])
                    held=[r['relative_l1'] for r in m if r['target']=='H'];check(np.mean(held),case['H_L1'])
                    recomputed_cases.append(dict(AB=abmean,H=held,branches=details))
                pair_checks=[];swap_checks=[]
                assert len(ev['pairs'])==56 and len(ev['swaps'])==112
                for p in ev['pairs']:
                    i,j,b=p['i'],p['j'],p['branch'];a=fields[i,b];z=fields[j,b];delta=diff(a,z)
                    for k,v in delta.items():check(v,p[k])
                    dt=truth[i]-truth[j];dv=a-z;den=np.sqrt(np.sum(dt*dt)*np.sum(dv*dv));cos=None if den<1e-30 else float(np.sum(dt*dv)/den)
                    check(cos,p['direction_cosine']);check(np.abs(dt).mean(),p['truth_mae']);check(np.sqrt(np.mean(dt*dt)),p['truth_rmse'])
                    check(np.abs(dv).sum()/np.abs(dt).sum(),p['relative_to_truth_difference']);assert p['repeat_bitwise']
                    pair_checks.append(p)
                    pairs_csv.append(dict(arm=arm,update=update,**p))
                for s in ev['swaps']:
                    i,j,b=s['recipient'],s['donor'],s['branch'];delta=diff(fields[j,b],fields[i,b])
                    for k,v in delta.items():check(v,s[k])
                    matched=l1(fields[i,b],truth[i]);mismatch=l1(fields[j,b],truth[i]);check(mismatch-matched,s['matching_advantage_L1'])
                    assert s['input_equal_to_donor'] and s['repeat_bitwise']
                    source=batch([ab[i]],b);donor=batch([ab[j]],b)
                    for key in source:
                        if key in ('input_projection_raw','input_projection'):source[key]=donor[key].clone()
                    assert value_hash(source)==value_hash(donor)
                    swap_checks.append(s);swaps_csv.append(dict(arm=arm,update=update,**s))
                history.append(dict(cases=recomputed_cases,pairs=pair_checks,swaps=swap_checks))
            first=history[0]['cases'];last=history[-1]['cases'];g=design()['gates'];checks={}
            checks['seen_training']=all(c['AB']<=.99 and first[i]['AB']-c['AB']>=.01 for i,c in enumerate(last[:6]))
            checks['independent_H']=all(h<=.99 for e in history[-2:] for c in e['cases'] for h in c['H'])
            checks['nontrivial_structure']=all(b['mean']>=1e-8 and b['cv']>=.05 for c in last for b in c['branches'].values())
            checks['pair_response']=all(p['fixed_scale_response']>=.001 and p['repeat_bitwise'] for p in history[-1]['pairs'])
            for b in ('A','B'):
                for group in ('seen_seen','seen_unseen','unseen_unseen'):
                    vals=[p['direction_cosine'] or 0. for p in history[-1]['pairs'] if p['branch']==b and p['group']==group]
                    checks[f'direction_{b}_{group}']=np.count_nonzero(np.asarray(vals)>0)/len(vals)>=.8 and np.median(vals)>=.1
                checks['seen_volume_'+b]=sum(c['branches'][b]['volume_L1']<=.99 for c in last[:6])>=4
                checks['unseen_volume_'+b]=all(c['branches'][b]['volume_L1']<=.99 for c in last[6:])
                for role in ('seen','unseen'):
                    vals=[s['matching_advantage_L1'] for s in history[-1]['swaps'] if s['branch']==b and s['recipient_role']==role]
                    checks['matching_'+b+'_'+role]=np.count_nonzero(np.asarray(vals)>=.01)/len(vals)>=.75
            checks['gradients']=all(b['gradient']['measurement_channel']['l2']>1e-12 for c in last for b in c['branches'].values())
            checks={k:bool(v) for k,v in checks.items()};assert checks==result['checks'],(arm,checks,result['checks'])
            nearzero=any(b['mean']<1e-8 or b['nearzero']>.95 for c in last for b in c['branches'].values())
            status='GO_DIAGNOSTIC_ONLY' if all(checks.values()) else 'NO_GO' if nearzero or not checks['pair_response'] else 'INCONCLUSIVE_BUDGET_EXHAUSTED'
            assert status==result['status'];summaries.append(dict(arm=arm,status=status,checks=checks,updates=200,checkpoint_sha256=result['checkpoint_sha256']))
        assert total==400 and read(ROOT.parent/'deep_baselines_v1/queue_status.json')['state']=='paused_by_user'
    for name,data in [('case_metrics.csv',cases_csv),('pair_metrics.csv',pairs_csv),('swap_metrics.csv',swaps_csv)]:csv_write(ROOT/name,data)
    freeze(ROOT/'independent_review.json',dict(at=now(),passed=True,independent_numpy=True,measurement_fields_recomputed=count,
        cases=len(cases_csv),pairs=len(pairs_csv),directed_swaps=len(swaps_csv),new_updates=400,cumulative_updates=2400,remaining_budget=100,
        historical_hashes_unchanged=True,main_checkpoint_hashes_unchanged=True,all_geometry_equal=True,
        training_process_only_read_train_AB=True,summaries=summaries,audit=access.summary()))
    print('INDEPENDENT REVIEW PASSED',total,'updates',count,'measurement fields',flush=True)

if __name__=='__main__':main()
