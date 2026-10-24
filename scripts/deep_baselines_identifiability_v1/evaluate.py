"""Evaluation-only H and synthetic truth capability; raw arrays remain in scratch."""
from diagnostics import *
from trace import layer_gradients, finite_difference

def evaluate(net,samples,indices,synthetic,target,update):
    net.eval();case_rows=[];raw_hashes={}
    with TrainGuard(synthetic_truth=True) as access,torch.no_grad():
        for pos,index in enumerate(indices):
            ab={k:batch([samples[pos]],k) for k in ('A','B')}
            h=batch([{'H':h_sample(index,synthetic)}],'H')
            targets=dict(ab,H=h);fields={k:net(ab[k]) for k in ('A','B')}
            arrays={};measurement=[];details={}
            for k,t in targets.items():
                arrays['y_'+k]=t['input_projection_raw'].cpu().numpy();arrays['mask_'+k]=valid(t).cpu().numpy()
            for label,field in fields.items():
                assert field.shape==(1,1,64,64,64) and field.dtype==torch.float32 and torch.isfinite(field).all() and field.min()>=0 and field.max()<=1
                arrays['field_'+label]=field.cpu().numpy()
                projections={k:project_input(field,t) for k,t in targets.items()}
                for key,pred in projections.items():
                    arrays['p_'+label+'_'+key]=pred.cpu().numpy()
                    value=scores(pred,targets[key])[0]
                    measurement.append(dict(prediction=label,target=key,**value))
                interventions={}
                for kind in ('zero','permuted'):
                    changed=net(intervene(ab[label],kind));repeated=net(intervene(ab[label],kind))
                    assert torch.equal(changed,repeated),'Intervention repeatability failed'
                    arrays['field_'+label+'_'+kind]=changed.cpu().numpy()
                    interventions[kind]=dict(**difference(changed,field),repeat_bitwise=True)
                details[label]=dict(field=field_stats(field),interventions=interventions)
                if synthetic:
                    truth=np.load(ROOT/f'inputs/synthetic_truth_{index}.npy',allow_pickle=False).astype(float)
                    value=field[0,0].cpu().double().numpy();error=value-truth
                    details[label]['synthetic_volume']=dict(relative_l1=float(np.sum(np.abs(error))/np.sum(np.abs(truth))),
                        mae=float(np.mean(np.abs(error))),rmse=float(np.sqrt(np.mean(error**2))),
                        truth_peak=float(truth.max()),truth_spatial_std=float(truth.std()),
                        truth_peak_xyz_m=[float(x*.6/63) for x in np.unravel_index(int(truth.argmax()),truth.shape)])
            train=[r for r in measurement if r['target']!='H'];held=[r for r in measurement if r['target']=='H']
            row=dict(case_index=index,case=samples[pos]['A']['case_id'],measurement=measurement,
                train_relative_l1=float(np.mean([r['relative_l1'] for r in train])),
                train_relative_l2=float(np.mean([r['relative_l2'] for r in train])),
                H_relative_l1=float(np.mean([r['relative_l1'] for r in held])),
                H_relative_l2=float(np.mean([r['relative_l2'] for r in held])),zero_train_l1=1.,zero_H_l1=1.,branches=details)
            path=target/f'raw/update{update:03d}_case{index}.npz';path.parent.mkdir(exist_ok=True)
            assert not path.exists();np.savez_compressed(path,**arrays);raw_hashes[str(path)]=c.sha(path)
            case_rows.append(row)
    # Gradient/FD uses A/B only; independent of all H/truth computations above.
    net.train()
    ab={k:batch([samples[0]],k) for k in ('A','B')}
    if update in (0,200):
        gradients=layer_gradients(net,ab,target.name.split('_')[2]);fd=finite_difference(net,ab['A'])
    else:gradients=None;fd=None
    return dict(update=update,cases=case_rows,train_relative_l1=float(np.mean([r['train_relative_l1'] for r in case_rows])),
        H_relative_l1=float(np.mean([r['H_relative_l1'] for r in case_rows])),
        train_relative_l2=float(np.mean([r['train_relative_l2'] for r in case_rows])),
        H_relative_l2=float(np.mean([r['H_relative_l2'] for r in case_rows])),
        raw_sha256=raw_hashes,layer_gradients=gradients,finite_difference=fd,evaluation_access=access.summary())

def decide(history):
    gate=design()['gate'];initial,last=history[0],history[-1]
    train_ok=all(r['train_relative_l1']<=gate['train_relative_l1_max'] and
        initial['cases'][i]['train_relative_l1']-r['train_relative_l1']>=gate['train_absolute_improvement_min'] for i,r in enumerate(last['cases']))
    h_ok=all(r['H_relative_l1']<=.99 for item in history[-2:] for r in item['cases'])
    healthy=all(b['field']['mean']>=gate['mean_min'] and b['field']['spatial_cv']>=gate['spatial_cv_min']
        for r in last['cases'] for b in r['branches'].values())
    sensitive=all(any(v['stabilized_relative_mae']>=gate['intervention_MAE_divided_by_fixed_init_scale_min'] and v['repeat_bitwise']
        for v in b['interventions'].values()) for r in last['cases'] for b in r['branches'].values())
    grads=last['layer_gradients'];g=grads['observed']['layers']['token_measurement_channel']
    grad_ok=g['finite_fraction']==1 and g['l2']>gate['measurement_input_gradient_l2_min']
    fd_ok=any(v['function']=='output_mean_div_fixed_scale' and v['observable'] and v['within_tolerance'] for v in last['finite_difference'])
    checks=dict(train_improvement=train_ok,independent_H=h_ok,nontrivial_structure=healthy,
                meaningful_repeated_response=sensitive,input_gradient=grad_ok,finite_difference=fd_ok)
    go=all(checks.values())
    nearzero=any(b['field']['mean']<gate['mean_min'] or b['field']['below_1e_10']>.95 for r in last['cases'] for b in r['branches'].values())
    if go:status='GO_DIAGNOSTIC_ONLY'
    elif nearzero or not sensitive:status='NO_GO'
    else:status='INCONCLUSIVE_BUDGET_EXHAUSTED'
    interpretation='A3_conditional_measurement_evidence' if go else 'A2_fits_training_not_independent_H' if train_ok and not h_ok else 'A1_no_substantial_training_fit' if not train_ok else 'unresolved_conditioning_or_structure'
    return dict(status=status,checks=checks,interpretation=interpretation,formal_training_authorized=False)
