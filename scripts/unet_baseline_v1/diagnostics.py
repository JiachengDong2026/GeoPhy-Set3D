"""Fixed normal/zero/wrong-case interventions, no volumetric targets."""
from unet_context import *
FIELDS={'input_projection_raw','input_projection'}
def fixed_identity(b):
    return tensor_identity(*[b[k] for k in sorted(b) if isinstance(b[k],torch.Tensor) and k not in FIELDS])

@torch.no_grad()
def diagnose(model,c,train,val,epoch,target):
    model.eval();rows=[];files=[]
    for split,ds,indices in [('train',train,[0,1,2]),('val',val,[0,1,2,3])]:
        old_epoch=int(ds._epoch.item());ds.set_epoch(0)
        for index in indices:
            sample=ds[index];cpu=dual_measurement_collate([sample]);b=move_dual_batch(cpu,'cuda:0')
            donor_row,p=ds._projections(index+1);arrays={}
            for branch in ['A','B']:
                original=b[branch];identity=fixed_identity(original)
                aligned=dict(sample[branch]);aligned.update(ds._views(p,sample[branch]['input_station_indices'].numpy(),'input'))
                donor=projection_pool_collate([aligned]);assert fixed_identity(donor)==fixed_identity(cpu[branch])
                preds={};scores={}
                h=b['H'];valid=h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]
                for mode in ['normal','zero','wrong_case']:
                    modified=dict(original)
                    if mode!='normal':
                        for k in FIELDS:modified[k]=torch.zeros_like(original[k]) if mode=='zero' else donor[k].to('cuda:0')
                    assert fixed_identity(modified)==identity
                    assert all(modified[k] is v for k,v in original.items() if k not in FIELDS)
                    pred,logits=model(modified,return_logits=True);check_prediction(pred,modified)
                    if mode=='normal':assert torch.equal(pred,model(modified)),'Repeat inference changed'
                    proj=project_input(pred,h);metric=relative_projection_metrics(proj,h['input_projection_raw'],valid)
                    out=pred.cpu().numpy();lg=logits.cpu().numpy();preds[mode]=out.astype('float64');scores[mode]=metric['relative_l1']
                    arrays[branch+'_'+mode+'_volume']=out;arrays[branch+'_'+mode+'_logits']=lg
                    arrays[branch+'_'+mode+'_H_projection']=proj.cpu().numpy()
                    arr=out.astype('float64');logit=lg.astype('float64')
                    row=dict(epoch=epoch,split=split,index=index,case_id=sample[branch]['case_id'],donor_case_id=donor_row['case_id'],branch=branch,mode=mode,
                        geometry_sha256=identity,mean=float(arr.mean()),std=float(arr.std()),nearzero=float((arr<1e-10).mean()),
                        min=float(arr.min()),max=float(arr.max()),logit_quantiles=np.quantile(logit,[0,.01,.5,.99,1]).tolist(),
                        sigmoid_derivative_mean=float((arr*(1-arr)).mean()),saturated=float((np.abs(logit)>10).mean()),H_relative_l1=metric['relative_l1'])
                    rows.append(row)
                rel=float(np.abs(preds['normal']-preds['zero']).sum()/max(np.abs(preds['normal']).sum(),1e-30))
                absolute=float(np.abs(preds['normal']-preds['zero']).mean());wrong=scores['wrong_case']-scores['normal'];benefit=scores['zero']-scores['normal']
                for r in rows:
                    if r['split']==split and r['index']==index and r['branch']==branch:
                        r.update(normal_zero_relative_MA=rel,normal_zero_absolute_MA=absolute,wrong_H_increase=wrong,zero_H_increase=benefit)
                
                for key in ['input_station_origin_m','input_ray_direction','input_t_near','input_path_length_m']:
                    arrays[branch+'_H_'+key]=h[key].cpu().numpy()
                arrays[branch+'_H_reference']=h['input_projection_raw'].cpu().numpy();arrays[branch+'_H_mask']=valid.cpu().numpy()
            path=target/f'{split}_{index}.npz';assert not path.exists();np.savez_compressed(path,**arrays);files.append(dict(path=str(path),sha256=sha(path)))
        ds.set_epoch(old_epoch)
    write(target/'diagnostics.json',dict(epoch=epoch,rows=rows,arrays=files))
    return rows

def gates(rows,validation,initial):
    checks={}
    normals=[r for r in rows if r['mode']=='normal']
    checks['healthy_fields']=all(r['mean']>=1e-8 and r['nearzero']<.95 and r['std']/max(r['mean'],1e-30)>=.05 and r['saturated']<.95 for r in rows)
    def cases(split):
        answer=[]
        for index in sorted({r['index'] for r in normals if r['split']==split}):
            pair=[r for r in normals if r['split']==split and r['index']==index]
            answer.append({k:sum(r[k] for r in pair)/len(pair) for k in ['normal_zero_relative_MA','normal_zero_absolute_MA','wrong_H_increase','zero_H_increase']})
        return answer
    tr,va=cases('train'),cases('val')
    checks['train_zero_2of3']=sum(r['normal_zero_relative_MA']>=1e-3 and r['normal_zero_absolute_MA']>=1e-8 for r in tr)>=2
    checks['train_wrong_2of3']=sum(r['wrong_H_increase']>=1e-6 for r in tr)>=2
    checks['val_zero_response_2of4']=sum(r['normal_zero_relative_MA']>=1e-3 and r['normal_zero_absolute_MA']>=1e-8 for r in va)>=2
    checks['val_zero_benefit_2of4']=sum(r['zero_H_increase']>=.001 for r in va)>=2
    score=selection_metric(validation);before=selection_metric(initial)
    checks['val_below_zero']=score<=.99
    checks['val_initial_improvement']=score<=.95*before and before-score>=.01
    return checks
