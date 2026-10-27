"""Independent final best-checkpoint field health and causal measurement checks."""
from feasible_context import *
sys.path.insert(0,str(PROJECT/'scripts/unet_baseline_v1'))
from numpy_projector import project as numpy_project


def main():
    setup();verify();registration=read(ART/'selected_checkpoint_review_registration.json')
    for path,digest in registration['sources'].items():assert sha(PROJECT/path)==digest,path
    summary=read(ART/'unet_summary.json');assert summary['formal_complete']
    validation=read(ART/'unet_validation_review.json');assert validation['passed']
    checkpoint=Path(summary['best_checkpoint']);assert sha(checkpoint)==summary['best_checkpoint_sha256']==validation['checkpoint_sha256']
    model=build('unet');ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model']);model.eval();del ck
    target=ROOT/'selected_unet_causal_review';target.mkdir(exist_ok=False)
    guard=GTReadGuard(DATA,ROOT);records=[];health=[];maximum=0.
    with guard,torch.no_grad():
        for split,count in [('val',4),('train',3)]:
            ds=make_dataset(config(),validation=split=='val');ds.set_epoch(0)
            for index in range(count):
                sample=ds[index];batch=dual_measurement_collate([sample]);h=batch['H']
                mask=(h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]).numpy()
                y=h['input_projection_raw'].numpy();denominator=max(np.abs(y[mask]).sum(),1e-6)
                _,raw=ds._projections(index+1);values=[]
                arrays={key:value.numpy() for key,value in h.items() if isinstance(value,torch.Tensor)};arrays['mask']=mask
                for branch in ['A','B']:
                    aligned=dict(sample[branch]);aligned.update(ds._views(raw,sample[branch]['input_station_indices'].numpy(),'input'))
                    donor=projection_pool_collate([aligned]);zero=dict(batch[branch]);wrong=dict(batch[branch])
                    for key in ['input_projection_raw','input_projection']:
                        zero[key]=torch.zeros_like(zero[key]);wrong[key]=donor[key]
                    for key in ['input_station_origin_m','input_ray_direction','input_t_near','input_path_length_m','input_valid_ray','view_mask','input_station_indices']:
                        assert torch.equal(batch[branch][key],zero[key]) and torch.equal(batch[branch][key],wrong[key])
                    fields={};scores={}
                    for intervention,inputs in [('normal',batch[branch]),('zero',zero),('wrong',wrong)]:
                        field=model(inputs);volume=field.numpy()
                        p=numpy_project(volume,h['input_station_origin_m'].numpy(),h['input_ray_direction'].numpy(),h['input_t_near'].numpy(),h['input_path_length_m'].numpy(),mask)
                        error=float(np.abs(p[mask]-project_input(field,h).numpy()[mask]).sum()/denominator)
                        maximum=max(maximum,error);assert error<2e-5
                        scores[intervention]=float(np.abs(p[mask]-y[mask]).sum()/denominator)
                        fields[intervention]=volume.astype('float64');arrays[branch+'_'+intervention]=volume
                        a=fields[intervention]
                        stats=dict(split=split,index=index,branch=branch,intervention=intervention,mean=float(a.mean()),
                            std=float(a.std()),nearzero=float((a<1e-10).mean()),saturated=float((volume*(1-volume)<4.54e-5).mean()))
                        assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
                        health.append(stats)
                    n,z=fields['normal'],fields['zero']
                    values.append(dict(normal_H=scores['normal'],zero_benefit=scores['zero']-scores['normal'],wrong_increase=scores['wrong']-scores['normal'],
                        response_relative=float(np.abs(n-z).sum()/max(np.abs(n).sum(),1e-30)),response_absolute=float(np.abs(n-z).mean())))
                row=dict(split=split,index=index,case_id=ds.records[index]['case_id'],**{key:float(np.mean([v[key] for v in values])) for key in values[0]})
                path=target/f'{split}{index:03d}.npz';np.savez_compressed(path,**arrays);row.update(path=str(path),sha256=sha(path));records.append(row)
                print('SELECTED_UNET_REVIEW',split,index,'normalH',row['normal_H'],'zero_benefit',row['zero_benefit'],flush=True)
    normal_health=[r for r in health if r['intervention']=='normal']
    val=[r for r in records if r['split']=='val'];train=[r for r in records if r['split']=='train']
    checks=dict(healthy=healthy(normal_health),Val_H=validation['independent_Val_H']<=.99,
        improvement=validation['independent_Val_H']<=.95*summary['initial_val_H'] and summary['initial_val_H']-validation['independent_Val_H']>=.01,
        val_benefit=sum(r['zero_benefit']>=.001 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in val)>=2,
        train_dependency=sum(r['wrong_increase']>=1e-6 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in train)>=2)
    write(ART/'selected_unet_causal_review.json',dict(passed=all(checks.values()),checks=checks,records=records,health=health,
        checkpoint_sha256=sha(checkpoint),independent_projection_integrals=42,full_fields=42,maximum_projection_error=maximum,
        access=guard.summary(),GT_used=False,new_optimizer_updates=0))
    assert all(checks.values()),'Selected final checkpoint failed registered causal qualification; do not test'
    verify()


if __name__=='__main__':main()
