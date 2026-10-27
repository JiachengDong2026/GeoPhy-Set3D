"""GT-free independent NumPy review of terminal multi-case pilots."""
from feasible_context import *
sys.path.insert(0,str(PROJECT/'scripts/unet_baseline_v1'))
from numpy_projector import project as numpy_project


def main():
    setup();verify();arm=sys.argv[1]
    assert arm in ['deeponet','fno','senseiver','gnot','deepsets']
    registration=read(ART/'multicase_review_registration.json')
    for path,digest in registration['sources'].items():assert sha(PROJECT/path)==digest,path
    summary=read(ART/(arm+'_pilot_summary.json'));source=ROOT/('pilot_'+arm)
    assert not summary['formal_complete'] and not summary['test_allowed']
    checkpoint=source/'last.pt';assert sha(checkpoint)==summary['last_checkpoint_sha256']
    rows=[json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
    physical=[r for r in rows if r['stage']=='physical']
    assert physical[-1]['epoch']==summary['epochs']
    assert len([r for r in rows if r['stage']=='preheat'])==60
    for stage,total in [('preheat',4980),('physical',summary['physical_updates'])]:
        journal=[json.loads(line) for line in (source/(stage+'_updates.jsonl')).read_text().splitlines()]
        assert len(journal)==total and [r['update'] for r in journal]==list(range(1,total+1))
        assert all(r['parameter_delta']>0 and r['gradient_norm_after_clip']>0 for r in journal)
    model=build(arm);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['epoch']==summary['epochs'] and ck['optimizer_steps']==summary['physical_updates']
    assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={summary['physical_updates']}
    model.load_state_dict(ck['model']);model.eval();del ck
    out=ROOT/('review_pilot_'+arm);out.mkdir(exist_ok=False)
    guard=GTReadGuard(DATA,ROOT);records=[];maximum=0.
    with guard,torch.no_grad():
        for split in ['val','train']:
            ds=make_dataset(config(),validation=split=='val');ds.set_epoch(0)
            for index in range(36 if split=='val' else 3):
                sample=ds[index];batch=dual_measurement_collate([sample]);h=batch['H']
                mask=(h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]).numpy()
                y=h['input_projection_raw'].numpy();denominator=max(np.abs(y[mask]).sum(),1e-6)
                arrays={key:value.numpy() for key,value in h.items() if isinstance(value,torch.Tensor)}
                arrays['mask']=mask;values=[]
                for branch in ['A','B']:
                    variants={'normal':batch[branch]}
                    if index<4:
                        _,raw=ds._projections(index+1)
                        aligned=dict(sample[branch]);aligned.update(ds._views(raw,sample[branch]['input_station_indices'].numpy(),'input'))
                        donor=projection_pool_collate([aligned]);zero=dict(batch[branch]);wrong=dict(batch[branch])
                        for key in ['input_projection_raw','input_projection']:
                            zero[key]=torch.zeros_like(zero[key]);wrong[key]=donor[key]
                        variants.update(zero=zero,wrong=wrong)
                    fields={};scores={}
                    for intervention,inputs in variants.items():
                        field=model(inputs);p=numpy_project(field.numpy(),h['input_station_origin_m'].numpy(),h['input_ray_direction'].numpy(),h['input_t_near'].numpy(),h['input_path_length_m'].numpy(),mask)
                        original=project_input(field,h).numpy()
                        error=float(np.abs(p[mask]-original[mask]).sum()/denominator);maximum=max(maximum,error)
                        assert error<2e-5
                        scores[intervention]=float(np.abs(p[mask]-y[mask]).sum()/denominator)
                        fields[intervention]=field.numpy();arrays[branch+'_'+intervention]=field.numpy()
                    item=dict(normal_H=scores['normal'])
                    if 'zero' in fields:
                        n,z=fields['normal'],fields['zero']
                        item.update(zero_benefit=scores['zero']-scores['normal'],wrong_increase=scores['wrong']-scores['normal'],
                            response_relative=float(np.abs(n-z).sum()/max(np.abs(n).sum(),1e-30)),response_absolute=float(np.abs(n-z).mean()))
                    values.append(item)
                row=dict(split=split,index=index,case_id=ds.records[index]['case_id'],**{key:float(np.mean([v[key] for v in values])) for key in values[0]})
                path=out/f'{split}{index:03d}.npz';np.savez_compressed(path,**arrays)
                row.update(path=str(path),sha256=sha(path));records.append(row)
                print('REVIEW_MULTICASE',arm,split,index,'H',row['normal_H'],flush=True)
    val_H=float(np.mean([r['normal_H'] for r in records if r['split']=='val']))
    np.testing.assert_allclose(val_H,summary['last_val_H'],rtol=5e-5,atol=3e-6)
    for split,key in [('val','validation_interventions'),('train','training_interventions')]:
        for expected in physical[-1].get(key,[]):
            observed=next(r for r in records if r['split']==split and r['index']==expected['index'])
            for metric in ['normal_H','zero_benefit','wrong_increase','response_relative','response_absolute']:
                np.testing.assert_allclose(observed[metric],expected[metric],rtol=2e-3,atol=3e-6)
    assert not any(guard.summary()[key] for key in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
    write(ART/(arm+'_multicase_review.json'),dict(passed=True,terminal_epoch=summary['epochs'],
        independent_val_H=val_H,recorded_val_H=summary['last_val_H'],checkpoint_sha256=sha(checkpoint),
        maximum_projection_error=maximum,records=records,access=guard.summary(),new_updates=0,
        training_decision_preserved=summary['decision'],no_test_or_GT=True))
    verify()


if __name__=='__main__':main()
