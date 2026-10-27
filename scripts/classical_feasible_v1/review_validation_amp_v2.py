"""Independent complete36Val H recomputation from frozen selected model, no GT."""
from feasible_context import *
from amplitude_adapter import AmplitudeModel
sys.path.insert(0,str(PROJECT/'scripts/unet_baseline_v1'))
from numpy_projector import project as numpy_project


def main():
    setup();verify();summary=read(ART/'unet_amp_summary.json');assert summary['formal_complete']
    assert sha(summary['best_checkpoint'])==summary['best_checkpoint_sha256']
    records=[json.loads(line) for line in (ROOT/'unet_amp_v2/metrics.jsonl').read_text().splitlines()]
    physical=[r for r in records if r['stage']=='physical'];assert len(physical)==300
    assert [r['epoch'] for r in physical]==list(range(1,301))
    chosen=min(physical,key=lambda r:selection_metric(r['val']))
    assert all(r['updates']==r['epoch']*83 and r['train']['sample_count']==166 and r['val']['sample_count']==36 for r in physical)
    preheat=[r for r in records if r['stage']=='preheat']
    assert len(preheat)==60 and all(r['updates']==r['epoch']*83 and r['train']['cases']==166 for r in preheat)
    for stage,total in [('preheat',4980),('physical',24900)]:
        journal=[json.loads(line) for line in (ROOT/'unet_amp_v2'/(stage+'_updates.jsonl')).read_text().splitlines()]
        assert len(journal)==total and [v['update'] for v in journal]==list(range(1,total+1))
        assert all(np.isfinite(v['parameter_delta']) and v['parameter_delta']>0 and v['gradient_norm_after_clip']>0 for v in journal)
    checkpoint=torch.load(ROOT/'unet_amp_v2/last.pt',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==300 and checkpoint['optimizer_steps']==24900 and checkpoint['scheduler']['last_epoch']==300
    assert {int(v['step']) for v in checkpoint['optimizer']['state'].values()}=={24900}
    del checkpoint
    target=ROOT/'validation_review_amp_v2';target.mkdir(exist_ok=False)
    model=AmplitudeModel(build('unet'),summary['reference_chi']);ck=torch.load(summary['best_checkpoint'],map_location='cpu',weights_only=False)
    assert ck['epoch']==chosen['epoch'];model.load_state_dict(ck['model']);model.eval();del ck
    scores=[];maximum=0.;guard=GTReadGuard(DATA,ROOT)
    with guard,torch.no_grad():
        ds=make_dataset(config(),validation=True)
        for index in range(36):
            batch=dual_measurement_collate([ds[index]]);h=batch['H'];mask=(h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]).numpy()
            arrays={k:v.numpy() for k,v in h.items() if isinstance(v,torch.Tensor)}
            arrays['mask']=mask;per_case=[]
            for branch in ['A','B']:
                field=model(batch[branch]);p=numpy_project(field.numpy(),h['input_station_origin_m'].numpy(),h['input_ray_direction'].numpy(),h['input_t_near'].numpy(),h['input_path_length_m'].numpy(),mask)
                original=project_input(field,h).numpy();y=h['input_projection_raw'].numpy()
                error=float(np.abs(p[mask]-original[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6));maximum=max(maximum,error);assert error<2e-5
                score=float(np.abs(p[mask]-y[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6));per_case.append(score)
                arrays[branch+'_volume']=field.numpy();arrays[branch+'_H_projection']=original
            path=target/f'val{index:03d}.npz';np.savez_compressed(path,**arrays)
            scores.append(dict(index=index,case_id=ds.records[index]['case_id'],H=float(np.mean(per_case)),path=str(path),sha256=sha(path)))
            print('INDEPENDENT_VAL',index,'H',scores[-1]['H'],flush=True)
    mean=float(np.mean([v['H'] for v in scores]));np.testing.assert_allclose(mean,summary['best_val_H'],rtol=5e-5,atol=3e-6)
    assert not any(guard.summary()[k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
    write(ART/'unet_amp_validation_review.json',dict(passed=True,full_fields=72,independent_H_integrations=72,Val_cases=36,
        best_epoch=chosen['epoch'],independent_Val_H=mean,training_reported_Val_H=summary['best_val_H'],
        maximum_projection_error_relative_to_target=maximum,preheat_updates_verified=4980,physical_updates_verified=24900,
        checkpoint=summary['best_checkpoint'],checkpoint_sha256=summary['best_checkpoint_sha256'],records=scores,access=guard.summary()))
    verify()


if __name__=='__main__':main()
