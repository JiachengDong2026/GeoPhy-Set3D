"""GT-free frozen-checkpoint audit of sparse background versus global collapse."""
from feasible_context import *
sys.path.insert(0,str(PROJECT/'scripts/unet_baseline_v1'))
from numpy_projector import project as numpy_project


def main():
    setup();verify()
    registration=read(ART/'health_guard_audit_registration.json')
    for path,digest in registration['sources'].items():assert sha(PROJECT/path)==digest
    target=ROOT/'health_guard_audit';target.mkdir(exist_ok=False)
    checkpoint=target/'frozen_best.pt';os.link(ROOT/'unet/best.pt',checkpoint)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    epoch=ck['epoch'];model=build('unet');model.load_state_dict(ck['model']);model.eval();del ck
    records=[];guard=GTReadGuard(DATA,ROOT)
    with guard,torch.no_grad():
        ds=make_dataset(config());ds.set_epoch(0)
        for index in [0,1,2]:
            batch=dual_measurement_collate([ds[index]]);h=batch['H']
            mask=(h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]).numpy()
            y=h['input_projection_raw'].numpy();denominator=max(np.abs(y[mask]).sum(),1e-6)
            for branch in ['A','B']:
                a=model(batch[branch]).numpy().astype('float64');zero=dict(batch[branch])
                for key in ['input_projection','input_projection_raw']:zero[key]=torch.zeros_like(zero[key])
                z=model(zero).numpy()
                scores=[]
                for field in [a,z]:
                    p=numpy_project(field,h['input_station_origin_m'].numpy(),h['input_ray_direction'].numpy(),h['input_t_near'].numpy(),h['input_path_length_m'].numpy(),mask)
                    scores.append(float(np.abs(p[mask]-y[mask]).sum()/denominator))
                low=a<1/(1+np.exp(10));high=a>1/(1+np.exp(-10))
                path=target/f'train{index:03d}_{branch}.npz';np.savez_compressed(path,normal=a,zero=z)
                records.append(dict(index=index,branch=branch,case_id=ds.records[index]['case_id'],
                    mean=float(a.mean()),std=float(a.std()),nearzero_fraction=float((a<1e-10).mean()),
                    negative_saturation_fraction=float(low.mean()),positive_saturation_fraction=float(high.mean()),
                    negative_saturation_mass_fraction=float(a[low].sum()/a.sum()),
                    positive_saturation_mass_fraction=float(a[high].sum()/a.sum()),
                    normal_H=scores[0],zero_H=scores[1],zero_benefit=scores[1]-scores[0],
                    response_absolute=float(np.abs(a-z).mean()),response_relative=float(np.abs(a-z).sum()/a.sum()),
                    output_quantiles=np.quantile(a,[0,.5,.9,.95,.99,.999,1]).tolist(),path=str(path),sha256=sha(path)))
                print('HEALTH_AUDIT',index,branch,'low_background',low.mean(),'background_mass',records[-1]['negative_saturation_mass_fraction'],'normal_H',scores[0],flush=True)
    write(ART/'health_guard_audit.json',dict(checkpoint_sha256=sha(checkpoint),checkpoint_epoch=epoch,
        records=records,access=guard.summary(),GT_used=False,new_optimizer_updates=0,
        training_gate_changed=False,formal_test_allowed_by_this_audit=False))
    verify()


if __name__=='__main__':main()
