"""Stage A: measurements-only physical/gradient/optimization evidence before repair."""
from repair_context import *
from geophy_set3d.models.ray_senseiver_ss import Y_SCALE,ray_tokens
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.physics.projectors import differentiable_project
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics


def gradient_audit(model,batch):
    # These models have no dropout/batchnorm. Training mode activates their
    # existing chunk checkpointing; eval-with-grad retains every262144-query
    # expert activation and can exhaust24GB. Forward semantics are identical.
    model.train();model.zero_grad(set_to_none=True)
    saved=[]
    head=model.decoder[-1] if hasattr(model,'phi') else model.head[-1]
    hook=head.register_forward_hook(lambda m,args,out:saved.append(out.detach().flatten()))
    total,parts,volumes,_=ss.objective(model,batch)
    hook.remove()
    logits=torch.cat(saved)
    params=list(model.named_parameters());vectors={};details={}
    for label,loss in dict(parts,total=total).items():
        grads=torch.autograd.grad(loss,[p for _,p in params],retain_graph=True,allow_unused=True)
        flat=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for (_,p),g in zip(params,grads)]).detach()
        vectors[label]=flat
        groups={}
        for (name,param),g in zip(params,grads):
            group=('output' if param is head.weight or param is head.bias else
                   'encoder' if any(k in name for k in ('phi.','measurement_encoder','geometry_encoder')) else 'decoder_latent')
            groups[group]=groups.get(group,0.)+(0. if g is None else float(g.double().square().sum()))
        details[label]=dict(loss=float(loss.detach()),gradient_l2=float(flat.double().norm()),
            groups={k:v**.5 for k,v in groups.items()},
            output_bias_gradient=float(grads[next(i for i,(_,p) in enumerate(params) if p is head.bias)].sum()),
            output_weight_gradient_l2=float(grads[next(i for i,(_,p) in enumerate(params) if p is head.weight)].double().norm()))
    cos={}
    for a in parts:
        for d in parts:
            if a<d:
                cos[a+'__'+d]=float(torch.dot(vectors[a],vectors[d])/(vectors[a].norm()*vectors[d].norm()+1e-30))
    quantiles=torch.quantile(logits.double(),torch.tensor([0.,.01,.5,.99,1.],device=logits.device,dtype=torch.double)).tolist()
    return dict(parts=details,pairwise_gradient_cosines=cos,
        weighted_gradient_norms={k:details[k]['gradient_l2']*ss.FROZEN_LOSS[k] for k in parts},
        logits_quantiles=quantiles,fraction_logits_below_minus12=float((logits < -12).float().mean()),
        outputs=[tensor_stats(x) for x in volumes],all_gradients_finite=all(torch.isfinite(v).all().item() for v in vectors.values()))


def physical(batch):
    t=batch['H'];mask=t['input_valid_ray'].bool()&t['view_mask'][:,:,None,None,None]
    result={}
    for label in ('A','B','H'):
        a=batch[label];v=a['input_valid_ray'].bool()&a['view_mask'][:,:,None,None,None]
        y=a['input_projection_raw'][v];length=a['input_path_length_m'][v]
        direction=a['input_ray_direction'][v[:,:,0]]
        result[label]=dict(unit='ppm_m',raw=tensor_stats(y),normalized=tensor_stats(y/Y_SCALE),
            valid_rays=int(v.sum()),valid_exact_zero=int((y==0).sum()),valid_below_1e_6=int((y<1e-6).sum()),
            path_length_m=tensor_stats(length),max_direction_norm_error=float((direction.norm(dim=-1)-1).abs().max()),
            stations=[a['input_station_indices'][i,a['view_mask'][i]].tolist() for i in range(len(y.shape) and len(a['case_id']))])
    ss.assert_disjoint(batch)
    # Independent NumPy midpoint computation for an XYZ-linear mass fraction.
    axis=torch.linspace(0.,.6,64,device='cuda:0',dtype=torch.float64)
    xyz=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1)
    field=.0002+.001*xyz[...,0]+.002*xyz[...,1]+.004*xyz[...,2]
    projected=project_input(field[None,None].expand(len(t['case_id']),-1,-1,-1,-1),t)
    maxdiff=0.
    for i in range(len(t['case_id'])):
        for j in t['view_mask'][i].nonzero().flatten().tolist():
            valid=t['input_valid_ray'][i,j,0].cpu().numpy().astype(bool)
            origin=t['input_station_origin_m'][i,j].cpu().numpy().astype(float)
            direction=t['input_ray_direction'][i,j].cpu().numpy().astype(float)
            near=t['input_t_near'][i,j,0].cpu().numpy().astype(float)
            length=t['input_path_length_m'][i,j,0].cpu().numpy().astype(float)
            points=origin+direction[...,None,:]*(near[...,None]+length[...,None]*(np.arange(96)+.5)/96)[...,None]
            cc=.0002+.001*points[...,0]+.002*points[...,1]+.004*points[...,2]
            yy=mass_to_mole(cc).mean(-1)*length*1e6
            difference=np.max(np.abs(projected[i,j,0].cpu().numpy()[valid]-yy[valid]));maxdiff=max(maxdiff,float(difference))
    assert maxdiff<1e-6,maxdiff
    zero=torch.zeros((len(t['case_id']),1,64,64,64),device='cuda:0')
    z=project_input(zero,t)
    exact=[relative_projection_metrics(z[i],t['input_projection_raw'][i],mask[i])['relative_l1'] for i in range(len(z))]
    # Full physical chain directional derivative, FP64, one fixed positive field.
    scalar=torch.tensor(.001,device='cuda:0',dtype=torch.double,requires_grad=True)
    def loss(value):
        pred=project_input(torch.ones_like(zero,dtype=torch.double)*value,t)
        return ss.measurement_discrepancy(pred,t)
    ad=float(torch.autograd.grad(loss(scalar),scalar)[0]);eps=1e-7
    fd=float((loss(scalar.detach()+eps)-loss(scalar.detach()-eps))/(2*eps))
    assert abs(ad-fd)<1e-4*max(1.,abs(ad)),(ad,fd)
    zero_scalar=torch.tensor(0.,device='cuda:0',dtype=torch.double,requires_grad=True)
    zero_gradient=float(torch.autograd.grad(loss(zero_scalar),zero_scalar)[0])
    return dict(measurements=result,zero_relative_l1=exact,zero_projection_max=float(z.max()),
        XYZ_numpy_reference_max_abs_ppm_m=maxdiff,AD=ad,FD=fd,relative_AD_FD_error=abs(ad-fd)/max(abs(fd),1e-12),
        zero_constant_field_loss_derivative=zero_gradient,
        note='Constant direction can increase L1 due to valid near-zero rays; a negative gradient on illuminated localized directions is separately checked')


def local_zero_gradient(batch):
    target=batch['H'];zero=torch.zeros((len(target['case_id']),1,64,64,64),device='cuda:0',requires_grad=True)
    loss=ss.measurement_discrepancy(project_input(zero,target),target)
    grad=torch.autograd.grad(loss,zero)[0]
    return dict(loss=float(loss),gradient=tensor_stats(grad),fraction_negative=float((grad<0).float().mean()),
                negative_nodes=int((grad<0).sum()),positive_nodes=int((grad>0).sum()),
                conclusion='Negative voxel gradient supports increasing illuminated regions; background-positive gradients suppress spatially uniform predictions')


def calibration(cfg):
    # Closed-form homogeneous mole-fraction least squares, TRAIN MEASUREMENTS ONLY.
    ds=ss.make_dataset(cfg)
    num=den=0.;files={};rays=0
    for i in range(len(ds)):
        for label in ('A','B'):
            t=ds[i][label]
            valid=t['input_valid_ray'].bool()[:,0]
            y=t['input_projection_raw'][:,0][valid].double();length=t['input_path_length_m'][:,0][valid].double()
            num+=float((length*y).sum());den+=float((length.square()).sum())*1e6;rays+=len(y)
        if i%30==0:print('train scale audit',i,'/166',flush=True)
    mole=num/den
    mass=(mole*16.043)/(mole*16.043+(1-mole)*28.965)
    assert 0<mass<1
    return dict(train_cases=len(ds),station_epoch=0,branches=['A','B'],valid_rays=rays,
                numerator_length_y=num,denominator_1e6_length_squared=den,mole_fraction=mole,
                mass_fraction=mass,output_bias_logit=float(np.log(mass/(1-mass))),
                derivation='argmin_chi sum_valid(y_ppm_m -1e6*path_length_m*chi)^2; then exact mole_to_mass',
                uses_GT_statistics=False,uses_val_statistics=False,uses_test=False)


def main():
    assert not (ROOT/'audit.json').exists()
    ROOT.mkdir(exist_ok=True)
    c.backend();torch.set_num_threads(2)
    cfg=config('deepsets',ROOT/'audit')
    regression=b.regression()
    old_files={}
    for arm in ARMS:
        run=old_checkpoint(arm).parent.parent
        for name in ('protocol.json','metrics.jsonl','summary.json'):
            path=run/name;old_files[str(path)]=c.sha(path)
        old_files[str(old_checkpoint(arm))]=c.sha(old_checkpoint(arm))
    with guard([old_checkpoint(a) for a in ARMS]) as access:
        batch=fixed_batch(cfg)
        physics=physical(batch);local=local_zero_gradient(batch)
        scale=calibration(cfg)
        models={}
        for arm in ARMS:
            models[arm]={}
            for stage in ('fresh','failed'):
                m=b.build_model(arm,'cuda:0',SEED)
                if stage=='failed':
                    ck=torch.load(old_checkpoint(arm),map_location='cpu',weights_only=False)
                    assert ck['volumetric_gt_used'] is False
                    m.load_state_dict(ck['model'],strict=True);del ck
                models[arm][stage]=dict(gradients=gradient_audit(m,batch),health=health(m,batch))
                del m;torch.cuda.empty_cache()
                print('audited',arm,stage,flush=True)
        logs={}
        for arm in ARMS:
            allrows=c.rows(old_checkpoint(arm).parent.parent/'metrics.jsonl')
            logs[arm]=[x for x in allrows if x['epoch'] in (1,2,3,5,10,25,100,300)]
        assert not access.denied
    freeze('audit.json',dict(at=c.now(),physical=physics,localized_zero_gradient=local,calibration=scale,
        models=models,historical_training_trajectories=logs,main_regression=regression,
        old_evidence_sha256=old_files,sources=sources(),access=access.summary(),GT_used=False,GT128_used=False,test_used=False))
    print('AUDIT COMPLETE',json.dumps(scale),flush=True)


if __name__=='__main__':main()
