"""A/B-only objective, measurement scores and interventions, no experiment state."""
from diag_context import *
from geophy_set3d.training.selfsup import measurement_discrepancy, consistency_discrepancy
from geophy_set3d.training.baseline_consistency_repair_v3 import homogeneous_consistency
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input

def valid(target):
    return target['input_valid_ray'].bool() & target['view_mask'][:,:,None,None,None]

def objective(net, ab, version):
    if set(ab)!= {'A','B'}:raise ValueError('Training capability is exactly A/B; H is forbidden')
    for i in range(len(ab['A']['view_mask'])):
        a=set(ab['A']['input_station_indices'][i,ab['A']['view_mask'][i]].tolist())
        b=set(ab['B']['input_station_indices'][i,ab['B']['view_mask'][i]].tolist())
        if a&b:raise ValueError('A/B stations overlap')
    ca,cb=net(ab['A']),net(ab['B'])
    projections={label:{k:project_input(field,ab[k]) for k in ('A','B')} for label,field in [('A',ca),('B',cb)]}
    loss=lambda label,key:measurement_discrepancy(projections[label][key],ab[key])
    parts=dict(cross=.5*(loss('A','B')+loss('B','A')),observed=.5*(loss('A','A')+loss('B','B')),
        consistency=(consistency_discrepancy if version=='v2' else homogeneous_consistency)(ca,cb))
    total=sum(design()['weights'][k]*v for k,v in parts.items())
    assert torch.isfinite(total)
    return total,parts,(ca,cb),projections

def scores(projected,target):
    mask=valid(target);rows=[]
    for i in range(len(projected)):
        p=projected[i][mask[i]].double();y=target['input_projection_raw'][i][mask[i]].double();e=p-y
        rows.append(dict(relative_l1=float(e.abs().sum()/y.abs().sum().clamp_min(1e-6)),
                         relative_l2=float(e.norm()/y.norm().clamp_min(1e-6)),
                         discrepancy=float(measurement_discrepancy(projected[i:i+1],slice_batch(target,i))),
                         valid_rays=len(y)))
    return rows

def slice_batch(value,i):
    return {k:(v[i:i+1] if isinstance(v,(torch.Tensor,list)) else v) for k,v in value.items()}

def field_stats(x):
    value=x.detach().double().reshape(-1)
    mean=float(value.mean());std=float(value.std(unbiased=False))
    arg=int(value.argmax());peak=np.unravel_index(arg,(64,64,64)) if value.numel()==64**3 else None
    return dict(mean=mean,std=std,min=float(value.min()),max=float(value.max()),spatial_cv=std/max(mean,1e-15),
                peak_xyz_m=None if peak is None else [float(k*.6/63) for k in peak],
                below_1e_10=float((value<1e-10).double().mean()))

def intervene(inp,kind):
    out=dict(inp);raw=inp['input_projection_raw'];mask=valid(inp)
    values=raw.clone()
    if kind in ('zero','half','one_half'):
        values[mask]=values[mask]*dict(zero=0.,half=.5,one_half=1.5)[kind]
    elif kind=='permuted':
        gen=torch.Generator(device='cpu').manual_seed(design()['seed']+91)
        for i in range(len(raw)):
            selected=values[i][mask[i]].clone();permutation=torch.randperm(len(selected),generator=gen).to(raw.device)
            values[i][mask[i]]=selected[permutation]
    elif kind!='normal':raise ValueError(kind)
    out['input_projection_raw']=values
    # Keep normalized projection exactly consistent; no other fields change.
    out['input_projection']=values/Y_SCALE
    assert torch.equal(mask,valid(out))
    return out

def difference(changed,normal):
    a=changed.detach().double().reshape(-1);b=normal.detach().double().reshape(-1);e=a-b
    ac,bc=a-a.mean(),b-b.mean();den=ac.norm()*bc.norm()
    return dict(mae=float(e.abs().mean()),rmse=float(e.square().mean().sqrt()),
                stabilized_relative_mae=float(e.abs().mean()/design()['initial_mass_fraction']),
                spatial_correlation=None if den<1e-30 else float((ac@bc)/den),
                peak_displacement_m=float(np.linalg.norm(np.array(np.unravel_index(int(a.argmax()),(64,)*3))-np.array(np.unravel_index(int(b.argmax()),(64,)*3)))*.6/63))

def gradient_stats(x):
    if x is None:return dict(present=False,l2=0.,max_abs=0.,finite_fraction=1.)
    v=x.detach().double()
    return dict(present=True,l2=float(v.norm()),max_abs=float(v.abs().max()),finite_fraction=float(torch.isfinite(v).double().mean()))

def parameter_groups(net):
    groups={}
    for name,p in net.named_parameters():
        key=('phi' if name.startswith('phi') else 'measurement_encoder' if name.startswith('measurement_encoder') else
             'geometry_encoder' if name.startswith('geometry_encoder') else 'readout' if name.startswith(('head.2','decoder.4')) else 'decoder_operator')
        groups.setdefault(key,[]).append(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1))
    return {k:gradient_stats(torch.cat(v)) for k,v in groups.items()}
