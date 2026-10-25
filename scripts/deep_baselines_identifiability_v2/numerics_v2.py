"""Frozen-model adapters and A/B-only diagnostic objective."""
from ctx_v2 import *
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training.selfsup import measurement_discrepancy
from geophy_set3d.training.baseline_consistency_repair_v3 import homogeneous_consistency
from geophy_set3d.models.ray_senseiver_ss import ray_tokens

def valid(inp):return inp['input_valid_ray'].bool()&inp['view_mask'][:,:,None,None,None]
def objective(net,ab):
    if set(ab)!={'A','B'}:raise ValueError('Exactly A/B, no H or truth')
    ca,cb=net(ab['A']),net(ab['B'])
    observed=.5*(measurement_discrepancy(project_input(ca,ab['A']),ab['A'])+measurement_discrepancy(project_input(cb,ab['B']),ab['B']))
    cross=.5*(measurement_discrepancy(project_input(ca,ab['B']),ab['B'])+measurement_discrepancy(project_input(cb,ab['A']),ab['A']))
    consistency=homogeneous_consistency(ca,cb)
    return .5*cross+.1*observed+.05*consistency,dict(cross=cross,observed=observed,consistency=consistency)

def intervene(inp,kind,other=None):
    out=dict(inp);raw=inp['input_projection_raw'].clone();mask=valid(inp)
    if kind=='zero':raw[mask]=0
    elif kind=='permuted':
        gen=torch.Generator().manual_seed(design()['seed']+91)
        for i in range(len(raw)):
            v=raw[i][mask[i]].clone();raw[i][mask[i]]=v[torch.randperm(len(v),generator=gen).to(raw.device)]
    elif kind=='swap':
        assert value_hash(nonmeasurement(inp))==value_hash(nonmeasurement(other))
        raw[mask]=other['input_projection_raw'][mask]
    elif kind!='normal':raise ValueError(kind)
    out['input_projection_raw']=raw;out['input_projection']=raw/Y_SCALE
    assert value_hash(nonmeasurement(out))==value_hash(nonmeasurement(inp))
    assert torch.equal(raw[~mask],inp['input_projection_raw'][~mask])
    return out

def stats(x):
    x=x.detach().double().reshape(-1)
    return dict(l2=float(x.norm()),max_abs=float(x.abs().max()),finite_fraction=float(torch.isfinite(x).double().mean()))

def sensitivity(net,inp):
    """Full token and full-volume observed gradient, no hooks/optimizer."""
    net.train();net.zero_grad(set_to_none=True)
    tokens,excluded,counts=ray_tokens(inp);tokens=tokens.detach().requires_grad_(True)
    if hasattr(net,'phi'):
        phi=net.phi(tokens).masked_fill(excluded.unsqueeze(-1),0.)
        latent=phi.sum(1)/counts.clamp_min(1).unsqueeze(-1)
        out=net.decode_queries(latent,net.query_coordinates_m).transpose(1,2).reshape(len(tokens),1,64,64,64)
        head=net.decoder[-1]
    else:
        out=net.forward_tokens(tokens,excluded,counts).transpose(1,2).reshape(len(tokens),1,64,64,64)
        head=net.head[-1]
    loss=measurement_discrepancy(project_input(out,inp),inp)
    gradients=torch.autograd.grad(loss,[tokens,head.weight,head.bias])
    result=dict(measurement_channel=stats(gradients[0][...,0]),geometry_channels=stats(gradients[0][...,1:]),
                readout_weight=stats(gradients[1]),readout_bias=stats(gradients[2]),observed_loss=float(loss.detach()),optimizer_updates=0)
    del out,loss,gradients,tokens
    net.eval();return result
