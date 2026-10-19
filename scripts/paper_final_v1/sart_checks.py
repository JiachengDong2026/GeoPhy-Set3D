"""Independent SART matrix, update, nonlinear-discrepancy and recovery checks."""
from common import *
import tempfile
import numpy as np
import torch
from sart import Block,solve,BASE
from geophy_set3d.reconstruction.classical import RayMatrix,sart,mole_to_mass,geometry_tensor,project
from geophy_set3d.physics.projectors import mass_to_mole


def phantom(n=4):
    origins=[];directions=[]
    for axis in range(3):
        for a in np.linspace(0,.6,n):
            for b in np.linspace(0,.6,n):
                o=np.zeros(3);o[axis]=-1;o[(axis+1)%3]=a;o[(axis+2)%3]=b
                d=np.zeros(3);d[axis]=1;origins.append(o);directions.append(d)
    v=len(origins)
    return dict(station_origin_m=np.array(origins),ray_direction=np.array(directions)[:,None,None,:],
        t_near=np.ones((v,1,1)),path_length_m=np.full((v,1,1),.6),valid_ray=np.ones((v,1,1),bool))


def main():
    backend();rng=np.random.default_rng(20261019);g=phantom();n=8
    blocks=[Block({k:v[i:i+16] for k,v in g.items()},n=n,samples=24,dtype=np.float64) for i in range(0,48,16)]
    adjoint_errors=[]
    for b in blocks:
        x=rng.normal(size=n**3);y=rng.normal(size=b.a.shape[0]);lhs=float(b.forward(x)@y);rhs=float(x@b.adjoint(y))
        error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12);assert error<1e-12;adjoint_errors.append(error)
        np.testing.assert_allclose(b.forward(np.ones(n**3)),600000,rtol=1e-14,atol=1e-8)
    gt=np.full((n,)*3,.04);gt[2:6,2:6,2:6]=.12;chi=mass_to_mole(gt)
    ys=[b.forward(chi).reshape(16,1,1) for b in blocks]
    config=dict(iterations=50,minimum_iterations=20,patience=5,tolerance=1e-7,relaxation=.5)
    result,stats=solve(blocks,ys,config)
    assert stats['solver_relative_l2']<.05 and stats['curve'][-1]['objective']<stats['curve'][0]['objective']
    # Independent historical torch implementation must agree with our CPU CSR sweeps.
    oldblocks=[RayMatrix({k:v[i:i+16] for k,v in g.items()},n=n,samples=24,dtype=torch.float64) for i in range(0,48,16)]
    old,oldstats=sart(oldblocks,[torch.as_tensor(y) for y in ys],config,n=n)
    np.testing.assert_allclose(result,old.numpy().ravel(),rtol=1e-12,atol=1e-14)
    assert oldstats['iterations']==stats['iterations']
    with tempfile.TemporaryDirectory(dir=BASE) as tmp:
        checkpoint=Path(tmp)/'resume.pt';identity=dict(case='phantom',stations=list(range(48)))
        _,partial=solve(blocks,ys,config,checkpoint=checkpoint,identity=identity,stop_after=7)
        assert partial['iterations']==7 and partial['status']=='running'
        resumed,recovered=solve(blocks,ys,config,checkpoint=checkpoint,identity=identity)
        np.testing.assert_array_equal(result,resumed);assert recovered==stats
        terminal,terminalstats=solve(blocks,ys,config,checkpoint=checkpoint,identity=identity)
        np.testing.assert_array_equal(terminal,resumed);assert terminalstats==recovered
        rejected=False
        try:solve(blocks,ys,dict(config,relaxation=1.),checkpoint=checkpoint,identity=identity)
        except AssertionError:rejected=True
        assert rejected
    zero,stopped=solve(blocks,[np.zeros_like(y) for y in ys],dict(config,minimum_iterations=3,patience=2))
    assert stopped['iterations']==3 and stopped['status']=='objective_stable' and not zero.any()
    c=torch.as_tensor(rng.uniform(0,.3,size=(n,)*3));a=Block(g,n=n,samples=24,dtype=np.float64)
    exact=project(c,geometry_tensor(g,'cpu'),24).numpy().ravel();approximate=a.forward(mass_to_mole(c).numpy())
    assert np.max(np.abs(exact-approximate))>1
    np.testing.assert_allclose(mole_to_mass(mass_to_mole(c.numpy())),c.numpy(),rtol=1e-14,atol=1e-14)
    # Real64 CPU timing and units, with one training measurement and no GT/H inputs.
    registration=verified(BASE/'registration.json');case=registration['train_cases'][0];setting=registration['settings'][0]
    source=next(r for r in rows(DATA/'source_index.jsonl') if r['case_id']==case)
    import time
    with AccessGuard(splits=['train']) as guard:
        with np.load(DATA/'geometry/train_geometry.npz') as z:geometry={k:z[k] for k in z.files}
        ids=setting['input_ids'][case];t=time.perf_counter()
        real=[Block({k:v[[i]] for k,v in geometry.items() if k not in ('horizontal_angle_deg','vertical_angle_deg')}) for i in ids]
        setup=time.perf_counter()-t;measurement=np.load(DATA/source['train_projection_path'],mmap_mode='r',allow_pickle=False)
        t=time.perf_counter();x,smoke=solve(real,np.array(measurement[ids]),dict(config,iterations=10));seconds=time.perf_counter()-t
        assert x.shape==(64**3,) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1
        assert smoke['curve'][-1]['objective']<smoke['curve'][0]['objective']
        audit=guard.summary();assert not audit['denied']
    freeze(BASE/'numerical_checks.json',dict(at=now(),passed=True,true_adjoint_errors=adjoint_errors,units='ppm*m, constant chi=1 ->600000 for .6m',
        phantom_relative_l2=stats['solver_relative_l2'],torch_reference_agreement=True,interrupted_resume_exact=True,terminal_resume_exact=True,
        wrong_config_resume_rejected=True,stable_stop_verified=True,discretization_difference_demonstrated=True,real64_shape_range_passed=True,
        smoke_10_sweeps_seconds=seconds,geometry_setup_seconds=setup,sparse_matrix_bytes=sum(b.bytes for b in real),audit=audit,source_sha256=sha(__file__)))
    print(json.dumps(read(BASE/'numerical_checks.json')),flush=True)


if __name__=='__main__':BASE.mkdir(parents=True,exist_ok=True);main()
