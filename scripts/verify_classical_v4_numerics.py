#!/usr/bin/env python3
"""Record measured numerical checks and train GT64 discretization diagnostics."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
import torch,numpy as np,yaml
from run_classical_v4 import verify,verify_selection,ReadGuard,write
from geophy_set3d.reconstruction.classical import RayMatrix,geometry_tensor,project,mass_to_mole,mole_to_mass,physical_gradient,sart,variational
from test_classical_reconstruction import phantom_geometry


def main():
    cfg=yaml.safe_load((ROOT/'configs/classical_v4_v1.yaml').read_text());out=Path(cfg['output']);p=verify(cfg,out);verify_selection(out);torch.set_num_threads(2)
    if (out/'numerical_validation.json').exists():print('Numerical diagnostics already saved; no repeat');return
    with ReadGuard(cfg,out,'calibrate',p['calibration_cases']):
        with np.load(Path(cfg['dataset'])/'geometry/train_geometry.npz') as a:g={k:a[k] for k in a.files}
        def subset(ids):return {k:v[ids] for k,v in g.items() if k not in ['horizontal_angle_deg','vertical_angle_deg']}
        geom=subset([0]);a=RayMatrix(geom,dtype=torch.float64);torch.manual_seed(2026101305)
        x=torch.randn(64**3,dtype=torch.float64);y=torch.randn(a.a.shape[0],dtype=torch.float64);lhs=a.forward(x)@y;rhs=x@a.adjoint(y)
        adjoint=float(abs(lhs-rhs)/max(abs(lhs),abs(rhs)))
        c=torch.full((64,)*3,.2,dtype=torch.float64);pred=project(c,geometry_tensor(geom,'cpu'))
        delta=a.forward(mass_to_mole(c))-a.measurement(pred);constant=float(delta.abs().sum()/a.measurement(pred).abs().sum())
        roundtrip=float((mole_to_mass(mass_to_mole(torch.linspace(0,1,1001,dtype=torch.float64)))-torch.linspace(0,1,1001,dtype=torch.float64)).abs().max())
        checks=dict(actual_PoolA_adjoint_relative_discrepancy=adjoint,constant_C02_sparse_vs_frozen_Pdiff_relative_l1=constant,mass_mole_roundtrip_max_absolute_error=roundtrip,
                    frozen_projector_source_sha256=p['source_sha256']['src/geophy_set3d/physics/projectors.py'],GT128_PoolB_test_read=False,torch_version=str(torch.__version__),sparse_weight_build='float64 then cast FP32 for actual runs; numerical audit float64')
        assert adjoint<1e-11 and constant<1e-6 and roundtrip<1e-14
        n=6;gg=phantom_geometry(n);aa=RayMatrix(gg,n=n,samples=24);axis=torch.linspace(0,.6,n);xx,yy,zz=torch.meshgrid(axis,axis,axis,indexing='ij');gt=.04*torch.exp(-((xx-.3)**2+(yy-.3)**2+(zz-.3)**2)/.012)
        target=aa.forward(mass_to_mole(gt)).reshape(gg['valid_ray'].shape)
        chi,stat=sart([aa],[target],dict(iterations=100,relaxation=.5,tolerance=1e-7,minimum_iterations=20,patience=5),n=n)
        checks['phantom']={'SART':dict(relative_l1=float((mole_to_mass(chi)-gt).abs().sum()/gt.abs().sum()),solver_relative_l2=stat['solver_relative_l2'],iterations=stat['iterations'],status=stat['status'])}
        tg=geometry_tensor(gg,'cpu');target=project(gt,tg)
        conf=dict(iterations=80,initial_step=100.,step_bounds=[1e-4,1e6],max_backtracks=16,armijo=1e-4,minimum_iterations=20,patience=5,tolerance=1e-6,epsilon=1e-4,**{'lambda':1e-5})
        for method in ['Tikhonov','TV']:
            rec,s=variational(tg,target,method,conf,cfg['normalization_scale_ppm_m'],n=n)
            checks['phantom'][method]=dict(relative_l1=float((rec-gt).abs().sum()/gt.abs().sum()),solver_relative_l2=s['solver_relative_l2'],iterations=s['iterations'],status=s['status'],first_objective=s['curve'][0]['objective'],last_objective=s['curve'][-1]['objective'])
        write(out/'numerical_validation.json',checks)
        # These diagnostics are post-calibration, never fed back into candidate selection.
        entries=json.loads((out/'calibration_stations.json').read_text());records={r['case_id']:r for r in p['calibration_cases']};cache={};rows=[]
        for e in entries:
            c=torch.as_tensor(np.array(np.load(Path(cfg['dataset'])/records[e['case_id']]['gt_volume_64_path'],mmap_mode='r')))
            tg=geometry_tensor(subset(e['input_station_ids']),'cpu')
            with torch.no_grad():pred=project(c,tg)
            delta=[];reference=[]
            for j,station in enumerate(e['input_station_ids']):
                if station not in cache:cache[station]=RayMatrix(subset([station]))
                a=cache[station];reference.append(a.measurement(pred[j]));delta.append(a.forward(mass_to_mole(c))-reference[-1])
            delta=torch.cat(delta);reference=torch.cat(reference)
            rows.append(dict(case_id=e['case_id'],layout=e['layout'],relative_l1=float(delta.abs().sum()/reference.abs().sum().clamp_min(1e-6)),relative_l2=float((delta.square().sum()/reference.square().sum().clamp_min(1e-12)).sqrt())))
        write(out/'gt_discretization.json',dict(purpose='Post-calibration diagnostic only; no selection/parameter feedback',rows=rows))
    print(checks,flush=True)
if __name__=='__main__':main()
