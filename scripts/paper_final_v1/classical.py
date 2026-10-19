"""Pre-registered finite calibration and 17-layout classical prediction workers."""
from common import *
import argparse
import time
import numpy as np
import torch
from geophy_set3d.reconstruction.classical import geometry_tensor
from geophy_set3d.evaluation.research import volume_metrics
from solver import CachedProjector,solve

METHODS=['NNLS-NL','Tikhonov','TV']
FIXED=dict(iterations=200,minimum_iterations=40,tolerance=1e-5,patience=8,initial_step=1000.,step_bounds=[1e-5,1e6],max_backtracks=20,armijo=1e-4,resolution=64)


def register():
    verified(OUT/'registration.json');assert not (OUT/'classical/search.json').exists()
    source=[r for r in rows(DATA/'source_index.jsonl') if r['split']=='train']
    indices=np.rint(np.linspace(0,len(source)-1,6)).astype(int)
    from audit import layouts
    with np.load(DATA/'geometry/train_geometry.npz') as g:settings=layouts([source[i]['case_id'] for i in indices],g['station_azimuth_deg'])
    settings=[s for s in settings if (s['condition'],s['draw']) in [('Val-A-V4',0),('clustered-V4',0)]]
    candidates={'NNLS-NL':[dict(FIXED,iterations=100),dict(FIXED)],
        'Tikhonov':[dict(FIXED,**{'lambda':v}) for v in (1e-4,1e-3)],
        'TV':[dict(FIXED,epsilon=1e-4,**{'lambda':v}) for v in (1e-5,1e-4)]}
    freeze(OUT/'classical/search.json',dict(registered_at=now(),train_cases=[source[i]['case_id'] for i in indices],settings=settings,candidates=candidates,
        prior_evidence='classical_v4_v1: Tikhonov stronger smoothing best; TV lower lambda better; SART approximate forward rejected for primary comparison',
        selection='minimum mean train-case 3DrelativeL1 over two layouts, ties candidate index; all calibration predictions sealed before calibration GT scoring',
        total_reconstructions=72,initialization='zero; no network or GT',bounds=[0,1],operator='true nonlinear K96; cached identical geometry grids',
        objective='mean_valid_rays 0.5*((P(C)-Y)/32284.894580078173)^2 + lambda*R(C)',
        regularizers=dict(Tikhonov='mean_nodes 0.5*sum_axes (forward_diff/spacing_m)^2',TV='mean_nodes(sqrt(sum_axes(forward_diff/spacing_m)^2+epsilon^2)-epsilon); smooth isotropic TV',boundary='zero outward derivative'),
        optimizer='projected Barzilai-Borwein gradient with monotone Armijo; fixed cap, last accepted iterate',
        sources={str(ROOT/'scripts/paper_final_v1'/n):sha(ROOT/'scripts/paper_final_v1'/n) for n in ('solver.py','common.py','classical.py')},
        historical_sources=verified(OUT/'model_manifest.json')['models'][0]['source_sha256']))
    print('Registered 72 finite calibration reconstructions before GT',flush=True)


def task_directory(split,method,candidate):
    return OUT/'classical'/('calibration' if split=='calibration' else split)/method/(f'candidate{candidate}' if candidate is not None else 'frozen')


def work(split,method,candidate=None):
    backend();search=verified(OUT/'classical/search.json')
    amendment=OUT/'classical/implementation_amendment_v2.json'
    if amendment.exists():
        amended=verified(amendment);assert amended['search_sha256']==sha(OUT/'classical/search.json');verify_hashes(amended['sources'])
    else:verify_hashes(search['sources'])
    if split=='calibration':
        cases=search['train_cases'];settings=search['settings'];config=search['candidates'][method][candidate];real_split='train'
    else:
        selected=verified(OUT/'classical/selected.json');config=selected['selected'][method]['config'];real_split=split
        reg=verified(OUT/'registration.json');cases=reg['cases'][split];settings=verified(OUT/'layouts.json')[split]
        if split=='test':
            lock=verified(OUT/'final_lock.json');assert method in [m['key'] for m in lock['methods']];verify_hashes(lock['sources'])
    target=task_directory(split,method,candidate);target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists():
        check=read(target/'complete.json');assert sha(target/'manifest.jsonl')==check['manifest_sha256'];print('Previously complete',target);return
    source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')}
    records=rows(target/'manifest.jsonl') if (target/'manifest.jsonl').exists() else []
    known={(r['condition'],r['draw'],r['case_id']):r for r in records}
    assert len(known)==len(records)
    for r in records:assert sha(target/r['prediction'])==r['sha256']
    started=time.monotonic()
    with AccessGuard(splits=[real_split]) as guard:
        with np.load(DATA/'geometry/train_geometry.npz') as z:geometry={k:z[k] for k in z.files}
        projector_cache={}
        for s in settings:
            for i,case in enumerate(cases):
                key=(s['condition'],s['draw'],case)
                if key in known:continue
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
                ids=s['input_ids'][case];hids=s['heldout_ids'][case]
                gtensor=geometry_tensor({k:v[ids] for k,v in geometry.items() if k not in ('horizontal_angle_deg','vertical_angle_deg')},'cuda:0')
                ikey=tuple(ids)
                # Bound geometry cache; repeated clustered layouts reuse their grid, random layouts stay bounded.
                if ikey not in projector_cache:
                    if len(projector_cache)>=8:projector_cache.pop(next(iter(projector_cache)))
                    projector_cache[ikey]=CachedProjector(gtensor)
                projector=projector_cache[ikey]
                p=np.load(DATA/source[case]['train_projection_path'],mmap_mode='r',allow_pickle=False)
                y=torch.as_tensor(np.array(p[ids]),device='cuda:0');torch.cuda.synchronize();setup=time.perf_counter()-t
                resume=target/'resume'/f'{s["condition"]}_d{s["draw"]}_{i:03d}.pt'
                torch.cuda.synchronize();t=time.perf_counter()
                c,stats=solve(projector,gtensor['valid_ray'],y,method,config,resume=resume if resume.exists() else None,save=resume)
                torch.cuda.synchronize();seconds=time.perf_counter()-t
                assert torch.isfinite(c).all() and c.min()>=0 and c.max()<=1
                path=target/'predictions'/f'{s["condition"]}_d{s["draw"]}_{i:03d}.npz';path.parent.mkdir(exist_ok=True)
                temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,C1=c.cpu().numpy());temp.replace(path)
                curve=target/'curves'/f'{s["condition"]}_d{s["draw"]}_{i:03d}.json';write(curve,stats)
                row=dict(condition=s['condition'],draw=s['draw'],case_id=case,input_ids=ids,heldout_ids=hids,prediction=str(path.relative_to(target)),sha256=sha(path),
                    config=config,solver_status=stats['status'],iterations=stats['iterations'],reconstruct_seconds=seconds,setup_seconds=setup,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),curve=str(curve))
                records.append(row);known[key]=row;jsonl(target/'manifest.jsonl',records)
                if len(records)%12==0:print(f'{split} {method} candidate{candidate}: {len(records)}/{len(cases)*len(settings)} {time.monotonic()-started:.1f}s',flush=True)
        audit=guard.summary();assert not audit['denied']
    assert len(records)==len(cases)*len(settings)
    write(target/'complete.json',dict(completed_at=now(),reconstructions=len(records),manifest_sha256=sha(target/'manifest.jsonl'),audit=audit,
        seconds=time.monotonic()-started,fields=len(records),split=real_split,method=method,candidate=candidate,config=config))


def select():
    search=verified(OUT/'classical/search.json');source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')};allrows=[]
    # Verify ALL candidates before opening even calibration GT.
    for method in METHODS:
        for candidate in range(2):
            d=task_directory('calibration',method,candidate);b=read(d/'complete.json');assert sha(d/'manifest.jsonl')==b['manifest_sha256']
            rr=rows(d/'manifest.jsonl');assert len(rr)==12
            for x in rr:assert sha(d/x['prediction'])==x['sha256'];allrows.append((method,candidate,d,x))
    freeze(OUT/'classical/calibration_seal.json',dict(at=now(),reconstructions=len(allrows),manifests={str(task_directory('calibration',m,c)/'manifest.jsonl'):sha(task_directory('calibration',m,c)/'manifest.jsonl') for m in METHODS for c in range(2)}))
    results=[];gt={}
    with AccessGuard(gt_splits=['train']) as guard:
        for method,candidate,d,x in allrows:
            case=x['case_id']
            if case not in gt:gt[case]=np.load(DATA/source[case]['gt_volume_64_path'],allow_pickle=False).squeeze()
            with np.load(d/x['prediction'],allow_pickle=False) as z:p=z['C1']
            results.append(dict(method=method,candidate=candidate,case_id=case,condition=x['condition'],volume=volume_metrics(p,gt[case])))
    scores=[];selected={}
    for method in METHODS:
        for c in range(2):
            rr=[x for x in results if x['method']==method and x['candidate']==c]
            scores.append(dict(method=method,candidate=c,mean_l1=float(np.mean([x['volume']['relative_l1'] for x in rr])),config=search['candidates'][method][c]))
        selected[method]=min([x for x in scores if x['method']==method],key=lambda x:(x['mean_l1'],x['candidate']))
    jsonl(OUT/'classical/calibration_metrics.jsonl',results)
    freeze(OUT/'classical/selected.json',dict(frozen_at=now(),selected=selected,scores=scores,search_sha256=sha(OUT/'classical/search.json'),audit=guard.summary(),final_GT_read=False))
    print(json.dumps(selected),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','work','select']);p.add_argument('--split',choices=['calibration','val','test']);p.add_argument('--method',choices=METHODS);p.add_argument('--candidate',type=int);a=p.parse_args()
    if a.action=='register':register()
    elif a.action=='select':select()
    else:work(a.split,a.method,a.candidate)
