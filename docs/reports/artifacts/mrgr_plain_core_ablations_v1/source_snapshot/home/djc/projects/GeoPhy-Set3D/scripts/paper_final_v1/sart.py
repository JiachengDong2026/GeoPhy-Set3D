"""Supplementary nodal-mole-fraction SART, registered before final test access.

CPU sparse CSR arithmetic leaves occupied GPUs available to registered training.
This solves A chi = Y; it does not claim equality with P(C).
"""
from common import *
import argparse
import fcntl
import resource
import time
import numpy as np
import scipy.sparse as sp
import torch
from geophy_set3d.reconstruction.classical import RayMatrix, mole_to_mass, geometry_tensor, project
from geophy_set3d.physics.projectors import mass_to_mole

BASE=OUT/'sart'


class Block:
    def __init__(self, geometry, n=64, samples=96, dtype=np.float32):
        a=RayMatrix(geometry,n=n,samples=samples,device='cpu',dtype=torch.float64 if dtype==np.float64 else torch.float32)
        self.valid=np.asarray(geometry['valid_ray'],bool).reshape(-1)
        self.a=sp.csr_matrix((a.a.values().numpy(),a.a.col_indices().numpy(),a.a.crow_indices().numpy()),shape=a.a.shape)
        self.at=self.a.T.tocsr()
        self.row_sum=np.asarray(self.a.sum(1)).ravel()
        col=np.asarray(self.a.sum(0)).ravel();self.inv_col=np.zeros_like(col)
        np.divide(1,col,out=self.inv_col,where=col>0)
        assert np.all(self.row_sum>0)
        self.bytes=sum(x.nbytes for m in (self.a,self.at) for x in (m.data,m.indices,m.indptr))

    def forward(self,x):return self.a@x.reshape(-1)
    def adjoint(self,y):return self.at@y


def solve(blocks, measurements, config, *, checkpoint=None, identity=None, stop_after=None):
    """One sweep visits each station in input order. Resume only at sweep boundaries."""
    ys=[np.asarray(y).reshape(-1)[b.valid].astype(b.a.dtype) for b,y in zip(blocks,measurements)]
    assert len(blocks)==len(measurements)>0 and all(np.isfinite(y).all() for y in ys)
    if checkpoint is not None and Path(checkpoint).exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['identity']==identity and state['config']==config
        x=state['x'];curve=state['curve'];stable=state['stable'];previous=state['previous'];status=state['status']
    else:
        x=np.zeros(blocks[0].a.shape[1],dtype=blocks[0].a.dtype);curve=[];stable=0;previous=None;status='running'
    denom=max(sum(float(np.dot(y.astype(np.float64),y.astype(np.float64))) for y in ys),1e-12)
    begin=len(curve)
    if status=='running':
        for step in range(begin+1,config['iterations']+1):
            for b,y in zip(blocks,ys):
                correction=b.adjoint((y-b.forward(x))/b.row_sum)
                x=np.clip(x+config['relaxation']*correction*b.inv_col,0,1)
            value=sum(float(np.dot(r.astype(np.float64),r.astype(np.float64))) for b,y in zip(blocks,ys) for r in [b.forward(x)-y])/denom
            if not np.isfinite(value) or not np.isfinite(x).all():raise FloatingPointError('Nonfinite SART iterate')
            stable=stable+1 if previous is not None and abs(previous-value)<=config['tolerance']*max(previous,1e-12) else 0
            curve.append(dict(iteration=step,objective=.5*value,solver_relative_l2=value**.5));previous=value
            if step>=config['minimum_iterations'] and stable>=config['patience']:status='objective_stable'
            elif step==config['iterations']:status='budget_exhausted'
            interrupted=stop_after is not None and step>=stop_after
            if checkpoint is not None and (step%10==0 or status!='running' or interrupted):
                path=Path(checkpoint);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.pt')
                torch.save(dict(identity=identity,config=config,x=x,curve=curve,stable=stable,previous=previous,status=status),temp);temp.replace(path)
            if status!='running' or interrupted:break
    assert np.isfinite(x).all() and x.min()>=0 and x.max()<=1
    return x,dict(iterations=len(curve),status=status,curve=curve,solver_relative_l2=previous**.5,
        checkpoint_io_in_solver_time=True,implementation='CPU SciPy CSR, station-block SART')


def directory(split,candidate=None):
    return OUT/'classical'/split/'SART'/(f'candidate{candidate}' if candidate is not None else 'frozen')


def register():
    assert not (OUT/'final_method_choices_closed.json').exists() and not (OUT/'final_lock.json').exists()
    search=verified(OUT/'classical/search.json');reg=verified(OUT/'registration.json')
    fixed=dict(iterations=200,minimum_iterations=40,tolerance=1e-5,patience=8,resolution=64,samples=96)
    freeze(BASE/'registration.json',dict(registered_at=now(),reason='User explicitly requested addition of SART before final test method closure',
        parent_registration_sha256=sha(OUT/'registration.json'),train_cases=search['train_cases'],settings=search['settings'],
        candidates=[dict(fixed,relaxation=v) for v in (.5,1.)],calibration_reconstructions=24,
        selection='minimum mean train 3DrelativeL1 over six fixed cases and two layouts; ties candidate index; seal all candidates before calibration GT64',
        initialization='zero nodal mole fraction',bounds=[0,1],station_order='frozen input_ids order, one station block at a time',
        unknown='nodal mole fraction chi; return mass fraction C=mole_to_mass(chi)',
        forward='A interpolates nodal chi then integrates K96 midpoint, ppm*m; frozen P interpolates nodal C then converts to chi; different discrete representation',
        comparison='supplementary alternative-discretization baseline; all reported projection metrics use the exact frozen nonlinear P(C)',
        numerical='float32 CPU CSR; FP64 accumulation of residual norms; true sparse transpose, never dense volume matrix',
        statistics=reg['statistics'],final_test_reconstructions=595,final_methods=13,final_fields_total=13090,
        final_primary_comparisons=12,no_test_access_before_final_lock=True,final_GT_read=False,
        sources={str(p):sha(p) for p in [Path(__file__),ROOT/'scripts/paper_final_v1/common.py',ROOT/'src/geophy_set3d/reconstruction/classical.py',ROOT/'src/geophy_set3d/physics/projectors.py']}))


def method():
    s=verified(BASE/'selected.json')
    return dict(key='SART',kind='sart',config=s['config'],operator='nodal mole fraction A; alternative discretization',registration_sha256=sha(BASE/'registration.json'),selected_sha256=sha(BASE/'selected.json'))


def cached_blocks(cache,ids,geometry,capacity=12):
    # Keep references for this reconstruction even when later stations evict
    # earlier requested entries from the bounded cross-case cache.
    blocks=[]
    for station in ids:
        if station not in cache:
            if len(cache)>=capacity:cache.pop(next(iter(cache)))
            cache[station]=Block({k:v[[station]] for k,v in geometry.items() if k not in ('horizontal_angle_deg','vertical_angle_deg')})
        blocks.append(cache[station])
    return blocks


def work(split,candidate=None,*,limit=None):
    backend();reg=verified(BASE/'registration.json');started=time.monotonic()
    amendment=BASE/'implementation_amendment_v2.json'
    if amendment.exists():
        amended=verified(amendment);assert amended['registration_sha256']==sha(BASE/'registration.json');verify_hashes(amended['sources'])
    else:verify_hashes(reg['sources'])
    if split=='calibration':cases=reg['train_cases'];settings=reg['settings'];config=reg['candidates'][candidate];real_split='train'
    else:
        config=verified(BASE/'selected.json')['config'];real_split=split
        cases=verified(OUT/'registration.json')['cases'][split];settings=verified(OUT/'layouts.json')[split]
    if split=='test':
        lock=verified(OUT/'final_lock.json');assert method() in lock['methods'];verify_hashes(lock['sources']);verify_hashes(lock['evidence']);verify_hashes(lock['data_identity'])
    target=directory(split,candidate);target.mkdir(parents=True,exist_ok=True)
    mutex=(target/'worker.lock').open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (target/'complete.json').exists():assert sha(target/'manifest.jsonl')==read(target/'complete.json')['manifest_sha256'];return
    rr=rows(target/'manifest.jsonl') if (target/'manifest.jsonl').exists() else []
    known={(r['condition'],r['draw'],r['case_id']) for r in rr};assert len(known)==len(rr)
    for r in rr:assert sha(target/r['prediction'])==r['sha256'] and r['config']==config
    source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')};cache={}
    with AccessGuard(splits=[real_split]) as guard:
        with np.load(DATA/'geometry/train_geometry.npz') as z:geometry={k:z[k] for k in z.files}
        for s in settings:
            for i,case in enumerate(cases):
                key=(s['condition'],s['draw'],case)
                if key in known:continue
                t=time.perf_counter();ids=s['input_ids'][case]
                blocks=cached_blocks(cache,ids,geometry)
                measurement=np.load(DATA/source[case]['train_projection_path'],mmap_mode='r',allow_pickle=False)
                ys=np.array(measurement[ids]);setup=time.perf_counter()-t
                stem=f'{s["condition"]}_d{s["draw"]}_{i:03d}'
                checkpoint=target/'resume'/(stem+'.pt');identity=dict(case_id=case,input_ids=ids,registration_sha256=sha(BASE/'registration.json'))
                t=time.perf_counter();chi,stats=solve(blocks,ys,config,checkpoint=checkpoint,identity=identity)
                c=mole_to_mass(chi.reshape((64,)*3));seconds=time.perf_counter()-t
                assert c.shape==(64,)*3 and np.isfinite(c).all() and c.min()>=0 and c.max()<=1
                path=target/'predictions'/(stem+'.npz');path.parent.mkdir(exist_ok=True);temp=path.with_suffix('.tmp.npz')
                np.savez_compressed(temp,C1=c);temp.replace(path)
                curve=target/'curves'/(stem+'.json');write(curve,stats)
                rr.append(dict(case_id=case,condition=s['condition'],draw=s['draw'],input_ids=ids,heldout_ids=s['heldout_ids'][case],
                    prediction=str(path.relative_to(target)),sha256=sha(path),config=config,iterations=stats['iterations'],solver_status=stats['status'],
                    reconstruct_seconds=seconds,setup_seconds=setup,peak_allocated_bytes=0,gpu_uuid=None,device='cpu',
                    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,operator='nodal chi A',curve=str(curve)))
                known.add(key);jsonl(target/'manifest.jsonl',rr)
                if len(rr)%12==0:print(f'SART {split} candidate{candidate}: {len(rr)}/{len(cases)*len(settings)}, {time.monotonic()-started:.1f}s',flush=True)
                if limit is not None and len(rr)>=limit:return
        audit=guard.summary();assert not audit['denied']
    assert len(rr)==len(cases)*len(settings)
    write(target/'complete.json',dict(completed_at=now(),reconstructions=len(rr),fields=len(rr),manifest_sha256=sha(target/'manifest.jsonl'),
        config=config,method='SART',split=real_split,audit=audit,device='cpu',seconds=time.monotonic()-started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def select():
    from geophy_set3d.evaluation.research import volume_metrics
    reg=verified(BASE/'registration.json');source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')};worklist=[];manifests={}
    for candidate in range(len(reg['candidates'])):
        d=directory('calibration',candidate);completion=read(d/'complete.json');manifest=d/'manifest.jsonl'
        assert sha(manifest)==completion['manifest_sha256'];rr=rows(manifest);assert len(rr)==12
        for r in rr:assert sha(d/r['prediction'])==r['sha256'];worklist.append((candidate,d,r))
        manifests[str(manifest)]=sha(manifest)
    assert len(worklist)==24
    freeze(BASE/'calibration_seal.json',dict(at=now(),manifests=manifests,reconstructions=24))
    scored=[];gt={};discrepancy=[]
    with AccessGuard(gt_splits=['train']) as guard:
        with np.load(DATA/'geometry/train_geometry.npz') as z:geometry={k:z[k] for k in z.files}
        for candidate,d,r in worklist:
            case=r['case_id']
            if case not in gt:gt[case]=np.load(DATA/source[case]['gt_volume_64_path'],allow_pickle=False).squeeze()
            with np.load(d/r['prediction'],allow_pickle=False) as z:c=z['C1']
            scored.append(dict(candidate=candidate,case_id=case,condition=r['condition'],volume=volume_metrics(c,gt[case])))
            if candidate==0:
                ids=r['input_ids'];g={k:v[ids] for k,v in geometry.items() if k not in ('horizontal_angle_deg','vertical_angle_deg')}
                b=Block(g);valid=np.asarray(g['valid_ray'],bool);cgt=torch.as_tensor(gt[case],dtype=torch.float32)
                with torch.no_grad():exact=project(cgt,geometry_tensor(g,'cpu')).numpy()[valid]
                approximate=b.forward(mass_to_mole(cgt).numpy());delta=approximate-exact
                discrepancy.append(dict(case_id=case,condition=r['condition'],relative_l1=float(np.abs(delta).sum()/max(np.abs(exact).sum(),1e-6)),
                    relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(exact),1e-6)),max_absolute_ppm_m=float(np.abs(delta).max())))
    scores=[dict(candidate=i,mean_l1=float(np.mean([r['volume']['relative_l1'] for r in scored if r['candidate']==i])),config=c) for i,c in enumerate(reg['candidates'])]
    choice=min(scores,key=lambda r:(r['mean_l1'],r['candidate']))
    jsonl(BASE/'calibration_metrics.jsonl',scored);jsonl(BASE/'operator_discrepancy_train.jsonl',discrepancy)
    freeze(BASE/'selected.json',dict(frozen_at=now(),**choice,scores=scores,registration_sha256=sha(BASE/'registration.json'),
        calibration_seal_sha256=sha(BASE/'calibration_seal.json'),audit=guard.summary(),final_GT_read=False))
    print(json.dumps(choice),flush=True)


def wait_development():
    verified(BASE/'registration.json')
    while not (BASE/'development_complete.json').exists():
        status=read(BASE/'pipeline_status.json')
        if status['state']=='failed':raise RuntimeError('SART development failed; inspect preserved logs')
        try:os.kill(status['pid'],0)
        except ProcessLookupError:raise RuntimeError('SART controller ended without development completion')
        print('Waiting for registered SART development completion',flush=True);time.sleep(15)
    completion=verified(BASE/'development_complete.json');verify_hashes(completion['evidence'])
    assert read(OUT/'evaluation/sart_dev/independent_verification.json')['passed']
    assert not read(OUT/'evaluation/sart_dev/summary.json')['failures']
    return method()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','work','select']);p.add_argument('--split',choices=['calibration','val','test']);p.add_argument('--candidate',type=int);a=p.parse_args()
    if a.action=='register':register()
    elif a.action=='select':select()
    else:work(a.split,a.candidate)
