"""Full-budget, fresh ablations with authoritative epoch-commit recovery."""
from common import *
import argparse
import copy
import math
import time
import fcntl
import torch
from torch.utils.data import Subset
import probe_mrgr_multiscale_v1 as base
from geophy_set3d.training import selfsup
from models import model_for,install_loss


def source_identity():
    return {str(p):sha(p) for p in [*sorted((ROOT/'src').rglob('*.py')),*sorted((ROOT/'src').rglob('*.cpp')),
        ROOT/'scripts/probe_mrgr_multiscale_v1.py',*[ROOT/'scripts/paper_final_v1'/n for n in ('common.py','models.py','train.py')]]}

def register(arm,seed,*,smoke=None):
    target=OUT/('training' if smoke is None else 'resume_smoke')/f'{arm}_seed{seed}'
    if smoke:target=target/smoke
    if (target/'protocol.json').exists():return target
    config=copy.deepcopy(verified(OUT/'model_manifest.json')['models'][0]['config']);config['training']['seed']=seed;config['training']['device']='cuda:0'
    config['experiment']=dict(name=target.name,output_root=str(target))
    if arm=='no_consistency':config['loss']['consistency']=0.
    counts=sum(p.numel() for p in model_for(config,arm,'cpu').parameters())
    freeze(target/'protocol.json',dict(registered_at=now(),arm=arm,seed=seed,config=config,epochs=300,parameters=counts,
        sources=source_identity(),train_cases=verified(OUT/'registration.json')['cases']['train'],validation_cases=verified(OUT/'registration.json')['cases']['val'],
        selection='strict lowest mean validation heldout measurement relative L1, no volume GT',fresh_initialization=True,
        data_files=verified(OUT/'model_manifest.json')['training_data_sha256'],
        subset_smoke=smoke is not None,planned_updates=24900 if smoke is None else None,
        resume='last.pt authoritative complete epoch; model+best+optimizer+scheduler+python/numpy/torch/CUDA/loader RNG+metrics; deterministic epoch/case sampling; interrupted epoch replayed'))
    return target

def publish_best(target,saved):
    base.atomic_checkpoint(target/'checkpoints/best.pt',dict(model=saved['best_model'],config=saved['config'],epoch=saved['best_epoch'],selection_metric=saved['best_score'],arm=saved['arm'],protocol_sha256=saved['protocol_sha256'],volumetric_gt_used=False))

def train(target,stop_after=300):
    backend();target=Path(target);p=verified(target/'protocol.json');verify_hashes(p['sources']);config=p['config'];arm=p['arm'];seed=p['seed']
    install_loss(config);last=target/'checkpoints/last.pt';target.joinpath('checkpoints').mkdir(exist_ok=True)
    lock=(target/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with base.GTReadGuard(DATA,target/'checkpoints') as guard:
        for name,digest in p['data_files'].items():assert sha(DATA/name)==digest,name
        model=model_for(config,arm,'cuda:0');initial=base.state_hash(model.state_dict())
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
        generator=torch.Generator().manual_seed(seed);ds=selfsup.make_dataset(config);vd=selfsup.make_dataset(config,validation=True)
        if p['subset_smoke']:train_ds=Subset(ds,range(4));val_ds=Subset(vd,range(2))
        else:train_ds=ds;val_ds=vd
        loader=selfsup.make_loader(train_ds,config,True,generator);vloader=selfsup.make_loader(val_ds,config)
        metrics=[];best=math.inf;best_epoch=0;best_model=None;elapsed=0.
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False);assert saved['protocol_sha256']==sha(target/'protocol.json') and saved['config']==config
            model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            metrics=saved['metrics'];best=saved['best_score'];best_epoch=saved['best_epoch'];best_model=saved['best_model'];elapsed=saved['elapsed_seconds']
            assert len(metrics)==saved['epoch'] and saved['updates']==saved['epoch']*len(loader)
            # Preserve generator stream despite worker construction at process restart.
            for dl in (loader,vloader):
                if dl.num_workers and dl.persistent_workers:iter(dl)
            base.restore_rng(saved['rng'],generator)
            jsonl(target/'metrics.jsonl',metrics);publish_best(target,saved)
            append(target/'events.jsonl',dict(event='resume',at=now(),epoch=len(metrics),updates=saved['updates'],rng_restored=True,optimizer_scheduler_restored=True))
            del saved
        started=time.monotonic()
        write(target/'environment.json',dict(base.environment(config),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),initial_model_sha256=initial))
        for epoch in range(len(metrics),stop_after):
            ds.set_epoch(epoch);vd.set_epoch(0);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();lr=opt.param_groups[0]['lr']
            tr=selfsup.run_epoch(model,loader,config,opt);va=selfsup.run_epoch(model,vloader,config);torch.cuda.synchronize();scheduler.step();score=selfsup.selection_metric(va)
            assert (tr['sample_count'],va['sample_count'])==((4,2) if p['subset_smoke'] else (166,36))
            assert math.isfinite(score)
            if score<best:best=score;best_epoch=epoch+1;best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            row=dict(epoch=epoch+1,train=tr,val=va,learning_rate=lr,seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),loader_rng_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
            metrics.append(row)
            saved=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=base.rng_state(generator),config=config,arm=arm,
                epoch=epoch+1,updates=(epoch+1)*len(loader),protocol_sha256=sha(target/'protocol.json'),volumetric_gt_used=False,best_model=best_model,best_epoch=best_epoch,best_score=best,
                metrics=metrics,elapsed_seconds=elapsed+time.monotonic()-started)
            base.atomic_checkpoint(last,saved)
            if epoch+1==best_epoch:publish_best(target,saved)
            jsonl(target/'metrics.jsonl',metrics);write(target/'gt_access_audit.json',guard.summary())
            write(target/'status.json',dict(state='training',epoch=epoch+1,max_epochs=300,best_epoch=best_epoch,best_val_H=best,val_H=score,
                updates=(epoch+1)*len(loader),last_epoch_seconds=row['seconds'],elapsed_seconds=saved['elapsed_seconds'],gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),updated_at=now()))
            print(f'{arm} seed={seed} epoch={epoch+1}/300 valH={score:.7f} best={best:.7f}@{best_epoch} sec={row["seconds"]:.2f}',flush=True)
            del saved
        verify_hashes(p['sources']);audit=guard.summary()
        assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        if len(metrics)==300 and not p['subset_smoke']:
            write(target/'summary.json',dict(completed=True,epochs=300,updates=24900,arm=arm,seed=seed,parameters=p['parameters'],best_epoch=best_epoch,best_val_H=best,
                best_checkpoint_sha256=sha(target/'checkpoints/best.pt'),last_checkpoint_sha256=sha(last),gt_audit=audit,finished_at=now(),elapsed_seconds=elapsed+time.monotonic()-started))
            write(target/'status.json',dict(read(target/'summary.json'),state='complete',epoch=300))
        elif p['subset_smoke']:write(target/'smoke_complete.json',dict(epoch=len(metrics),at=now()))
        else:append(target/'events.jsonl',dict(event='controlled_stop',epoch=len(metrics),at=now()))
    fcntl.flock(lock,fcntl.LOCK_UN)


def resume_check():
    # Separate processes reproduce actual worker startup and CUDA RNG restoration.
    for name,stop in [('uninterrupted',2),('resumed',1),('resumed',2)]:
        target=register('single_scale',20261011,smoke=name)
        subprocess.run([sys.executable,__file__,'train','--target',str(target),'--stop-after',str(stop)],check=True)
    a=torch.load(OUT/'resume_smoke/single_scale_seed20261011/uninterrupted/checkpoints/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(OUT/'resume_smoke/single_scale_seed20261011/resumed/checkpoints/last.pt',map_location='cpu',weights_only=False)
    def same(a,b):
        if isinstance(a,torch.Tensor):
            if a.is_floating_point():torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
            else:assert torch.equal(a,b)
        elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
        elif isinstance(a,dict):assert a.keys()==b.keys();[same(a[k],b[k]) for k in a]
        elif isinstance(a,(tuple,list)):assert len(a)==len(b);[same(x,y) for x,y in zip(a,b)]
        elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-4,abs_tol=1e-6),(a,b)
        else:assert a==b,(a,b)
    import numpy as np
    for key in ('model','best_model','optimizer','scheduler','rng','epoch','updates','best_epoch','best_score'):same(a[key],b[key])
    for ra,rb in zip(a['metrics'],b['metrics']):
        for key in ('epoch','learning_rate','loader_rng_sha256'):same(ra[key],rb[key])
        for split in ('train','val'):
            same({k:v for k,v in ra[split].items() if k!='seconds'},{k:v for k,v in rb[split].items() if k!='seconds'})
    write(OUT/'training_resume_verification.json',dict(at=now(),passed=True,checked_state_keys=['model','best_model','optimizer','scheduler','rng','epoch','updates','best_epoch','best_score','metrics_except_timing'],float_tolerance=dict(rtol=1e-4,atol=1e-6),reason='CUDA grid_sample backward uses nondeterministic atomics; integer/RNG states exact, floats checked with tolerance',epochs=2,workers=4,train_subset=4,validation_subset=2,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')))
    print('Subprocess training interruption/resume PASSED within registered floating tolerance',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['register','train','resume-check']);a.add_argument('--target');a.add_argument('--arm');a.add_argument('--seed',type=int);a.add_argument('--stop-after',type=int,default=300);args=a.parse_args()
    if args.action=='register':print(register(args.arm,args.seed))
    elif args.action=='resume-check':resume_check()
    else:train(args.target,args.stop_after)
