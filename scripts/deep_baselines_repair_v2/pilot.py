"""Frozen bounded pilot and scientific GO; no GT or historical warm starts."""
from repair_context import *
import argparse
import importlib
import math
from audit import gradient_audit


def model_config(arm,version):return c.read(PROJECT/'configs/baselines'/arm/f'model_repair_{version}.json')

def build(arm,version,device='cpu'):
    seed_everything(SEED);m=model_config(arm,version)
    module,name=m['factory'].rsplit('.',1)
    return getattr(importlib.import_module(module),name)(**m['kwargs']).to(device)

def atomic(path,value):c.canonical_worker.probe.atomic_checkpoint(path,value)


def dependency(model,cfg):
    model.eval();rows=[]
    for i in range(4):
        batch=fixed_batch(cfg,validation=True,indices=(i,))
        altered=dict(batch['A'],input_projection_raw=batch['A']['input_projection_raw']*0)
        with torch.no_grad():
            normal,zero=model(batch['A']),model(altered)
            target=batch['H'];valid=target['input_valid_ray'].bool()&target['view_mask'][:,:,None,None,None]
            def score(x):
                p=ss.project_input(x,target)
                return ss.relative_projection_metrics(p[0],target['input_projection_raw'][0],valid[0])['relative_l1']
            n,z=score(normal),score(zero)
            rel=float((normal-zero).abs().mean()/(normal.abs().mean()+1e-15))
        rows.append(dict(case_id=batch['A']['case_id'][0],normal_H_L1=n,zero_input_H_L1=z,
            H_benefit=z-n,relative_output_change=rel,passed=rel>=1e-3 and z-n>=.001))
    return dict(rows=rows,passed_cases=sum(r['passed'] for r in rows),passed=sum(r['passed'] for r in rows)>=2)


def register(arm,version):
    target=ROOT/'pilots'/f'{arm}_{version}_seed{SEED}'
    target.mkdir(parents=True)
    cfg=config(arm,target);cfg['model']=dict(name='ray_'+arm+'_repair_'+version,baseline=model_config(arm,version))
    with guard() as access:
        ds=ss.make_dataset(cfg);vd=ss.make_dataset(cfg,validation=True)
        assert (len(ds),len(vd))==(166,36)
        m=build(arm,version)
        identities={str(c.DATA/'source_index.jsonl'):c.sha(c.DATA/'source_index.jsonl'),
            str(c.DATA/'train_statistics.json'):c.sha(c.DATA/'train_statistics.json'),
            str(c.DATA/'geometry/train_geometry.npz'):c.sha(c.DATA/'geometry/train_geometry.npz')}
        for row in ds.records+vd.records:
            path=c.DATA/row['train_projection_path'];identities[str(path)]=c.sha(path)
        src=sources();src[str(PROJECT/'configs/baselines'/arm/f'model_repair_{version}.json')]=c.sha(PROJECT/'configs/baselines'/arm/f'model_repair_{version}.json')
        src[str(PROJECT/'docs/deep_baselines_collapse_repair_v2_protocol_zh.md')]=c.sha(PROJECT/'docs/deep_baselines_collapse_repair_v2_protocol_zh.md')
        p=dict(at=c.now(),arm=arm,version=version,seed=SEED,config=cfg,
            train_cases=[r['case_id'] for r in ds.records[:32]],validation_cases=[r['case_id'] for r in vd.records],
            initial_state_sha256=state_hash(m.state_dict()),parameters=sum(p.numel() for p in m.parameters()),
            pilot_initial_epochs=10,pilot_hard_max_epochs=30,updates_per_epoch=16,
            extension='only if H improvement over last5epochs>=.005 and no collapse',
            gate='last2full-valH<=.99; >=.01 improvement vs initial; healthy output/case differences; >=2/4val input-zero H benefit>=.001 and output relative MA>=.001',
            sources=src,data_identity=identities,GT_used=False,test_used=False,selection='strict minimum full fixed36case val H4 L1',registration_audit=access.summary())
    c.freeze(target/'protocol.json',p)
    print(target,flush=True)


def smoke(model,batch,cfg,target):
    model.train();before=state_hash(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
    generator=torch.Generator().manual_seed(SEED)
    total,parts,volumes,_=ss.objective(model,batch)
    assert all(x.shape==(2,1,64,64,64) and x.dtype==torch.float32 and torch.isfinite(x).all() and x.min()>=0 and x.max()<=1 for x in volumes)
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));assert norm>0
    opt.step();opt.zero_grad(set_to_none=True);scheduler.step()
    assert state_hash(model.state_dict())!=before
    model.eval()
    with torch.no_grad():expected=model(batch['A'])
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator),volumetric_gt_used=False)
    path=target/'smoke_roundtrip.pt';atomic(path,payload)
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(loaded['model'],strict=True);opt.load_state_dict(loaded['optimizer']);scheduler.load_state_dict(loaded['scheduler']);restore_rng(loaded['rng'],generator)
    with torch.no_grad():actual=model(batch['A'])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert scheduler.last_epoch==1 and all(int(s['step'])==1 for s in opt.state.values())
    return dict(passed=True,gradient_norm=norm,parameter_updated=True,checkpoint_roundtrip_bitwise=True,
        optimizer_scheduler_rng_restored=True,checkpoint_sha256=c.sha(path),loss=float(total.detach()),loss_parts={k:float(v.detach()) for k,v in parts.items()})


def run(arm,version):
    target=ROOT/'pilots'/f'{arm}_{version}_seed{SEED}'
    p=c.verified(target/'protocol.json');c.verify_hashes(p['sources'])
    assert not (target/'metrics.jsonl').exists(), 'No silent replay of a pilot'
    c.backend();torch.set_num_threads(2);cfg=p['config']
    with guard() as access:
        c.verify_hashes(p['data_identity'])
        fixed=fixed_batch(cfg)
        model=build(arm,version,'cuda:0')
        engineering=smoke(model,fixed,cfg,target)
        c.freeze(target/'smoke.json',engineering)
        del model;torch.cuda.empty_cache()
        # Fresh seed and tensors: engineering optimizer state is never reused.
        model=build(arm,version,'cuda:0')
        assert state_hash(model.state_dict())==p['initial_state_sha256']
        ds=ss.make_dataset(cfg);vd=ss.make_dataset(cfg,validation=True)
        train=Subset(ds,range(32));generator=torch.Generator().manual_seed(SEED)
        loader=ss.make_loader(train,cfg,True,generator);vloader=ss.make_loader(vd,cfg)
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
        zero=ss.zero_volume_baseline(vloader,cfg);assert zero['relative_l1']==1.
        initial=ss.run_epoch(model,vloader,cfg)
        initial_health=health(model,fixed)
        c.freeze(target/'initial.json',dict(val=initial,zero=zero,health=initial_health,
            dependency=dependency(model,cfg),gradients=gradient_audit(model,fixed)))
        history=[];best=math.inf;best_epoch=0;began=time.monotonic();decision=None
        for epoch in range(1,31):
            ds.set_epoch(epoch-1);vd.set_epoch(0);torch.cuda.reset_peak_memory_stats()
            lr=opt.param_groups[0]['lr'];tr=ss.run_epoch(model,loader,cfg,opt);va=ss.run_epoch(model,vloader,cfg)
            torch.cuda.synchronize();scheduler.step();score=ss.selection_metric(va)
            assert (tr['sample_count'],va['sample_count'])==(32,36) and math.isfinite(score)
            h=health(model,fixed)
            dep=dependency(model,cfg) if epoch>=9 else None
            row=dict(epoch=epoch,updates=epoch*16,train=tr,val=va,health=h,dependency=dep,learning_rate=lr,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-began)
            if epoch in (1,5,10,20,30):row['gradients']=gradient_audit(model,fixed)
            history.append(row)
            payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator),
                config=cfg,epoch=epoch,updates=epoch*16,protocol_sha256=c.sha(target/'protocol.json'),volumetric_gt_used=False,
                metrics=history,best_epoch=best_epoch,best_score=best)
            if score<best:
                best=score;best_epoch=epoch;atomic(target/'best.pt',dict(model=model.state_dict(),epoch=epoch,selection_metric=score,config=cfg,volumetric_gt_used=False))
            payload.update(best_epoch=best_epoch,best_score=best)
            atomic(target/'last.pt',payload);c.jsonl(target/'metrics.jsonl',history)
            c.write(target/'status.json',dict(at=c.now(),state='pilot_running',epoch=epoch,updates=epoch*16,val_H=score,best_val_H=best,health=h))
            c.write(target/'access_audit.json',access.summary())
            print(arm,version,'epoch',epoch,'H',score,'mean',h['mean'],'caseRel',h['case_relative_difference'],'dep',None if dep is None else dep['passed_cases'],flush=True)
            healthy=h['mean']>=1e-8 and h['fraction_below_1e_10']<.95 and h['case_relative_difference']>=1e-3
            last2=epoch>=10 and all(ss.selection_metric(x['val'])<=.99 for x in history[-2:])
            improved=ss.selection_metric(initial)-score>=.01
            if epoch>=10 and last2 and improved and healthy and dep['passed']:
                decision='GO';break
            if epoch==10:
                trend=ss.selection_metric(history[4]['val'])-score
                if trend<.005 or not healthy:
                    decision='NO_GO_NO_SCIENTIFIC_TREND';break
            if epoch>=10 and epoch%5==0 and (h['mean']<1e-8 or h['fraction_below_1e_10']>=.95):
                decision='NO_GO_NEAR_ZERO_COLLAPSE';break
        decision=decision or 'NO_GO_PILOT_BUDGET_EXHAUSTED'
        c.verify_hashes(p['sources']);c.verify_hashes(p['data_identity']);assert not access.denied
        final=history[-1]
        c.freeze(target/'decision.json',dict(at=c.now(),decision=decision,eligible_for_formal=decision=='GO',
            completed_epochs=final['epoch'],updates=final['updates'],initial_val_H=ss.selection_metric(initial),
            final_val_H=ss.selection_metric(final['val']),zero_H=zero['relative_l1'],best_val_H=best,best_epoch=best_epoch,
            final_health=final['health'],final_dependency=final['dependency'],
            checkpoint_sha256=c.sha(target/'last.pt'),best_checkpoint_sha256=c.sha(target/'best.pt'),
            engineering=engineering,audit=access.summary(),sources_unchanged=True))
        c.write(target/'status.json',dict(at=c.now(),state=decision,epoch=final['epoch'],updates=final['updates']))
    print('PILOT DECISION',arm,version,decision,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['register','run']);a.add_argument('--arm',required=True,choices=ARMS);a.add_argument('--version',default='v2');args=a.parse_args()
    globals()[args.action](args.arm,args.version)
