#!/usr/bin/env python3
"""Independent smoke and full fresh/resumable DeepSets measurement-only training."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src')]
import torch
from torch.utils.data import Subset
from geophy_set3d.data.measurement_only import DualMeasurementDataset,dual_measurement_collate
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.deep_comparison import (load_comparison_config,build_comparison_model,model_identity,sha256,verify_registration,check_prediction)
from geophy_set3d.training.selfsup import make_dataset,make_loader,run_epoch,selection_metric,objective,move_dual_batch,assert_disjoint
from geophy_set3d.training.utils import seed_everything,directories,write_json,environment,atomic_checkpoint,rng_state,restore_rng,permute_views
from geophy_set3d.training.rvtc_matched import state_hash

REPO=Path(__file__).resolve().parents[1]


class CountedAdamW(torch.optim.AdamW):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.updates=0;self.post_clip_norms=[]
    def step(self,closure=None):
        values=[p.grad for g in self.param_groups for p in g['params'] if p.grad is not None]
        norm=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(v.detach()) for v in values])))
        if not torch.isfinite(torch.tensor(norm)): raise FloatingPointError('Nonfinite clipped gradient')
        result=super().step(closure);self.updates+=1;self.post_clip_norms.append(norm)
        return result


class InputLedger:
    """Observe actual model A/B inputs without modifying original run_epoch."""
    def __init__(self,model):
        self.rows=[];self.calls=0;self.handle=model.register_forward_pre_hook(self.capture)
    def capture(self,module,args):
        b=args[0]
        ids=[b['input_station_indices'][i,b['view_mask'][i]].tolist() for i in range(len(b['case_id']))]
        self.rows.append(dict(subset='A' if self.calls%2==0 else 'B',case_ids=list(b['case_id']),station_ids=ids))
        self.calls+=1
    def reset(self):self.rows=[];self.calls=0
    def close(self):self.handle.remove()


def smoke(c):
    out=Path(c['comparison']['output']); target=out/'engineering_smoke.json'
    if target.exists():raise FileExistsError('Smoke already exists')
    paths=directories(c,'_smoke'); guard=GTReadGuard(c['dataset']['root'],paths['checkpoints'])
    with guard:
        seed_everything(c['training']['seed']);torch.set_num_threads(2)
        model=build_comparison_model(c);initial=state_hash(model.state_dict())
        ds=DualMeasurementDataset(c['dataset']['root'],split='train',a_range=(5,5),b_range=(5,5),hold_views=2,
                                  base_seed=c['training']['seed'],normalization=c['normalization']['method'])
        b=move_dual_batch(dual_measurement_collate([ds[0],ds[1]]),c['training']['device']);assert_disjoint(b)
        opt=CountedAdamW(model.parameters(),lr=c['optimizer']['lr'],weight_decay=c['optimizer']['weight_decay'])
        calls=[]
        handle=model.register_forward_pre_hook(lambda m,args:calls.append(tuple(args[0]['case_id'])))
        # Explicit identity test: network must receive the exact A/B dictionaries.
        seen=[];h2=model.register_forward_pre_hook(lambda m,args:seen.append(id(args[0])))
        model.train();torch.cuda.reset_peak_memory_stats(c['training']['device']);began=time.perf_counter()
        loss,parts,volumes,proj=objective(model,b)
        for x,k in zip(volumes,['A','B']):check_prediction(x,b[k])
        assert seen==[id(b['A']),id(b['B'])] and id(b['H']) not in seen
        loss.backward();grads={name:float(p.grad.abs().sum()) for name,p in model.named_parameters()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert grads['encoder.0.weight']>0 and grads['decoder.0.body.0.weight']>0
        pre=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
        assert opt.updates==1 and state_hash(model.state_dict())!=initial
        handle.remove();h2.remove();model.eval()
        with torch.no_grad():
            base=model(b['A']); perm=model(permute_views(b['A'],torch.tensor([4,2,0,3,1],device=base.device)))
            permutation_error=float((base-perm).abs().max());assert torch.allclose(base,perm,atol=2e-6,rtol=2e-5)
            padded=dict(b['A'])
            for key,val in b['A'].items():
                if isinstance(val,torch.Tensor) and (key.startswith('input_') or key=='view_mask'):
                    extra=val[:,:1].clone()
                    if key=='view_mask':extra.fill_(False)
                    elif extra.is_floating_point():extra.fill_(float('nan'))
                    padded[key]=torch.cat([val,extra],1)
            pad=model(padded); padding_error=float((base-pad).abs().max());assert torch.allclose(base,pad,atol=2e-6,rtol=2e-5)
            invalid=dict(b['A']);invalid['input_valid_ray']=torch.zeros_like(invalid['input_valid_ray'])
            for key in ['input_projection','input_path_length_m']:invalid[key]=torch.full_like(invalid[key],float('nan'))
            check_prediction(model(invalid),invalid)
            allpad=dict(padded);allpad['view_mask']=torch.zeros_like(padded['view_mask']);check_prediction(model(allpad),allpad)
            # Own smoke checkpoint only; no formal initialization from it.
            atomic_checkpoint(paths['checkpoints']/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict()))
            saved=torch.load(paths['checkpoints']/'last.pt',map_location=c['training']['device'],weights_only=False)
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
            assert not model.training
        summary=dict(passed=True,initial_state_sha256=initial,optimizer_steps=1,max_views=dict(A=5,B=5,H=2),batch_size=2,
                     loss=float(loss),parts={k:float(v) for k,v in parts.items()},all_gradients_finite=True,encoder_gradient_l1=grads['encoder.0.weight'],
                     pre_clip_norm=pre,post_clip_norm=opt.post_clip_norms[-1],permutation_max_abs=permutation_error,padding_max_abs=padding_error,
                     H_excluded_from_network=True,disjoint=True,invalid_nan_safe=True,all_padding_safe=True,resume_smoke_passed=True,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated(c['training']['device']),seconds=time.perf_counter()-began,
                     identity=model_identity(model,c),access=guard.summary(),formal_reinitialization_required=True)
    write_json(target,summary);print(json.dumps(summary,indent=2),flush=True)


def formal(c,resume):
    out=Path(c['comparison']['output']);p=verify_registration(c);paths=directories(c)
    manifest=json.loads((out/'training_source_manifest.json').read_text())
    for path,h in manifest['sha256'].items():assert sha256(REPO/path)==h,path
    assert subprocess.check_output(['git','show','HEAD:docs/reports/artifacts/deepsets_selfsup_v1/training_source_manifest.json'],cwd=REPO)==(out/'training_source_manifest.json').read_bytes()
    assert json.loads((out/'engineering_smoke.json').read_text())['passed']
    last=paths['checkpoints']/'last.pt'
    if last.exists() and not resume:raise FileExistsError('Own run exists; explicit --resume required')
    guard=GTReadGuard(c['dataset']['root'],paths['checkpoints'])
    with guard:
        seed_everything(c['training']['seed']);torch.set_num_threads(2);model=build_comparison_model(c)
        initial=state_hash(model.state_dict());smoke_initial=json.loads((out/'engineering_smoke.json').read_text())['initial_state_sha256'];assert initial==smoke_initial
        optimizer=CountedAdamW(model.parameters(),lr=c['optimizer']['lr'],weight_decay=c['optimizer']['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        generator=torch.Generator().manual_seed(c['training']['seed'])
        train,val=make_dataset(c),make_dataset(c,validation=True);assert (len(train),len(val))==(166,36)
        train_loader=make_loader(train,c,True,generator);val_loader=make_loader(val,c)
        protocol={val.records[i]['case_id']:{k:val[i][k]['input_station_indices'].tolist() for k in ['A','B','H']} for i in range(36)}
        historical=json.loads(Path('/data/djc/GeoPhy-Set3D/outputs/logs/selfsup_oari_mrgr_v1/validation_station_ids.json').read_text())
        assert protocol==historical;write_json(out/'validation_station_ids.json',protocol)
        env=environment(c);env.update(identity=model_identity(model,c),initial_state_sha256=initial,
             initialization='fresh independent random seed; no weights read',TF32=False,cudnn_benchmark=False,cudnn_deterministic=torch.backends.cudnn.deterministic,
             deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),effective_batch=2)
        write_json(out/'environment.json',env)
        start=0;best=float('inf');rows=[]
        if resume:
            ck=torch.load(last,map_location=c['training']['device'],weights_only=False)
            assert ck['config']==c and ck['source_manifest_sha256']==sha256(out/'training_source_manifest.json')
            model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
            restore_rng(ck['rng'],generator);start=ck['epoch']+1;best=ck['best_val_metric'];optimizer.updates=ck['optimizer_steps']
            rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines() if json.loads(l)['epoch']<start]
            (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        ledger=InputLedger(model);index={r['case_id']:i for i,r in enumerate(train.records)}
        histrows=[json.loads(l) for l in Path('/data/djc/GeoPhy-Set3D/outputs/logs/selfsup_oari_mrgr_v1/metrics.jsonl').read_text().splitlines()]
        for epoch in range(start,300):
            train.set_epoch(epoch);val.set_epoch(0);ledger.reset();optimizer.post_clip_norms=[]
            torch.cuda.reset_peak_memory_stats(c['training']['device']);began=time.perf_counter();lr=optimizer.param_groups[0]['lr'];before=optimizer.updates
            tr=run_epoch(model,train_loader,c,optimizer)
            order=[case for b in ledger.rows if b['subset']=='A' for case in b['case_ids']]
            assert len(order)==166 and set(order)==set(index)
            for record in ledger.rows:
                for case,ids in zip(record['case_ids'],record['station_ids']):
                    i=index[case];a,b,h=train.sampler.sample(train.station_count,epoch=epoch,sample_index=i,case_id=case)
                    assert ids==(a if record['subset']=='A' else b).tolist()
            ordered=dict(epoch=epoch,batches=ledger.rows,case_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())
            with (out/'actual_training_inputs.jsonl').open('a') as f:f.write(json.dumps(ordered)+'\n')
            assert optimizer.updates-before==83
            ledger.reset();va=run_epoch(model,val_loader,c);scheduler.step();score=selection_metric(va);improved=score<best;best=min(best,score)
            row=dict(epoch=epoch,train=tr,val=va,learning_rate=lr,seconds=time.perf_counter()-began,
                     gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(c['training']['device']),optimizer_steps=optimizer.updates,
                     optimizer_steps_this_epoch=optimizer.updates-before,post_clip_gradient_norm_max=max(optimizer.post_clip_norms),
                     case_order_sha256=ordered['case_order_sha256'],historical_station_digest_equal=tr['case_station_digest']==histrows[epoch]['train']['case_station_digest'],
                     best=best,is_best=improved)
            rows.append(row)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,
                         best_val_metric=best,config=c,rng=rng_state(generator),optimizer_steps=optimizer.updates,git_commit=env['git_commit'],
                         source_manifest_sha256=sha256(out/'training_source_manifest.json'),training_signal='TDLAS measurements only',volumetric_gt_used=False,
                         selection_metric='mean held-out measurement relative L1',initial_state_sha256=initial)
            atomic_checkpoint(last,payload)
            if improved:atomic_checkpoint(paths['checkpoints']/'best.pt',payload)
            write_json(out/'training_access_audit.json',guard.summary())
            print(f"DeepSets {epoch+1}/300 steps={optimizer.updates} trainH={tr['measurement']['heldout']['relative_l1']:.6f} valH={score:.6f} best={best:.6f} seconds={row['seconds']:.2f}",flush=True)
        ledger.close();assert len(rows)==300 and optimizer.updates==24900
        audit=guard.summary();assert not any(audit[k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
        bestrow=min(rows,key=lambda r:selection_metric(r['val']))
        summary=dict(completed=True,epochs=300,optimizer_steps=optimizer.updates,steps_per_epoch=83,best_epoch=bestrow['epoch']+1,
                     best_val_heldout_relative_l1=best,total_epoch_seconds=sum(r['seconds'] for r in rows),mean_epoch_seconds=sum(r['seconds'] for r in rows)/300,
                     peak_gpu_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in rows),all_historical_station_digests_equal=all(r['historical_station_digest_equal'] for r in rows),
                     access=audit,identity=model_identity(model,c),stopping_reason='full preregistered budget; no early stopping')
        write_json(out/'training_complete.json',summary)
        write_json(out/'frozen_best.json',dict(checkpoint=str(paths['checkpoints']/'best.pt'),sha256=sha256(paths['checkpoints']/'best.pt'),
                     best_epoch=bestrow['epoch']+1,completed_epochs=300,optimizer_steps=24900,selection_metric='mean held-out measurement relative L1',
                     volumetric_gt_used=False,source_manifest_sha256=sha256(out/'training_source_manifest.json')))
        print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['smoke','train']);ap.add_argument('--config',default='configs/train_deepsets_selfsup_v1.yaml');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    c=load_comparison_config(a.config);verify_registration(c)
    if a.phase=='smoke':smoke(c)
    else:formal(c,a.resume)

if __name__=='__main__':main()
