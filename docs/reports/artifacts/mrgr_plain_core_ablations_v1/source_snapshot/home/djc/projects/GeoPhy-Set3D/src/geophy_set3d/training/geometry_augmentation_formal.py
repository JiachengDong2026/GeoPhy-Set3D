"""Independent instrumentation of the unchanged v1 numeric training sequence.
Only scalar observations and optional trace hashes added. Original selfsup.py untouched.
"""
import hashlib,json,math,time
import numpy as np
import torch
from geophy_set3d.training.selfsup import assert_no_gt,move_dual_batch,objective,projection_scores
from geophy_set3d.training.rvtc_matched import state_hash

def tree_hash(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()

def runtime_flags():
    return dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),precision='fp32')

def run_epoch(model,loader,config,optimizer=None):
    training=optimizer is not None; model.train(training)
    totals={k:0. for k in ('total','hold','cross','observed','consistency')}; count=0; started=time.perf_counter()
    scores={k:[] for k in ('heldout','cross','observed')}; moments=[]; examples={}; pairings=[]
    accumulation=config['training']['gradient_accumulation']; gradient_max=0.; gradient_rows=[]
    if training: optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step,raw in enumerate(loader):
            assert_no_gt(raw); batch=move_dual_batch(raw,config['training']['device'])
            loss,parts,volumes,projections=objective(model,batch)
            n=len(volumes[0]); count+=n
            if training:
                start=(step//accumulation)*accumulation; stop=min(start+accumulation,len(loader))
                group_samples=min(stop*loader.batch_size,len(loader.dataset))-start*loader.batch_size
                (loss*n/group_samples).backward()
                if (step+1)%accumulation==0 or step+1==len(loader):
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['training']['gradient_clip_norm'],error_if_nonfinite=True)
                    gradient_max=max(gradient_max,float(norm))
                    clipped=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in model.parameters() if p.grad is not None])))
                    gradient_rows.append(dict(preclip=float(norm),postclip=clipped))
                    optimizer.step()
                    if getattr(optimizer,'formal_trace',None) is not None:
                        optimizer.formal_trace.append(dict(step=len(gradient_rows),loss={k:float(v.detach()) for k,v in dict(total=loss,**parts).items()},lr=optimizer.param_groups[0]['lr'],gradient=gradient_rows[-1],parameter_hash=state_hash(dict(model.named_parameters())),optimizer_hash=tree_hash(optimizer.state_dict())))
                    optimizer.zero_grad(set_to_none=True)
            for k,v in dict(total=loss,**parts).items(): totals[k]+=float(v.detach())*n
            values=projection_scores(projections,batch)
            for k in scores: scores[k].extend(values[k])
            for c in volumes:
                for pred in c.detach():
                    mean=float(pred.mean()); std=float(pred.std(unbiased=False))
                    moments.append((float(pred.min()),float(pred.max()),mean,std*std+mean*mean,
                                    float((pred>=.999).float().mean())))
            for i,case_id in enumerate(batch['A']['case_id']):
                if len(examples)<8 and case_id not in examples:
                    examples[case_id]=volumes[0][i].detach().cpu()
                pairings.append((case_id, *[batch[k]['input_station_indices'][i, batch[k]['view_mask'][i]].tolist() for k in ('A', 'B', 'H')]))
    mean=sum(v[2] for v in moments)/len(moments)
    numerics=dict(min=min(v[0] for v in moments),max=max(v[1] for v in moments),mean=mean,
        std=math.sqrt(max(0.,sum(v[3] for v in moments)/len(moments)-mean*mean)),
        fraction_near_one=sum(v[4] for v in moments)/len(moments))
    ordered=[examples[k] for k in sorted(examples)]
    numerics['case_to_case_mean_absolute_difference']=sum(float((a-b).abs().mean()) for a,b in zip(ordered,ordered[1:]))/max(len(ordered)-1,1)
    return dict(loss={k:v/count for k,v in totals.items()},
        measurement={k:{metric:sum(v[metric] for v in rows)/len(rows) for metric in ('relative_l1','relative_l2')} for k,rows in scores.items()},
        output=numerics,gradient_norm_max=gradient_max,
        case_station_digest=hashlib.sha256(json.dumps(sorted(pairings)).encode()).hexdigest(),
        gradient_norms=gradient_rows,sample_count=count,gt_used=False,seconds=time.perf_counter()-started)
