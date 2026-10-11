"""GT-free dual-input MRGR: held-out/cross/observed measurements and consistency."""
import hashlib
import json
import math
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from geophy_set3d.models import build_model
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input, Y_SCALE
from geophy_set3d.data.measurement_only import DualMeasurementDataset, dual_measurement_collate, assert_no_gt
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from .utils import move_batch

FROZEN_LOSS=dict(hold=1.0,cross=.5,observed=.1,consistency=.05,beta=.01,relative_l1=.1,relative_epsilon=1e-6,measurement_scale=Y_SCALE)


def validate_selfsup_config(config):
    expected_train=dict(split='train',geometry_pool='train',a_views=[3,5],b_views=[3,5],heldout_views=2)
    expected_val=dict(split='val',geometry_pool='train',a_views=[4,4],b_views=[4,4],heldout_views=4)
    if config['train']!=expected_train or config['validation']!=expected_val or config['loss']!=FROZEN_LOSS:
        raise ValueError('Selfsup v1 freezes disjoint subsets and all loss weights/scales')
    if config['model']['name']!='geolift_oari_mrgr_res3d':
        raise ValueError('Selfsup uses the exact frozen full MRGR architecture')
    tr=config['training']
    expected=dict(seed=20261011,max_epochs=300,precision='fp32',gradient_clip_norm=1.,early_stopping_patience=None)
    if any(tr[k]!=v for k,v in expected.items()) or (tr['batch_size'],tr['gradient_accumulation']) not in ((2,1),(1,2)):
        raise ValueError('Selfsup v1 requires full 300 epochs and effective batch2')
    if config['optimizer']!=dict(name='adamw',lr=2e-4,weight_decay=1e-4) or config['scheduler']!=dict(name='cosine',T_max=300):
        raise ValueError('Optimizer/schedule differs from frozen selfsup v1')
    if config['normalization']!=dict(method='linear_p99_5',statistics_file='train_statistics.json'):
        raise ValueError('Use frozen train projection normalization')


def initialize_fresh_model(config):
    """Only model construction: no checkpoint or warm-start argument exists."""
    return build_model(config).to(config['training']['device'])


def make_dataset(config, validation=False, fixed_gate=False):
    settings=config['validation'] if validation else config['train']
    ds=DualMeasurementDataset(config['dataset']['root'],split=settings['split'],geometry_pool=settings['geometry_pool'],
        a_range=settings['a_views'],b_range=settings['b_views'],hold_views=settings['heldout_views'],
        base_seed=config['training']['seed'],normalization=config['normalization']['method'],
        statistics_file=config['normalization']['statistics_file'])
    ds.set_epoch(0)
    return ds


def make_loader(dataset, config, shuffle=False, generator=None):
    workers=config['loader']['num_workers']
    return DataLoader(dataset,batch_size=config['training']['batch_size'],shuffle=shuffle,
        num_workers=workers,pin_memory=config['loader']['pin_memory'],
        persistent_workers=config['loader']['persistent_workers'] and workers>0,
        collate_fn=dual_measurement_collate,generator=generator,
        **({'multiprocessing_context':'fork'} if workers else {}))


def move_dual_batch(batch, device):
    assert_no_gt(batch)
    return {k: move_batch(batch[k], device) for k in ('A', 'B', 'H')}


def measurement_discrepancy(projected, target, *, beta=.01,relative_l1=.1,relative_epsilon=1e-6):
    """Mean per-case robust discrepancy across all valid rays, including Y=0."""
    valid=target['input_valid_ray'].detach().bool() & target['view_mask'][:,:,None,None,None]
    reference=target['input_projection_raw'].detach()
    p=torch.where(valid,projected,0)/Y_SCALE
    y=torch.where(valid,reference,0)/Y_SCALE
    smooth=F.smooth_l1_loss(p,y,beta=beta,reduction='none').flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)
    relative=(p-y).abs().flatten(1).sum(1)/y.abs().flatten(1).sum(1).clamp_min(relative_epsilon)
    return (smooth+relative_l1*relative).mean()


def consistency_discrepancy(ca,cb):
    numerator=(ca-cb).abs().flatten(1).mean(1)
    denominator=.5*(ca.abs().flatten(1).mean(1)+cb.abs().flatten(1).mean(1))+1e-6
    return (numerator/denominator).mean()


def assert_disjoint(batch):
    for i in range(len(batch['A']['case_id'])):
        sets=[set(batch[k]['input_station_indices'][i,batch[k]['view_mask'][i]].tolist()) for k in ('A','B','H')]
        if any(sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2))):
            raise ValueError('A/B/H must be pairwise disjoint')
        if not all(batch[k]['case_id'][i]==batch['A']['case_id'][i] for k in ('B','H')):
            raise ValueError('A/B/H must refer to the same plume')


def objective(model,batch):
    assert_no_gt(batch); assert_disjoint(batch)
    # H is never passed to the reconstruction network.
    ca,cb=model(batch['A']),model(batch['B'])
    if not torch.isfinite(ca).all() or not torch.isfinite(cb).all():
        raise FloatingPointError('Nonfinite dual MRGR output')
    projections={label:{k:project_input(c,batch[k]) for k in ('A','B','H')}
                 for label,c in (('C_A',ca),('C_B',cb))}
    m=lambda label,key:measurement_discrepancy(projections[label][key],batch[key])
    parts=dict(hold=.5*(m('C_A','H')+m('C_B','H')),
               cross=.5*(m('C_A','B')+m('C_B','A')),
               observed=.5*(m('C_A','A')+m('C_B','B')),
               consistency=consistency_discrepancy(ca,cb))
    total=sum(FROZEN_LOSS[k]*value for k,value in parts.items())
    if not all(bool(torch.isfinite(v)) for v in [total,*parts.values()]):
        raise FloatingPointError('Nonfinite measurement-only objective')
    return total,parts,(ca,cb),projections


def projection_scores(projections,batch):
    groups={'heldout':(('C_A','H'),('C_B','H')),
            'cross':(('C_A','B'),('C_B','A')),
            'observed':(('C_A','A'),('C_B','B'))}
    results={k:[] for k in groups}
    for category,pairs in groups.items():
        for label,key in pairs:
            t=batch[key]; valid=t['input_valid_ray'].bool() & t['view_mask'][:,:,None,None,None]
            for i in range(len(t['case_id'])):
                row=relative_projection_metrics(projections[label][key][i],t['input_projection_raw'][i],valid[i])
                results[category].append(row)
    return results


def selection_metric(validation):
    """The sole model-selection quantity is mean held-out measurement relative L1."""
    return validation['measurement']['heldout']['relative_l1']


def zero_volume_baseline(loader,config):
    totals=[]; losses=[]
    with torch.no_grad():
        for raw in loader:
            assert_no_gt(raw); b=move_dual_batch(raw,config['training']['device']); h=b['H']
            zero=torch.zeros_like(h['input_projection_raw'])
            losses.append((float(measurement_discrepancy(zero,h)),len(h['case_id'])))
            valid=h['input_valid_ray'].bool() & h['view_mask'][:,:,None,None,None]
            for i in range(len(h['case_id'])):
                totals.append(relative_projection_metrics(zero[i],h['input_projection_raw'][i],valid[i]))
    return dict(relative_l1=sum(r['relative_l1'] for r in totals)/len(totals),
                relative_l2=sum(r['relative_l2'] for r in totals)/len(totals),
                measurement_discrepancy=sum(x*n for x,n in losses)/sum(n for x,n in losses))


def run_epoch(model,loader,config,optimizer=None):
    training=optimizer is not None; model.train(training)
    totals={k:0. for k in ('total','hold','cross','observed','consistency')}; count=0; started=time.perf_counter()
    scores={k:[] for k in ('heldout','cross','observed')}; moments=[]; examples={}; pairings=[]
    accumulation=config['training']['gradient_accumulation']; gradient_max=0.
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
                    gradient_max=max(gradient_max,float(norm)); optimizer.step(); optimizer.zero_grad(set_to_none=True)
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
        sample_count=count,gt_used=False,seconds=time.perf_counter()-started)
