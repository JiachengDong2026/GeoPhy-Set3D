"""Separate, GT-free sparse-field guard correction; original decision stays frozen."""
from feasible_context import *
from feasible_context import verify as verify_original
BASE_ROOT=ROOT
BASE_ART=ART
ROOT=BASE_ROOT/'sparse_guard_v2'
ART=BASE_ART/'sparse_guard_v2'


def verify():
    verify_original()
    registration=read(ART/'registration.json')
    for path,digest in registration['sources'].items():assert sha(PROJECT/path)==digest,path
    for path,digest in registration['boundary'].items():assert sha(path)==digest,path


def field_stats(value):
    p=value.detach().double();saturated=value*(1-value)<4.54e-5
    return dict(mean=p.mean().item(),std=p.std(correction=0).item(),nearzero=(p<1e-10).double().mean().item(),
        saturated=saturated.double().mean().item(),
        saturated_mass_fraction=(p*saturated).sum().item()/max(p.sum().item(),1e-30))


def healthy(rows):
    return all(r['mean']>=1e-8 and r['nearzero']<.95 and r['std']/max(r['mean'],1e-30)>=.05
        and r['saturated_mass_fraction']<.95 for r in rows)


@torch.no_grad()
def score_health(model,dataset,indices=(0,1,2)):
    rows=[];model.eval();old=int(dataset._epoch.item());dataset.set_epoch(0)
    for index in indices:
        batch=move_dual_batch(dual_measurement_collate([dataset[index]]),'cuda:0')
        for branch in ['A','B']:
            p=model(batch[branch]);assert torch.isfinite(p).all() and p.min()>=0 and p.max()<=1
            rows.append(dict(index=index,branch=branch,**field_stats(p)))
    dataset.set_epoch(old);return rows
