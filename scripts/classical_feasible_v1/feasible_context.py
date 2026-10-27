"""Independent measured-field pretraining protocol; historical code is read-only."""
import sys,os,json,time,hashlib,copy,math
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'scripts/voxel_probe_v1')]
import numpy as np
import torch
from probe_context import sha,read,write,append,verify as verify_old,fresh
from geophy_set3d.training.selfsup import (make_dataset,make_loader,objective,run_epoch,selection_metric,move_dual_batch,projection_scores)
from geophy_set3d.training.utils import seed_everything,rng_state,restore_rng,atomic_checkpoint,move_batch
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.unet_baseline_v1 import load_config
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
ROOT=Path('/scratch/djc/GeoPhy-Set3D/classical_feasible_v1')
ART=PROJECT/'docs/reports/artifacts/classical_feasible_v1'
DATA=Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')


def setup():
    torch.set_num_threads(2);seed_everything(20261011)


def config():
    c=copy.deepcopy(load_config());c['loader'].update(num_workers=0,persistent_workers=False)
    return c


def verify():
    p=read(ART/'registration.json')
    for name,digest in p['sources'].items():assert sha(PROJECT/name)==digest,name
    verify_old()
    for directory in ['voxel_probe_v1','unet_continuation_v1']:
        for name,digest in read(PROJECT/f'docs/reports/artifacts/{directory}/delivery_manifest.json').items():assert sha(PROJECT/name)==digest,name
    for name,digest in read(PROJECT/'docs/reports/artifacts/unet_baseline_v1/history_audit.json')['frozen_sha256'].items():assert sha(name)==digest,name
    return p


class SharedField(torch.nn.Module):
    def __init__(self,value):super().__init__();self.value=torch.nn.Parameter(value)
    def forward(self,batch):return self.value


class TrackedAdam(torch.optim.AdamW):
    def __init__(self,model,journal):
        super().__init__(model.parameters(),lr=2e-4,weight_decay=1e-4);self.updates=0;self.journal=journal
    @torch.no_grad()
    def step(self,closure=None):
        parameters=[p for group in self.param_groups for p in group['params']]
        before=[p.detach().clone() for p in parameters]
        gradients=[p.grad for p in parameters if p.grad is not None]
        norm=torch.stack([v.double().square().sum() for v in gradients]).sum().sqrt().item()
        assert np.isfinite(norm) and norm>0
        result=super().step(closure)
        delta=torch.stack([(p-v).double().square().sum() for p,v in zip(parameters,before)]).sum().sqrt().item()
        assert np.isfinite(delta) and delta>0
        self.updates+=1
        append(self.journal,dict(update=self.updates,gradient_norm_after_clip=norm,parameter_delta=delta,lr=self.param_groups[0]['lr']))
        return result


def build(arm):
    seed_everything(20261011)
    if arm=='unet':return fresh()
    from geophy_set3d.training.classical_operators import REGISTRY,SETTINGS
    if arm in REGISTRY:model=REGISTRY[arm](**SETTINGS[arm])
    else:
        import importlib
        m=read(PROJECT/f'configs/baselines/{arm}/model_v1.json')
        factory=m.get('factory',m.get('implementation')).replace(':','.')
        module,name=factory.rsplit('.',1)
        model=getattr(importlib.import_module(module),name)(**m.get('kwargs',m.get('settings',{})))
    c0=read(PROJECT/'docs/reports/artifacts/unet_baseline_v1/initialization.json')['c0'];z0=math.log(c0/(1-c0))
    with torch.no_grad():
        if arm=='deeponet':
            model.branch[-1].weight.mul_(.01);model.branch[-1].bias.mul_(.01);model.bias.fill_(z0)
        else:
            head={'fno':lambda:model.readout[-1],'senseiver':lambda:model.decoder.postproc,
                'gnot':lambda:model.head[-1],'deepsets':lambda:model.decoder[-1]}[arm]()
            head.weight.mul_(.01);head.bias.fill_(z0)
    return model


def save_checkpoint(path,model,opt,scheduler,generator,**extra):
    atomic_checkpoint(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        rng=rng_state(generator),optimizer_steps=opt.updates,**extra))


def score_health(model,dataset,indices=(0,1,2)):
    rows=[];model.eval()
    old=int(dataset._epoch.item());dataset.set_epoch(0)
    with torch.no_grad():
        for index in indices:
            batch=move_dual_batch(dual_measurement_collate([dataset[index]]),'cuda:0')
            for branch in ['A','B']:
                p=model(batch[branch]);a=p.double()
                assert torch.isfinite(a).all() and a.min()>=0 and a.max()<=1
                rows.append(dict(index=index,branch=branch,mean=a.mean().item(),std=a.std(correction=0).item(),
                    nearzero=(a<1e-10).double().mean().item(),saturated=(p*(1-p)<4.54e-5).double().mean().item()))
    dataset.set_epoch(old);return rows


def healthy(rows):return all(r['mean']>=1e-8 and r['nearzero']<.95 and r['std']/max(r['mean'],1e-30)>=.05 and r['saturated']<.95 for r in rows)
