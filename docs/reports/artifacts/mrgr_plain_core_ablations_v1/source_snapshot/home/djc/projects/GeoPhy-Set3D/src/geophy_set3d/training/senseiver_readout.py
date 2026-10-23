"""Bounded frozen-latent readout; statistics fitted on indices 0..7 only."""
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.models.senseiver_core import Decoder
from geophy_set3d.models.ray_senseiver_ss import fourier


class ProbeGuard(GTReadGuard):
    def __init__(self, root, checkpoint_root, records, count, exact_checkpoint=None):
        super().__init__(root, checkpoint_root)
        permitted={str((Path(root)/r['train_projection_path']).absolute()) for r in records[:count]}
        self.allowed={p:i for p,i in self.allowed.items() if p in permitted}
        self.exact_checkpoint=str(Path(exact_checkpoint).absolute()) if exact_checkpoint else None
        self.count=count

    def _audit(self,event,args):
        if self.active and event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.path.abspath(os.fsdecode(args[0]));mode,flags=args[1:3]
            reading=('r' in mode or '+' in mode) if isinstance(mode,str) else not bool(flags&os.O_WRONLY)
            if reading and self.exact_checkpoint and p.endswith(('.pt','.pth','.ckpt')) and p!=self.exact_checkpoint:
                with self.supervised_checkpoint_attempts.get_lock():self.supervised_checkpoint_attempts.value+=1
                raise PermissionError('Only registered consistency Treatment last35 is readable in Stage A')
        super()._audit(event,args)

    def summary(self):
        return dict(super().summary(),policy=f'Only first {self.count} train Pool A; no GT/Val/Pool B/test; exact source checkpoint in A, own checkpoint directory in C')


def standardize(latents, targets):
    x=torch.as_tensor(latents,dtype=torch.float64);y=torch.as_tensor(targets,dtype=torch.float64)
    mu=x[:8].mean(0);scale=(x[:8]-mu).square().mean().sqrt()
    ym=y[:8].mean();ys=y[:8].std(unbiased=False)
    if not torch.isfinite(scale) or not torch.isfinite(ys) or scale<=1e-12 or ys<=1e-12:
        raise ValueError('Degenerate training-only latent/target scale; no check-set fallback')
    return (x-mu)/scale,(y-ym)/ys,dict(latent_mean=mu,latent_global_rms=scale,target_mean=ym,target_std=ys)


def ridge(z, y):
    z=z.double().flatten(1);y=y.double()
    gram=z[:8]@z[:8].T
    lam=1e-3*gram.trace()/8
    matrix=gram+lam*torch.eye(8,dtype=torch.float64)
    alpha=torch.linalg.solve(matrix,y[:8])
    prediction=z@z[:8].T@alpha
    return prediction,dict(lambda_value=float(lam),condition_number=float(torch.linalg.cond(matrix)),
        solve_relative_residual=float((matrix@alpha-y[:8]).norm()/y[:8].norm()),dtype='float64',intercept='training target mean; no extra intercept')


class ScalarQueryReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=Decoder(96,64,64,1,1,num_cross_attention_heads=4,dropout=0.)
        self.register_buffer('center_query',fourier(torch.zeros(1,3)).reshape(1,1,96))

    def forward(self,z):
        return self.decoder(z,self.center_query.expand(len(z),-1,-1)).reshape(-1)


def errors(prediction,target):
    prediction=torch.as_tensor(prediction,dtype=torch.float64);target=torch.as_tensor(target,dtype=torch.float64)
    return {name:dict(MSE=float((prediction[s]-target[s]).square().mean()),MAE=float((prediction[s]-target[s]).abs().mean()))
        for name,s in [('fit',slice(0,8)),('check',slice(8,16))]}
