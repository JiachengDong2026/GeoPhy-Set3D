"""Independent fixed-eight-case output-initialization probe; never changes v1."""
import copy
import math
from pathlib import Path
import torch
from geophy_set3d.training.ray_senseiver import MODEL
from geophy_set3d.training.selfsup import FROZEN_LOSS
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.reconstruction.classical import mole_to_mass

NAME = 'ray_senseiver_init_probe_v1'
PROBE = dict(case_indices=list(range(8)), sampler_epoch=0, shuffle=False,
             diagnostic_case_indices=[0,1,2], health_every=5, weight_factor=.01,
             logit_clamp=[1e-6,1-1e-6], dependence_epsilon=1e-12)


def validate(c):
    if set(c) != {'dataset','experiment','loader','loss','model','normalization','optimizer','train','training','scheduler','runtime','probe'}:
        raise ValueError('Independent probe schema required')
    if c['model'] != MODEL or c['loss'] != FROZEN_LOSS or c['probe'] != PROBE:
        raise ValueError('Frozen model/loss/probe settings required')
    if c['experiment']['name'] != NAME or c['experiment']['output_root'] != '/data/djc/GeoPhy-Set3D/outputs':
        raise ValueError('Independent output identity required')
    if c['dataset'] != dict(root='/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1'):
        raise ValueError('Frozen dataset required')
    if c['train'] != dict(split='train',geometry_pool='train',a_views=[3,5],b_views=[3,5],heldout_views=2):
        raise ValueError('Original natural disjoint subsets required')
    if c['normalization'] != dict(method='linear_p99_5',statistics_file='train_statistics.json'):
        raise ValueError('Original normalization required')
    t=c['training']
    if set(t) != {'seed','max_epochs','precision','gradient_clip_norm','batch_size','gradient_accumulation','device'} or any(t[k]!=v for k,v in dict(seed=20261011,max_epochs=40,precision='fp32',gradient_clip_norm=1.).items()):
        raise ValueError('Exact bounded budget required')
    if (t['batch_size'],t['gradient_accumulation']) not in [(2,1),(1,2)]:
        raise ValueError('Effective batch2 required')
    if c['optimizer'] != dict(name='adamw',lr=2e-4,weight_decay=1e-4) or c['scheduler'] != dict(name='cosine',T_max=300):
        raise ValueError('Original optimizer/schedule required')
    if c['loader'] != dict(num_workers=0,persistent_workers=False,pin_memory=True):
        raise ValueError('Single-process auditable loader required')
    if set(c['runtime']) != {'query_chunk_size','decoder_checkpoint'} or type(c['runtime']['query_chunk_size']) is not int or not 1<=c['runtime']['query_chunk_size']<=262144 or type(c['runtime']['decoder_checkpoint']) is not bool:
        raise ValueError('Only complete-query engineering changes allowed')
    return c


class TrainEightGuard(GTReadGuard):
    def __init__(self,root,checkpoint_root,records):
        super().__init__(root,checkpoint_root)
        paths={str(Path(root)/r['train_projection_path']) for r in records[:8]}
        self.allowed={p:i for p,i in self.allowed.items() if p in paths}


def fitted_scale(length, measurement):
    length=torch.as_tensor(length,dtype=torch.float64)
    measurement=torch.as_tensor(measurement,dtype=torch.float64)
    if length.numel()==0 or length.shape!=measurement.shape or not torch.isfinite(length).all() or not torch.isfinite(measurement).all() or (length<=0).any() or (measurement<0).any():
        raise ValueError('Invalid valid-ray length/measurement')
    numerator=float((length*measurement).sum())
    denominator=float(1e6*length.square().sum())
    if not math.isfinite(denominator) or denominator<=0:
        raise ValueError('Invalid scale denominator')
    chi=numerator/denominator;mass=float(mole_to_mass(chi))
    if not math.isfinite(chi) or not 0<chi<1 or not math.isfinite(mass) or not 0<mass<1:
        raise ValueError('Illegal physical scale; no fallback')
    clamp=min(max(mass,1e-6),1-1e-6)
    return dict(numerator=numerator,denominator=denominator,numerator_units='ppm*m^2',
                denominator_units='1e6*m^2',chi_init=chi,C_init=mass,
                C_for_logit=clamp,clamp_applied=mass!=clamp,
                bias_logit=math.log(clamp/(1-clamp)),valid_rays=length.numel())


def initialize_pair(fresh,scale):
    control=copy.deepcopy(fresh);treatment=copy.deepcopy(fresh)
    with torch.no_grad():
        treatment.decoder.postproc.weight.mul_(.01)
        treatment.decoder.postproc.bias.fill_(scale['bias_logit'])
    changed=[]
    for (n,p),(n2,q) in zip(control.named_parameters(),treatment.named_parameters()):
        assert n==n2
        if not torch.equal(p,q):changed.append(n)
        if not n.startswith('decoder.postproc.'):assert torch.equal(p,q),n
    assert changed==['decoder.postproc.weight','decoder.postproc.bias']
    assert torch.count_nonzero(treatment.decoder.postproc.weight)>0
    return control,treatment


def exit_event(epoch,history,health,c_init,previous):
    """Strict 'after' means epoch>5 / epoch>20: first eligible checks 10 /25."""
    h=health['full_fixed_subset']; log=health['logits']; out=h['output']
    collapse=(epoch>5 and h['measurement']['heldout']['relative_l1']>=.98 and
              log['fraction_below_minus12']>.95 and out['mean']<.01*c_init)
    collapse_streak=previous.get('collapse_streak',0)+1 if collapse else 0
    recent=sum(r['eval']['measurement']['heldout']['relative_l1'] for r in history[-5:])/5 if epoch>=10 else None
    older=sum(r['eval']['measurement']['heldout']['relative_l1'] for r in history[-10:-5])/5 if epoch>=10 else None
    improvement=(older-recent)/older if older is not None and older>0 else None
    failure=epoch>20 and recent is not None and recent>=.98 and improvement<.01
    failure_streak=previous.get('failure_streak',0)+1 if failure else 0
    reason='NEAR_ZERO_COLLAPSE' if collapse_streak>=2 else 'FAILED_TO_LEARN' if failure_streak>=2 else 'BUDGET_COMPLETE' if epoch==40 else None
    return dict(epoch=epoch,collapse_condition=collapse,collapse_streak=collapse_streak,
                failure_condition=failure,failure_streak=failure_streak,
                recent5_H=recent,previous5_H=older,relative_improvement=improvement,reason=reason)


def group_norm(named,values):
    groups={'encoder':lambda n:n.startswith('encoder.') and n!='encoder.latent',
            'latents':lambda n:n=='encoder.latent','decoder':lambda n:n.startswith('decoder.')}
    return {g:float(torch.stack([v.detach().double().square().sum() for (n,p),v in zip(named,values) if keep(n) and v is not None]).sum().sqrt()) for g,keep in groups.items()}


class ObservedAdamW(torch.optim.AdamW):
    def __init__(self,model,**kw):
        self.named=list(model.named_parameters());self.updates=0;self.observations=[]
        super().__init__(model.parameters(),**kw)
    def step(self,closure=None):
        before=[p.detach().clone() for n,p in self.named]
        grads=[p.grad for n,p in self.named]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('Missing or nonfinite parameter gradient')
        norms=group_norm(self.named,grads)
        result=super().step(closure)
        changes=[p.detach()-v for (n,p),v in zip(self.named,before)]
        if any(not torch.isfinite(p).all() for n,p in self.named):
            raise FloatingPointError('Nonfinite updated parameter')
        delta=group_norm(self.named,changes)
        if any(v<=0 for v in delta.values()):raise RuntimeError('Optimizer group did not update')
        self.updates+=1
        self.observations.append(dict(step=self.updates,post_clip_gradient_l2=norms,update_l2=delta))
        return result
