"""Read-only signal-path diagnostics; never changes the frozen numerical model."""
import hashlib
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from geophy_set3d.models.ray_senseiver_ss import Y_SCALE
from geophy_set3d.training.senseiver_init_probe import TrainEightGuard

MEASUREMENT_FIELDS={'input_projection_raw','input_projection'}
CHANNELS={'measurement':(0,1),'raw_geometry':(1,11),'fourier_geometry':(11,203)}
PROBE_STAGES=['encoder.preproc','encoder.final','decoder.logits','concentration']


class SignalGuard(TrainEightGuard):
    """Tightens the original guard; only four registered data files and two weights."""
    def __init__(self,root,checkpoint_paths,records,output):
        self.exact_checkpoints={str(Path(p).absolute()) for p in checkpoint_paths}
        super().__init__(root,Path(checkpoint_paths[0]).parent,records)
        first4={str((Path(root)/r['train_projection_path']).absolute()) for r in records[:4]}
        self.allowed={p:i for p,i in self.allowed.items() if p in first4}
        self.output=Path(output).absolute();self.exact_checkpoint_opens=[]
    def _audit(self,event,args):
        if self.active and event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.path.abspath(os.fsdecode(args[0]));mode,flags=args[1:3]
            reading=('r' in mode or '+' in mode) if isinstance(mode,str) else not bool(flags&os.O_WRONLY)
            if reading and path.endswith(('.pt','.pth','.ckpt')):
                if path not in self.exact_checkpoints:
                    with self.supervised_checkpoint_attempts.get_lock():self.supervised_checkpoint_attempts.value+=1
                    raise PermissionError('Only the two registered Treatment checkpoints are readable')
                self.exact_checkpoint_opens.append(path)
            if reading and path.endswith('.npz') and path!=str(self.root/'geometry/train_geometry.npz') and not Path(path).is_relative_to(self.output):
                raise PermissionError('Only frozen Pool A geometry and own diagnostic arrays are readable')
        super()._audit(event,args)
    def summary(self):
        return dict(super().summary(),policy='Only registered train0..3 Pool A files and exact Treatment epoch0/last; own diagnostic arrays; no GT/Val/PoolB/test/foreign weights',checkpoint_opens=list(self.exact_checkpoint_opens))


def model_hashes(model):
    def digest(items):
        h=hashlib.sha256()
        for name,x in items:
            a=x.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
        return h.hexdigest()
    return dict(parameters=digest(model.named_parameters()),buffers=digest(model.named_buffers()))


def valid_measurements(b):
    return b['input_valid_ray'].bool() & b['view_mask'][:,:,None,None,None]


def intervene(original,mode,source=None):
    """All nonmeasurement objects retain identity, including station/path/mask tensors."""
    result=dict(original);raw=original['input_projection_raw']
    if mode=='normal':return result
    if mode=='zero':value=torch.zeros_like(raw)
    elif mode=='wrong_case':
        if source is None:raise ValueError('Aligned source is required')
        for key in original:
            if key.startswith('input_') and key not in MEASUREMENT_FIELDS and key in source:
                if not torch.equal(original[key],source[key]):raise ValueError('Wrong-case ray identity mismatch: '+key)
        value=source['input_projection_raw']
        if value.shape!=raw.shape:raise ValueError('Wrong-case station/ray shape mismatch')
    elif mode in ('gain_plus','gain_minus'):value=raw*(1.01 if mode=='gain_plus' else .99)
    else:raise ValueError('Unregistered input mode')
    result['input_projection_raw']=value
    result['input_projection']=value/Y_SCALE
    assert all(result[k] is v for k,v in original.items() if k not in MEASUREMENT_FIELDS)
    return result


def differentiable_input(original):
    u=(original['input_projection_raw']/Y_SCALE).detach().clone().requires_grad_()
    b=dict(original,input_projection_raw=u*Y_SCALE,input_projection=u)
    assert all(b[k] is v for k,v in original.items() if k not in MEASUREMENT_FIELDS)
    return b,u


class Moments:
    """Element-weighted streaming moments and paired metrics, never chunk means."""
    def __init__(self):
        self.n=0;self.nonfinite=0;self.minimum=math.inf;self.maximum=-math.inf
        self.s=0.;self.ss=0.;self.sa=0.;self.bs=0.;self.bss=0.;self.bsa=0.;self.cross=0.
        self.ds=0.;self.dss=0.;self.dmax=0.
    def add(self,x,base=None):
        a=np.asarray(x).astype(np.float64,copy=False).reshape(-1)
        self.nonfinite+=int((~np.isfinite(a)).sum())
        if self.nonfinite:raise FloatingPointError('Nonfinite intermediate value')
        b=a if base is None else np.asarray(base).astype(np.float64,copy=False).reshape(-1)
        if a.shape!=b.shape:raise ValueError('Paired tensor shape mismatch')
        d=a-b;self.n+=a.size;self.minimum=min(self.minimum,float(a.min()));self.maximum=max(self.maximum,float(a.max()))
        self.s+=float(a.sum());self.ss+=float((a*a).sum());self.sa+=float(np.abs(a).sum())
        self.bs+=float(b.sum());self.bss+=float((b*b).sum());self.bsa+=float(np.abs(b).sum());self.cross+=float((a*b).sum())
        self.ds+=float(np.abs(d).sum());self.dss+=float((d*d).sum());self.dmax=max(self.dmax,float(np.abs(d).max()))
    def finish(self,epsilon=1e-12):
        n=self.n;mean=self.s/n;variance=max(0.,self.ss/n-mean*mean);brms=math.sqrt(self.bss/n);drms=math.sqrt(self.dss/n)
        dotden=math.sqrt(self.ss*self.bss);covden=math.sqrt(max(0.,self.ss-self.s*self.s/n)*max(0.,self.bss-self.bs*self.bs/n))
        return dict(effective_elements=n,min=self.minimum,max=self.maximum,mean=mean,std=math.sqrt(variance),rms=math.sqrt(self.ss/n),mean_abs=self.sa/n,nonfinite_count=self.nonfinite,
            paired=dict(MAE=self.ds/n,RMS=drms,max_absolute=self.dmax,normal_RMS=brms,relative_RMS=drms/brms if brms>epsilon else None,
                relative_RMS_near_zero_denominator_count=int(brms<=epsilon),normal_mean_abs=self.bsa/n,
                cosine=max(-1.,min(1.,self.cross/dotden)) if dotden>epsilon else None,cosine_near_zero_denominator_count=int(dotden<=epsilon),
                correlation=max(-1.,min(1.,(self.cross-self.s*self.bs/n)/covden)) if covden>epsilon else None,correlation_near_zero_denominator_count=int(covden<=epsilon)))


def describe(x):
    a=x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
    m=Moments();m.add(a);result=m.finish();result.pop('paired')
    result['quantiles']={str(q):float(np.quantile(a.astype(np.float64),q)) for q in [0,.01,.5,.99,1.]}
    return result


def probe_signs(shape,stage,index,seed=20261015):
    # Identical signs for logit and concentration probes, same coordinates and normalization.
    key=PROBE_STAGES.index(stage);key=2 if key==3 else key
    generator=torch.Generator().manual_seed(seed+100*key+index)
    return torch.randint(0,2,tuple(shape),generator=generator,dtype=torch.int8).mul_(2).sub_(1)


def probe_scalars(x,stage):
    a=np.asarray(x);return [float((a.astype(np.float64)*probe_signs(a.shape,stage,i).numpy()).sum()/math.sqrt(a.size)) for i in range(4)]


def linear_contributions(model,tokens,actual):
    valid=tokens
    groups={};mapped=[];layer=model.encoder.preproc
    for key,(start,end) in CHANNELS.items():
        w=layer.weight[:,start:end];part=F.linear(valid[...,start:end],w,None);mapped.append(part)
        groups[key]=dict(channels=[start,end],input=describe(valid[...,start:end]),mapped_contribution=describe(part),weight=describe(w),weight_frobenius=float(w.detach().double().norm()))
    bias=layer.bias.expand_as(actual);reconstruction=sum(mapped)+bias
    torch.testing.assert_close(reconstruction,actual,atol=2e-6,rtol=2e-5)
    return dict(groups=groups,bias=describe(layer.bias),bias_contribution=describe(bias),reconstruction_error=describe(reconstruction-actual),reconstruction_passed=True,measurement_weight_column=layer.weight[:,0].detach().cpu().tolist())


class Capture:
    """Observe actual calls, including shared blocks and every decoder chunk."""
    def __init__(self,model,batch,baseline=None):
        self.model=model;self.batch=batch;self.baseline=baseline;self.moments={};self.arrays={};self.small={};self.shapes={};self.offsets={};self.handles=[];self.calls=[];self.hashers={};self.decomposition=None
        self.block_call=1;self.cross_call=1;self.norm_call=1
    def record(self,name,tensor):
        a=tensor.detach().cpu().contiguous().numpy();offset=self.offsets.get(name,0)
        base=None if self.baseline is None else self.baseline[name][:,offset:offset+a.shape[1]]
        self.moments.setdefault(name,Moments()).add(a,base);self.offsets[name]=offset+a.shape[1]
        self.shapes.setdefault(name,[]).append(list(a.shape));self.hashers.setdefault(name,hashlib.sha256()).update(a.tobytes())
        if self.baseline is None:self.arrays.setdefault(name,[]).append(a)
        if name in PROBE_STAGES:self.small.setdefault(name,[]).append(a)
    def __enter__(self):
        def hook(module,fn):self.handles.append(module.register_forward_hook(fn))
        def pre(module,fn):self.handles.append(module.register_forward_pre_hook(fn))
        def encoder_input(m,args):
            tokens,mask=args;assert len(tokens)==1 and int((~mask).sum())==int(valid_measurements(self.batch).sum())
            valid=~mask;self.tokens=tokens[valid].reshape(1,-1,203)
            raw=self.batch['input_projection_raw'][valid_measurements(self.batch)].reshape(1,-1,1)
            normalized=self.batch['input_projection'][valid_measurements(self.batch)].reshape(1,-1,1)
            torch.testing.assert_close(self.tokens[...,:1],raw/Y_SCALE,atol=0,rtol=0)
            if self.baseline is not None:
                assert np.array_equal(self.tokens[...,1:].detach().cpu().numpy(),self.baseline['ray_token'][...,1:]),'Geometry token changed'
            self.record('measurement.raw',raw);self.record('measurement.normalized_field',normalized);self.record('ray_token',self.tokens)
        pre(self.model.encoder,encoder_input)
        def enc_preproc(m,args,y):
            selected=y.reshape(1,-1,64) # batch1 has no padding; encoder_input checked real-ray count.
            self.record('encoder.preproc',selected)
            if self.baseline is None:self.decomposition=linear_contributions(self.model,self.tokens,selected)
        hook(self.model.encoder.preproc,enc_preproc)
        def first_cross(m,args,y):self.calls.append('after_cross_1');self.record('encoder.after_cross_1',y)
        hook(self.model.encoder.layer_1[0][0],first_cross)
        def shared_cross(m,args,y):
            self.cross_call+=1;self.calls.append('after_cross_'+str(self.cross_call));self.record('encoder.after_cross_'+str(self.cross_call),y)
        hook(self.model.encoder.layer_n[0][0],shared_cross)
        def kvfirst(m,args,y):self.record('encoder.kv_norm_1',y)
        hook(self.model.encoder.layer_1[0][0].module.kv_norm,kvfirst)
        def kvshared(m,args,y):
            self.norm_call+=1;self.record('encoder.kv_norm_'+str(self.norm_call),y)
        hook(self.model.encoder.layer_n[0][0].module.kv_norm,kvshared)
        def first_block(m,args,y):self.calls.append('block_1');self.record('encoder.block_1',y)
        hook(self.model.encoder.layer_1,first_block)
        def shared_block(m,args,y):
            self.block_call+=1;self.calls.append('block_'+str(self.block_call));self.record('encoder.block_'+str(self.block_call),y)
        hook(self.model.encoder.layer_n,shared_block)
        hook(self.model.encoder,lambda m,a,y:self.record('encoder.final',y))
        def query_input(m,args):
            self.record('decoder.query_coordinates_fixed',args[0][...,:96]);self.record('decoder.learned_query_fixed',args[0][...,96:])
        pre(self.model.decoder.preproc,query_input)
        hook(self.model.decoder.preproc,lambda m,a,y:self.record('decoder.query_preproc_fixed',y))
        hook(self.model.decoder.cross_attention[0].module,lambda m,a,y:self.record('decoder.cross_attention_raw',y))
        hook(self.model.decoder.cross_attention,lambda m,a,y:self.record('decoder.cross_attention',y))
        hook(self.model.decoder.postproc,lambda m,a,y:self.record('decoder.logits',y))
        return self
    def finish(self,output):
        assert output.shape==(1,1,64,64,64)
        self.record('concentration',output.flatten(2).transpose(1,2))
        assert self.calls==['after_cross_1','block_1','after_cross_2','block_2','after_cross_3','block_3'],self.calls
        stages={}
        for name,m in self.moments.items():
            shape=self.shapes[name][0].copy();shape[1]=self.offsets[name]
            if name.startswith('decoder.') or name=='concentration':assert shape[1]==262144
            stages[name]=dict(**m.finish(),shape=shape,call_shapes=self.shapes[name],calls=len(self.shapes[name]),tensor_sha256=self.hashers[name].hexdigest())
        arrays={k:np.concatenate(v,axis=1) for k,v in self.arrays.items()}
        small={k:np.concatenate(v,axis=1) for k,v in self.small.items()}
        indices=torch.linspace(0,262143,259).long().numpy()
        scalars={k:probe_scalars(v if k not in ['decoder.logits','concentration'] else v[:,indices],k) for k,v in small.items()}
        return stages,arrays,small,scalars
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        self.handles=[]
