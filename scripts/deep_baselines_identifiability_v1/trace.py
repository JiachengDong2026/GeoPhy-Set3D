"""Read-only representation taps; all original model methods remain unchanged on disk."""
from diagnostics import *
from geophy_set3d.models.ray_senseiver_ss import ray_tokens

class Trace:
    def __init__(self,net):
        self.net=net;self.values={};self.handles=[];self.restores=[];self.logits=[]
    def save(self,name,x):
        if name not in self.values:self.values[name]=x
        return x
    def wrap(self,obj,name,label):
        original=getattr(obj,name)
        def tapped(*args,**kwargs):
            out=original(*args,**kwargs)
            self.save(label,out)
            return out
        setattr(obj,name,tapped);self.restores.append((obj,name,original))
    def hook(self,module,name):
        self.handles.append(module.register_forward_hook(lambda m,a,o:self.save(name,o)))
    def __enter__(self):
        n=self.net
        if hasattr(n,'phi'):
            self.hook(n.phi,'phi');head=n.decoder[-1]
        else:
            self.hook(n.measurement_encoder,'measurement_encoder')
            self.hook(n.geometry_encoder,'geometry_encoder')
            for i,block in enumerate(n.blocks):
                self.wrap(block.cross,'apply_statistics',f'operator_{i}_first_chunk')
                self.wrap(block,'after_self',f'query_{i}_first_chunk')
            head=n.head[-1]
        def readout(m,args,out):
            self.save('decoder_query_first_chunk',args[0]);self.save('logits_first_chunk',out);self.logits.append(out)
        self.handles.append(head.register_forward_hook(readout))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        for obj,name,original in reversed(self.restores):setattr(obj,name,original)
    def forward(self,inp):
        n=self.net;tokens,excluded,counts=ray_tokens(inp);self.save('tokens',tokens)
        if hasattr(n,'phi'):
            encoded=n.phi(tokens).masked_fill(excluded.unsqueeze(-1),0.)
            pooled=encoded.sum(1)/counts.clamp_min(1).unsqueeze(-1)
            latent=torch.where(counts[:,None]>0,pooled,torch.zeros_like(pooled))
            self.save('masked_mean',latent);self.save('decoder_latent',latent)
            value=n.decode_queries(latent,n.query_coordinates_m)
        else:value=n.forward_tokens(tokens,excluded,counts)
        return value.transpose(1,2).reshape(len(tokens),1,64,64,64)

def finite_difference(net,inp):
    original=inp['input_projection_raw'];raw=original.detach().clone().requires_grad_(True)
    current=dict(inp,input_projection_raw=raw)
    with Trace(net) as trace:out=trace.forward(current)
    scalar=out.double().mean()/design()['initial_mass_fraction']
    logit=torch.cat([x.reshape(-1) for x in trace.logits]).double().mean()
    derivatives={}
    for name,value in [('output_mean_div_fixed_scale',scalar),('mean_logit',logit)]:
        g=torch.autograd.grad(value,raw,retain_graph=True)[0]
        derivatives[name]=float((g.double()*original.double()).sum())
    del out,scalar,logit,trace,current,raw
    result=[]
    for eps in (.01,.001):
        measured={}
        with torch.no_grad():
            for sign in (-1,1):
                test=intervene(inp,'normal');test['input_projection_raw']=original*(1+sign*eps)
                test['input_projection']=test['input_projection_raw']/Y_SCALE
                with Trace(net) as tr:out=tr.forward(test)
                measured[sign]=dict(output_mean_div_fixed_scale=float(out.double().mean()/design()['initial_mass_fraction']),
                    mean_logit=float(torch.cat([v.reshape(-1) for v in tr.logits]).double().mean()))
        for key,ad in derivatives.items():
            fd=(measured[1][key]-measured[-1][key])/(2*eps);error=abs(ad-fd)
            floor=design()['gate']['finite_difference_absolute_tolerance']
            observable=max(abs(ad),abs(fd))>floor
            result.append(dict(function=key,epsilon=eps,AD=ad,FD=fd,abs_error=error,
                relative_error=error/max(abs(ad),abs(fd),floor),observable=observable,
                within_tolerance=error<=floor+.05*max(abs(ad),abs(fd)),
                interpretation='resolved' if observable else 'precision_limited'))
    return result

def layer_gradients(net,ab,version):
    """H is not accepted; no optimizer exists in this routine."""
    if set(ab)!={'A','B'}:raise ValueError('No H gradient capability')
    net.train();net.zero_grad(set_to_none=True)
    raw=ab['A']['input_projection_raw'].detach().clone().requires_grad_(True)
    aa=dict(ab['A'],input_projection_raw=raw)
    with Trace(net) as tr:ca=tr.forward(aa)
    cb=net(ab['B'])
    parts=dict(observed=.5*(measurement_discrepancy(project_input(ca,ab['A']),ab['A'])+
                               measurement_discrepancy(project_input(cb,ab['B']),ab['B'])),
               cross=.5*(measurement_discrepancy(project_input(ca,ab['B']),ab['B'])+
                           measurement_discrepancy(project_input(cb,ab['A']),ab['A'])),
               consistency=(consistency_discrepancy if version=='v2' else homogeneous_consistency)(ca,cb))
    targets=dict(tr.values);targets['raw_measurement']=raw
    head=net.decoder[-1] if hasattr(net,'phi') else net.head[-1]
    targets['readout_weight']=head.weight;targets['readout_bias']=head.bias
    keys=list(targets);result={}
    for name,loss in parts.items():
        grads=torch.autograd.grad(loss,list(targets.values()),retain_graph=True,allow_unused=True)
        values={k:gradient_stats(g) for k,g in zip(keys,grads)}
        tokens=grads[keys.index('tokens')]
        values['token_measurement_channel']=gradient_stats(None if tokens is None else tokens[...,0])
        values['token_geometry_channels']=gradient_stats(None if tokens is None else tokens[...,1:])
        result[name]=dict(loss=float(loss.detach()),layers=values)
    result['representation_shapes']={k:list(v.shape) for k,v in targets.items()}
    result['query_scope']='first query chunk only for operator/query/readout taps; complete token/encoder/latent tensors'
    result['logits']=field_stats(torch.cat([x.detach().reshape(-1) for x in tr.logits]))
    result['logits']['fraction_below_minus12']=float(torch.cat([x.detach().reshape(-1) for x in tr.logits]).lt(-12).float().mean())
    result['sigmoid_derivative_mean']=float((ca.detach()*(1-ca.detach())).mean())
    return result
