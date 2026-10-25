from latent_context import *
import argparse,time
from geophy_set3d.models.ray_senseiver_ss import ray_tokens,Y_SCALE

class Capture:
    """Observational hooks and wrappers: always return the exact original object."""
    def __init__(self,net,arm):self.net=net;self.arm=arm;self.data={};self.handles=[];self.restore=[]
    def add(self,key,x):self.data.setdefault(key,[]).append(x.detach().cpu().numpy().copy())
    def hook(self,module,key,pre=False):
        def h(m,args,out):
            if key=='prehead':self.add(key,args[0])
            else:self.add(key,out)
        self.handles.append(module.register_forward_hook(h))
    def wrap(self,obj,name,fn):
        original=getattr(obj,name);self.restore.append((obj,name,original));setattr(obj,name,fn(original))
    def __enter__(self):
        n=self.net
        if self.arm=='deepsets':
            self.hook(n.phi,'measurement_encoder');self.hook(n.decoder[0],'decoder_first_linear')
            self.hook(n.decoder[-1],'prehead');self.hook(n.decoder[-1],'logit')
            def pooled(m,args,out):
                if 'pooled_latent' not in self.data:self.add('pooled_latent',args[0][:,0,:128])
            self.handles.append(n.decoder[0].register_forward_hook(pooled))
        else:
            self.hook(n.measurement_encoder,'measurement_encoder')
            self.hook(n.head[-1],'prehead');self.hook(n.head[-1],'logit')
            for j,block in enumerate(n.blocks):
                def before(orig,j=j):
                    def f(*args,**kw):
                        out=orig(*args,**kw);self.add(f'block{j}_cross_state',out);return out
                    return f
                def after(orig,j=j):
                    def f(*args,**kw):
                        key=f'block{j}_global_operator'
                        if key not in self.data:self.add(key,torch.cat([args[2].flatten(),args[3].flatten()]))
                        out=orig(*args,**kw);self.add(f'block{j}_final_state',out);return out
                    return f
                def stats(orig,j=j):
                    def f(*args,**kw):
                        out=orig(*args,**kw)
                        if (args[1] if len(args)>1 else kw.get('branch',0))==0:self.add(f'block{j}_measurement_operator',torch.cat([out[0].flatten(),out[1].flatten()]))
                        return out
                    return f
                self.wrap(block,'before_self',before);self.wrap(block,'after_self',after);self.wrap(block.cross,'statistics',stats)
        return self
    def arrays(self):
        return {k:(v[0] if len(v)==1 else np.concatenate(v,axis=1)) for k,v in self.data.items()}
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        for obj,key,orig in self.restore:setattr(obj,key,orig)

def savearrays(path,arrays):
    path.mkdir(parents=True,exist_ok=False)
    for k,x in arrays.items():assert np.isfinite(x).all();np.save(path/f'{k}.npy',x,allow_pickle=False)

def main(arm):
    reg=verify_seal();old.backend();start=time.monotonic();target=ROOT/f'extraction_{arm}.json';assert not target.exists()
    checks=[];alias=[];manifest={}
    with Access('extract_'+arm) as access:
        ss=samples();net=netload(arm,'cuda:0');state=old.state_hash(net.state_dict());torch.cuda.reset_peak_memory_stats()
        # Whole original output reproduction phase precedes all internal extraction.
        with torch.inference_mode():
            for i,s in enumerate(ss):
                with np.load(OLD/f'runs/{arm}/raw/update200_case{i}.npz',allow_pickle=False) as z:
                    for b in ('A','B'):
                        inp=old.batch([s],b,'cuda:0');pred=net(inp).cpu().numpy();assert np.array_equal(pred,z['field_'+b]),(arm,i,b,'archive mismatch')
                        pred0=net(zero(inp)).cpu().numpy();assert np.array_equal(pred0,z['field_'+b+'_zero'])
                        p=ROOT/f'original/{arm}/{b}';p.mkdir(parents=True,exist_ok=True);np.save(p/f'{i}.npy',pred)
                        checks.append(dict(case=i,branch=b,archived_normal_bitwise=True,archived_zero_bitwise=True))
                print('REPRODUCED',arm,i,flush=True)
            for b in ('A','B'):
                baseline0=None
                for kind in ('normal','zero'):
                    for i,s in enumerate(ss):
                        inp=old.batch([s],b,'cuda:0');inp=zero(inp) if kind=='zero' else inp
                        if kind=='zero':
                            h=old.value_hash(inp)
                            if baseline0 is None:baseline0=h
                            assert h==baseline0
                            if i>0:alias.append(dict(case=i,branch=b,kind=kind,features_from=0,input_sha256=h));continue
                        off=net(inp)
                        with Capture(net,arm) as cap:on=net(inp)
                        assert torch.equal(off,on),(arm,i,b,kind,'hook mismatch')
                        arrays=cap.arrays();tokens,excluded,counts=ray_tokens(inp)
                        arrays.update(raw_measurement=(tokens[...,:1]*Y_SCALE).cpu().numpy(),valid_mask=(~excluded).cpu().numpy(),concentration=on.cpu().numpy())
                        p=ROOT/f'features/{arm}/{b}/{kind}/{i}';savearrays(p,arrays)
                        for k in arrays:manifest[str(p/f'{k}.npy')]=sha(p/f'{k}.npy')
                        checks.append(dict(case=i,branch=b,kind=kind,hook_bitwise=True,stages=list(arrays)))
                        print('CAPTURED',arm,b,kind,i,flush=True)
                # All registered swaps: full donor equivalence, then alias valid at every layer.
                for i,j in old.design()['pairs']:
                    for recipient,donor in ((i,j),(j,i)):
                        x=old.batch([ss[recipient]],b,'cuda:0');d=old.batch([ss[donor]],b,'cuda:0');sw=swap(x,d)
                        alias.append(dict(case=recipient,donor=donor,branch=b,kind='swap',features_from=donor,input_sha256=old.value_hash(sw)))
                # Explicit representative swap with all hooks, supplementing complete input proof.
                sw=swap(old.batch([ss[0]],b,'cuda:0'),old.batch([ss[1]],b,'cuda:0'))
                with Capture(net,arm) as cap:out=net(sw)
                for k,x in cap.arrays().items():assert np.array_equal(x,np.load(ROOT/f'features/{arm}/{b}/normal/1/{k}.npy')),(b,k)
                assert np.array_equal(out.cpu().numpy(),np.load(ROOT/f'original/{arm}/{b}/1.npy'))
        assert old.state_hash(net.state_dict())==state
        assert all(not p.requires_grad and p.grad is None for p in net.parameters())
        info=dict(at=now(),arm=arm,checkpoint_sha256=sha(checkpoint(arm)),parameter_count=sum(p.numel() for p in net.parameters()),state_hash=state,
            neural_updates=0,checks=checks,aliases=alias,arrays_sha256=manifest,representative_swap_all_layers_exact=True,elapsed_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        freeze(target,info)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=list(EXPECTED));main(p.parse_args().arm)
