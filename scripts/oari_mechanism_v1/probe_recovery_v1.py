"""Numerical-check recovery: direct SciPy KL sum avoids near-zero sqrt NaNs."""
from runtime import *
import math
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch


def digest_tensor(t):
    a=t.detach().cpu().contiguous().numpy();return hashlib.sha256(a.tobytes()).hexdigest()


def input_hash(batch):
    h=hashlib.sha256()
    for k,v in sorted(batch.items()):
        h.update(k.encode())
        if isinstance(v,torch.Tensor):h.update(str((v.shape,v.dtype)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        else:h.update(repr(v).encode())
    return h.hexdigest()


class Capture:
    def __init__(self,model):self.model=model;self.data={};self.calls=0;self.qkv_calls=0
    def before(self,module,args):
        features,geometry,visibility,view_mask,relations,observability,*_=args
        self.data.update(mask=(visibility&view_mask[:,:,None,None,None]).permute(0,2,3,4,1).reshape(-1,features.shape[1]).cpu().numpy(),
            relations=relations.cpu().numpy(),observability=observability.cpu().numpy(),
            features_sha256=digest_tensor(features),geometry_sha256=digest_tensor(geometry),visibility_sha256=digest_tensor(visibility))
    def qkv(self,module,args,out):self.qkv_calls+=1;self.data['qkv']=out.cpu().numpy()
    def after(self,module,args,out):
        self.calls+=1;context,weights,bias=out
        b,v,c,x,y,z=context.shape
        self.data.update(context=context.permute(0,3,4,5,1,2).reshape(-1,v,c).cpu().numpy(),
            weights=weights.permute(0,4,5,6,1,2,3).reshape(-1,module.num_heads,v,v).cpu().numpy(),bias=bias.cpu().numpy())
    def __enter__(self):
        self.hooks=[self.model.interaction.register_forward_pre_hook(self.before),self.model.interaction.qkv.register_forward_hook(self.qkv),self.model.interaction.register_forward_hook(self.after)];return self
    def __exit__(self,*args):
        for h in self.hooks:h.remove()
        if args[0] is None:assert self.calls==self.qkv_calls==1


def array_stats(x):
    x=np.asarray(x,dtype=np.float64).ravel();assert x.size and np.isfinite(x).all()
    q=np.quantile(x,[.5,.9,.95,.99]);a=np.abs(x);qa=np.quantile(a,[.5,.9,.95,.99])
    return dict(n=int(x.size),mean=float(x.mean()),std=float(x.std()),rms=float(np.sqrt(np.mean(x*x))),mean_abs=float(a.mean()),
        **{f'p{p}':float(v) for p,v in zip((50,90,95,99),q)},**{f'abs_p{p}':float(v) for p,v in zip((50,90,95,99),qa)})


def corr(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    xx=x-x.mean();yy=y-y.mean();den=np.sqrt(np.dot(xx,xx)*np.dot(yy,yy))
    return None if den<=1e-24 else float(np.dot(xx,yy)/den)


def softmax_content(data):
    n,v,_=data['qkv'].shape;heads=data['weights'].shape[1];dim=32//heads
    qkv=data['qkv'].reshape(n,v,3,heads,dim).transpose(2,0,3,1,4)
    q,k=qkv[0].astype(np.float64),qkv[1].astype(np.float64)
    content=(q@k.swapaxes(-1,-2))/math.sqrt(dim)
    mask=data['mask'];pair=mask[:,:,None]&mask[:,None,:]
    def softmax(x):
        xx=np.where(mask[:,None,None,:],x,-np.inf);xx=np.where(mask.any(1)[:,None,None,None],xx,0)
        xx=np.exp(xx-xx.max(-1,keepdims=True));xx=xx/xx.sum(-1,keepdims=True)
        return np.where(pair[:,None],xx,0)
    no_bias=softmax(content);actual=softmax(content+data['bias'])
    np.testing.assert_allclose(actual,data['weights'],atol=2e-6,rtol=2e-5)
    return content,no_bias


def distribution_changes(a,b):
    # Distributions over key view; invalid query rows are excluded by caller.
    raw_a=a.astype(np.float64);raw_b=b.astype(np.float64)
    a=raw_a/np.maximum(raw_a.sum(-1,keepdims=True),1e-300);b=raw_b/np.maximum(raw_b.sum(-1,keepdims=True),1e-300);m=.5*(a+b)
    def xlogratio(x,y):return np.where(x>0,x*np.log(np.maximum(x,1e-300)/np.maximum(y,1e-300)),0)
    def entropy(x):return -np.sum(xlogratio(x,np.ones_like(x)),axis=-1)
    return dict(l1=np.abs(raw_b-raw_a).sum(-1),l2=np.sqrt(((raw_b-raw_a)**2).sum(-1)),
        js=np.maximum(.5*np.sum(xlogratio(a,m)+xlogratio(b,m),axis=-1),0),entropy_delta=entropy(b)-entropy(a))



def independent_distribution_check(a,b,mask):
    from scipy.special import rel_entr
    from scipy.stats import entropy
    distances=distribution_changes(a,b)
    for head in range(a.shape[1]):
        aa=a[:,head][mask].astype(np.float64);bb=b[:,head][mask].astype(np.float64)
        ids=np.linspace(0,len(aa)-1,min(1024,len(aa)),dtype=int)
        an=aa[ids]/aa[ids].sum(-1,keepdims=True);bn=bb[ids]/bb[ids].sum(-1,keepdims=True);mid=.5*(an+bn)
        expected=.5*(rel_entr(an,mid)+rel_entr(bn,mid)).sum(-1)
        assert np.min(expected)>=-1e-12
        expected=np.maximum(expected,0)
        np.testing.assert_allclose(distances['js'][:,head][mask][ids],expected,atol=1e-14,rtol=1e-7)
        np.testing.assert_allclose(distances['entropy_delta'][:,head][mask][ids],entropy(bb[ids],axis=-1)-entropy(aa[ids],axis=-1),atol=1e-14,rtol=1e-7)

def summarize_A(plan,data):
    content,no_bias=softmax_content(data);independent_distribution_check(no_bias,data['weights'],data['mask']);mask=data['mask'];pair=mask[:,:,None]&mask[:,None,:]
    n,heads,v,_=data['bias'].shape;scale=[];case=[]
    change=distribution_changes(no_bias,data['weights'])
    meta={k:plan[k] for k in ('sample_index','case_id','condition','draw','views')}
    for h in range(heads):
        bs=array_stats(data['bias'][:,h][pair]);cs=array_stats(content[:,h][pair])
        # Softmax-invariant offsets can make raw bias amplitude misleading.
        count=mask.sum(1).clip(1)
        centered=data['bias'][:,h]-(np.where(pair,data['bias'][:,h],0).sum(-1)/count[:,None])[:,:,None]
        row=dict(meta,head=h,scope='all_valid_pairs',**{f'bias_{k}':v for k,v in bs.items()},**{f'content_{k}':v for k,v in cs.items()},
            rms_ratio=bs['rms']/cs['rms'] if cs['rms']>0 else None,std_ratio=bs['std']/cs['std'] if cs['std']>0 else None,
            key_centered_bias_rms=float(np.sqrt(np.mean(centered[pair]**2))),
            no_bias_to_bias_JS_mean=float(change['js'][:,h][mask].mean()),no_bias_to_bias_L1_mean=float(change['l1'][:,h][mask].mean()),
            entropy_delta_mean=float(change['entropy_delta'][:,h][mask].mean()),valid_query_rows=int(mask.sum()),valid_voxels=int(mask.any(1).sum()))
        scale.append(row);case.append(row)
        for qi in range(v):
            for ki in range(v):
                valid=pair[:,qi,ki]
                if not valid.any():continue
                b=data['bias'][valid,h,qi,ki];c=content[valid,h,qi,ki];rel=data['relations'][valid,qi,ki]
                bb=array_stats(b);cc=array_stats(c)
                scale.append(dict(meta,head=h,scope='ordered_valid_pair',query_view=qi,key_view=ki,self_pair=qi==ki,
                    **{f'bias_{k}':v for k,v in bb.items()},**{f'content_{k}':v for k,v in cc.items()},
                    rms_ratio=bb['rms']/cc['rms'] if cc['rms']>0 else None,std_ratio=bb['std']/cc['std'] if cc['std']>0 else None,
                    relation_std=json.dumps(np.std(rel.astype(np.float64),axis=0).tolist()),
                    relation_bias_pearson=json.dumps([corr(rel[:,i],b) for i in range(3)]),
                    relation_mean=json.dumps(np.mean(rel.astype(np.float64),axis=0).tolist())))
    return scale,case


def summarize_C(plan,baseline,changed,mode):
    softmax_content(changed)
    independent_distribution_check(baseline['weights'],changed['weights'],baseline['mask'])
    assert np.array_equal(baseline['mask'],changed['mask']) and np.array_equal(baseline['qkv'],changed['qkv'])
    for k in ('features_sha256','geometry_sha256','visibility_sha256'):assert baseline[k]==changed[k]
    np.testing.assert_array_equal(baseline['observability'],changed['observability'])
    mask=baseline['mask'];pair=mask[:,:,None]&mask[:,None,:];heads=baseline['weights'].shape[1];v=mask.shape[1]
    diff=distribution_changes(baseline['weights'],changed['weights']);rows=[]
    feature=np.abs(changed['context'].astype(np.float64)-baseline['context'].astype(np.float64)).mean(-1)
    feature=np.where(mask,feature,0).sum(-1)/mask.sum(1).clip(1)
    for h in range(heads):
        aa=baseline['weights'][:,h];bb=changed['weights'][:,h]
        d=changed['bias'][:,h].astype(np.float64)-baseline['bias'][:,h].astype(np.float64)
        pervoxel=np.where(mask,diff['l1'][:,h],0).sum(1)/mask.sum(1).clip(1);vv=mask.any(1)
        key_delta=[];key_before=[];key_after=[]
        for k in range(v):
            effective=pair[:,:,k]
            key_delta.append(float((bb[:,:,k]-aa[:,:,k])[effective].mean()) if effective.any() else None)
            key_before.append(float(aa[:,:,k][effective].mean()) if effective.any() else None)
            key_after.append(float(bb[:,:,k][effective].mean()) if effective.any() else None)
        row={k:plan[k] for k in ('sample_index','case_id','condition','draw','views')}
        row.update(experiment=mode,head=h,valid_query_rows=int(mask.sum()),valid_voxels=int(vv.sum()),
            **{f'attention_{k}_mean':float(a[:,h][mask].mean()) for k,a in diff.items()},
            **{f'attention_{k}_p95':float(np.quantile(a[:,h][mask],.95)) for k,a in diff.items()},
            bias_change_mean_abs=float(np.abs(d[pair]).mean()),bias_change_rms=float(np.sqrt(np.mean(d[pair]**2))),
            key_weight_before=json.dumps(key_before),key_weight_after=json.dumps(key_after),key_weight_delta=json.dumps(key_delta),
            context_mean_abs_change=float(feature[vv].mean()),
            context_relative_change=float(np.abs(changed['context'][mask].astype(np.float64)-baseline['context'][mask].astype(np.float64)).sum()/max(np.abs(baseline['context'][mask].astype(np.float64)).sum(),1e-12)),
            voxel_attention_context_pearson=corr(pervoxel[vv],feature[vv]),content_QKV_unchanged=True)
        rows.append(row)
    return rows


def run():
    backend();recovery=verified(BASE/'numerical_recovery_v1.json');verify_hashes(recovery['sources']);p=verified(BASE/'protocol.json');verify_hashes(p['sources']);assert verified(BASE/'attenuation_summary.json')['completed']
    m=p['model'];assert sha(m['checkpoint'])==HIST.EXPECTED
    references={(r['sample_index'],r['mode']):r for r in verified(BASE/'sampled_historical_predictions.json')}
    scales=[];case_bias=[];changes=[];manifest=[];checks=[]
    with Guard() as guard,torch.inference_mode():
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
        model=HIST.model_for(m['config'],'uniform','cuda:0').eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True);del ck
        initial=HIST.state_hash(model);assert initial==verified(HIST.BASE/'experiment_registry.json')['main_state_sha256']
        layouts=verified(HIST.BASE/'layouts.json')['test'];current_li=None
        for plan in p['sample_plans']:
            if current_li!=plan['layout_index']:ds=dataset('test',layouts[plan['layout_index']]);current_li=plan['layout_index']
            sample=ds[plan['case_index']];assert sample['case_id']==plan['case_id']
            batch=move_batch(projection_pool_collate([sample]),'cuda:0');batch={k:v for k,v in batch.items() if not k.startswith('hold_')}
            ih=input_hash(batch);collected={}
            for mode in ('A','P1','P2'):
                torch.cuda.reset_peak_memory_stats()
                with HIST.Intervention(model,mode,plan['permutation'] if mode=='P1' else None) as intervene,Capture(model) as capture:
                    st=model(batch,return_stages=True)
                assert all(not x.training for x in model.modules()) and all(x.grad is None and not x.requires_grad for x in model.parameters())
                assert input_hash(batch)==ih and HIST.state_hash(model)==initial
                record=references[plan['sample_index'],mode];assert sha(record['path'])==record['sha256']
                comparisons={};fields={k:st[k][0,0].cpu().numpy() for k in ('C0','C1')}
                with np.load(record['path'],allow_pickle=False) as old:
                    for k,a in fields.items():
                        assert a.shape==(64,)*3 and a.dtype==np.float32 and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
                        absmax=float(np.abs(a.astype(np.float64)-old[k]).max());relative=float(np.abs(a.astype(np.float64)-old[k]).mean()/(np.abs(old[k]).mean()+1e-6))
                        assert absmax<=1e-5 and relative<=1e-4,(plan,mode,k,absmax,relative)
                        comparisons[k]=dict(max_absolute=absmax,relative_mean_absolute=relative)
                path=BASE/'sampled_predictions_recovery_v1'/f's{plan["sample_index"]:02d}_{mode}.npz';path.parent.mkdir(exist_ok=True);assert not path.exists();np.savez_compressed(path,**fields)
                manifest.append(dict(plan,experiment=mode,checkpoint_sha256=HIST.EXPECTED,input_sha256=ih,prediction=str(path.relative_to(BASE)),sha256=sha(path),
                    output_tensor_sha256={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in fields.items()},historical_prediction_path=record['path'],historical_sha256=record['sha256'],
                    historical_comparison=comparisons,actual_intervention=intervene.audit(),capture_calls=capture.calls,qkv_calls=capture.qkv_calls,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                jsonl(BASE/'prediction_manifest.jsonl',manifest)
                collected[mode]=capture.data;intervene.capture.clear();del st,fields,capture,intervene
            ss,cc=summarize_A(plan,collected['A']);scales.extend(ss);case_bias.extend(cc)
            for mode in ('P1','P2'):changes.extend(summarize_C(plan,collected['A'],collected[mode],mode))
            checks.append(dict(sample_index=plan['sample_index'],case_id=plan['case_id'],condition=plan['condition'],all_modes_match_history=True,geometry_features_masks_QKV_unchanged=True,state_unchanged=True))
            del collected,batch,sample;torch.cuda.empty_cache()
            print(f'A/C sampled input {plan["sample_index"]+1}/15 complete; cumulative forwards={len(manifest)}',flush=True)
        assert not guard.denied
    csv_write(BASE/'bias_scale_statistics.csv',scales);csv_write(BASE/'per_case_bias.csv',case_bias);csv_write(BASE/'attention_change_statistics.csv',changes)
    assert len(manifest)==45 and len(case_bias)==60 and len(changes)==120
    freeze(BASE/'probe_complete.json',dict(at=now(),completed=True,forward_calls=45,interaction_calls=45,checkpoint_sha256=HIST.EXPECTED,
        state_before=initial,state_after=HIST.state_hash(model),model_eval=True,all_requires_grad_false=True,all_grad_none=True,
        checks=checks,audit=guard.summary(),GT64_read=False,GT128_read=False,training_runs=0,backward_calls=0,weight_updates=0,
        manifest_sha256=sha(BASE/'prediction_manifest.jsonl'),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        scipy_JS_entropy_check='direct SciPy rel_entr JS sum, bounded roundoff clipped to0 before comparison; up to1024 deterministic valid query rows per head/input/condition, independent SciPy formulas',
        attention_reconstruction='independent float64 CPU QK/softmax from read-only actual QKV hook matched actual weights, atol2e-6/rtol2e-5',
        summary_files={str(BASE/n):sha(BASE/n) for n in ('bias_scale_statistics.csv','per_case_bias.csv','attention_change_statistics.csv')}))
    print('A/C COMPLETE:45 frozen forwards, all historical predictions and model integrity verified; no GT',flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        import traceback
        write(BASE/'probe_recovery_failure.json',dict(at=now(),traceback=traceback.format_exc()));raise
