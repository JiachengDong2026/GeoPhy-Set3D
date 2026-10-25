"""FP64 correction reviewer; original failed reviewer retained. Independent review: does not import execution ridge, metrics or decision functions."""
from latent_context import *
import itertools

def loadcsv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def close(a,b):
    if a is None:assert b in ('',None)
    else:assert np.isclose(float(a),float(b),atol=2e-9,rtol=2e-6),(a,b)

def independent_metrics(a,t):
    a=a.reshape(64,64,64).astype('float64');t=t.reshape(a.shape);delta=abs(a-t)
    positive=np.clip(a,0,None);axes=np.linspace(0,.6,64)
    def center(x):
        if x.sum()<=0:return None
        return np.array([(x*np.reshape(axes,[64 if j==k else 1 for j in range(3)])).sum()/x.sum() for k in range(3)])
    ca,ct=center(positive),center(t);den=np.std(a)*np.std(t)
    return dict(relative_l1=delta.mean()/np.abs(t).mean(),mae=delta.mean(),mean=a.mean(),std=a.std(),max=a.max(),min=a.min(),nonzero_fraction=np.count_nonzero(abs(a)>1e-10)/a.size,positive_structure_fraction=np.count_nonzero(a>1e-6)/a.size,negative_fraction=np.count_nonzero(a<0)/a.size,above_one_fraction=np.count_nonzero(a>1)/a.size,centroid_error_m=np.linalg.norm(ca-ct) if ca is not None else None,mass_relative_error=abs(a.mean()-t.mean())/abs(t.mean()),integral=a.sum()*(.6/63)**3,spatial_correlation=np.mean((a-a.mean())*(t-t.mean()))/den if den>0 else None)

def independent_moments(x,y):
    d=x.shape[-1];flat=x.reshape(-1,d);yt=y.ravel();n=0;mean=np.zeros(d);gram=np.zeros((d,d));target=np.zeros(d);ym=0.
    for at in range(0,len(flat),16384):
        q=np.array(flat[at:at+16384],dtype='float64');v=yt[at:at+16384];n+=len(q);mean+=q.sum(0);gram+=q.T@q;target+=q.T@v;ym+=v.sum()
    return n,mean,gram,target,ym

def independent_fit(moments,lam):
    n=sum(t[0] for t in moments);mean=sum(t[1] for t in moments)/n;gram=sum(t[2] for t in moments);target=sum(t[3] for t in moments);ym=sum(t[4] for t in moments)/n;d=len(mean)
    gram=gram/n-mean[:,None]*mean[None,:];gram=(gram+gram.T)*.5
    scale=np.sqrt(np.maximum(gram.diagonal(),0)).clip(1e-6);matrix=gram/scale[:,None]/scale[None,:]+np.identity(d)*lam
    rhs=(target/n-mean*ym)/scale;beta=np.linalg.solve(matrix,rhs);w=beta/scale
    return w,ym-mean@w

def main():
    reg=verify_seal();ev=read(ROOT/'evaluation_fp64.json');seal=read(ROOT/'probe_seal.json');counts=dict(metrics=0,pairs=0,cv=0,representations=0,layer_stats=0)
    for a in EXPECTED:
        ext=read(ROOT/f'extraction_{a}.json')
        for p,h in ext['arrays_sha256'].items():assert sha(p)==h
        assert ext['representative_swap_all_layers_exact'] and all(c.get('hook_bitwise',True) and c.get('archived_normal_bitwise',True) and c.get('archived_zero_bitwise',True) for c in ext['checks'])
        assert sum(r['kind']=='swap' for r in ext['aliases'])==112
    supplement=read(ROOT/'exact_raw_supplement.json')
    for p,h in supplement['arrays_sha256'].items():assert sha(p)==h
    for p,h in seal['coefficients_sha256'].items():assert sha(p)==h
    for p,h in ev['predictions_sha256'].items():assert sha(p)==h
    assert reg['seen']==list(range(6)) and reg['unseen']==[6,7]
    fitaccess=read(ROOT/'access_fit.json');assert not any(f'truth_{i}.npy' in p for p in fitaccess['reads'] for i in (6,7))
    assert not any('unseen_AB' in p for p in fitaccess['reads'])
    rows=loadcsv(ROOT/'case_metrics.csv');pairrows=loadcsv(ROOT/'pair_metrics_fp64.csv');cv=loadcsv(ROOT/'cross_validation.csv');reps=loadcsv(ROOT/'representation_differences.csv')
    with Access('review_fp64'):
        truth=[np.load(OLD/f'inputs/truth_{i}.npy') for i in range(8)]
        for r in rows:
            p=np.load(ROOT/f"predictions/{r['arm']}/{r['branch']}/{r['method']}/{r['case']}.npy");m=independent_metrics(p,truth[int(r['case'])])
            for k,v in m.items():close(v,r[k]);counts['metrics']+=1
        for r in pairrows:
            i,j=int(r['i']),int(r['j']);path=ROOT/f"predictions/{r['arm']}/{r['branch']}/{r['method']}"
            dp=(np.load(path/f'{i}.npy').astype(np.float64)-np.load(path/f'{j}.npy').astype(np.float64)).ravel();dt=(truth[i]-truth[j]).ravel();den=np.sqrt(np.sum(dp**2)*np.sum(dt**2))
            close(float(np.sum(dp*dt)/den) if den>0 else None,r['difference_cosine']);close(np.abs(dp-dt).mean()/np.abs(dt).mean(),r['difference_relative_l1']);counts['pairs']+=1
        # Refit all six held-out-plume folds independently from raw native features.
        for arm in EXPECTED:
            for b in ('A','B'):
                for kind in ('normal','zero'):
                    xs=[np.load(feature_path(arm,b,kind,i),mmap_mode='r') for i in range(6)]
                    moments=[independent_moments(x,t) for x,t in zip(xs,truth[:6])]
                    # Independently accumulated per-plume stats; full-plume exclusion.
                    for hold in range(6):
                        rs=[r for r in cv if r['arm']==arm and r['branch']==b and r['kind']==kind and int(r['holdout'])==hold]
                        for r in rs:
                            assert [int(i) for i in r['train_ids'].split()]==[i for i in range(6) if i!=hold]
                            w,offset=independent_fit([m for i,m in enumerate(moments) if i!=hold],float(r['lambda_value']))
                            p=xs[hold].reshape(-1,len(w)).astype(np.float64)@w+offset;score=abs(p-truth[hold].ravel()).sum()/abs(truth[hold]).sum();close(score,r['relative_l1']);counts['cv']+=1
                    rs=[r for r in cv if r['arm']==arm and r['branch']==b and r['kind']==kind]
                    means={l:np.mean([float(r['relative_l1']) for r in rs if float(r['lambda_value'])==l]) for l in reg['ridge_grid']};minimum=min(means.values());chosen=max(l for l in means if means[l]<=minimum+1e-12)
                    s=next(s for s in seal['selected'] if s['arm']==arm and s['branch']==b and s['kind']==kind);assert chosen==s['lambda_value']
                    with np.load(ROOT/f'probes/{arm}_{b}_{kind}.npz') as c:
                        assert c['train_ids'].tolist()==list(range(6));w,offset=independent_fit(moments,chosen)
                        np.testing.assert_allclose(w,c['coef'],rtol=2e-5,atol=2e-7);close(offset,c['intercept'])
                        for case in range(8):
                            feature=np.load(feature_path(arm,b,kind,case),mmap_mode='r');prediction=(feature.reshape(-1,len(w)).astype(np.float64)@c['coef']+c['intercept']).reshape(64,64,64)
                            saved=np.load(ROOT/f'predictions/{arm}/{b}/{kind}_ridge/{case}.npy');np.testing.assert_allclose(prediction,saved,rtol=1e-10,atol=1e-12)

                    print('REVIEWED CV',arm,b,kind,flush=True)
        # Full layer statistics, and full-feature pair/swap/zero distances independently recomputed.
        cache={}
        for r in loadcsv(ROOT/'layer_statistics.csv'):
            x=np.load(feature_path(r['arm'],r['branch'],r['condition'],int(r['case']),r['layer']),mmap_mode='r');v=np.asarray(x,dtype=np.float64)
            close(v.mean(),r['mean']);close(v.std(),r['std']);assert np.isfinite(v).sum()==int(r['finite_count'])==int(r['total_count']);counts['layer_stats']+=1
        for arm in EXPECTED:
            for b in ('A','B'):
                for layer in sorted({r['layer'] for r in reps if r['arm']==arm and r['branch']==b}):
                    xs=[np.load(feature_path(arm,b,'normal',i,layer),mmap_mode='r') for i in range(8)];z=np.load(feature_path(arm,b,'zero',0,layer),mmap_mode='r');dist={}
                    subset=[r for r in reps if r['arm']==arm and r['branch']==b and r['layer']==layer]
                    for r in subset:
                        i,j=int(r['i']),int(r['j']);key=(i,j) if r['condition']=='normal_zero' else tuple(sorted((i,j)))
                        key=(r['condition']=='normal_zero',key)
                        if key not in dist:
                            a=np.asarray(xs[i],dtype=np.float64).ravel();v=np.asarray(z if r['condition']=='normal_zero' else xs[j],dtype=np.float64).ravel();na=np.sqrt(a@a);nv=np.sqrt(v@v)
                            dist[key]=(np.sqrt(np.sum((a-v)**2))/max(na,nv,1e-30),np.sum(a*v)/(na*nv) if na*nv>0 else None,np.sqrt(np.mean((a-v)**2)))
                        for k,value in zip(('normalized_l2','cosine','rms_difference'),dist[key]):close(value,r[k])
                        counts['representations']+=1
                    print('REVIEWED REPRESENTATIONS',arm,b,layer,flush=True)
        # Reconstruct classifications directly, not by calling the execution decision.
        results=[]
        for arm in EXPECTED:
            cats=[]
            for b in ('A','B'):
                layer='pooled_latent' if arm=='deepsets' else 'block1_measurement_operator';med=np.median([float(r['normalized_l2']) for r in reps if r['arm']==arm and r['branch']==b and r['layer']==layer and r['condition']=='normal_pair']);response=med>=1e-4
                good=True
                for i in (6,7):
                    case={r['method']:float(r['relative_l1']) for r in rows if r['arm']==arm and r['branch']==b and int(r['case'])==i}
                    good &= all(case[m]-case['normal_ridge']>=.01 for m in ('original','zero_ridge','zero_field'))
                pair=next(r for r in pairrows if r['arm']==arm and r['branch']==b and r['method']=='normal_ridge' and r['i']=='6' and r['j']=='7')
                direction=pair['difference_cosine']!='' and float(pair['difference_cosine'])>=.1 and float(pair['difference_relative_l1'])<.99
                stable=all(s['best_lambda_agreement']>=3 and s['solve_residual']<1e-6 for s in seal['selected'] if s['arm']==arm and s['branch']==b)
                cat='B' if response and good and direction and stable else 'D' if not stable else 'C' if response else 'A';cats.append(cat)
            status='GO' if set(cats)=={'B'} else 'INCONCLUSIVE' if 'D' in cats or ('B' in cats and len(set(cats))>1) else 'NO-GO'
            assert next(d['status'] for d in ev['decisions'] if d['arm']==arm)==status;results.append(dict(arm=arm,status=status,categories=cats))
    for p,h in reg['main_checkpoints'].items():assert sha(p)==h
    assert sha(reg['queue_path'])==reg['queue_sha256'];assert read(reg['queue_path'])['state']=='paused_by_user'
    verify_seal()
    freeze(ROOT/'independent_review.json',dict(at=now(),passed=True,counts=counts,decisions=results,checkpoints_unchanged=True,history_unchanged=True,main_checkpoints_unchanged=True,queue_unchanged_paused=True,neural_updates=0,source_sha256=sha(__file__)))
    print('INDEPENDENT REVIEW PASS',results,counts,flush=True)
if __name__=='__main__':main()
