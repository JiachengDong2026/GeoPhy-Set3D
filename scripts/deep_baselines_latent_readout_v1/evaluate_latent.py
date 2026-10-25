"""Final label-capable stage, only after analytical readout coefficients are sealed."""
from latent_context import *
from ridge_latent import predict
import itertools

def metric(p,y):
    p=np.asarray(p,dtype=np.float64).reshape(64,64,64);y=y.reshape(p.shape);assert np.isfinite(p).all()
    pos=np.maximum(p,0);yp=np.maximum(y,0);axis=np.linspace(0,.6,64)
    def centroid(a):
        mass=a.sum()
        return np.array([np.dot(a.sum(axis=tuple(k for k in range(3) if k!=j)),axis)/mass for j in range(3)]) if mass>0 else None
    cp,cy=centroid(pos),centroid(yp);pc=p-p.mean();yc=y-y.mean();den=np.linalg.norm(pc)*np.linalg.norm(yc)
    return dict(relative_l1=float(np.abs(p-y).sum()/np.abs(y).sum()),mae=float(np.abs(p-y).mean()),mean=float(p.mean()),std=float(p.std()),max=float(p.max()),min=float(p.min()),nonzero_fraction=float((np.abs(p)>1e-10).mean()),positive_structure_fraction=float((p>1e-6).mean()),negative_fraction=float((p<0).mean()),above_one_fraction=float((p>1).mean()),centroid_error_m=float(np.linalg.norm(cp-cy)) if cp is not None else None,mass_relative_error=float(abs(p.sum()-y.sum())/abs(y.sum())),integral=float(p.sum()*(.6/63)**3),spatial_correlation=float((pc*yc).sum()/den) if den>0 else None)

def diff(x,y):
    x=np.asarray(x,dtype=np.float64).reshape(-1);y=np.asarray(y,dtype=np.float64).reshape(-1);d=x-y;nx=np.linalg.norm(x);ny=np.linalg.norm(y)
    return dict(normalized_l2=float(np.linalg.norm(d)/max(nx,ny,1e-30)),cosine=float(x@y/(nx*ny)) if nx*ny>0 else None,rms_difference=float(np.sqrt(np.mean(d*d))))

def stats(x):
    a=np.asarray(x,dtype=np.float64);return dict(shape='x'.join(map(str,a.shape)),mean=float(a.mean()),std=float(a.std()),finite_count=int(np.isfinite(a).sum()),total_count=a.size,finite_fraction=float(np.isfinite(a).sum()/a.size))

def decision(rows,pairs,reps,seal):
    decisions=[]
    for arm in EXPECTED:
        evidence=[]
        for b in ('A','B'):
            # Actual upstream conditional representation, not input channels or coordinate features.
            stage='pooled_latent' if arm=='deepsets' else 'block1_measurement_operator'
            rr=[r['normalized_l2'] for r in reps if r['arm']==arm and r['branch']==b and r['layer']==stage and r['condition']=='normal_pair']
            response=float(np.median(rr))>=1e-4
            conditions=[]
            for i in (6,7):
                cases={r['method']:r for r in rows if r['arm']==arm and r['branch']==b and r['case']==i}
                conditions.append(all(cases[m]['relative_l1']-cases['normal_ridge']['relative_l1']>=.01 for m in ('original','zero_ridge','zero_field')))
            p=next(r for r in pairs if r['arm']==arm and r['branch']==b and r['i']==6 and r['j']==7 and r['method']=='normal_ridge')
            direction=p['difference_cosine'] is not None and p['difference_cosine']>=.1 and p['difference_relative_l1']<.99
            fits=[s for s in seal['selected'] if s['arm']==arm and s['branch']==b]
            stable=all(s['best_lambda_agreement']>=3 and s['solve_residual']<1e-6 for s in fits)
            # Failure of stability forces D when a positive readout claim is at issue;
            # do not convert an otherwise negative outcome into a GO.
            category='B' if response and all(conditions) and direction and stable else 'D' if not stable else 'C' if response else 'A'
            evidence.append(dict(branch=b,upstream_layer=stage,median_normalized_l2=float(np.median(rr)),meaningful_response=response,unseen_improvement=conditions,unseen_difference_gate=direction,probe_stable=stable,category=category))
        categories={e['category'] for e in evidence}
        status='GO' if categories=={'B'} else 'INCONCLUSIVE' if 'D' in categories or ('B' in categories and len(categories)>1) else 'NO-GO'
        decisions.append(dict(arm=arm,status=status,evidence=evidence))
    return decisions

def main():
    verify_seal();seal=read(ROOT/'probe_seal.json')
    for p,h in seal['coefficients_sha256'].items():assert sha(p)==h
    rows=[];pairs=[];layerstats=[];representations=[];alignments=[];predhash={}
    with Access('evaluate'):
        truth=[np.load(OLD/f'inputs/truth_{i}.npy',allow_pickle=False) for i in range(8)]
        for arm in EXPECTED:
            for b in ('A','B'):
                preds={}
                for i in range(8):
                    methods={'original':np.load(ROOT/f'original/{arm}/{b}/{i}.npy').reshape(64,64,64),'zero_field':np.zeros((64,64,64))}
                    for kind in ('normal','zero'):
                        with np.load(ROOT/f'probes/{arm}_{b}_{kind}.npz') as coef:
                            methods[kind+'_ridge']=predict(np.load(feature_path(arm,b,kind,i),mmap_mode='r'),coef['coef'],coef['intercept'])
                    for method,p in methods.items():
                        path=ROOT/f'predictions/{arm}/{b}/{method}/{i}.npy';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.save(path,p);predhash[str(path)]=sha(path)
                        rows.append(dict(arm=arm,branch=b,case=i,role='seen' if i<6 else 'unseen',method=method,**metric(p,truth[i])));preds[i,method]=p
                layers=sorted(p.stem for p in (ROOT/f'features/{arm}/{b}/normal/0').glob('*.npy'))
                for layer in layers:
                    xs=[np.load(feature_path(arm,b,'normal',i,layer),mmap_mode='r') for i in range(8)]
                    z=np.load(feature_path(arm,b,'zero',0,layer),mmap_mode='r');values=[];truthdiff=[]
                    for i in range(8):
                        layerstats.append(dict(arm=arm,branch=b,case=i,condition='normal',layer=layer,**stats(xs[i])))
                        layerstats.append(dict(arm=arm,branch=b,case=i,condition='zero',layer=layer,**stats(z)))
                        representations.append(dict(arm=arm,branch=b,i=i,j=i,layer=layer,condition='normal_zero',**diff(xs[i],z)))
                    for i,j in old.design()['pairs']:
                        d=diff(xs[i],xs[j]);td=diff(truth[i],truth[j]);values.append(d['rms_difference']);truthdiff.append(td['rms_difference'])
                        representations.append(dict(arm=arm,branch=b,i=i,j=j,layer=layer,condition='normal_pair',**d))
                        for recipient,donor in ((i,j),(j,i)):
                            representations.append(dict(arm=arm,branch=b,i=recipient,j=donor,layer=layer,condition='swap',**d))
                    a=np.array(values);t=np.array(truthdiff);corr=np.corrcoef(a,t)[0,1] if a.std()>0 and t.std()>0 else None
                    alignments.append(dict(arm=arm,branch=b,layer=layer,truth_pair_distance_correlation=float(corr) if corr is not None else None,pair_count=28,independent_samples=False))
                    print('REPRESENTATIONS',arm,b,layer,flush=True)
                for i,j in old.design()['pairs']:
                    dy=(truth[i]-truth[j]).reshape(-1)
                    for method in ('original','normal_ridge','zero_ridge','zero_field'):
                        dp=(preds[i,method]-preds[j,method]).reshape(-1);den=np.linalg.norm(dp)*np.linalg.norm(dy)
                        pairs.append(dict(arm=arm,branch=b,i=i,j=j,group='seen_seen' if j<6 else 'unseen_unseen' if i>=6 else 'seen_unseen',method=method,difference_cosine=float(dp@dy/den) if den>0 else None,difference_relative_l1=float(np.abs(dp-dy).sum()/np.abs(dy).sum()),prediction_rms=float(np.sqrt(np.mean(dp*dp))),truth_rms=float(np.sqrt(np.mean(dy*dy)))))
        improvements=[]
        for arm in EXPECTED:
            for b in ('A','B'):
                for i in range(8):
                    rs={r['method']:r for r in rows if r['arm']==arm and r['branch']==b and r['case']==i}
                    improvements.append(dict(arm=arm,branch=b,case=i,role='seen' if i<6 else 'unseen',**{f'L1_improvement_vs_{m}':rs[m]['relative_l1']-rs['normal_ridge']['relative_l1'] for m in ('original','zero_ridge','zero_field')}))
        for name,data in [('case_metrics',rows),('pair_metrics',pairs),('layer_statistics',layerstats),('representation_differences',representations),('truth_distance_alignment',alignments),('probe_improvements',improvements)]:csvwrite(ROOT/f'{name}.csv',data)
        decisions=decision(rows,pairs,representations,seal)
        freeze(ROOT/'evaluation.json',dict(at=now(),decisions=decisions,predictions_sha256=predhash,neural_updates=0,synthetically_supervised_diagnostic=True,unseen_read_after_probe_seal=True))
        print('DECISIONS',decisions,flush=True)
if __name__=='__main__':main()
