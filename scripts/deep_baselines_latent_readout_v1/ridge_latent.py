"""Independent FP64 affine ridge, analytical solve; no neural optimizer."""
from latent_context import *

def sufficient(x,y,chunk=32768):
    d=x.shape[-1];x=x.reshape(-1,d);y=y.reshape(-1)
    sx=np.zeros(d);xx=np.zeros((d,d));xy=np.zeros(d)
    for start in range(0,len(x),chunk):
        a=x[start:start+chunk].astype(np.float64);b=y[start:start+chunk]
        sx+=a.sum(0);xx+=a.T@a;xy+=a.T@b
    return dict(n=len(x),sx=sx,xx=xx,xy=xy,sy=float(y.sum()))

def solve(stats,lam,floor=1e-6):
    n=sum(s['n'] for s in stats);mu=sum(s['sx'] for s in stats)/n;ym=sum(s['sy'] for s in stats)/n
    cov=sum(s['xx'] for s in stats)/n-np.outer(mu,mu)
    cov=(cov+cov.T)/2;scale=np.maximum(np.sqrt(np.maximum(np.diag(cov),0)),floor)
    a=cov/np.outer(scale,scale)+lam*np.eye(len(mu));rhs=(sum(s['xy'] for s in stats)/n-mu*ym)/scale
    beta=np.linalg.solve(a,rhs);coef=beta/scale;intercept=ym-mu@coef
    residual=float(np.linalg.norm(a@beta-rhs)/max(np.linalg.norm(rhs),1e-30))
    return dict(coef=coef,intercept=intercept,mean=mu,scale=scale,beta=beta,condition=float(np.linalg.cond(a)),residual=residual)

def predict(x,coef,intercept):
    return (x.reshape(-1,x.shape[-1]).astype(np.float64)@coef+intercept).reshape(64,64,64)

def main():
    reg=verify_seal();out=ROOT/'probes';out.mkdir(exist_ok=False);cv=[];selected=[];hashes={}
    with Access('fit') as access:
        # No unseen feature or label is opened anywhere in this stage.
        labels=[np.load(OLD/f'inputs/truth_{i}.npy',allow_pickle=False) for i in range(6)]
        for arm in EXPECTED:
            for b in ('A','B'):
                for kind in ('normal','zero'):
                    xs=[np.load(feature_path(arm,b,kind,i),mmap_mode='r') for i in range(6)]
                    stats=[sufficient(x,y) for x,y in zip(xs,labels)]
                    scores={};winners=[]
                    for lam in reg['ridge_grid']:
                        errs=[]
                        for hold in range(6):
                            fit=solve([s for i,s in enumerate(stats) if i!=hold],lam,reg['std_floor'])
                            p=predict(xs[hold],fit['coef'],fit['intercept']);l1=float(np.abs(p-labels[hold]).sum()/np.abs(labels[hold]).sum());errs.append(l1)
                            cv.append(dict(arm=arm,branch=b,kind=kind,lambda_value=lam,holdout=hold,train_ids=' '.join(str(i) for i in range(6) if i!=hold),relative_l1=l1,condition=fit['condition'],solve_residual=fit['residual']))
                        scores[lam]=float(np.mean(errs))
                    best=min(scores.values());chosen=max(l for l in scores if scores[l]<=best+1e-12)
                    for hold in range(6):
                        rs=[r for r in cv if r['arm']==arm and r['branch']==b and r['kind']==kind and r['holdout']==hold]
                        minimum=min(r['relative_l1'] for r in rs);winners.append(max(r['lambda_value'] for r in rs if r['relative_l1']<=minimum+1e-12))
                    fit=solve(stats,chosen,reg['std_floor']);assert fit['residual']<1e-6 and np.isfinite(fit['coef']).all()
                    p=out/f'{arm}_{b}_{kind}.npz';np.savez(p,**fit,lambda_value=chosen,train_ids=np.arange(6));hashes[str(p)]=sha(p)
                    selected.append(dict(arm=arm,branch=b,kind=kind,lambda_value=chosen,mean_CV_relative_l1=scores[chosen],fold_winners=winners,best_lambda_agreement=winners.count(chosen),condition=fit['condition'],solve_residual=fit['residual'],features=len(fit['coef']),coefficient_count=len(fit['coef'])+1))
                    print('FITTED',arm,b,kind,chosen,scores,flush=True)
        csvwrite(ROOT/'cross_validation.csv',cv)
        freeze(ROOT/'probe_seal.json',dict(at=now(),selected=selected,coefficients_sha256=hashes,seen_ids=list(range(6)),unseen_ids=[6,7],unseen_labels_read=False,neural_updates=0,cv_sha256=sha(ROOT/'cross_validation.csv')))
if __name__=='__main__':main()
