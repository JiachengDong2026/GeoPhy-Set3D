"""Algebraic counterfactual: isolate consistency's incentive to shrink amplitude."""
from repair_context import *


def scale_invariant(a,b):
    a,b=a.double(),b.double()
    n=(a-b).abs().flatten(1).mean(1)
    d=.5*(a.abs().flatten(1).mean(1)+b.abs().flatten(1).mean(1))
    return torch.where(d>0,n/d.clamp_min(torch.finfo(torch.double).tiny),torch.zeros_like(d)).mean()


def main():
    rows=[]
    a=torch.tensor([[.001,.002,.003,.004]],dtype=torch.double)
    b=a*torch.tensor([[.9,1.1,.8,1.2]],dtype=torch.double)
    for alpha in (1.,.001,1e-6,1e-10):
        for name,fn in [('original',ss.consistency_discrepancy),('homogeneous_ratio',scale_invariant)]:
            x=torch.tensor(alpha,dtype=torch.double,requires_grad=True)
            loss=fn(a*x,b*x);grad=float(torch.autograd.grad(loss,x)[0])
            eps=alpha*1e-4
            fd=float((fn(a*(alpha+eps),b*(alpha+eps))-fn(a*(alpha-eps),b*(alpha-eps)))/(2*eps))
            n=float((a-b).abs().mean());d=float(.5*(a.abs().mean()+b.abs().mean()))
            analytic=n*1e-6/(alpha*d+1e-6)**2 if name=='original' else 0.
            # The well-conditioned perturbation is log amplitude. Absolute
            # dL/dalpha subtracts O(1/alpha) terms as alpha approaches zero.
            assert abs(alpha*(grad-analytic))<1e-12
            assert abs(alpha*(fd-analytic))<1e-8
            rows.append(dict(function=name,alpha=alpha,loss=float(loss.detach()),dL_dalpha=grad,
                finite_difference=fd,analytic=analytic,
                dL_dlogalpha=alpha*grad,analytic_dL_dlogalpha=alpha*analytic))
    freeze('consistency_causal_probe.json',dict(at=c.now(),passed=True,rows=rows,
        causal_claim='For the SAME fixed pair of fields, original epsilon rewards shrinking common amplitude; exact homogeneous ratio removes only this incentive',
        limitation='Algebraic incentive established; whether it explains failed training sufficiently requires matched fresh training pilots',
        GT_used=False,test_used=False))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
