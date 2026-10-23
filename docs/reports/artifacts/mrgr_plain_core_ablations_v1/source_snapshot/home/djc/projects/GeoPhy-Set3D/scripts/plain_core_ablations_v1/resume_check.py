"""Cross-process canonical epoch commit recovery; engineering subsets only."""
from common_stage import *
import math


def main():
    backend();arm,seed=TASKS[0];source=verified(BASE/'training'/f'{arm}_seed{seed}'/'protocol.json')
    for name,stop in [('uninterrupted',2),('resumed',1),('resumed',2)]:
        target=BASE/'resume_smoke'/name
        if not (target/'protocol.json').exists():
            p=copy.deepcopy(source);p['subset_smoke']=True;p['planned_updates']=4
            p['config']['experiment']=dict(name='engineering_'+name,output_root=str(target))
            freeze(target/'protocol.json',p)
        subprocess.run(command('train_worker.py','--target',target,'--stop',stop),cwd=PROJECT,check=True)
    a=torch.load(BASE/'resume_smoke/uninterrupted/checkpoints/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(BASE/'resume_smoke/resumed/checkpoints/last.pt',map_location='cpu',weights_only=False)
    maximum=0.
    def same(a,b):
        nonlocal maximum
        if isinstance(a,torch.Tensor):
            if a.is_floating_point():
                maximum=max(maximum,float((a-b).abs().max()) if a.numel() else 0.)
                torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
            else:assert torch.equal(a,b)
        elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-4,abs_tol=1e-6),(a,b)
        else:assert a==b,(a,b)
    for k in ('model','best_model','optimizer','scheduler','rng','epoch','updates','best_epoch','best_score'):same(a[k],b[k])
    for x,y in zip(a['metrics'],b['metrics']):
        for k in ('epoch','learning_rate','loader_rng_sha256'):same(x[k],y[k])
        for split in ('train','val'):same({k:v for k,v in x[split].items() if k!='seconds'},{k:v for k,v in y[split].items() if k!='seconds'})
    freeze(BASE/'checks/resume.json',dict(at=now(),passed=True,epochs=2,committed_updates=4,workers=4,rtol=1e-4,atol=1e-6,
        max_tensor_abs_difference=maximum,integer_RNG_exact=True,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],formal_models_trained=0))
    print('Cross-process resume PASS',flush=True)


if __name__=='__main__':main()
