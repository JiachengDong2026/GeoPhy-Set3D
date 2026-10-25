from latent_context import *
import subprocess,platform

def main():
    assert not ROOT.exists(),ROOT;ROOT.mkdir()
    registered=read(OLD/'registration.json')
    for p,h in registered['inputs'].items():assert sha(p)==h,p
    for a,h in EXPECTED.items():assert sha(checkpoint(a))==h
    ss=samples();ident=read(OLD/'input_registration.json')['identity']
    for i,s in enumerate(ss):
        for b in ('A','B'):assert old.value_hash(old.nonmeasurement(s[b]))==ident[i]['geometry'][b]==ident[0]['geometry'][b]
    tracked=subprocess.check_output(['git','ls-files'],cwd=PROJECT,text=True).splitlines()
    history={str(PROJECT/p):sha(PROJECT/p) for p in tracked if (PROJECT/p).is_file()}
    anchor=read('/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/main_anchor.json')
    mains={m['checkpoint']:sha(m['checkpoint']) for m in (anchor['model'],anchor['reference'])}
    inputs=registered['inputs']
    sources={str(p):sha(p) for p in HERE.glob('*.py')}
    sources[str(PROJECT/'docs/deep_baselines_latent_readout_v1_protocol_zh.md')]=sha(PROJECT/'docs/deep_baselines_latent_readout_v1_protocol_zh.md')
    reg=dict(at=now(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        inputs=inputs,checkpoints={str(checkpoint(a)):h for a,h in EXPECTED.items()},sources=sources,history=history,main_checkpoints=mains,
        queue_path='/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/queue_status.json',queue_sha256=sha('/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/queue_status.json'),
        seen=list(range(6)),unseen=[6,7],ridge_grid=[1e-6,1e-4,1e-2,1.0],std_floor=1e-6,
        neural_updates=0,remaining_diagnostic_updates=100,dependencies=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),identity_verified=True)
    freeze(ROOT/'registration.json',reg);print('REGISTERED',reg['git_head'],flush=True)
if __name__=='__main__':main()
