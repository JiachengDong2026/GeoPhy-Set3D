"""Reviewed zero-update C OOM recovery; original finite jobs and budgets only."""
from diag_context import *
import subprocess

def main():
    amendment=verified('engineering_memory_v2.json');pre=verified('preregistration.json')
    assert c.verified(ROOT/'C/deepsets/result.json')['optimizer_updates']==0
    jobs=[('C_gnot',['--script','experiment_c','--','--arm','gnot'])]
    jobs.extend((f"{j['experiment']}_{j['arm']}_{j['version']}_n{j['n']}",
        ['--script','train','--','--experiment',j['experiment'],'--arm',j['arm'],'--version',j['version'],'--n',str(j['n'])]) for j in pre['jobs'])
    for key,args in jobs:
        c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='running',current=key,engineering_adapter='weak_taps_v2',formal_training_authorized=False))
        command=[sys.executable,str(HERE/'launch.py'),'--key',key]
        if key=='C_gnot':command+=['--recover-reason','Inspected C GNOT input-gradient OOM; zero optimizer updates; preserved failed arrays/log; weak tap closures and unreachable-graph collection only, unchanged thresholds/full64/full rays']
        command+=['--',sys.executable,str(HERE/'engineering_memory_v2.py'),*args]
        with (ROOT/f'{key}_memory_v2_controller.log').open('x') as log:
            p=subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:
            c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='STOPPED_ENGINEERING_OR_RESOURCE',current=key,exit_code=p.returncode,formal_training_authorized=False))
            raise RuntimeError(f'Inspect failed finite job {key}, no automatic retry')
    c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='DIAGNOSTICS_COMPLETED_PENDING_REVIEW',A_updates=1200,B_updates=800,C_updates=0,
        formal_training_authorized=False,second_seed_authorized=False))

if __name__=='__main__':main()
