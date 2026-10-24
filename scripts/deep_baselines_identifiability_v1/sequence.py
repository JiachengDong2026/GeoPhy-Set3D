"""Finite C-first sequence; no formal worker or unbounded search is available."""
from diag_context import *
import subprocess

def main():
    pre=verified('preregistration.json');done=0
    tasks=[(f'C_{a}',[str(HERE/'experiment_c.py'),'--arm',a]) for a in ARMS]
    tasks += [(f"{j['experiment']}_{j['arm']}_{j['version']}_n{j['n']}",
        [str(HERE/'train.py'),'--experiment',j['experiment'],'--arm',j['arm'],'--version',j['version'],'--n',str(j['n'])]) for j in pre['jobs']]
    for key,args in tasks:
        c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='running',current=key,completed_jobs=done,
            formal_training_authorized=False,second_seed_authorized=False))
        log=ROOT/f'{key}_controller.log'
        with log.open('x') as stream:
            result=subprocess.run([sys.executable,str(HERE/'launch.py'),'--key',key,'--',sys.executable,*args],
                cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='STOPPED_ENGINEERING_OR_RESOURCE',current=key,exit_code=result.returncode,
                completed_jobs=done,formal_training_authorized=False));raise RuntimeError(f'Blocked sequence: {key}; inspect {log}')
        done+=1
    c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='DIAGNOSTICS_COMPLETED_PENDING_REVIEW',
        completed_jobs=done,A_updates=1200,B_updates=800,C_updates=0,formal_training_authorized=False,second_seed_authorized=False))

if __name__=='__main__':main()
