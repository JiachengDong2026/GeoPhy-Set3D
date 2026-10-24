"""Continue the original ten diagnostics after inspected zero-update import failure."""
from diag_context import *
import subprocess

def main():
    pre=verified('preregistration.json');verified('engineering_dispatch_v3.json')
    assert all(c.verified(ROOT/f'C/{a}/result.json')['optimizer_updates']==0 for a in ARMS)
    assert not list((ROOT/'runs').glob('*/updates.jsonl'))
    for j in pre['jobs']:
        key=f"{j['experiment']}_{j['arm']}_{j['version']}_n{j['n']}"
        command=[sys.executable,str(HERE/'launch.py'),'--key',key]
        if key=='A_deepsets_v2_n1':command+=['--recover-reason','Inspected cached historical train module collision; failed before run directory/optimizer updates; exact-file dispatch to sealed diagnostic train/evaluate; unchanged numerical code and thresholds']
        command+=['--',sys.executable,str(HERE/'engineering_dispatch_v3.py'),'--experiment',j['experiment'],'--arm',j['arm'],'--version',j['version'],'--n',str(j['n'])]
        c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='running',current=key,engineering_adapter='weak_taps_and_exact_dispatch',formal_training_authorized=False))
        with (ROOT/f'{key}_dispatch_v3_controller.log').open('x') as log:p=subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:
            c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='STOPPED_ENGINEERING_OR_RESOURCE',current=key,exit_code=p.returncode,formal_training_authorized=False))
            raise RuntimeError(f'Inspect {key}; no automatic retry')
    c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='DIAGNOSTICS_COMPLETED_PENDING_REVIEW',A_updates=1200,B_updates=800,C_updates=0,
        formal_training_authorized=False,second_seed_authorized=False))

if __name__=='__main__':main()
