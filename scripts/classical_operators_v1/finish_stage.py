"""Finish independent reviews after the current FNO pilot exits; no new training."""
import subprocess,time,sys
from audit import PROJECT,ROOT,read
HERE=PROJECT/'scripts/classical_operators_v1'
def run(name,*args):
    command=[sys.executable,str(HERE/name),*args]
    print('Executing',name,*args,flush=True)
    subprocess.run(command,cwd=PROJECT,check=True)
def main():
    began=time.monotonic()
    while True:
        p=ROOT/'scheduling/pilot_fno.json'
        if p.exists() and read(p)['state'] in ['complete','failed','interrupted']:break
        if time.monotonic()-began>1800:raise TimeoutError('FNO did not finish; no implicit intervention')
        time.sleep(5)
    if (ROOT/'fno/pilot/result.json').exists():run('review.py','fno')
    else:
        run('terminal_controller.py','fno','--gpu','1')
        assert read(ROOT/'scheduling/terminal_fno.json')['state']=='complete'
        run('terminal_review.py','fno')
    run('terminal_controller.py','deeponet','--gpu','1')
    assert read(ROOT/'scheduling/terminal_deeponet.json')['state']=='complete'
    run('terminal_review.py','deeponet')
    print('Independent review stages finished; no formal training started',flush=True)
if __name__=='__main__':main()
