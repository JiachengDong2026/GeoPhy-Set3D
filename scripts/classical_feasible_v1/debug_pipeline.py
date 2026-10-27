"""Sequential bounded multi-case checks for architectures passing independent local review."""
from feasible_context import *
import subprocess


def call(name,args):
    with (ROOT/(name+'.log')).open('x') as log:
        process=subprocess.Popen([sys.executable,*args],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
        assert process.wait()==0,(name,'failed; inspect preserved log')


def main():
    verify();deadline=time.monotonic()+7200
    while not (ART/'debug_summary.json').exists():
        if (ROOT/'scheduling/debug.json').exists():assert read(ROOT/'scheduling/debug.json')['state'] not in ['failed','interrupted']
        assert time.monotonic()<deadline,'Debug dependency wait exceeded bounded2h'
        time.sleep(15)
    call('review_debug',[str(PROJECT/'scripts/classical_feasible_v1/review_debug.py')])
    while not (ART/'teachers.json').exists():time.sleep(15)
    for arm in ['deeponet','fno','senseiver','gnot','deepsets']:
        local=next(v for v in read(ART/'debug_independent_review.json')['models'] if v['arm']==arm)
        if not local['local_fit_passed']:continue
        call(arm+'_pilot_controller',[str(PROJECT/'scripts/classical_feasible_v1/launch.py'),'pilot_'+arm,'train_multicase_debug.py',arm])
    write(ART/'other_algorithms_status.json',dict(state='bounded_multicase_checks_complete'))


if __name__=='__main__':main()
