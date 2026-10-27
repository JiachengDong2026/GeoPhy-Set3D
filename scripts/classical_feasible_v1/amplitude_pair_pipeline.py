"""Use the next freed diagnostic GPU for a bounded paired UNet scale check."""
from feasible_context import *
import subprocess


def main():
    deadline=time.monotonic()+3600
    while True:
        task=ROOT/'scheduling/pilot_deeponet.json'
        if task.exists() and read(task)['state'] in ['complete','failed','interrupted']:
            assert read(task)['state']=='complete';break
        assert time.monotonic()<deadline
        time.sleep(10)
    with (ROOT/'amplitude_pair_controller.log').open('x') as log:
        code=subprocess.call([sys.executable,str(PROJECT/'scripts/classical_feasible_v1/launch.py'),'amplitude_pair','amplitude_pair.py'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
    assert code==0,'Paired check failed; preserve log'


if __name__=='__main__':main()
