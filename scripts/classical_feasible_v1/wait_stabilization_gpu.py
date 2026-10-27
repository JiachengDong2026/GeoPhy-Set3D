"""Bounded resource wait for the one registered DeepONet stabilization check."""
from feasible_context import *
from launch import controller
import subprocess


def main():
    verify()
    for path,digest in read(ART/'deeponet_stabilization_registration.json')['sources'].items():assert sha(PROJECT/path)==digest,path
    scheduler=controller.c.scheduler();began=time.monotonic()
    while time.monotonic()-began<7200:
        snapshot=scheduler.snapshot()
        if any(device['idle'] for device in scheduler.devices(snapshot)):
            with (ROOT/'deeponet_stabilization_controller.log').open('x') as log:
                process=subprocess.Popen([sys.executable,str(PROJECT/'scripts/classical_feasible_v1/launch.py'),
                    'pilot_deeponet_lr2e5','train_deeponet_stabilization.py'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
                write(ART/'deeponet_stabilization_status.json',dict(state='controller',pid=process.pid))
                code=process.wait()
            assert code==0,'Inspect registered stabilization log; no hidden retry'
            assert (ART/'deeponet_lr2e5_summary.json').exists()
            write(ART/'deeponet_stabilization_status.json',dict(state='complete'))
            return
        write(ART/'deeponet_stabilization_status.json',dict(state='waiting_for_idle_gpu',elapsed_seconds=time.monotonic()-began,optimizer_updates=0))
        time.sleep(30)
    write(ART/'deeponet_stabilization_status.json',dict(state='deferred_resource_wait_limit',optimizer_updates=0))


if __name__=='__main__':main()
