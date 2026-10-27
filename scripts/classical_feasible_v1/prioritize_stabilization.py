"""Reserve the next queue gap for the already registered short rate control."""
from feasible_context import *
import signal


def main():
    pid=int(sys.argv[1]);paused=False;deadline=time.monotonic()+3600
    try:
        while time.monotonic()<deadline:
            state=read(ART/'deeponet_stabilization_status.json')['state']
            if state!='waiting_for_idle_gpu':return
            rows=[json.loads(line) for line in (ROOT/'pilot_fno/metrics.jsonl').read_text().splitlines()]
            latest=rows[-1]
            if latest['stage']=='physical' and latest['epoch']>=28:
                assert 'remaining_pilots.py' in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
                assert read(ROOT/'scheduling/pilot_fno_workers.json')['state']=='running'
                os.kill(pid,signal.SIGSTOP);paused=True
                write(ROOT/'stabilization_priority.json',dict(action='pause queue supervisor only',pid=pid,
                    reason='Run the one registered short DeepONet rate control before the next long architecture pilot',
                    training_workers_interrupted=0))
                break
            if read(ROOT/'scheduling/pilot_fno_workers.json')['state']!='running':return
            time.sleep(5)
        if not paused:return
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            state=read(ART/'deeponet_stabilization_status.json')['state']
            if state in ['complete','deferred_resource_wait_limit']:return
            record=ROOT/'scheduling/pilot_deeponet_lr2e5.json'
            if record.exists() and read(record)['state'] in ['failed','interrupted']:
                raise RuntimeError('Rate-control engineering failure; preserve logs')
            time.sleep(5)
        raise TimeoutError('Priority window expired; remaining queue will be resumed')
    finally:
        if paused:
            os.kill(pid,signal.SIGCONT)
            write(ROOT/'stabilization_priority_resumed.json',dict(pid=pid,action='resume pending architecture queue',training_workers_interrupted=0))


if __name__=='__main__':main()
