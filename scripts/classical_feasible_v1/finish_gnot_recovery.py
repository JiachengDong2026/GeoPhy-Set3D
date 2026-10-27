"""Finish bookkeeping after exact GNOT recovery and the existing independent observer."""
from feasible_context import *
import subprocess


def main():
    verify();deadline=time.monotonic()+8*3600
    while time.monotonic()<deadline:
        path=ROOT/'scheduling/pilot_gnot_resume.json'
        if path.exists():
            state=read(path)['state'];assert state not in ['failed','interrupted'],'Preserve resumed GNOT failure; no replay'
            if state=='complete':break
        time.sleep(10)
    else:raise TimeoutError('GNOT recovery wait limit')
    assert (ART/'gnot_pilot_summary.json').exists() and (ART/'deepsets_pilot_summary.json').exists()
    write(ART/'other_algorithms_status.json',dict(state='bounded_multicase_checks_complete',
        foreign_conflict_preserved=True,gnot_authoritative_task='pilot_gnot_resume',no_optimizer_replay=True))
    while time.monotonic()<deadline:
        path=ART/'completion_observer_status.json'
        if path.exists() and read(path)['state']=='all_training_decisions_and_reviews_complete':break
        time.sleep(10)
    else:raise TimeoutError('Independent observer wait limit')
    write(ART/'sparse_guard_v2/remaining_status.json',dict(state='remaining_training_decisions_complete',
        interrupted_waiting_supervisor_superseded=True,all_original_independent_reviews_complete=True))
    with (ROOT/'post_recovery_report.log').open('x') as log:
        subprocess.run([sys.executable,str(PROJECT/'scripts/classical_feasible_v1/delivery_report.py')],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
    verify()


if __name__=='__main__':main()
