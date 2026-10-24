"""Inspected post-result bookkeeping repair; never replay committed updates."""
from diag_context import *
import argparse
import importlib.util
import subprocess
import engineering_memory_v2 as memory

def worker(job):
    amendment=verified('engineering_bookkeeping_v4.json');c.verify_hashes(amendment['sources'])
    memory.install()
    spec=importlib.util.spec_from_file_location('diag_eval_final_exact',HERE/'evaluate.py')
    evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation);sys.modules['evaluate']=evaluation
    spec=importlib.util.spec_from_file_location('diag_train_bookkeeping_v2',HERE/'train_fixed_v2.py')
    training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
    training.run(job['experiment'],job['arm'],job['version'],job['n'])

def sequence():
    amendment=verified('engineering_bookkeeping_v4.json');pre=verified('preregistration.json')
    existing=c.verified(ROOT/'runs/A_deepsets_v2_n1/result.json');assert existing['updates']==200
    for index,j in enumerate(pre['jobs']):
        if index==0:continue
        key=f"{j['experiment']}_{j['arm']}_{j['version']}_n{j['n']}"
        c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='running',current=key,committed_prior_updates=200,
            numerical_protocol_unchanged=True,formal_training_authorized=False))
        command=[sys.executable,str(HERE/'launch.py'),'--key',key,'--',sys.executable,str(HERE/'resume_v4.py'),'--job-index',str(index)]
        with (ROOT/f'{key}_bookkeeping_v4_controller.log').open('x') as log:p=subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:
            c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='STOPPED_ENGINEERING_OR_RESOURCE',current=key,exit_code=p.returncode,formal_training_authorized=False))
            raise RuntimeError(f'Inspect {key}; no automatic retry')
    c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='DIAGNOSTICS_COMPLETED_PENDING_REVIEW',A_updates=1200,B_updates=800,C_updates=0,
        formal_training_authorized=False,second_seed_authorized=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job-index',type=int);a=ap.parse_args()
    if a.job_index is None:sequence()
    else:
        assert 1<=a.job_index<10;worker(verified('preregistration.json')['jobs'][a.job_index])
