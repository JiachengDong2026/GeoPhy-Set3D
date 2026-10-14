#!/usr/bin/env python3
"""Finite orchestration after the already-registered formal run: no training/search."""
import json,os,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];OUT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/ray_senseiver_supervised_v1');ART=REPO/'docs/reports/artifacts/ray_senseiver_supervised_v1'
PYTHON='/scratch/djc/envs/gas-torch/bin/python'

def command(args,log):
    with (OUT/log).open('a',buffering=1) as f:subprocess.run(args,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)

def checkpoint_status():
    for name in ['engineering_error.json','pause.json']:
        if (OUT/'science'/name).exists():raise RuntimeError('Formal engineering pause: '+str(OUT/'science'/name))

def main():
    while not (OUT/'science/complete.json').exists():
        checkpoint_status();time.sleep(30)
    result=json.loads((OUT/'science/complete.json').read_text());checkpoint_status()
    if result['reason'] not in ['FAILED_TO_LEARN','STALLED_WITHOUT_DEPENDENCE']:
        command([PYTHON,'scripts/evaluate_senseiver_supervised.py','freeze'],'freeze.log')
        frozen=json.loads((ART/'frozen_best.json').read_text())
        if frozen['effective_learning']:
            subprocess.run(['git','add','docs/reports/artifacts/ray_senseiver_supervised_v1/evaluation_protocol.json','docs/reports/artifacts/ray_senseiver_supervised_v1/frozen_best.json'],cwd=REPO,check=True)
            subprocess.run(['git','commit','-m','Freeze Val-selected supervised Senseiver and register fixed 900-row evaluation'],cwd=REPO,check=True)
            for phase in ['infer','gt','timing']:command([PYTHON,'scripts/evaluate_senseiver_supervised.py',phase],phase+'.log')
    command([PYTHON,'scripts/report_senseiver_supervised.py'],'report.log')
    summary=dict(formal_complete=result,evaluation_complete=(ART/'evaluation_complete.json').exists())
    (OUT/'lifecycle_complete.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
