"""Independent delivery checks, including GT-before-freeze refusal in subprocess."""
import os,subprocess
from context import *
def main():
    status=read(ART/'experiment_status.json');protected=read(ROOT/'protected_tracked.json')
    assert all(sha(PROJECT/p)==h for p,h in protected.items())
    assert read(ART/'github_main.json')['verified']
    for arm in ['deeponet','fno']:
        assert status[arm]['status']=='NO_GO' and status[arm]['formal_updates']==0 and not status[arm]['GT64_evaluated']
        assert read(ART/(arm+'_independent_review.json'))['all_passed']
    manifest=read(ART/'checkpoint_manifest.json');assert all(sha(x['path'])==x['sha256'] for x in manifest.values())
    for p,h in read(ART/'history_audit.json')['evidence_sha256'].items():assert sha(p)==h
    source=read(ART/'source_manifest_deeponet_pilot.json');ck=torch.load(ROOT/'deeponet/pilot/last.pt',map_location='cpu',weights_only=False)
    assert ck['source_manifest_sha256']==sha(ART/'source_manifest_deeponet_pilot.json')
    fno=torch.load(ROOT/'fno/pilot/last.pt',map_location='cpu',weights_only=False)
    assert fno['source_manifest_sha256']==sha(ART/'source_manifest.json')
    # Refuse loading checkpoint/measurements/GT before any dataset tensor read.
    snippet=r'''
import sys,runpy,json
from pathlib import Path
seen=[]
def audit(event,args):
 if event=='open' and isinstance(args[0],str) and args[0].endswith(('.npy','.npz','.pt','.pth')):
  seen.append(args[0]);raise RuntimeError('No tensor reads permitted in incomplete-formal boundary test')
sys.addaudithook(audit)
sys.path.insert(0,'scripts/classical_operators_v1')
sys.argv=['evaluate.py','gt','deeponet']
try:runpy.run_path('scripts/classical_operators_v1/evaluate.py',run_name='__main__')
except FileNotFoundError as e:
 assert 'training_complete.json' in str(e)
else:raise AssertionError('Incomplete formal evaluation must refuse')
assert not seen
print(json.dumps(dict(blocked_before_any_checkpoint_measurement_or_GT_tensor=True,tensor_open_attempts=seen)))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(PROJECT/'src'),OMP_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-c',snippet],cwd=PROJECT,env=env,text=True,capture_output=True,check=True)
    boundary=json.loads(result.stdout.strip().splitlines()[-1])
    out=dict(all_passed=True,protected_tracked_files=len(protected),new_formal_updates=0,no_failed_model_in_formal_ranking=True,
        all_checkpoint_hashes_verified=True,pilot_source_manifests_verified=True,github_main_verified=True,
        evaluator_boundary=boundary,GT_reads=0,new_updates=0)
    write(ART/'delivery_review.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
