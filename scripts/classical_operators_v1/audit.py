"""Read-only freeze of historical evidence; never reads GT arrays."""
import hashlib,json,subprocess,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path('/scratch/djc/GeoPhy-Set3D/classical_operators_v1')
ART=PROJECT/'docs/reports/artifacts/classical_operators_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 import torch
 ROOT.mkdir(exist_ok=False)
 hist=PROJECT/'docs/reports/artifacts/deepsets_selfsup_v1'
 freeze=read(hist/'frozen_best.json');assert sha(freeze['checkpoint'])==freeze['sha256']
 ck=torch.load(freeze['checkpoint'],map_location='cpu',weights_only=False)
 assert ck['epoch']==291 and not ck['volumetric_gt_used']
 assert read(hist/'training_complete.json')['optimizer_steps']==24900
 assert read(hist/'independent_audit.json')['passed'] and read(hist/'integrity.json')['passed']
 out=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/deepsets_selfsup_v1')
 seal=read(hist/'predictions_complete.json');assert seal['rows']==900
 rows=[]
 for token,h in seal['prediction_rows_sha256'].items():
  p=out/'prediction_rows'/f'{token}.json'; assert sha(p)==h
  r=read(p);assert sha(r['prediction_file'])==r['prediction_sha256'];rows.append(r)
 assert len(rows)==900
 a=read('/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/main_anchor.json')
 for m in (a['model'],a['reference']):assert sha(m['checkpoint'])==m['checkpoint_sha256']
 protected={p:sha(PROJECT/p) for p in subprocess.check_output(['git','ls-files','-z'],cwd=PROJECT).decode().split('\0') if p and (PROJECT/p).is_file()}
 write(ROOT/'protected_tracked.json',protected)
 artifacts={str(p):sha(p) for p in hist.iterdir() if p.is_file()}
 artifacts.update({freeze['checkpoint']:freeze['sha256'],a['model']['checkpoint']:a['model']['checkpoint_sha256'],a['reference']['checkpoint']:a['reference']['checkpoint_sha256']})
 queue=Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/queue_status.json');artifacts[str(queue)]=sha(queue)
 write(ART/'history_audit.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
  github_main_verified=False,github_issue='configured proxies 7897 refused; 17890/direct bounded verification attempted separately',
  deepsets=freeze,training_complete=read(hist/'training_complete.json'),independent_review=read(hist/'independent_audit.json'),
  predictions=dict(count=900,all_prediction_row_and_array_hashes_verified=True,seal_sha256=sha(hist/'predictions_complete.json'),station_manifest_sha256=sha(hist/'station_subsets.json')),
  current_primary={k:v for k,v in a['model'].items() if k in ['key','config','checkpoint','checkpoint_sha256','parameters','best_epoch']},
  historical_reference_checkpoint=read(hist/'protocol.json')['reference_checkpoint'],
  comparison_warning='Historical DeepSets table refers to SelfSup-MRGR v1, not current Plain primary. Reuse its checkpoint/predictions only; no relabelled primary scores.',
  protected_file_count=len(protected),protected_manifest_sha256=sha(ROOT/'protected_tracked.json'),evidence_sha256=artifacts,new_updates=0))
 print('Historical freeze verified: 900 arrays; 300 epochs / 24900 updates; protected files',len(protected),flush=True)
if __name__=='__main__':main()
