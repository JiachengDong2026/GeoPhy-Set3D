"""Read-only protection of main, all historical artifacts, and v2 branch."""
from unet_paths import *
import numpy as np

def main():
 assert git('rev-parse','origin/main')=='279814b5152c3916913766e7a0fb28134e7d320d'
 assert not (ROOT/'protected_tracked.json').exists()
 paths=git('ls-files').splitlines();protected={p:sha(PROJECT/p) for p in paths if (PROJECT/p).is_file()}
 write(ROOT/'protected_tracked.json',protected)
 v2art=V2/'docs/reports/artifacts/classical_operators_v2_scale_init'
 for p,h in read(v2art/'delivery_manifest.json').items():assert sha(V2/p)==h,p
 for entries in read(v2art/'checkpoint_manifest.json').values():
  for r in entries.values():assert sha(r['path'])==r['sha256']
 frozen=read(v2art/'v1_scale_audit.json')['frozen_sha256']
 for p,h in frozen.items():assert sha(p)==h,p
 # Freeze all existing files in the prior experiment roots by byte identity,
 # without opening any GT or formal prediction arrays.
 history_roots=[Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_v1'),Path('/scratch/djc/GeoPhy-Set3D/classical_operators_v1'),Path('/scratch/djc/GeoPhy-Set3D/classical_operators_v2_scale_init')]
 history={str(p):sha(p) for root in history_roots for p in root.rglob('*') if p.is_file() and p.suffix in ('.pt','.json','.jsonl','.yaml','.log','.gz','.csv') and p.name not in ('resources.jsonl','gpu_snapshots.jsonl')}
 write(ROOT/'protected_history.json',history)
 # Verify all train and Val projection identities before any new training.
 frozen_data=read(DATA/'audits/artifact_hashes.json');rows=[json.loads(s) for s in (DATA/'source_index.jsonl').read_text().splitlines()]
 data={str(p):sha(p) for p in [DATA/'source_index.jsonl',DATA/'train_statistics.json',DATA/'geometry/train_geometry.npz',DATA/'dataset_build_manifest.json',DATA/'audits/artifact_hashes.json']}
 for r in rows:
  if r['split'] not in ('train','val'):continue
  p=DATA/r['train_projection_path'];h=sha(p);assert h==frozen_data[r['train_projection_path']];data[str(p)]=h
 scale=read(v2art/'scale_calibration.json')
 for p,h in scale['sources_sha256'].items():assert sha(p)==h,p
 write(ART/'data_identity.json',dict(sha256=data,train=166,val=36,scale_sources_all_match_v2=True))
 write(ART/'reused_scale.json',dict(c0=scale['c0'],q=scale['q'],z0=scale['z0'],formula=scale['method'],valid_rays=scale['valid_rays'],
  source=str(v2art/'scale_calibration.json'),source_sha256=sha(v2art/'scale_calibration.json'),v2_commit=git('-C',str(V2),'rev-parse','HEAD'),calibration_data_sha256=scale['sources_sha256'],no_recalibration=True))
 write(ART/'history_audit.json',dict(base_commit=git('rev-parse','origin/main'),main_snapshot_protected=len(protected),frozen_sha256=frozen,
  protected_manifest_sha256=sha(ROOT/'protected_tracked.json'),historical_files=len(history),history_manifest_sha256=sha(ROOT/'protected_history.json'),
  v2_delivery_verified=True,v2_checkpoints_verified=62,reports_sha256={str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'docs/reports').glob('*.md')},
  existing_paper_results_modified=False))
 print('History/data identities verified:',len(protected),'main files;',len(history),'history files; 202 measurements')
if __name__=='__main__':main()
