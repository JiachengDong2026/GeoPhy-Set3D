"""Freeze existing predictions and Senseiver checkpoint/logs without GT access."""
from unet_paths import *
def main():
 ds=PROJECT/'docs/reports/artifacts/deepsets_selfsup_v1';seal=read(ds/'predictions_complete.json')
 out=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/deepsets_selfsup_v1');protected={}
 for token,h in seal['prediction_rows_sha256'].items():
  p=out/'prediction_rows'/f'{token}.json';assert sha(p)==h;protected[str(p)]=h
  row=read(p);p=Path(row['prediction_file']);assert sha(p)==row['prediction_sha256'];protected[str(p)]=row['prediction_sha256']
 for root in [Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/ray_senseiver_selfsup_v1'),Path('/data/djc/GeoPhy-Set3D/outputs/checkpoints/ray_senseiver_selfsup_v1'),Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/deepsets_selfsup_v1'),Path('/data/djc/GeoPhy-Set3D/outputs/checkpoints/deepsets_selfsup_v1')]:
  for p in root.rglob('*'):
   if p.is_file() and p.suffix in ('.pt','.yaml','.json','.jsonl','.log','.gz','.csv'):protected[str(p)]=sha(p)
 assert seal['rows']==900
 write(ROOT/'protected_external.json',protected)
 write(ART/'external_freeze.json',dict(existing_deepsets_predictions=900,protected_count=len(protected),manifest=str(ROOT/'protected_external.json'),sha256=sha(ROOT/'protected_external.json'),GT_opened=False))
 print('External formal predictions/checkpoints/logs protected:',len(protected),'files')
if __name__=='__main__':main()
