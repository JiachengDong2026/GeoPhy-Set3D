"""Independent U-Net experiment paths and identity utilities."""
import hashlib,json,subprocess,os
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path('/scratch/djc/GeoPhy-Set3D/unet_baseline_v1')
ART=PROJECT/'docs/reports/artifacts/unet_baseline_v1'
V2=Path('/scratch/djc/GeoPhy-Set3D-classical-operators-v2-scale-init')
DATA=Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:
  json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
