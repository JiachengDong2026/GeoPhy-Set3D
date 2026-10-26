"""Independent v2 audit paths; no historical file mutation."""
import hashlib,json,subprocess
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path('/scratch/djc/GeoPhy-Set3D/classical_operators_v2_scale_init')
ART=PROJECT/'docs/reports/artifacts/classical_operators_v2_scale_init'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
