"""v2 execution context, reusing frozen v1 utilities without editing them."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'classical_operators_v1'))
from context import *
from scale_context_base import PROJECT,ROOT,ART,sha,read,write
from geophy_set3d.training.classical_operators_v2_scale_init import load_config,build_model as initialized_model

def build_model(c,device=None):
 scale=read(ART/'initialization_calibration.json');arm=c['model']['name'].removesuffix('_v2_scale_init')
 return initialized_model(c,scale['c0'],scale['arms'][arm]['alpha'],device or c['training']['device'])

def verify():
 r=read(ART/'registration.json')
 for p,h in r['source_sha256'].items():assert sha(PROJECT/p)==h,p
 for p,h in r['input_sha256'].items():assert sha(p)==h,p
 return r
