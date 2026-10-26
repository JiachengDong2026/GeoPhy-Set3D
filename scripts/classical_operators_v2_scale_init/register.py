"""Seal exact sources and inputs before CPU optimization checks and GPU pilot."""
from common import *
def main():
 assert not (ART/'registration.json').exists()
 scale=read(ART/'scale_calibration.json');init=read(ART/'initialization_calibration.json')
 paths=list((PROJECT/'src').rglob('*.py'))+list((PROJECT/'scripts/classical_operators_v2_scale_init').glob('*.py'))
 paths+=list((PROJECT/'scripts/classical_operators_v1').glob('*.py'))+list((PROJECT/'scripts/deep_baselines_v1').glob('*.py'))
 paths+=list((PROJECT/'scripts/plain_final_v1').glob('*.py'))
 paths+=list((PROJECT/'configs/classical_operators_v2_scale_init').glob('*.yaml'))+[PROJECT/'docs/classical_operators_v2_scale_init_protocol_zh.md']
 sources={str(p.relative_to(PROJECT)):sha(p) for p in sorted(set(paths))}
 write(ART/'source_manifest.json',sources)
 reg=dict(base_commit='279814b5152c3916913766e7a0fb28134e7d320d',design_commit=scale['design_commit'],registration_parent_commit=git('rev-parse','HEAD'),
  source_sha256=sources,input_sha256=scale['sources_sha256'],calibration_sha256=sha(ART/'scale_calibration.json'),
  initialization_calibration_sha256=sha(ART/'initialization_calibration.json'),c0=scale['c0'],z0=scale['z0'],
  arms={arm:{k:v for k,v in a.items() if k not in ('before','after')} for arm,a in init['arms'].items()},
  allowed_parameter_changes=dict(deeponet=['bias','branch.2.weight','branch.2.bias'],fno=['readout.2.weight','readout.2.bias']),
  training=dict(epochs=30,steps_per_epoch=16,max_updates=480,train_indices=list(range(32)),val_count=36,batch=2,precision='fp32',seed=20261011,optimizer='AdamW',lr=.0002,weight_decay=.0001,clip=1,cosine_T_max=300),
  gates_protocol_sha256=sha(PROJECT/'docs/classical_operators_v2_scale_init_protocol_zh.md'),resource='Global UUID flock; 15second double idle; bounded60second wait; no foreign termination; serial',
  no_formal=True,no_GT=True,no_main_merge=True)
 write(ART/'registration.json',reg);print('registered',len(sources),'source files',flush=True)
if __name__=='__main__':main()
