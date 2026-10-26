"""Seal scientific and execution inputs before CPU optimization and GPU pilot."""
from unet_paths import *
def main():
 assert not (ART/'registration.json').exists()
 design=read(ART/'design_registration.json');assert sha(PROJECT/'docs/unet_baseline_v1_protocol_zh.md')==design['protocol_sha256']
 init=read(ART/'initialization.json');assert init['c0']==read(ART/'reused_scale.json')['c0']
 paths=list((PROJECT/'src').rglob('*.py'))+list((PROJECT/'scripts/unet_baseline_v1').glob('*.py'))
 paths+=list((PROJECT/'scripts/deep_baselines_v1').glob('*.py'))+list((PROJECT/'scripts/plain_final_v1').glob('*.py'))
 paths+=[PROJECT/'scripts/train_deepsets_selfsup.py',PROJECT/'configs/unet_baseline_v1/unet.yaml',PROJECT/'docs/unet_baseline_v1_protocol_zh.md']
 sources={str(p.relative_to(PROJECT)):sha(p) for p in sorted(set(paths))}
 data=read(ART/'data_identity.json')['sha256']
 for p,h in data.items():assert sha(p)==h,p
 write(ART/'source_manifest.json',sources)
 write(ART/'registration.json',dict(design_commit=git('rev-parse','HEAD'),base_commit=git('rev-parse','origin/main'),source_sha256=sources,data_sha256=data,
  initialization_sha256=sha(ART/'initialization.json'),initial_parameter_sha256=init['initialized_state_sha256'],c0=init['c0'],alpha=init['alpha'],parameters=init['parameters'],
  input_adapter=dict(source='src/geophy_set3d/models/baseline_fno_v1.py',sha256=sha(PROJECT/'src/geophy_set3d/models/baseline_fno_v1.py'),channels=['Tt_y_over_Tt_1','log1p_Tt_1','support','X','Y','Z']),
  protocol_sha256=sha(PROJECT/'docs/unet_baseline_v1_protocol_zh.md'),objective_audit_sha256=sha(ART/'objective_direction_audit.json'),
  budget=dict(train_indices=list(range(32)),val_count=36,epochs=30,updates_per_epoch=16,max_updates=480),
  memory_estimate=dict(parameter_bytes=4*init['parameters'],AdamW_parameter_grad_moment_bytes=16*init['parameters'],conservative_activation_workspace_GiB=12,actual_batch2_forward_backward_check_required=True),
  registration_complete_before_any_optimizer_update=True,model_parameters_only_from_fresh=True,no_search=True,no_formal=True,no_main_merge=True))
 print('Registered',len(sources),'sources;',len(data),'data inputs; initial hash',init['initialized_state_sha256'])
if __name__=='__main__':main()
