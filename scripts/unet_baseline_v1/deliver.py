"""Seal U-Net results and verify historical byte identity after training."""
import gzip,csv,shutil,sys,datetime
from unet_paths import *
sys.path.insert(0,str(PROJECT/'src'))
import torch,numpy as np

def main():
 root=ROOT/'unet/pilot';result=read(root/'result.json');review=read(ART/'unet_independent_review.json')
 assert review['all_passed'] and review['status']==result['status']
 assert read(ART/'unet_numpy_projected_interventions.json')['all_intervention_gates_match']
 assert read(ART/'full_validation_independent_review.json')['passed'] and read(ART/'full_validation_independent_review.json')['gate_A_matches']
 protected={}
 for name in ['protected_tracked.json','protected_history.json','protected_external.json']:
  m=read(ROOT/name)
  for p,h in m.items():assert sha(PROJECT/p if name=='protected_tracked.json' else p)==h,p
  protected[name]=dict(count=len(m),sha256=sha(ROOT/name))
 history=read(ART/'history_audit.json')
 for p,h in history['frozen_sha256'].items():assert sha(p)==h,p
 # v2 source/artifact files and every saved checkpoint remain frozen.
 for p,h in read(V2/'docs/reports/artifacts/classical_operators_v2_scale_init/delivery_manifest.json').items():assert sha(V2/p)==h,p
 registration=read(ART/'registration.json')
 for p,h in registration['source_sha256'].items():assert sha(PROJECT/p)==h,p
 for p,h in registration['data_sha256'].items():assert sha(p)==h,p
 checkpoints={}
 for epoch in range(31):
  p=root/f'epoch{epoch}.pt';entry=read(root/f'epoch{epoch}_checkpoint.json');assert sha(p)==entry['sha256']
  ck=torch.load(p,map_location='cpu',weights_only=False)
  assert ck['epoch']==epoch and ck['optimizer_steps']==16*epoch and ck['scheduler']['last_epoch']==epoch and ck['rng']
  if epoch:assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={16*epoch}
  assert all(torch.isfinite(v).all() for v in ck['model'].values());del ck
  checkpoints[str(epoch)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,updates=16*epoch)
 write(ART/'checkpoint_manifest.json',checkpoints)
 journal=[json.loads(s) for s in (root/'actual_updates.jsonl').read_text().splitlines()]
 assert len(journal)==480 and [r['update'] for r in journal]==list(range(1,481));assert journal==read(root/'updates.json')
 assert all(r['postclip_gradient_norm']>0 and r['actual_parameter_update_l2']>0 for r in journal)
 distributions=[]
 for epoch in [0,10,20,29,30]:
  diagnostic=read(root/f'diagnostic_epoch{epoch}/diagnostics.json');write(ART/f'diagnostic_epoch{epoch}.json',diagnostic)
  for entry in diagnostic['arrays']:
   assert sha(entry['path'])==entry['sha256']
   with np.load(entry['path'],allow_pickle=False) as z:
    for key in z.files:
     if not key.endswith('_volume'):continue
     v=z[key].astype('float64');lg=z[key.replace('_volume','_logits')].astype('float64')
     distributions.append(dict(epoch=epoch,file=entry['path'],key=key,mean=float(v.mean()),std=float(v.std()),min=float(v.min()),max=float(v.max()),spatial_cv=float(v.std()/max(v.mean(),1e-30)),nearzero=float((v<1e-10).mean()),saturated=float((np.abs(lg)>10).mean()),logit_mean=float(lg.mean()),logit_std=float(lg.std()),logit_quantiles=np.quantile(lg,[0,.01,.25,.5,.75,.99,1]).tolist()))
 write(ART/'full_distributions.json',distributions)
 rows=[json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
 with (ART/'curve.csv').open('w') as f:
  writer=csv.DictWriter(f,lineterminator='\n',fieldnames=['epoch','updates','train_loss','train_H','val_H','val_mean','val_std']);writer.writeheader()
  for r in rows:writer.writerow(dict(epoch=r['epoch'],updates=r['optimizer_steps'],train_loss=r['train']['loss']['total'],train_H=r['train']['measurement']['heldout']['relative_l1'],val_H=r['val']['measurement']['heldout']['relative_l1'],val_mean=r['val']['output']['mean'],val_std=r['val']['output']['std']))
 for name in ['metrics.jsonl','actual_updates.jsonl','actual_training_inputs.jsonl']:
  with (root/name).open('rb') as src,gzip.open(ART/(name+'.gz'),'wb') as dst:shutil.copyfileobj(src,dst)
 for name in ['epoch0.json','access.json','gpu_memory_check.json']:shutil.copyfile(root/name,ART/name)
 resources=[]
 for line in (ROOT/'gpu_snapshots.jsonl').read_text().splitlines():
  row=json.loads(line);resources.append({k:row[k] for k in ['at','gpus','gpu_processes']})
 write(ART/'resource_execution.json',dict(double_idle_snapshots=resources,task=read(ROOT/'scheduling/pilot_unet.json'),full_local_resources=str(ROOT/'scheduling/resources.jsonl'),foreign_processes_terminated=False,resource_configuration_reduced=False))
 status=dict(status=result['status'],model='standard_3d_unet_v1',branch='experiment/3d_unet_baseline_v1',base_commit=registration['base_commit'],parameters=result['parameters'],
  epochs=30,actual_pilot_updates=480,actual_CPU_test_updates=1,actual_GPU_memory_check_updates=0,CPU_passed=True,primary_C0_C1_bitwise=True,independent_full_fields=review['recomputed_full_fields'],independent_NumPy_integrals=True,additional_full_Val_independent_fields=216,full_Val_gate_A_verified=True,
  c0=registration['c0'],alpha=registration['alpha'],initial_val_H=result['initial_val_H'],epoch29_val_H=rows[28]['val']['measurement']['heldout']['relative_l1'],epoch30_val_H=result['last_val_H'],last_checks={str(e):result['final_checks'][str(e)] for e in [29,30]},
  histories_protected=protected,original_main_files_unchanged=True,v2_branch_unchanged=True,data_identity_reverified=True,registration_scientific_sources_unchanged=True,
  checkpoint_manifest_sha256=sha(ART/'checkpoint_manifest.json'),last_checkpoint_sha256=result['checkpoint_sha256'],best_checkpoint_sha256=result['pilot_best_sha256'],
  formal_epochs=0,formal_updates=0,new_formal_predictions=0,GT64_reads=0,GT128_reads=0,PoolB_reads=0,test_truth_reads=0,merged_into_main=False,scientific_deviations=[],
  engineering_incidents=['Initial YAML generation used system python lacking PyYAML; configuration absent before model initialization, fixed using frozen environment and original log preserved'],finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 write(ART/'final_status.json',status);print(json.dumps(status,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
