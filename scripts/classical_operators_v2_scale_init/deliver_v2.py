"""Seal results after experiments; never changes registration, sources or training."""
import gzip,csv,sys,datetime,shutil
import torch
import numpy as np
from scale_context_base import *

def main():
 original=read(ROOT/'protected_tracked.json')
 for p,h in original.items():assert sha(PROJECT/p)==h,p
 audit=read(ART/'v1_scale_audit.json')
 for p,h in audit['frozen_sha256'].items():assert sha(p)==h,p
 reg=read(ART/'registration.json')
 for p,h in reg['source_sha256'].items():assert sha(PROJECT/p)==h,p
 models={};checkpoints={};comparison=[]
 for arm in ['deeponet','fno']:
  root=ROOT/arm/'pilot';r=read(root/'result.json');ind=read(ART/f'{arm}_independent_review.json')
  assert ind['all_passed'] and r['status']==ind['status']
  updates=[json.loads(s) for s in (root/'actual_updates.jsonl').read_text().splitlines()]
  assert len(updates)==480 and [x['update'] for x in updates]==list(range(1,481))
  assert updates==read(root/'updates.json')
  m={}
  for epoch in range(31):
   p=root/f'epoch{epoch}.pt';entry=read(root/f'epoch{epoch}_checkpoint.json');assert sha(p)==entry['sha256']
   ck=torch.load(p,map_location='cpu',weights_only=False)
   assert ck['epoch']==epoch and ck['optimizer_steps']==16*epoch and ck['scheduler']['last_epoch']==epoch and ck['rng']
   assert all(torch.isfinite(v).all() for v in ck['model'].values())
   if epoch:assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={16*epoch}
   del ck
   m[str(epoch)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,updates=16*epoch)
  checkpoints[arm]=m
  distributions=[]
  for epoch in [0,10,20,29,30]:
   diagnostic=read(root/f'diagnostic_epoch{epoch}/diagnostics.json')
   write(ART/f'{arm}_diagnostic_epoch{epoch}.json',diagnostic)
   for entry in diagnostic['arrays']:
    assert sha(entry['path'])==entry['sha256']
    with np.load(entry['path'],allow_pickle=False) as z:
     for key in z.files:
      if not key.endswith('_volume'):continue
      v=z[key].astype('float64');lg=z[key.replace('_volume','_logits')].astype('float64')
      distributions.append(dict(epoch=epoch,file=entry['path'],key=key,mean=float(v.mean()),std=float(v.std()),min=float(v.min()),max=float(v.max()),nearzero=float((v<1e-10).mean()),spatial_cv=float(v.std()/max(v.mean(),1e-30)),logit_mean=float(lg.mean()),logit_std=float(lg.std()),logit_quantiles=np.quantile(lg,[0,.01,.25,.5,.75,.99,1]).tolist(),logit_lt_minus10=float((lg< -10).mean()),logit_gt10=float((lg>10).mean())))
  write(ART/f'{arm}_full_distributions.json',distributions)
  history=[json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
  with (ART/f'{arm}_curve.csv').open('w') as f:
   writer=csv.DictWriter(f,lineterminator='\n',fieldnames=['epoch','optimizer_steps','train_loss','train_H','val_H','val_mean','val_std']);writer.writeheader()
   for row in history:writer.writerow(dict(epoch=row['epoch'],optimizer_steps=row['optimizer_steps'],train_loss=row['train']['loss']['total'],train_H=row['train']['measurement']['heldout']['relative_l1'],val_H=row['val']['measurement']['heldout']['relative_l1'],val_mean=row['val']['output']['mean'],val_std=row['val']['output']['std']))
  for name in ['actual_updates.jsonl','actual_training_inputs.jsonl','metrics.jsonl']:
   with (root/name).open('rb') as src,gzip.open(ART/f'{arm}_{name}.gz','wb') as dst:shutil.copyfileobj(src,dst)
  for name in ['epoch0.json','access.json']:shutil.copyfile(root/name,ART/f'{arm}_{name}')
  models[arm]=dict(status=r['status'],pilot_epochs=r['epochs'],actual_updates=r['optimizer_steps'],CPU_passed=read(ART/f'{arm}_cpu_tests.json')['passed'],independent_full_fields=ind['recomputed_full_fields'],initial_val_H=r['initial_val_H'],final_val_H=r['last_val_H'],final_checks={str(e):r['final_checks'][str(e)] for e in [29,30]},formal_updates=0)
 write(ART/'checkpoint_manifest.json',checkpoints)
 resources=[]
 for line in (ROOT/'gpu_snapshots.jsonl').read_text().splitlines():
  row=json.loads(line);resources.append({k:row[k] for k in ['at','gpus','gpu_processes']})
 write(ART/'resource_review.json',dict(initial_double_snapshots=resources,scheduler_tasks={arm:read(ROOT/f'scheduling/pilot_{arm}.json') for arm in models},full_local_resource_log=str(ROOT/'scheduling/resources.jsonl'),foreign_processes_terminated=False,serial=True))
 status=dict(status='COMPLETED',branch='experiment/classical_operators_v2_scale_init',base_commit=reg['base_commit'],models=models,c0=reg['c0'],
  protected_original_files=len(original),frozen_checkpoints_verified=len(audit['frozen_sha256']),registered_sources_unchanged=True,
  primary_C0_C1_bitwise=read(ART/'primary_regression.json')['passed'],formal_epochs=0,formal_updates=0,new_formal_predictions=0,GT64_reads=0,GT128_reads=0,Pool_B_reads=0,merged_into_main=False,
  no_scientific_deviations=True,engineering_preregistration_complete=False,input_hash_preregistration_limitation='Val36 projection per-file hashes were omitted from initial registration; post-run verification against frozen build manifest is supplemental, not retrospective preregistration',engineering_amendments=['CPU donor assignment retained batch padding; no update before fix','v2 helper renamed to avoid scheduler import collision; no GPU allocation before fix'],
  checkpoint_manifest_sha256=sha(ART/'checkpoint_manifest.json'),finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 write(ART/'final_status.json',status)
 # Hash every delivered artifact except this self-referential seal.
 write(ART/'delivery_manifest.json',{str(p.relative_to(PROJECT)):sha(p) for p in sorted(ART.iterdir()) if p.is_file() and p.name!='delivery_manifest.json'})
 print(json.dumps(status,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
