"""Read-only CPU replay of own epoch0/29/30 checkpoints; no optimizer exists."""
from unet_context import *

def main():
 verify();setup();c=load_config();target=ROOT/'unet/full_validation';target.mkdir(exist_ok=False)
 guard=GTReadGuard(DATA,ROOT/'unet');manifest={}
 history=[json.loads(s) for s in (ROOT/'unet/pilot/metrics.jsonl').read_text().splitlines()]
 initial=read(ROOT/'unet/pilot/epoch0.json')
 with guard,torch.no_grad():
  ds=make_dataset(c,validation=True);model=build_model(c,'cpu').eval()
  for epoch in [0,29,30]:
   ckpath=ROOT/f'unet/pilot/epoch{epoch}.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
   assert ck['epoch']==epoch and ck['optimizer_steps']==16*epoch and not ck['volumetric_gt_used']
   model.load_state_dict(ck['model']);before=state_hash(model.state_dict());entries=[];scores=[]
   directory=target/f'epoch{epoch}';directory.mkdir()
   for i in range(0,36,2):
    samples=[ds[i],ds[i+1]];b=dual_measurement_collate(samples)
    h=b['H'];mask=h['input_valid_ray'] & h['view_mask'][:,:,None,None,None]
    arrays={};metrics=[]
    for branch in ['A','B']:
     volume,logits=model(b[branch],return_logits=True);projection=project_input(volume,h)
     for offset in [0,1]:
      m=relative_projection_metrics(projection[offset],h['input_projection_raw'][offset],mask[offset]);scores.append(m['relative_l1']);metrics.append(dict(index=i+offset,branch=branch,**m))
     arrays[branch+'_volume']=volume.numpy();arrays[branch+'_logits']=logits.numpy();arrays[branch+'_H_projection']=projection.numpy()
    for key in ['input_station_origin_m','input_ray_direction','input_t_near','input_path_length_m','input_projection_raw']:
     arrays['H_'+key]=h[key].numpy()
    arrays['H_mask']=mask.numpy();path=directory/f'val_{i}_{i+1}.npz';np.savez_compressed(path,**arrays)
    entries.append(dict(path=str(path),sha256=sha(path),metrics=metrics,case_ids=samples[0]['A']['case_id']+';'+samples[1]['A']['case_id']))
   logged=initial['val']['measurement']['heldout']['relative_l1'] if epoch==0 else history[epoch-1]['val']['measurement']['heldout']['relative_l1']
   replay=float(np.mean(scores));assert np.isclose(logged,replay,rtol=2e-6,atol=2e-7),(epoch,logged,replay)
   assert state_hash(model.state_dict())==before
   manifest[str(epoch)]=dict(checkpoint=str(ckpath),sha256=sha(ckpath),arrays=entries,CPU_replay_H=replay,GPU_logged_H=logged,within_backend_tolerance=True,model_hash=before)
   print('CPU full Val replay',epoch,'H',replay,'logged',logged,flush=True)
  access=guard.summary();no_forbidden(access)
 write(target/'manifest.json',manifest);write(ART/'full_validation_manifest.json',dict(epochs=manifest,access=access,new_updates=0,optimizer_exists=False))
if __name__=='__main__':main()
