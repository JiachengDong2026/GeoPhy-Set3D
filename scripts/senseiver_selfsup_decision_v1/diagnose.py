"""Zero-update full-field interventions and official full-Val selector replay."""
import time
import numpy as np
from audit import *
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.selfsup import (make_dataset, make_loader, move_dual_batch,
    assert_disjoint, run_epoch)
from geophy_set3d.training.ray_senseiver import check_prediction, tensor_identity
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics

FIELDS = {'input_projection_raw', 'input_projection'}

class Guard(GTReadGuard):
    def __init__(self, root):
        super().__init__(root, CK)
        records=[json.loads(s) for s in (self.root/'source_index.jsonl').read_text().splitlines()]
        trains=[r for r in records if r['split']=='train'][:4]
        val=[r for r in records if r['split']=='val']
        allowed={str(self.root/r['train_projection_path']) for r in trains+val}
        self.allowed={p:i for p,i in self.allowed.items() if p in allowed}

    def _audit(self,event,args):
        if self.active and event=='open' and isinstance(args[0],(str,bytes)):
            import os
            path=Path(os.fsdecode(args[0])).absolute()
            mode,flags=args[1:3]
            reading=('r' in mode or '+' in mode) if isinstance(mode,str) else not bool(flags&os.O_WRONLY)
            if reading and str(path).endswith(('.pt','.pth','.ckpt')) and path!=CK/'last.pt':
                raise PermissionError('Only registered selfsup last checkpoint allowed')
            if reading and path.suffix=='.npz' and path!=self.root/'geometry/train_geometry.npz' and not path.is_relative_to(ROOT):
                raise PermissionError('Foreign geometry or arrays forbidden')
        super()._audit(event,args)

def frozen_identity(batch):
    return tensor_identity(*[batch[k] for k in sorted(batch) if isinstance(batch[k],torch.Tensor) and k not in FIELDS])

def main():
    target=ART/'final_checkpoint_diagnostic.json'
    assert not target.exists(), 'Do not repeat completed diagnostic'
    audit=read(ART/'history_audit.json');c=audit['config']
    assert all(sha(p)==h for p,h in audit['evidence_sha256'].items())
    # The committed protocol and code identities precede any model execution.
    reg=dict(protocol_sha256=sha(PROJECT/'docs/senseiver_selfsup_decision_v1_protocol_zh.md'),
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        config=c,checkpoint=audit['checkpoints']['last'],optimizer_created=False,new_updates=0,
        targets=[0,1,2],donors=[1,2,3],branches=['A','B'],modes=['normal','zero','wrong_case'])
    write(ROOT/'execution_registration.json',reg)
    torch.set_num_threads(2);seed_everything(c['training']['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    started=time.monotonic();rows=[];array_manifest={}
    guard=Guard(c['dataset']['root'])
    try:
        with guard:
            saved=torch.load(CK/'last.pt',map_location='cpu',weights_only=False)
            model=build_comparison_model(c,'cuda:0');model.load_state_dict(saved['model']);del saved
            before=state_hash(model.state_dict());model.eval()
            assert before==audit['checkpoints']['last']['model_state_sha256']
            ds=make_dataset(c);ds.set_epoch(0)
            with torch.inference_mode():
                for index,donor in zip([0,1,2],[1,2,3]):
                    sample=ds[index];cpu=dual_measurement_collate([sample]);assert_disjoint(cpu)
                    batch=move_dual_batch(cpu,'cuda:0');arrays={};case_rows=[]
                    for branch in ['A','B']:
                        original=batch[branch];identity=frozen_identity(original)
                        donor_row, donor_p=ds._projections(donor)
                        aligned=dict(sample[branch]);aligned.update(ds._views(donor_p,sample[branch]['input_station_indices'].numpy(),'input'))
                        aligned=projection_pool_collate([aligned])
                        assert frozen_identity(aligned)==frozen_identity(cpu[branch])
                        predicted={}
                        for mode in ['normal','zero','wrong_case']:
                            b=dict(original)
                            if mode!='normal':
                                for k in FIELDS:
                                    b[k]=torch.zeros_like(original[k]) if mode=='zero' else aligned[k].to('cuda:0')
                            assert frozen_identity(b)==identity
                            assert all(b[k] is v for k,v in original.items() if k not in FIELDS)
                            logits=[]
                            hook=model.decoder.postproc.register_forward_hook(lambda m,a,y:logits.append(y.detach().cpu()))
                            try:
                                pred=check_prediction(model(b),b)
                            finally:
                                hook.remove()
                            logit=torch.cat(logits,1).numpy();out=pred.detach().cpu().numpy()
                            assert np.isfinite(logit).all()
                            h=batch['H'];valid=h['input_valid_ray'].bool()&h['view_mask'][:,:,None,None,None]
                            proj=project_input(pred,h)
                            metrics=relative_projection_metrics(proj,h['input_projection_raw'],valid)
                            prefix=branch+'_'+mode
                            arrays[prefix+'_volume']=out
                            arrays[prefix+'_logit']=logit
                            arrays[prefix+'_H_projection']=proj.cpu().numpy()
                            arrays[prefix+'_H_reference']=h['input_projection_raw'].cpu().numpy()
                            arrays[prefix+'_H_mask']=valid.cpu().numpy()
                            predicted[mode]=out
                            case_rows.append(dict(index=index,case_id=sample[branch]['case_id'],donor_index=donor,
                                donor_case_id=donor_row['case_id'],branch=branch,mode=mode,geometry_mask_path_unchanged=True,
                                frozen_geometry_sha256=identity,station_ids={k:sample[k]['input_station_indices'].tolist() for k in ['A','B','H']},
                                output_mean=float(out.astype('float64').mean()),output_std=float(out.astype('float64').std()),
                                output_min=float(out.min()),output_max=float(out.max()),
                                spatial_neighbor_MA=[float(np.abs(np.diff(out.astype('float64'),axis=j)).mean()) for j in [2,3,4]],
                                logit_mean=float(logit.astype('float64').mean()),logit_std=float(logit.astype('float64').std()),
                                logit_fraction_below_minus10=float((logit < -10).mean()),H=metrics))
                        rel=float(np.abs(predicted['normal'].astype('float64')-predicted['zero']).sum()/max(np.abs(predicted['normal'].astype('float64')).sum(),1e-30))
                        for r in case_rows:
                            if r['branch']==branch:r['normal_zero_relative_MA']=rel
                    path=ROOT/f'case{index}.npz';np.savez_compressed(path,**arrays);array_manifest[str(path)]=sha(path)
                    rows.extend(case_rows);print('completed fixed case',index,flush=True)
                # Exact original objective/selector; only worker count changed for traceability.
                val_c=json.loads(json.dumps(c));val_c['loader'].update(num_workers=0,persistent_workers=False)
                val=make_dataset(c,validation=True)
                validation=run_epoch(model,make_loader(val,val_c),c,optimizer=None)
            after=state_hash(model.state_dict())
            assert before==after and all(p.grad is None for p in model.parameters())
            logged=json.loads((HIST/'metrics.jsonl').read_text().splitlines()[-1])['val']
            assert validation['case_station_digest']==logged['case_station_digest'] and validation['sample_count']==36
            assert abs(validation['measurement']['heldout']['relative_l1']-logged['measurement']['heldout']['relative_l1'])<1e-7
            access=guard.summary();access['policy']='registered first4 train +36 Val Pool A; exact selfsup last.pt; own NPZ; no volumetric GT/PoolB/test'
            assert not any(access[k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
            write(target,dict(registration=reg,rows=rows,validation_replay=validation,validation_logged=logged,
                access=access,model_hash_before=before,model_hash_after=after,parameter_grads_all_none=True,
                optimizer_created=False,new_updates=0,full_field_forwards=18,validation_cases=36,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
                saved_arrays_sha256=array_manifest))
    finally:
        write(ROOT/'diagnostic_access.json',guard.summary())

if __name__=='__main__':
    main()
