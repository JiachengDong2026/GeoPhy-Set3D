"""Seal design before synthetic targets; register everything before updates."""
from ctx_v2 import *
from synthetic_v2 import analytic,reference
from numerics_v2 import project_input,valid
import subprocess

def main():
    ROOT.mkdir(exist_ok=False)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=PROJECT).decode().split('\0')
    old={str(PROJECT/p):sha(PROJECT/p) for p in tracked if p and (PROJECT/p).is_file()}
    anchor=read(PROJECT/'docs/reports/artifacts/deep_baselines_v1/main_anchor.json') if (PROJECT/'docs/reports/artifacts/deep_baselines_v1/main_anchor.json').exists() else read(ROOT.parent/'deep_baselines_v1/main_anchor.json')
    checkpoints={anchor[k]['checkpoint']:sha(anchor[k]['checkpoint']) for k in ('model','reference')}
    freeze(ROOT/'preregistration.json',dict(at=now(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        config=design(),sources=sources(),historical_tracked=old,frozen_main_checkpoints=checkpoints,
        prior_status_sha256=sha(ROOT.parent/'deep_baselines_identifiability_v1/final_status.json'),
        preliminary_access='Only train geometry schema/origins inspected; no real projection getters or measurements used'))
    backend();(ROOT/'inputs').mkdir();precision=[];all_samples=[];identities=[]
    with Boundary('generate') as guard:
        geometry=geometry_only();save_pt(ROOT/'inputs/geometry.pt',geometry)
        base_hash={k:value_hash(nonmeasurement(v)) for k,v in geometry.items()}
        field_hashes={k:{key:tensor_hash(value) if torch.is_tensor(value) else value for key,value in nonmeasurement(v).items()} for k,v in geometry.items()}
        for index,spec in enumerate(design()['synthetic_cases']):
            sample=copy.deepcopy(geometry);measurement_hashes={}
            axis=np.linspace(0,.6,64,dtype=np.float64);xyz=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1)
            truth=analytic(xyz,spec);np.save(ROOT/f'inputs/truth_{index}.npy',truth,allow_pickle=False)
            for branch,inp in sample.items():
                low=reference(inp,spec,128);high=reference(inp,spec,256)
                assert np.abs(high-low).sum()/np.abs(high).sum()<1e-8
                inp['input_projection_raw']=torch.from_numpy(high.astype(np.float32))
                inp['input_projection']=inp['input_projection_raw']/Y_SCALE
                assert value_hash(nonmeasurement(inp))==base_hash[branch]
                measurement_hashes[branch]=tensor_hash(inp['input_projection_raw'])
                target=batch([{branch:inp}],branch)
                with torch.no_grad():p=project_input(torch.from_numpy(truth.astype(np.float32))[None,None],target)
                mask=valid(target);y=target['input_projection_raw']
                precision.append(dict(index=index,branch=branch,reference_relative_l1=float(np.abs(high-low).sum()/np.abs(high).sum()),
                    reference_max_abs_ppm_m=float(np.abs(high-low).max()),
                    grid64_K96_relative_l1=float((p[mask].double()-y[mask].double()).abs().sum()/y[mask].double().abs().sum()),
                    fp32_cast_max_abs_ppm_m=float(np.abs(high-high.astype(np.float32).astype(np.float64)).max())))
            save_pt(ROOT/f'inputs/H_{index}.pt',sample.pop('H'));all_samples.append(sample)
            identities.append(dict(index=index,name=spec['name'],role='seen' if index<6 else 'unseen',geometry=base_hash,
                measurements=measurement_hashes,truth=sha(ROOT/f'inputs/truth_{index}.npy')))
        for branch in ('A','B','H'):assert len({r['measurements'][branch] for r in identities})==8
        assert len({r['truth'] for r in identities})==8
        save_pt(ROOT/'inputs/train_AB.pt',all_samples[:6]);save_pt(ROOT/'inputs/unseen_AB.pt',all_samples[6:])
    states={arm:state_hash(model(arm).state_dict()) for arm in design()['arms']}
    old_states=read(ROOT.parent/'deep_baselines_identifiability_v1/registration.json')['initial_states'];assert states==old_states
    freeze(ROOT/'input_registration.json',dict(at=now(),geometry_file_sha256=sha(GEOMETRY),geometry_field_hashes=field_hashes,
        identity=identities,precision=precision,initial_states=states,inputs={str(p):sha(p) for p in (ROOT/'inputs').iterdir()},
        audit=guard.summary(),all_case_geometry_equal=True,all_measurements_and_truth_distinct=True,optimizer_updates=0))
    print('PREPARED fixed geometry,8 plumes,6/2 split; no real measurements read',flush=True)

if __name__=='__main__':main()
