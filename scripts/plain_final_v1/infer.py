"""Only missing main-model forward calls, measurement diagnostics, then global seal."""
from context import *


def diagnostics(fields,geometry,projection,setting,case,device):
    out={};operators={}
    for label,ids in [('observed',setting['input_ids'][case]),('heldout',setting['heldout_ids'][case])]:
        g=geometry_tensor({k:v[ids] for k,v in geometry.items() if k not in ('horizontal_angle_deg','vertical_angle_deg')},device)
        operators[label]=(CachedProjector(g),g,torch.as_tensor(np.array(projection[ids]),device=device))
    for stage,c in fields.items():
        x=torch.as_tensor(c,device=device,dtype=torch.float32);out[stage]={}
        for label,(operator,g,y) in operators.items():out[stage][label]=relative_projection_metrics(operator(x),y,g['valid_ray'])
    return out


def main():
    backend();lock=verified(BASE/'protocol.json');verify_hashes(lock['sources']);assert verified(BASE/'checks/cpu.json')['passed']
    mm=verified(BASE/'model_manifest.json');m=mm['models'][0];assert m['key']==MAIN and sha(m['checkpoint'])==m['checkpoint_sha256']
    settings=verified(BASE/'layouts.json');cases=verified(BASE/'cases.json');inputs=verified(BASE/'input_manifest.json')
    reused=verified(BASE/'reuse_predictions.json');old=[r for r in reused if r['method']==MAIN]
    lookup={(r['condition'],r['draw'],r['case_id']):r for r in old};inp={(r['condition'],r['draw'],r['case_id']):r for r in inputs}
    root=BASE/'inference';assert not root.exists(),'Partial inference requires audited resume; no silent replay'
    root.mkdir();(root/'predictions').mkdir();predictions=[];measurements=[];gpu_checks=[];archived={(r['method'],r['condition'],r['draw'],r['case_id']):r for r in rows(HISTORY/'evaluation/test/metrics.jsonl') if r['method'] in CLASSICS}
    with io_guard(splits=['test','val'],predictions=reused,checkpoints=[m['checkpoint']]) as guard,torch.inference_mode():
        model=model_for(m['config'],PLAIN,'cuda:0').eval().requires_grad_(False)
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);del ck
        initial=canonical_worker.probe.state_hash(model.state_dict());assert initial==m['state_sha256']
        b=batch_for(dataset('val',verified(HISTORY/'layouts.json')['val'][0]),0,'cuda:0');st=model(b,return_stages=True)
        assert all(st[k].shape==(1,1,64,64,64) and torch.isfinite(st[k]).all() and st[k].min()>=0 and st[k].max()<=1 for k in ('C0','C1'));del st,b
        with np.load(DATA/'geometry/train_geometry.npz',allow_pickle=False) as z:geometry={k:z[k] for k in z.files}
        # Fixed historical case/draw0 across every layout class, all four methods;
        # validates diagnostics reuse without solving any inverse problem.
        for s in [x for x in settings if x['draw']==0]:
            case=cases[0];source=next(r for r in rows(DATA/'source_index.jsonl') if r['case_id']==case)
            projection=np.load(DATA/source['train_projection_path'],allow_pickle=False)
            for method in CLASSICS:
                r=next(r for r in reused if (r['method'],r['condition'],r['draw'],r['case_id'])==(method,s['condition'],s['draw'],case))
                with np.load(r['prediction'],allow_pickle=False) as z:c=np.array(z['C1'])
                diag=diagnostics({'C1':c},geometry,projection,s,case,'cuda:0')['C1'];expected=archived[method,s['condition'],s['draw'],case]['measurement']
                difference=max(abs(diag[label][k]-expected[label][k]) for label in diag for k in diag[label]);assert difference<1e-7,(method,difference)
                ids=s['input_ids'][case];g=geometry_tensor({k:v[ids] for k,v in geometry.items() if k not in ('horizontal_angle_deg','vertical_angle_deg')},'cuda:0')
                torch.testing.assert_close(CachedProjector(g)(torch.as_tensor(c,device='cuda:0')),project(torch.as_tensor(c,device='cuda:0'),g),rtol=1e-6,atol=.02)
                gpu_checks.append(dict(method=method,condition=s['condition'],draw=0,case_id=case,max_measurement_difference=difference))
        freeze(BASE/'checks/gpu.json',dict(at=now(),passed=True,device=torch.cuda.get_device_name(),uuid=os.environ['CUDA_VISIBLE_DEVICES'],
            parameter_state_unchanged=initial==canonical_worker.probe.state_hash(model.state_dict()),classic_diagnostic_spotchecks=gpu_checks,no_solver_runs=True,GT64_read=False))
        source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')}
        for s in settings:
            ds=dataset('test',s)
            for i,c in enumerate(cases):
                key=(s['condition'],s['draw'],c);identity=inp[key]
                if key in lookup:
                    r=dict(lookup[key]);assert sha(r['prediction'])==r['prediction_sha256']
                    with np.load(r['prediction'],allow_pickle=False) as z:ff={k:np.array(z[k]) for k in ('C0','C1')}
                else:
                    b=batch_for(ds,i,'cuda:0');assert input_hash(b)==identity['input_sha256']
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                    stages=model(b,return_stages=True);torch.cuda.synchronize();seconds=time.perf_counter()-start
                    ff={k:stages[k][0,0].cpu().numpy().copy() for k in ('C0','C1')}
                    path=root/'predictions'/f"{s['condition']}_d{s['draw']}_{i:03d}.npz";assert not path.exists()
                    np.savez_compressed(path,**ff);check_fields(path,['C0','C1'])
                    r=dict(identity,prediction=str(path),prediction_sha256=sha(path),method=MAIN,checkpoint_sha256=m['checkpoint_sha256'],
                        seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],reused=False,forward_calls=1)
                    del stages,b
                p=np.load(DATA/source[c]['train_projection_path'],allow_pickle=False)
                diag=diagnostics(ff,geometry,p,s,c,'cuda:0')
                measurements.extend(dict(method=MAIN,case_id=c,condition=s['condition'],draw=s['draw'],stage=k,measurement=v) for k,v in diag.items())
                predictions.append(r)
                append(root/'manifest.jsonl',r);append(root/'measurement.jsonl',measurements[-2:][0]);append(root/'measurement.jsonl',measurements[-1])
                del ff,p,diag
            print(s['condition'],s['draw'],len(predictions),'/595; new',sum(not r['reused'] for r in predictions),flush=True)
        assert len(predictions)==595 and sum(not r['reused'] for r in predictions)==385
        assert canonical_worker.probe.state_hash(model.state_dict())==initial and all(p.grad is None for p in model.parameters())
        allpred=predictions+[r for r in reused if r['method'] in CLASSICS];assert len(allpred)==2975
        for r in allpred:assert sha(r['prediction'])==r['prediction_sha256']
        assert not guard.denied
    freeze(root/'complete.json',dict(at=now(),count=595,new=385,reused=210,manifest_sha256=sha(root/'manifest.jsonl'),measurement_sha256=sha(root/'measurement.jsonl'),
        state_unchanged=True,no_grad=True,audit=guard.summary()))
    freeze(BASE/'prediction_manifest.json',allpred)
    freeze(BASE/'all_predictions_sealed.json',dict(at=now(),count=2975,fields=3570,new_main=385,reused_main=210,reused_classic=2380,
        prediction_manifest_sha256=sha(BASE/'prediction_manifest.json'),model_manifest_sha256=sha(BASE/'model_manifest.json'),
        comparison_manifest_sha256=sha(BASE/'comparison_manifest.json'),measurement_sha256=sha(root/'measurement.jsonl'),
        all_predictions_saved_before_GT64=True,GT64_read=False,GT128_read=False,Pool_B_read=False,model_state_unchanged=True,audit=guard.summary()))
    print('SEALED; GT64 scoring now permitted only for35 registered cases',flush=True)


if __name__=='__main__':main()
