"""Zero-update information propagation and saturation diagnostics."""
from diagnostics import *
from trace import Trace,layer_gradients,finite_difference
import argparse

def run(arm):
    p=verified('registration.json');c.verify_hashes(p['sources']);c.backend()
    target=ROOT/'C'/arm;target.mkdir(parents=True,exist_ok=False)
    states=checkpoint_states(arm);checkpoints=[s['path'] for s in states if s['path']]
    results=[];seen={}
    with TrainGuard(checkpoints) as access:
        samples=torch.load(ROOT/'inputs/real_AB.pt',map_location='cpu',weights_only=False)
        for state in states:
            net=model(arm,'cuda:0',original=state['original'])
            if state['path']:
                ck=torch.load(state['path'],map_location='cpu',weights_only=False);net.load_state_dict(ck['model'],strict=True);del ck
            identity=state_hash(net.state_dict());net.train()
            state_result=dict(state=state,checkpoint_sha256=None if state['path'] is None else c.sha(state['path']),
                state_sha256=identity,optimizer_updates=0,cases=[])
            if identity in seen:
                state_result.update(reused_identical_state=seen[identity],cases=next(x for x in results if x['state']['name']==seen[identity])['cases'])
                results.append(state_result);del net;torch.cuda.empty_cache();continue
            seen[identity]=state['name']
            for index in design()['C_case_indices']:
                ab={k:batch([samples[index]],k) for k in ('A','B')}
                h=batch([{'H':h_sample(index)}],'H')
                records=[];arrays={};normal_reps={};normal_fields={}
                with torch.no_grad():
                    for branch in ('A','B'):
                        original=net(ab[branch])
                        for kind in ('normal','zero','half','one_half','permuted'):
                            inp=intervene(ab[branch],kind)
                            with Trace(net) as tr:field=tr.forward(inp)
                            if kind=='normal':
                                assert torch.equal(field,original),'Trace changed forward semantics'
                                normal_fields[branch]=field
                                normal_reps[branch]={k:v.detach().cpu().double().reshape(-1) for k,v in tr.values.items()}
                            repeated=net(inp);assert torch.equal(field,repeated)
                            representation={}
                            for key,value in tr.values.items():
                                cur=value.detach().cpu().double().reshape(-1);base=normal_reps[branch][key];delta=cur-base
                                representation[key]=dict(l2=float(cur.norm()),delta_l2=float(delta.norm()),delta_mae=float(delta.abs().mean()),
                                    relative_delta_l2=float(delta.norm()/base.norm().clamp_min(1e-12)),shape=list(value.shape))
                            pred=project_input(field,h);mask=valid(h)
                            arrays[branch+'_'+kind+'_field']=field.cpu().numpy()
                            arrays[branch+'_'+kind+'_pH']=pred.cpu().numpy()
                            input_delta=(inp['input_projection_raw']-ab[branch]['input_projection_raw'])[valid(inp)].double()
                            records.append(dict(branch=branch,intervention=kind,output=field_stats(field),difference=difference(field,original),
                                H=scores(pred,h)[0],zero_H_l1=1.,input_mae_ppm_m=float(input_delta.abs().mean()),
                                input_rmse_ppm_m=float(input_delta.square().mean().sqrt()),representation=representation,repeat_bitwise=True))
                arrays['y_H']=h['input_projection_raw'].cpu().numpy();arrays['mask_H']=valid(h).cpu().numpy()
                path=target/f"{state['name']}_case{index}.npz";np.savez_compressed(path,**arrays)
                gradients=layer_gradients(net,ab,'v3' if state['name'].startswith('v3') else 'v2')
                fd=finite_difference(net,ab['A'])
                case=dict(index=index,case=samples[index]['A']['case_id'],responses=records,gradients=gradients,finite_difference=fd,
                    raw=str(path),raw_sha256=c.sha(path),H_in_backward=False)
                state_result['cases'].append(case)
                print('C',arm,state['name'],'case',index,'normalH',records[0]['H']['relative_l1'],flush=True)
            assert state_hash(net.state_dict())==identity,'C modified parameters'
            results.append(state_result);c.freeze(target/f"{state['name']}.json",state_result)
            del net;torch.cuda.empty_cache()
        c.freeze(target/'result.json',dict(at=c.now(),arm=arm,states=results,optimizer_updates=0,
            audit=access.summary(),decision='SENSITIVITY_DIAGNOSIS_ONLY_NO_TRAINING_GATE',
            no_formal_authorization=True,sources_unchanged=True))
        assert not access.denied;c.verify_hashes(p['sources'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);a=ap.parse_args();run(a.arm)
