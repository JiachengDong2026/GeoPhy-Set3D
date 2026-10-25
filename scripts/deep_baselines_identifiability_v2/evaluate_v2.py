"""Independent evaluation process, sole model phase with H/truth capability."""
from numerics_v2 import *
from metrics_v2 import volume,difference,measurement,paired,decide
import argparse
import gc
import time

def main(arm):
    reg=verified(ROOT/'registration.json');verify(reg['sources']);verify(reg['inputs']);backend();began=time.monotonic()
    target=ROOT/'runs'/arm;assert not (target/'evaluation_result.json').exists()
    history=[];all_pairs=[];all_swaps=[]
    with Boundary('evaluate') as access:
        samples=torch.load(ROOT/'inputs/train_AB.pt',map_location='cpu',weights_only=False)+torch.load(ROOT/'inputs/unseen_AB.pt',map_location='cpu',weights_only=False)
        truth=[np.load(ROOT/f'inputs/truth_{i}.npy',allow_pickle=False) for i in range(8)]
        geometry=verified(ROOT/'input_registration.json')['identity'][0]['geometry']
        for sample in samples:
            for b in ('A','B'):assert value_hash(nonmeasurement(sample[b]))==geometry[b]
        net=model(arm,'cuda:0')
        for update in design()['evaluation_updates']:
            ckpath=target/f'checkpoint_{update:03d}.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
            assert ck['updates']==update;net.load_state_dict(ck['model'],strict=True);state_before=state_hash(net.state_dict());del ck;net.eval()
            case_rows=[];normal={};repeats={};pairs=[];swaps=[];raw_hashes={}
            for index,sample in enumerate(samples):
                inp={b:batch([sample],b,'cuda:0') for b in ('A','B')}
                h=torch.load(ROOT/f'inputs/H_{index}.pt',map_location='cpu',weights_only=False)
                targets=dict(inp,H=batch([{'H':h}],'H','cuda:0'));arrays={};branches={};measurements=[]
                for k,t in targets.items():arrays['y_'+k]=t['input_projection_raw'].cpu().numpy();arrays['mask_'+k]=valid(t).cpu().numpy()
                for branch in ('A','B'):
                    t=inp[branch]
                    with torch.no_grad():
                        field=net(t);repeat=net(t);assert torch.equal(field,repeat)
                        assert field.shape==(1,1,64,64,64) and field.dtype==torch.float32 and torch.isfinite(field).all() and field.min()>=0 and field.max()<=1
                        a=field.cpu().numpy();arrays['field_'+branch]=a;normal[index,branch]=a;repeats[index,branch]=True
                        interventions={}
                        for kind in ('zero','permuted'):
                            changed=intervene(t,kind);f=net(changed);ff=net(intervene(t,kind));assert torch.equal(f,ff)
                            arrays['field_'+branch+'_'+kind]=f.cpu().numpy()
                            interventions[kind]=dict(**difference(arrays['field_'+branch+'_'+kind],a),repeat_bitwise=True)
                        for key,target_inp in targets.items():
                            projected=project_input(field,target_inp).cpu().numpy();arrays['p_'+branch+'_'+key]=projected
                            measurements.append(dict(branch=branch,target=key,**measurement(projected,arrays['y_'+key],arrays['mask_'+key])))
                    gradient=sensitivity(net,t);gc.collect()
                    branches[branch]=dict(volume=volume(a,truth[index]),interventions=interventions,gradient=gradient)
                ab=[r for r in measurements if r['target']!='H'];held=[r for r in measurements if r['target']=='H']
                row=dict(index=index,name=design()['synthetic_cases'][index]['name'],role='seen' if index<6 else 'unseen',branches=branches,measurement=measurements,
                    AB_L1=float(np.mean([r['relative_l1'] for r in ab])),AB_L2=float(np.mean([r['relative_l2'] for r in ab])),
                    H_L1=float(np.mean([r['relative_l1'] for r in held])),H_L2=float(np.mean([r['relative_l2'] for r in held])),
                    geometry_sha256=geometry,measurement_sha256={b:tensor_hash(sample[b]['input_projection_raw']) for b in ('A','B')},checkpoint_sha256=sha(ckpath))
                raw=target/'raw'/f'update{update:03d}_case{index}.npz';raw.parent.mkdir(exist_ok=True);assert not raw.exists()
                np.savez_compressed(raw,**arrays);raw_hashes[str(raw)]=sha(raw);case_rows.append(row)
            for i,j in design()['pairs']:
                group='seen_seen' if j<6 else 'unseen_unseen' if i>=6 else 'seen_unseen'
                for branch in ('A','B'):
                    p=dict(i=i,j=j,branch=branch,group=group,**paired(normal[i,branch],normal[j,branch],truth[i],truth[j]),repeat_bitwise=True)
                    pairs.append(p)
                    for recipient,donor in ((i,j),(j,i)):
                        original=batch([samples[recipient]],branch,'cuda:0');other=batch([samples[donor]],branch,'cuda:0')
                        changed=intervene(original,'swap',other)
                        assert value_hash(changed)==value_hash(other)
                        # Exact same tensor input to a deterministic model: use
                        # donor's already bitwise-repeated full forward result.
                        matched=volume(normal[recipient,branch],truth[recipient])['relative_l1']
                        mismatched=volume(normal[donor,branch],truth[recipient])['relative_l1']
                        swaps.append(dict(recipient=recipient,donor=donor,branch=branch,recipient_role='seen' if recipient<6 else 'unseen',
                            **difference(normal[donor,branch],normal[recipient,branch]),matching_advantage_L1=mismatched-matched,
                            matched_L1=matched,swapped_vs_recipient_truth_L1=mismatched,repeat_bitwise=True,
                            input_equal_to_donor=True,output_reused_from='donor bitwise repeated normal forward'))
            assert state_hash(net.state_dict())==state_before
            ev=dict(update=update,cases=case_rows,pairs=pairs,swaps=swaps,raw_sha256=raw_hashes,checkpoint_sha256=sha(ckpath))
            freeze(target/f'evaluation_{update:03d}.json',ev);history.append(ev)
            print('EVALUATE',arm,update,'seen H',np.mean([r['H_L1'] for r in case_rows[:6]]),'unseen H',np.mean([r['H_L1'] for r in case_rows[6:]]),flush=True)
        result=decide(history,history[-1]['pairs'],history[-1]['swaps'])
        result.update(at=now(),arm=arm,updates=200,checkpoint_sha256=history[-1]['checkpoint_sha256'],
            elapsed_seconds=time.monotonic()-began,access=access.summary(),optimizer_updates_in_evaluation=0)
        freeze(target/'evaluation_result.json',result)
    print('DIAGNOSTIC',arm,result['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['gnot','deepsets'],required=True);a=p.parse_args();main(a.arm)
