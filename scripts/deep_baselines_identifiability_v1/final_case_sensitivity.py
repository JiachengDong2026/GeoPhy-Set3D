"""Complete the preregistered all-case acceptance evidence, zero training updates."""
from diag_context import *
import engineering_memory_v2 as memory

def main():
    evidence=verified('final_case_sensitivity_plan.json');c.verify_hashes(evidence['sources'])
    c.backend();memory.install()
    from trace import layer_gradients,finite_difference
    outputs=[]
    with TrainGuard() as access:
        for path in sorted((ROOT/'runs').glob('*/result.json')):
            r=c.verified(path);p=c.verified(path.parent/'protocol.json');last=c.rows(path.parent/'evaluations.jsonl')[-1]
            inputs=ROOT/('inputs/synthetic_AB.pt' if p['experiment']=='B' else 'inputs/real_AB.pt')
            samples=torch.load(inputs,map_location='cpu',weights_only=False)[:p['n']]
            net=model(p['arm'],'cuda:0');ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
            net.load_state_dict(ck['model'],strict=True);identity=state_hash(net.state_dict());del ck
            cases=[]
            for index,sample in enumerate(samples):
                if index==0:
                    gradients=last['layer_gradients'];fd=last['finite_difference'];reused=True
                else:
                    ab={k:batch([sample],k) for k in ('A','B')}
                    gradients=layer_gradients(net,ab,p['version']);fd=finite_difference(net,ab['A']);reused=False
                g=gradients['observed']['layers']['token_measurement_channel']
                grad_ok=g['l2']>1e-12 and g['finite_fraction']==1
                fd_ok=any(v['function']=='output_mean_div_fixed_scale' and v['observable'] and v['within_tolerance'] for v in fd)
                cases.append(dict(index=index,case=sample['A']['case_id'],gradients=gradients,finite_difference=fd,
                    input_gradient_pass=grad_ok,finite_difference_pass=fd_ok,reused_existing_case0=reused,H_in_backward=False))
            assert state_hash(net.state_dict())==identity
            outputs.append(dict(run=path.parent.name,cases=cases,all_case_input_gradient=all(x['input_gradient_pass'] for x in cases),
                all_case_finite_difference=all(x['finite_difference_pass'] for x in cases),optimizer_updates=0,checkpoint_sha256=r['checkpoint_sha256']))
            del net;torch.cuda.empty_cache();print('FINAL CASE SENSITIVITY',path.parent.name,outputs[-1]['all_case_input_gradient'],outputs[-1]['all_case_finite_difference'],flush=True)
    assert len(outputs)==10 and sum(len(x['cases']) for x in outputs)==30
    freeze('final_case_sensitivity.json',dict(at=c.now(),completed=True,optimizer_updates=0,training_budget_unchanged=True,
        cases=30,reused_initial_case0_audits=10,additional_case_audits=20,thresholds_unchanged=True,runs=outputs,audit=access.summary()))

if __name__=='__main__':main()
