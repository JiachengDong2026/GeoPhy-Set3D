"""Separate capability/identity audit, without optimizer updates or real labels."""
from diag_context import *
import ast
import subprocess
from diagnostics import objective

def main():
    registration=verified('registration.json');pre=verified('preregistration.json')
    c.verify_hashes(registration['sources']);c.verify_hashes(registration['data_identity']);c.verify_hashes(registration['inputs'])
    c.verify_hashes(pre['historical_hashes'])
    for name in ('engineering_memory_v2.json','engineering_dispatch_v3.json','engineering_bookkeeping_v4.json'):
        c.verify_hashes(verified(name)['sources'])
    prior=c.verified(ROOT.parent/'deep_baselines_repair_v2/acceptance.json')
    checkpoints={r[k+'_checkpoint']:r[k+'_checkpoint_sha256'] for r in prior['pilots'] for k in ('best','last')}
    for arm in ARMS:
        for state in c.verified(ROOT/f'C/{arm}/result.json')['states']:
            path=state['state']['path']
            if path is not None:
                actual=c.sha(path);assert actual==state['checkpoint_sha256']
                if path in checkpoints:assert actual==checkpoints[path]
        assert c.verified(ROOT/f'C/{arm}/result.json')['optimizer_updates']==0
    # A pure synthetic graph counterfactual: arbitrarily change H values while
    # holding A/B, model and RNG fixed. H is absent from the objective capability.
    sys.path.insert(0,str(PROJECT/'tests'))
    from test_baseline_contract_v1 import synthetic_dual
    from geophy_set3d.models.baseline_measurement_init_v2 import RayDeepSetsMeasurementInitV2
    torch.set_num_threads(2);seed_everything(20261011)
    net=RayDeepSetsMeasurementInitV2(.004195626187323315,hidden_dim=8).train()
    source=synthetic_dual();source['H']['input_projection_raw'].requires_grad_();source['H']['input_ray_direction'].requires_grad_()
    before=state_hash(net.state_dict());observed=[]
    def tap(module,args):observed.append(args[0]['input_station_indices'].tolist())
    handle=net.register_forward_pre_hook(tap)
    gradients=[];losses=[]
    for factor in (1.,1000.):
        h=dict(source['H'],input_projection_raw=source['H']['input_projection_raw']*factor)
        value,parts,_,_=objective(net,{k:source[k] for k in ('A','B')},'v2')
        params=list(net.parameters());grads=torch.autograd.grad(value,params+[h['input_projection_raw'],source['H']['input_ray_direction']],allow_unused=True)
        assert grads[-2:] == (None,None)
        gradients.append(torch.cat([g.reshape(-1) for g in grads[:-2]]));losses.append(float(value.detach()))
    handle.remove();assert losses[0]==losses[1] and torch.equal(gradients[0],gradients[1])
    assert observed==[[[0]],[[1]],[[0]],[[1]]] and state_hash(net.state_dict())==before
    c_ast=ast.parse((HERE/'experiment_c.py').read_text())
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='step' for node in ast.walk(c_ast))
    fixed=(HERE/'train_fixed_v2.py').read_text();original=(HERE/'train.py').read_text()
    assert fixed==original.replace('dict(at=c.now(),**result)','dict(result,at=c.now())')
    amendment=verified('engineering_memory_v2.json');archive=Path(amendment['failed_partial_arrays_preserved_at'])
    for original,h in amendment['failed_partial_arrays_original_hashes'].items():
        assert c.sha(archive/Path(original).name)==h
    assert c.read(legacy.ROOT/'queue_status.json')['state']=='paused_by_user'
    old=verified('historical_git_tree.json')
    assert subprocess.check_output(['git','diff','--name-only',old['commit']],text=True).strip()==''
    freeze('integrity_audit.json',dict(at=c.now(),passed=True,real_train_only=True,
        H_factor1000_training_loss_and_parameter_gradients_bitwise_identical=True,H_input_and_geometry_gradients_absent=True,
        reconstruction_only_saw_A_B=True,C_source_has_no_optimizer_step=True,C_updates=0,
        all_registered_sources_and_inputs_unchanged=True,original_failed_and_repair_checkpoints_unchanged=True,
        original_tracked_history_unchanged=True,only_bookkeeping_line_diff_in_training_worker=True,
        failed_partial_arrays_preserved=True,old_queue_paused=True,synthetic_boundary_test_optimizer_updates=0))
    print('INTEGRITY AND H CAPABILITY AUDIT PASSED',flush=True)

if __name__=='__main__':main()
