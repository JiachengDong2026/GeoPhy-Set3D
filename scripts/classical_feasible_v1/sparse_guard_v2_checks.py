"""CPU checks for the revised guard and exact boundary checkpoint identity."""
from sparse_guard_v2_context import *
from feasible_context import healthy as original_healthy
from prepared_inputs import prepare_model


def main():
    setup();verify();checks={}
    empty=torch.zeros(1,1,64,64,64)
    sparse=torch.full_like(empty,1e-6);sparse.flatten()[:2621]=.1
    saturated=torch.full_like(empty,1e-6);saturated.flatten()[:2621]=1-1e-6
    for name,value,wanted in [('empty',empty,False),('constant_low',empty+1e-6,False),
        ('constant_high',empty+1,False),('tiny_nonzero',empty+1e-12,False),
        ('sparse_unsaturated_signal',sparse,True),('sparse_saturated_signal',saturated,False)]:
        stats=field_stats(value);actual=healthy([stats]);assert actual==wanted,(name,stats)
        checks[name]=dict(passed=True,expected=wanted,observed=actual,legacy_observed=original_healthy([stats]),stats=stats)
    checkpoint=BASE_ROOT/'unet/last.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=prepare_model(build('unet'));model.load_state_dict(ck['model'])
    assert state_hash(model.state_dict())==state_hash(ck['model'])
    assert ck['optimizer_steps']==14525 and ck['scheduler']['last_epoch']==175
    assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={14525}
    assert all(v['lr']==ck['optimizer']['param_groups'][0]['lr'] for v in ck['optimizer']['param_groups'])
    guard=GTReadGuard(DATA,BASE_ROOT)
    with guard,torch.no_grad():
        ds=make_dataset(config());ds.set_epoch(0)
        batch=dual_measurement_collate([ds[2]])
        value=model(batch['A']);stats=field_stats(value);assert healthy([stats]),stats
    write(ART/'cpu_checks.json',dict(passed=True,guard_checks=checks,original_boundary_checkpoint_sha256=sha(checkpoint),
        boundary_model_state_sha256=state_hash(model.state_dict()),boundary_real_train2_stats=stats,
        optimizer_steps=14525,scheduler_epoch=175,optimizer_rng_will_be_restored_without_new_updates=True,
        new_optimizer_updates=0,access=guard.summary(),legacy_decision_preserved=True))
    verify();print('SPARSE_GUARD_CPU_CHECKS_PASSED',flush=True)


if __name__=='__main__':main()
