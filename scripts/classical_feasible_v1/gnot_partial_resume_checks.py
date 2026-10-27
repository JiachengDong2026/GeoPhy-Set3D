"""Reconstruct interrupted epoch48 data order without replaying optimizer updates."""
from feasible_context import *
from torch.utils.data import DataLoader


def main():
    setup();verify();target=ROOT/'pilot_gnot'
    failure=read(target/'failure.json');assert failure['stage']=='preheat' and failure['epoch']==48 and failure['updates']==3914
    assert 'loss.backward()' in failure['traceback'] and failure['error']=='Owned worker interrupted'
    for source,destination in [('last_preheat.pt','conflict_boundary47.pt'),('interrupted.pt','conflict_partial48.pt')]:
        os.link(target/source,target/destination)
    boundary=torch.load(target/'conflict_boundary47.pt',map_location='cpu',weights_only=False)
    partial=torch.load(target/'conflict_partial48.pt',map_location='cpu',weights_only=False)
    assert boundary['epoch']==47 and boundary['optimizer_steps']==3901
    assert partial['epoch']==48 and partial['optimizer_steps']==3914 and partial['scheduler']['last_epoch']==47
    assert {int(v['step']) for v in partial['optimizer']['state'].values()}=={3914}
    journal=[json.loads(s) for s in (target/'preheat_updates.jsonl').read_text().splitlines()]
    assert [v['update'] for v in journal]==list(range(1,3915))
    generator=torch.Generator();generator.set_state(boundary['rng']['loader'])
    trace=DataLoader(list(range(166)),batch_size=2,shuffle=True,num_workers=0,generator=generator,collate_fn=list)
    iterator=iter(trace);prefix=[next(iterator) for _ in range(13)];unupdated=next(iterator)
    assert torch.equal(generator.get_state(),partial['rng']['loader']),'Mid-epoch sampler state must exactly reproduce'
    batches=prefix+[unupdated]+list(iterator);end=generator.get_state()
    assert len(batches)==83 and sorted(index for batch in batches for index in batch)==list(range(166))
    guard=GTReadGuard(DATA,ROOT)
    with guard:
        ds=make_dataset(config());ds.set_epoch(47)
        actual_generator=torch.Generator();actual_generator.set_state(boundary['rng']['loader'])
        actual=iter(make_loader(ds,config(),True,actual_generator))
        for indices in batches[:14]:
            batch=next(actual)
            assert batch['A']['case_id']==[ds.records[index]['case_id'] for index in indices]
        assert torch.equal(actual_generator.get_state(),partial['rng']['loader'])
    model=build('gnot');assert not any(isinstance(m,torch.nn.Dropout) for m in model.modules())
    write(ART/'gnot_partial_resume_checks.json',dict(passed=True,stage='preheat',epoch=48,completed_batches=13,
        completed_updates=3914,boundary47_updates=3901,unupdated_fetched_batch=14,all_epoch48_batches=batches,
        remaining_batches=batches[13:],loader_rng_after_epoch48=end.tolist(),sampler_midstate_exact=True,
        actual_dataset_case_order_exact=True,no_stochastic_model_layers=True,new_optimizer_updates=0,
        no_optimizer_replay=True,prefix_loss_unavailable=True,access=guard.summary(),
        boundary47_sha256=sha(target/'conflict_boundary47.pt'),partial48_sha256=sha(target/'conflict_partial48.pt')))
    verify();print('GNOT_PARTIAL_RESUME_CHECKS_PASSED 3914 real updates,70 remaining batches in epoch48,zero replay',flush=True)


if __name__=='__main__':main()
