"""Fixed per-case diagnostic optimization; no changes to original objective."""
from probe_context import *
import traceback


def main():
    registration = verify()
    setup()
    arm, text_index = sys.argv[1:]
    index = int(text_index)
    assert arm in registration['arms'] and index in registration['indices']
    directory = ROOT / f'case{index}_{arm}'
    directory.mkdir(exist_ok=False)
    prepared = next(r for r in read(ART / 'prepared.json')['cases'] if r['index'] == index)
    assert sha(prepared['path']) == prepared['sha256']
    guard = GTReadGuard(DATA, ROOT)
    generator = torch.Generator().manual_seed(20261011)
    steps = 0
    started = time.monotonic()
    metrics = []
    with guard:
        payload = torch.load(prepared['path'], map_location='cpu', weights_only=False)
        batch, extra = payload['batch'], payload['extra']
        model = fresh() if arm == 'unet' else DirectField(payload['initial_shared'], registration['c0'], arm)
        settings = registration['arms'][arm]
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        last = directory / 'last.pt'

        def checkpoint(error=None):
            atomic_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), rng=rng_state(generator), steps=steps, arm=arm, index=index,
                error=error, metrics=metrics, elapsed_seconds=time.monotonic()-started))
            write(directory / 'last_sha256.json', dict(path=str(last), sha256=sha(last), steps=steps))

        try:
            initial = evaluate(model, batch, extra, directory, 0, True)
            metrics.append(initial)
            checkpoint()
            # All gradients must be finite and nonzero; journal actual parameter changes.
            for step in range(1, registration['updates_per_run']+1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, parts, _, _ = objective(model, batch)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                assert norm > 0
                before = [p.detach().clone() for p in model.parameters()]
                optimizer.step()
                if arm == 'voxel_scaled':
                    with torch.no_grad():
                        model.value.clamp_(0, 1/registration['c0'])
                steps = step
                delta = sum(float((p.detach().double()-v.double()).square().sum()) for p,v in zip(model.parameters(),before))**.5
                assert delta > 0 and np.isfinite(delta)
                assert all(torch.isfinite(p).all() for p in model.parameters())
                # Match original cosine schedule per 16 pilot updates, not each single-case update.
                if step % 16 == 0:
                    scheduler.step()
                append(directory / 'updates.jsonl', dict(update=step, loss=float(loss.detach()),
                    preclip_norm=float(norm), actual_parameter_delta=delta, lr=optimizer.param_groups[0]['lr']))
                if step in registration['evaluation_steps']:
                    row = evaluate(model, batch, extra, directory, step, step == registration['updates_per_run'])
                    metrics.append(row)
                    write(directory / 'metrics.json', metrics)
                    checkpoint()
                    print(arm, index, 'step', step, 'loss', row['loss'], 'H', row['measurement']['heldout'],
                          'E', row['extra_E_H'], 'seconds', round(time.monotonic()-started,1), flush=True)
            checkpoint()
            assert steps == registration['updates_per_run']
            assert {int(v['step']) for v in optimizer.state.values()} == {steps}
            result = dict(arm=arm, index=index, actual_updates=steps, metrics=metrics,
                          seconds=time.monotonic()-started, checkpoint_sha256=sha(last), access=guard.summary(),
                          formal_training=False, status='COMPLETE_DIAGNOSTIC')
            write(directory / 'result.json', result)
            write(ART / f'case{index}_{arm}.json', result)
        except BaseException:
            error = traceback.format_exc()
            checkpoint(error)
            write(directory / 'failure.json', dict(steps=steps, error=error, access=guard.summary()))
            raise


if __name__ == '__main__':
    main()
