"""Zero-update boundary and gradient verification before the bounded experiment."""
from probe_context import *


def main():
    r = verify()
    setup()
    prepared = read(ART / 'prepared.json')['cases'][0]
    results = []
    guard = GTReadGuard(DATA, ROOT)
    with guard:
        payload = torch.load(prepared['path'], map_location='cpu', weights_only=False)
        for arm in r['arms']:
            model = fresh() if arm == 'unet' else DirectField(payload['initial_shared'], r['c0'], arm)
            before = state_hash(model.state_dict())
            model.eval()
            loss, _, fields, _ = objective(model, payload['batch'])
            gradients = torch.autograd.grad(loss, tuple(model.parameters()))
            assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
            norm = sum(float(g.double().square().sum()) for g in gradients)**.5
            assert norm > 0
            for field in fields:
                assert field.shape == (1, 1, 64, 64, 64)
                assert torch.isfinite(field).all() and (field >= 0).all() and (field <= 1).all()
                if arm != 'unet':
                    torch.testing.assert_close(field, payload['initial_shared'], rtol=2e-6, atol=1e-10)
            assert state_hash(model.state_dict()) == before
            assert all(p.grad is None for p in model.parameters())
            results.append(dict(arm=arm, loss=float(loss.detach()), norm=norm, state_unchanged=True))
    write(ART / 'preflight.json', dict(passed=True, rows=results, optimizer_updates=0, access=guard.summary()))
    print('All three complete 64-cubed gradient preflights passed; zero updates')


if __name__ == '__main__':
    main()
