"""Meaningful CPU mechanism checks before preregistration, never training."""
from runtime import *
from interventions import controlled_maps, permute_relation, Intervention, Capture
_verify = load_unique('oari_v2_independent_verify', HERE / 'verify.py')
rebuild, check = _verify.rebuild, _verify.check
from measurements import distances
from scipy.special import rel_entr


def run():
    backend()
    rng = np.random.default_rng(SEED)
    n, v = 96, 4
    mask = np.zeros((n, v), dtype=bool)
    for i in range(n):
        mask[i, :i % 5] = True
    rel = rng.uniform(size=(n, v, v, 3)).astype(np.float32)
    rel = (rel + rel.swapaxes(1, 2)) / 2
    rel[:, np.arange(v), np.arange(v)] = [1, 0, 0]
    pair = mask[:, :, None] & mask[:, None, :]
    rel[~pair] = 0
    maps, eligible = controlled_maps(rel, mask, SEED)
    assert np.array_equal(eligible, mask.sum(1) >= 3)
    for mp in maps:
        moved = permute_relation(torch.from_numpy(rel), torch.from_numpy(mp.astype(np.int64))).numpy()
        assert np.array_equal(moved, moved.swapaxes(1, 2))
        assert np.array_equal(moved[~eligible], rel[~eligible])
        assert np.array_equal(np.diagonal(moved, axis1=1, axis2=2), np.diagonal(rel, axis1=1, axis2=2))
        assert np.all(moved[~pair] == 0)
    # Independent numerical reconstruction of all planned interventions on toy
    # real OARI modules, including token residual effects and zero-valid voxels.
    from geophy_set3d.models.ray_relational_interaction import ObservabilityAwareRayInteraction
    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.interaction = ObservabilityAwareRayInteraction()
    torch.manual_seed(SEED)
    model = Holder().eval().requires_grad_(False)
    # Nonzero bias output is necessary to test intervention algebra.
    with torch.no_grad():
        model.interaction.pairwise_bias[-1].weight.copy_(torch.randn_like(model.interaction.pairwise_bias[-1].weight) * .1)
    f = torch.randn(1, v, 32, 4, 4, 6)
    geo = torch.randn(1, v, 4, 4, 4, 6)
    vis = torch.from_numpy(mask.T.reshape(1, v, 4, 4, 6).copy())
    args = (f, geo, vis, torch.ones(1, v, dtype=torch.bool), torch.from_numpy(rel), torch.rand(n, 5), True)
    maxima = {}
    weights = {k: a.numpy().astype(np.float64) for k, a in model.interaction.state_dict().items()}
    with torch.inference_mode():
        for group in GROUPS:
            mp = maps[int(group.split('_')[1])] if group.startswith('A3_') else None
            with Intervention(model, group, mp, [3, 2, 1, 0]), Capture(model) as capture:
                model.interaction(*args)
            data = capture.cpu()
            ids = capture.sample_ids
            probe = {k: data[k][ids].astype(np.float64) if data[k].dtype.kind == 'f' else data[k][ids] for k in data}
            probe.update(token_input=capture.sample_token_input.astype(np.float64), observability=capture.args[5][ids].numpy().astype(np.float64))
            rebuilt = rebuild(probe, weights, group)
            for key, value in rebuilt.items():
                actual = capture.sample_qkv if key == 'qkv' else probe[key]
                check(actual, value, key, maxima)
        a = rng.dirichlet(np.ones(4), size=(32, 4, 4))
        b = rng.dirichlet(np.ones(4), size=(32, 4, 4))
        own = distances(a, b)
        middle = (a + b) / 2
        check(own['js'], (rel_entr(a, middle).sum(-1) + rel_entr(b, middle).sum(-1)) / 2, 'JS', maxima, atol=1e-14, rtol=1e-12)
    print(json.dumps(dict(passed=True, controlled_mask_symmetry_diagonal=True, ineligible_unchanged=True,
                         all18_interventions_independently_reconstructed=True, max_errors=maxima), indent=2))


if __name__ == '__main__':
    run()
