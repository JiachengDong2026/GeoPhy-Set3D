"""Real two-step convolutional network / explicit-ray-chain acceptance tests."""
import pytest
import torch

from geophy_set3d.models.geolift_deepsets_sum import GeoLiftDeepSetsSum
from geophy_set3d.models.geolift_oari_res3d import GeoLiftOARIRes3D
from geophy_set3d.models.set_unrolled3d import SetUnrolled3D, Y_SCALE
from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_batch(dtype=torch.float32, batch_size=1, views=2, hw=4):
    """Two opposing real ray bundles through a 0.6 m endpoint cube."""
    shape = (batch_size, views, 1, hw, hw)
    origins = torch.tensor([[-.2, .3, .3], [.8, .3, .3]], dtype=dtype)[:views]
    forward = torch.tensor([[1., 0, 0], [-1., 0, 0]], dtype=dtype)[:views]
    right = torch.tensor([[0., 1., 0], [0., -1., 0]], dtype=dtype)[:views]
    up = torch.tensor([[0., 0, 1.]], dtype=dtype).expand(views, -1)
    expand = lambda a: a[None].expand(batch_size, *a.shape).clone()
    batch = dict(view_mask=torch.ones((batch_size, views), dtype=torch.bool),
                 input_valid_ray=torch.ones(shape, dtype=torch.bool),
                 input_station_origin_m=expand(origins),
                 input_center_direction=expand(forward),
                 input_right_direction=expand(right), input_up_direction=expand(up),
                 input_station_azimuth_deg=torch.zeros((batch_size, views), dtype=dtype),
                 input_ray_direction=expand(forward[:, None, None].expand(-1, hw, hw, -1)),
                 input_t_near=torch.full(shape, .2, dtype=dtype),
                 input_path_length_m=torch.full(shape, .6, dtype=dtype),
                 horizontal_angle_deg=torch.linspace(-70, 70, hw, dtype=dtype)[None].expand(batch_size, -1),
                 vertical_angle_deg=torch.linspace(-70, 70, hw, dtype=dtype)[None].expand(batch_size, -1))
    op = ExplicitRayOperator(samples_per_ray=8)
    raw = op(torch.full((batch_size, 1, 8, 8, 8), .12, dtype=dtype), batch)
    batch.update(input_projection_raw=raw, input_projection=raw / Y_SCALE)
    return batch


def small_model(variant='A2', **settings):
    torch.manual_seed(314)
    return SetUnrolled3D(variant=variant, steps=0 if variant == 'A0' else 2,
                         lift_resolution=4, output_resolution=8,
                         prior_width=8, samples=8, ray_chunk=7, **settings)


def permute(batch, order):
    return {k: v[:, order] if k.startswith('input_') or k == 'view_mask' else v
            for k, v in batch.items()}


def pad(batch):
    result = {}
    for key, value in batch.items():
        if key.startswith('input_') or key == 'view_mask':
            fill = False if value.dtype == torch.bool else float('nan')
            result[key] = torch.cat((value, torch.full_like(value[:, :1], fill)), 1)
        else:
            result[key] = value
    return result


@pytest.mark.parametrize('variant', ['A0', 'A1', 'A2'])
def test_shared_initialization_matches_historical_sum4(variant):
    torch.manual_seed(25)
    historical = GeoLiftDeepSetsSum()
    torch.manual_seed(25)
    model = SetUnrolled3D(variant=variant)
    for name, value in historical.state_dict().items():
        torch.testing.assert_close(model.initializer.state_dict()[name], value, rtol=0, atol=0)
    if model.prior is not None:
        assert torch.count_nonzero(model.prior[-1].weight) == 0
        assert torch.count_nonzero(model.prior[-1].bias) == 0


def test_a3_preserves_historical_oari_initializer():
    torch.manual_seed(19)
    historical = GeoLiftOARIRes3D()
    torch.manual_seed(19)
    model = SetUnrolled3D(variant='A3')
    for name, value in historical.state_dict().items():
        torch.testing.assert_close(model.initializer.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize('variant', ['A0', 'A1', 'A2'])
def test_forward_no_grad_padding_permutation(variant):
    batch, model = make_batch(), small_model(variant).eval()
    with torch.no_grad():
        expected = model(batch)
        changed = model(permute(batch, torch.tensor([1, 0])))
        padded = model(pad(batch))
        diagnostics = model(batch, return_diagnostics=True)
    assert expected.shape == (1, 1, 8, 8, 8)
    assert torch.isfinite(expected).all()
    assert bool(((expected >= 0) & (expected <= 1)).all())
    torch.testing.assert_close(changed, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(padded, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(diagnostics['volume'], expected)
    assert len(diagnostics['steps']) == model.steps
    if model.steps:
        torch.testing.assert_close(diagnostics['tau'], torch.full((model.steps,), .5))
        assert bool((diagnostics['zero_coverage_fraction'] > 0).all())
        assert bool((diagnostics['D0'][diagnostics['D0'] > 0] > 0).all())
        assert all(s['E'].isfinite().all() for s in diagnostics['steps'])


def test_per_case_mixed_views_matches_individual_evaluation():
    model = small_model().eval()
    first, second = pad(make_batch(views=1)), make_batch()
    # A different scale AND ray count catches accidentally shared batch m/eps.
    second['input_projection_raw'] *= 2
    second['input_projection'] *= 2
    second['input_valid_ray'][:, 1, :, :2] = False
    batch = {k: torch.cat((first[k], second[k]), 0) for k in first}
    with torch.no_grad():
        mixed = model(batch, return_diagnostics=True)
        independent = [model(x, return_diagnostics=True) for x in (first, second)]
    torch.testing.assert_close(mixed['volume'], torch.cat([x['volume'] for x in independent]), atol=3e-6, rtol=0)
    torch.testing.assert_close(mixed['eps_D'], torch.cat([x['eps_D'] for x in independent]), atol=1e-6, rtol=1e-6)
    assert mixed['valid_counts'][0] != mixed['valid_counts'][1]


def test_a0_is_initializer_and_a1_has_no_learned_prior():
    batch, a0, a1 = make_batch(), small_model('A0'), small_model('A1')
    torch.testing.assert_close(a0(batch), a0.initializer(batch), rtol=0, atol=0)
    assert a1.prior is None and a1.step_embedding is None
    d = a1(batch, return_diagnostics=True)
    assert all(bool((s['prior_norm'] == 0).all()) for s in d['steps'])
    assert not torch.equal(d['volume'], d['C0'])
    stencil = a1.operator.prepare(batch, 8)
    torch.testing.assert_close(d['D0'], a1.operator.geometry_preconditioner(stencil))


def test_two_step_full_network_has_finite_initializer_step_prior_gradients():
    batch, model = make_batch(), small_model()
    # Activate the zero-initialized head to test every upstream prior layer.
    with torch.no_grad():
        model.prior[-1].weight.normal_(0, .005)
    d = model(batch, return_diagnostics=True)
    d['C0'].retain_grad()
    d['volume'].square().mean().backward()
    for module in (model.initializer, model.prior, model.step_embedding):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    assert bool(torch.isfinite(model.step_logits.grad).all())
    assert bool((model.step_logits.grad.abs() > 0).all())
    assert d['C0'].grad is not None and d['C0'].grad.abs().sum() > 0


def test_two_step_parameter_derivatives_match_finite_differences():
    batch, model = make_batch(torch.float64), small_model().double()
    with torch.no_grad():
        model.prior[-1].weight.normal_(0, .003)
    weights = torch.linspace(.7, 1.3, 8**3, dtype=torch.float64).reshape(1, 1, 8, 8, 8)
    objective = lambda: (model(batch) * weights).mean()
    objective().backward()
    checks = [(model.initializer.decoder[-2].bias, (0,)),
              (model.step_logits, (0,)), (model.step_logits, (1,)),
              (model.prior[-1].weight, (0, 0, 0, 0, 0))]
    for parameter, index in checks:
        analytic, original = parameter.grad[index].item(), parameter[index].item()
        eps = 1e-5
        with torch.no_grad():
            parameter[index] = original + eps
            plus = objective().item()
            parameter[index] = original - eps
            minus = objective().item()
            parameter[index] = original
        numerical = (plus-minus) / (2*eps)
        assert abs(analytic-numerical) < 2e-7 + abs(numerical)*3e-4


def test_prior_checkpoint_preserves_outputs_and_gradients():
    batch, plain, checked = make_batch(), small_model(), small_model(checkpoint_prior=True)
    with torch.no_grad():
        plain.prior[-1].weight.normal_(0, .003)
    checked.load_state_dict(plain.state_dict())
    p, q = plain(batch), checked(batch)
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    p.square().mean().backward()
    q.square().mean().backward()
    for a, b in zip(plain.parameters(), checked.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('variant', ['A0', 'A1', 'A2', 'A3', 'A-control'])
def test_no_input_rays_is_rejected(variant):
    batch = make_batch()
    batch['input_valid_ray'].zero_()
    with pytest.raises(ValueError, match='valid ray|usable input'):
        model = SetUnrolled3D(variant='A3') if variant == 'A3' else small_model(variant)
        model(batch)


def test_no_preconditioner_is_explicit_scaled_raw_gradient():
    batch = make_batch()
    model = small_model('A1', preconditioner='none', unpreconditioned_scale=1e-4)
    d = model(batch, return_diagnostics=True)
    stencil = model.operator.prepare(batch, 8)
    c0 = d['C0']
    residual = model.operator.project_normalized(c0, stencil) - batch['input_projection']
    g = model.operator.vjp(c0, residual, stencil) / stencil.valid_counts[:, None, None, None, None]
    expected = (c0 - .5 * 1e-4 * g).clamp(0, 1)
    torch.testing.assert_close(d['steps'][0]['volume'], expected, rtol=0, atol=0)


def test_control_exact_parameter_match_and_no_jacobian_action():
    a2, control = small_model(), small_model('A-control')
    assert sum(p.numel() for p in a2.parameters()) == sum(p.numel() for p in control.parameters())
    assert dict(a2.named_parameters()).keys() == dict(control.named_parameters()).keys()
    for name, parameter in a2.named_parameters():
        torch.testing.assert_close(parameter, dict(control.named_parameters())[name], rtol=0, atol=0)
    def forbidden(*args, **kwargs):
        raise AssertionError('A-control must never use a Jacobian action')
    control.operator.vjp = forbidden
    control.operator.jvp = forbidden
    with torch.no_grad():
        control.prior[-1].weight.normal_(0, .0002)
        # Test feedback derivatives away from tanh saturation, just as the
        # physical derivative test avoids clip boundaries.
        control.initializer.decoder[-2].weight.zero_()
        control.initializer.decoder[-2].bias.fill_(torch.logit(torch.tensor(.12)))
    batch = make_batch()
    with torch.no_grad():
        p0 = control.operator.project_normalized(control.initializer(batch), batch)
        batch['input_projection'] = p0 + .005
        batch['input_projection_raw'] = batch['input_projection'] * Y_SCALE
    d = control(batch, return_diagnostics=True)
    assert all(bool((s['physical_step_norm'] == 0).all()) for s in d['steps'])
    assert all(bool((s['feedback_norm'] > 0).all()) for s in d['steps'])
    assert not torch.equal(d['volume'], d['C0'])
    d['volume'].square().mean().backward()
    assert control.step_logits.grad is not None
    assert bool((control.step_logits.grad.abs() > 0).all())
