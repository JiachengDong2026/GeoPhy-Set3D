"""Plan B continuous-coordinate physics and differentiable set-conditioning.

All fixtures are synthetic measurements; no GT files or GPU are needed.
Small lift grids below exercise the implementation, not formal model accuracy.
"""
import copy
import os

import pytest
import torch
from torch import nn

from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator
from geophy_set3d.training.selfsup import consistency_discrepancy
from research.plan_b.model import FieldContext, SetConditionalField, sample_trilinear
from research.plan_b.physics import ContinuousRayIntegrator


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_num_threads(2)
    # Match the reference trainer's FP32 runtime. CUDA grid_sample backward
    # remains nondeterministic; TF32 convolution can amplify that roundoff.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(previous)
    torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
    torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def chi(c):
    methane = c / 16.043
    return methane / (methane + (1 - c) / 28.965)


def inverse_chi(x):
    return (16.043 * x) / (16.043 * x + 28.965 * (1 - x))


def make_batch(views=3, hw=3, dtype=torch.float32):
    """Six possible axis-aligned, pre-clipped bundles through the cube center."""
    direction = torch.tensor([[1., 0, 0], [-1., 0, 0], [0., 1, 0],
                              [0., -1., 0], [0., 0, 1.], [0., 0, -1.]], dtype=dtype)[:views]
    origin = .3 - .5 * direction
    right = torch.tensor([[0., 1, 0], [0., -1, 0], [-1., 0, 0],
                          [1., 0, 0], [1., 0, 0], [-1., 0, 0]], dtype=dtype)[:views]
    up = torch.linalg.cross(direction, right)
    shape = (1, views, 1, hw, hw)
    raw = torch.full(shape, float(chi(torch.tensor(.12, dtype=dtype)) * 6e5), dtype=dtype)
    return dict(view_mask=torch.ones((1, views), dtype=torch.bool),
                input_valid_ray=torch.ones(shape, dtype=torch.bool),
                input_station_origin_m=origin[None].clone(),
                input_center_direction=direction[None].clone(),
                input_right_direction=right[None], input_up_direction=up[None],
                input_station_azimuth_deg=torch.zeros((1, views), dtype=dtype),
                input_ray_direction=direction[None, :, None, None].expand(1, views, hw, hw, 3).clone(),
                input_t_near=torch.full(shape, .2, dtype=dtype),
                input_path_length_m=torch.full(shape, .6, dtype=dtype),
                input_projection_raw=raw, input_projection=raw / 32284.894580078173,
                horizontal_angle_deg=torch.linspace(-70, 70, hw, dtype=dtype)[None],
                vertical_angle_deg=torch.linspace(-70, 70, hw, dtype=dtype)[None])


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


def small_model(**kwargs):
    torch.manual_seed(314)
    return SetConditionalField(lift_resolution=4, samples=8, ray_chunk=2,
                               query_chunk=23, **kwargs)


@pytest.mark.parametrize('concentration', [0., .12, 1.])
def test_fp64_constant_integrals_and_exact_midpoint_queries(concentration):
    batch = make_batch(views=1, hw=3, dtype=torch.float64)
    integrator = ContinuousRayIntegrator(samples_per_ray=96, ray_chunk=2)
    queried = []

    def field(case, xyz):
        assert case == 0
        assert xyz.dtype == torch.float64
        queried.append(xyz)
        return torch.full_like(xyz[:, 0], concentration)

    actual = integrator.integrate(field, batch)
    expected = float(chi(torch.tensor(concentration, dtype=torch.float64)) * 6e5)
    torch.testing.assert_close(actual, torch.full_like(actual, expected), atol=1e-10, rtol=1e-14)
    points = torch.cat(queried)
    assert points.shape == (9 * 96, 3)
    x = (.5 + torch.arange(96, dtype=torch.float64)) * .6 / 96
    torch.testing.assert_close(points[:96, 0], x, atol=1e-15, rtol=0)
    torch.testing.assert_close(points[:, 1:], torch.full_like(points[:, 1:], .3))
    normalized = integrator.integrate(field, batch, normalized=True)
    torch.testing.assert_close(normalized, actual / 32284.894580078173)


def test_midpoint_q96_192_384_convergence_on_fixed_continuous_field():
    batch = make_batch(views=1, hw=3, dtype=torch.float64)
    field = lambda case, xyz: inverse_chi(.01 + .2 * (xyz[:, 0] / .6).square())
    truth = 6e5 * (.01 + .2 / 3)
    errors = []
    for q in (96, 192, 384):
        actual = ContinuousRayIntegrator(samples_per_ray=q, ray_chunk=2).integrate(field, batch)
        errors.append(abs(actual[0, 0, 0, 0, 0].item() - truth))
    assert errors[0] > errors[1] > errors[2] > 0
    assert 3.99 < errors[0] / errors[1] < 4.01
    assert 3.99 < errors[1] / errors[2] < 4.01
    assert errors[-1] / truth < 2e-6


def test_continuous_projection_parameter_derivative_finite_difference():
    batch = make_batch(views=3, hw=3, dtype=torch.float64)
    theta = torch.tensor(-2., dtype=torch.float64, requires_grad=True)
    integrator = ContinuousRayIntegrator(samples_per_ray=96, ray_chunk=2)

    def objective():
        return integrator.integrate(lambda case, xyz: torch.sigmoid(
            theta + xyz @ xyz.new_tensor([.5, -.7, .9])), batch, normalized=True).square().mean()

    objective().backward()
    analytic = theta.grad.item()
    with torch.no_grad():
        theta.add_(1e-5)
        plus = objective().item()
        theta.sub_(2e-5)
        minus = objective().item()
        theta.add_(1e-5)
    numerical = (plus - minus) / 2e-5
    assert abs(analytic - numerical) < 1e-7 + abs(numerical) * 1e-8


def test_xyz_endpoint_trilinear_interpolation_and_coordinate_gradients():
    axis = torch.linspace(0, .6, 4, dtype=torch.float64)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
    grid = torch.stack((x + 2 * y + 4 * z, 3 * x - y + .5 * z))[None]
    xyz = torch.tensor([[[0., 0., 0.], [.6, 0., .6], [0., .6, 0.],
                         [.6, .6, .6], [.15, .33, .49]]], dtype=torch.float64, requires_grad=True)
    sampled = sample_trilinear(grid, xyz)
    matrix = xyz.new_tensor([[1., 2., 4.], [3., -1., .5]])
    torch.testing.assert_close(sampled, xyz @ matrix.T, atol=1e-14, rtol=1e-14)
    sampled[:, -1].sum().backward()
    torch.testing.assert_close(xyz.grad[:, -1], matrix.sum(0)[None], atol=1e-13, rtol=1e-13)


def test_segment_box_clipping_and_parallel_miss():
    batch = make_batch(2, dtype=torch.float64)
    batch['input_t_near'].fill_(0)
    batch['input_path_length_m'].fill_(1)
    # A ray parallel to the X faces, but outside Y, misses the cube.
    batch['input_ray_direction'][:, 1, 0, 0] = torch.tensor([0., 0., 1.])
    integrator = ContinuousRayIntegrator(samples_per_ray=96, ray_chunk=2)
    geometry = integrator.prepare(batch)
    assert not geometry.valid_ray[0, 1, 0, 0, 0]
    actual = integrator.integrate(lambda case, xyz: xyz.new_full((len(xyz),), .12), geometry)
    expected = chi(torch.tensor(.12, dtype=torch.float64)) * 6e5
    torch.testing.assert_close(actual[geometry.valid_ray], expected.expand_as(actual[geometry.valid_ray]),
                               rtol=1e-14, atol=1e-10)
    assert actual[0, 1, 0, 0, 0] == 0


def test_empty_case_and_nonfinite_valid_ray_are_rejected():
    empty = make_batch(1)
    empty['input_valid_ray'].zero_()
    with pytest.raises(ValueError, match='valid|usable'):
        small_model().encode(empty)
    malformed = make_batch(1)
    malformed['input_ray_direction'][0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        ContinuousRayIntegrator().prepare(malformed)
    bad_measurement = make_batch(1)
    bad_measurement['input_projection'][0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        small_model().encode(bad_measurement)


def test_coverage_matches_independent_voxel_operator_at_32_cubed():
    batch = make_batch(3, dtype=torch.float64)
    continuous = ContinuousRayIntegrator(samples_per_ray=96, ray_chunk=2)
    voxel = ExplicitRayOperator(samples_per_ray=96, ray_chunk=2)
    expected, _ = voxel.coverage(voxel.prepare(batch, 32))
    actual = continuous.coverage(batch, 32)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    expected_total = batch['input_path_length_m'].sum() * 1e6 / 32284.894580078173
    torch.testing.assert_close(actual.sum(), expected_total, rtol=1e-14, atol=1e-12)


@pytest.mark.parametrize('views', [1, 3, 4, 6])
def test_variable_view_set_symmetry_nan_padding_and_bounded_output(views):
    batch, model = make_batch(views), small_model().eval()
    points = torch.tensor([[.07, .21, .49], [.51, .13, .31]])
    with torch.no_grad():
        expected = model.query(model.encode(batch), points)
        swapped = model.query(model.encode(permute(batch, torch.arange(views - 1, -1, -1))), points)
        padded = model.query(model.encode(pad(batch)), points)
    assert torch.isfinite(expected).all() and ((expected >= 0) & (expected <= 1)).all()
    torch.testing.assert_close(swapped, expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(padded, expected, atol=2e-6, rtol=0)


def test_mixed_variable_views_matches_separate_contexts():
    first, second = pad(make_batch(1)), make_batch(2)
    second['input_projection'] *= 1.6
    second['input_projection_raw'] *= 1.6
    mixed = {key: torch.cat((first[key], second[key]), 0) for key in first}
    model = small_model().eval()
    points = torch.tensor([[.11, .21, .37], [.49, .28, .13]])
    with torch.no_grad():
        actual = model.query(model.encode(mixed), points)
        expected = torch.cat([model.query(model.encode(b), points) for b in (first, second)])
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=0)


def test_padded_invalid_geometry_is_ignored_by_continuous_projection():
    batch = make_batch(3, dtype=torch.float64)
    padded = pad(batch)
    # An invalid pixel inside an active station may also contain arbitrary NaNs.
    padded['input_valid_ray'][:, 1, :, 0, 0] = False
    padded['input_ray_direction'][:, 1, 0, 0] = float('nan')
    padded['input_t_near'][:, 1, :, 0, 0] = float('nan')
    padded['input_path_length_m'][:, 1, :, 0, 0] = float('nan')
    integrator = ContinuousRayIntegrator(samples_per_ray=96, ray_chunk=2)
    actual = integrator.integrate(lambda case, xyz: .1 + .2 * xyz[:, 0], padded)
    expected = integrator.integrate(lambda case, xyz: .1 + .2 * xyz[:, 0], batch)
    assert torch.isfinite(actual).all()
    mask = padded['input_valid_ray'][:, :3]
    torch.testing.assert_close(actual[:, :3][mask], expected[mask])
    assert torch.count_nonzero(actual[:, 3]) == 0
    assert actual[0, 1, 0, 0, 0] == 0


def _checkpoint_comparison(device):
    plain, checked = small_model(checkpoint_query=False), small_model(checkpoint_query=True)
    checked.load_state_dict(plain.state_dict())
    plain, checked = plain.to(device), checked.to(device)
    batch = {key: value.to(device) for key, value in make_batch(3).items()}
    results = []
    for model in (plain, checked):
        context = model.encode(batch)
        projection = model.integrate(context, batch)
        volume = model.sample_grid(context, 4)
        # Both output routes share the encoder graph, as the real loss does.
        loss = (projection / 32284.894580078173).square().mean() + volume.square().mean()
        loss.backward()
        results.append((projection, volume))
    for a, b in zip(results[0], results[1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for (name, a), (_, b) in zip(plain.named_parameters(), checked.named_parameters()):
        assert a.grad is not None and b.grad is not None, name
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all(), name
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4 if device == 'cuda' else 1e-5,
                                   atol=1e-6 if device == 'cuda' else 1e-7, msg=name)
    encoder_grads = [p.grad for p in plain.encoder.parameters()]
    assert sum(g.abs().sum().item() for g in encoder_grads) > 0


def test_checkpoint_preserves_full_encoder_query_projection_and_consistency_gradients():
    _checkpoint_comparison('cpu')


def test_both_continuous_contexts_receive_consistency_gradients():
    model, first = small_model(checkpoint_query=True), make_batch(3)
    second = copy.deepcopy(first)
    second['input_projection'] *= 1.8
    second['input_projection_raw'] *= 1.8
    ca, cb = model.encode(first), model.encode(second)
    ca.features.retain_grad()
    cb.features.retain_grad()
    consistency_discrepancy(model.sample_grid(ca, 4), model.sample_grid(cb, 4)).backward()
    for context in (ca, cb):
        grad = context.features.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.skipif(os.environ.get('PLAN_B_TEST_CUDA') != '1',
                    reason='GPU test is opt-in; set CUDA_VISIBLE_DEVICES to a non-A GPU')
def test_cuda_full_encoder_query_projection_checkpoint_gradients():
    _checkpoint_comparison('cuda')


def test_formal_decoder_contract_and_arbitrary_grid_queries():
    model = small_model().eval()
    linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(61, 64)] + [(64, 64)] * 3 + [(64, 1)]
    with torch.no_grad():
        context = model.encode(make_batch(3))
        assert context.features.shape == (1, 32, 4, 4, 4)
        assert context.coverage.shape == (1, 2, 4, 4, 4)
        volume = model.sample_grid(context, 5)
        endpoint = model.query(context, torch.tensor([[0., 0., 0.], [.6, .6, .6]]))
    assert volume.shape == (1, 1, 5, 5, 5)
    torch.testing.assert_close(volume[:, 0, 0, 0, 0], endpoint[..., 0].reshape(-1))
    torch.testing.assert_close(volume[:, 0, -1, -1, -1], endpoint[..., 1].reshape(-1))


def test_query_uses_all_four_fourier_bands_and_two_coverage_features():
    model = small_model(checkpoint_query=False).eval()
    context = FieldContext(torch.zeros((1, 32, 4, 4, 4)),
                           torch.stack((torch.full((4, 4, 4), .25),
                                        torch.full((4, 4, 4), .75)))[None])
    xyz = torch.tensor([[.07, .23, .51]])
    captured = []
    hook = model.decoder[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    with torch.no_grad():
        model.query(context, xyz)
    hook.remove()
    vector = captured[0]
    norm = xyz * (2 / .6) - 1
    torch.testing.assert_close(vector[0, :, :32], torch.zeros((1, 32)))
    torch.testing.assert_close(vector[0, :, 32:35], norm)
    for band in range(4):
        phase = norm * (2 ** band * torch.pi)
        torch.testing.assert_close(vector[0, :, 35 + 6 * band:38 + 6 * band], phase.sin())
        torch.testing.assert_close(vector[0, :, 38 + 6 * band:41 + 6 * band], phase.cos())
    torch.testing.assert_close(vector[0, :, -2:], torch.tensor([[.25, .75]]))


def test_projection_does_not_construct_voxel_concentration_grid():
    model = small_model().eval()
    batch = make_batch(1)
    def forbidden(*args, **kwargs):
        raise AssertionError('Plan B projection must query the continuous field directly')
    model.sample_grid = forbidden
    with torch.no_grad():
        result = model.integrate(model.encode(batch), batch)
    assert torch.isfinite(result).all()


def test_model_predictions_depend_on_measured_values():
    model, batch = small_model().eval(), make_batch(3)
    changed = copy.deepcopy(batch)
    changed['input_projection_raw'] *= 2
    changed['input_projection'] *= 2
    points = torch.tensor([[.09, .17, .23], [.43, .39, .51]])
    with torch.no_grad():
        first = model.query(model.encode(batch), points)
        second = model.query(model.encode(changed), points)
    assert torch.max(torch.abs(first - second)) > 1e-7
