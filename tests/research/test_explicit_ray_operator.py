"""Numerical contracts for the explicit, higher-order physical ray operator."""
import copy
import unittest

import numpy as np
import torch

from geophy_set3d.geometry.tdlas import ray_box_intersection
from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator
from geophy_set3d.physics.projectors import differentiable_project, mass_to_mole


def asymmetric_batch(dtype=torch.float64):
    origins = np.array([[-.2, .11, .07], [.12, -.2, .19]], dtype=np.float64)
    directions = np.array([[[[1, .21, .39], [1, .38, .16], [1, -.1, .43]]],
                           [[[.18, 1, .31], [.41, 1, -.12], [.07, 1, .47]]]], dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    near, far, length, valid = ray_box_intersection(origins[:, None, None], directions)
    tensor = lambda a: torch.as_tensor(a, dtype=dtype)
    return dict(input_station_origin_m=tensor(origins)[None],
                input_ray_direction=tensor(directions)[None],
                input_t_near=tensor(near)[None, :, None],
                input_t_far=tensor(far)[None, :, None],
                input_path_length_m=tensor(length)[None, :, None],
                input_valid_ray=torch.as_tensor(valid)[None, :, None],
                view_mask=torch.ones(1, len(origins), dtype=torch.bool))


def old_geometry(batch):
    return dict(station_origin_m=batch['input_station_origin_m'][0],
                ray_direction=batch['input_ray_direction'][0],
                t_near=batch['input_t_near'][0, :, 0],
                path_length_m=batch['input_path_length_m'][0, :, 0],
                valid_ray=batch['input_valid_ray'][0, :, 0])


class ExplicitRayOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(314159)

    def test_fp64_constant_and_asymmetric_xyz_against_independent_midpoints(self):
        batch = asymmetric_batch()
        op = ExplicitRayOperator(samples_per_ray=96, ray_chunk=2)
        stencil = op.prepare(batch, 9)
        for value in (0., .001, .5, 1.):
            volume = torch.full((1, 1, 9, 9, 9), value, dtype=torch.float64)
            expected = 1e6 * mass_to_mole(value) * batch['input_path_length_m']
            torch.testing.assert_close(op(volume, stencil), expected, rtol=1e-12, atol=1e-8)
        axis = torch.linspace(0, .6, 9, dtype=torch.float64)
        x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
        for coefficients in ((.019, 0, 0), (0, .051, 0), (0, 0, .087), (.019, .051, .087)):
            c = (.013 + coefficients[0]*x + coefficients[1]*y + coefficients[2]*z)[None, None]
            actual = op(c, stencil)
            expected = torch.zeros_like(actual)
            # This independently evaluates the analytic affine concentration at
            # ray midpoints; it never calls a volume interpolation routine.
            for v in range(2):
                for col in range(3):
                    near = batch['input_t_near'][0, v, 0, 0, col].item()
                    length = batch['input_path_length_m'][0, v, 0, 0, col].item()
                    times = near + (np.arange(96)+.5)/96 * length
                    xyz = (batch['input_station_origin_m'][0, v].numpy() +
                           times[:, None]*batch['input_ray_direction'][0, v, 0, col].numpy())
                    concentration = .013 + xyz @ np.asarray(coefficients)
                    expected[0, v, 0, 0, col] = mass_to_mole(concentration).mean()*length*1e6
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-8)

    def test_forward_matches_unchanged_training_projector_fp32_fp64(self):
        for dtype, rtol, atol in ((torch.float64, 1e-12, 1e-8), (torch.float32, 2e-6, .03)):
            batch = asymmetric_batch(dtype)
            c = torch.rand(1, 1, 11, 11, 11, dtype=dtype)*.4
            op = ExplicitRayOperator(ray_chunk=2)
            expected = differentiable_project(c[0, 0], old_geometry(batch), 96)[None, :, None]
            actual = op(c, batch)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            torch.testing.assert_close(actual/op.projection_scale_ppm_m,
                                       op.project_normalized(c, batch))

    def test_jvp_vjp_adjoint_autograd_and_finite_difference(self):
        batch = asymmetric_batch()
        c = (.05 + torch.rand(1, 1, 5, 5, 5, dtype=torch.float64)*.2).requires_grad_()
        op = ExplicitRayOperator(samples_per_ray=13, ray_chunk=2)
        stencil = op.prepare(batch, 5)
        v = torch.randn_like(c)
        p = op.project_normalized(c, stencil)
        u = torch.randn_like(p)
        jv, jtu = op.jvp(c, v, stencil), op.vjp(c, u, stencil)
        torch.testing.assert_close((jv*u).sum(), (v*jtu).sum(), rtol=1e-12, atol=1e-12)
        reference, = torch.autograd.grad((p*u).sum(), c, create_graph=True)
        torch.testing.assert_close(jtu, reference, rtol=1e-12, atol=1e-12)
        _, autograd_jv = torch.autograd.functional.jvp(lambda x: op.project_normalized(x, stencil), c, v)
        torch.testing.assert_close(jv, autograd_jv, rtol=1e-12, atol=1e-12)
        epsilon = 1e-6
        difference = (op.project_normalized(c+epsilon*v, stencil)-op.project_normalized(c-epsilon*v, stencil))/(2*epsilon)
        torch.testing.assert_close(jv, difference, rtol=1e-8, atol=1e-8)
        with torch.no_grad():
            torch.testing.assert_close(op.vjp(c, u, stencil), jtu)
            torch.testing.assert_close(op.jvp(c, v, stencil), jv)

    def test_higher_derivatives_of_full_residual_gradient(self):
        c = (.1 + .1*torch.rand(1, 1, 3, 3, 3, dtype=torch.float64)).requires_grad_()
        op = ExplicitRayOperator(samples_per_ray=7, ray_chunk=2)
        stencil = op.prepare(asymmetric_batch(), 3)
        target = torch.randn(stencil.projection_shape, dtype=c.dtype)
        def analytic_gradient(x):
            residual = op.project_normalized(x, stencil)-target
            return op.vjp(x, residual, stencil)/stencil.valid_counts.reshape(-1, 1, 1, 1, 1)
        energy = .5*(op.project_normalized(c, stencil)-target).square().sum()/stencil.valid_counts[0]
        reference, = torch.autograd.grad(energy, c, create_graph=True)
        actual = analytic_gradient(c)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        probe = torch.randn_like(c)
        hv, = torch.autograd.grad((actual*probe).sum(), c)
        epsilon = 1e-6
        hv_fd = (analytic_gradient(c+epsilon*probe)-analytic_gradient(c-epsilon*probe))/(2*epsilon)
        torch.testing.assert_close(hv, hv_fd, rtol=1e-8, atol=1e-8)
        self.assertTrue(torch.autograd.gradcheck(analytic_gradient, (c,), eps=1e-6, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.autograd.gradgradcheck(lambda x: op.project_normalized(x, stencil),
                                                    (c,), eps=1e-6, atol=1e-6, rtol=1e-5))

    def test_fused_residual_gradient_matches_unfused_through_second_derivatives(self):
        op = ExplicitRayOperator(samples_per_ray=7, ray_chunk=2)
        stencil = op.prepare(asymmetric_batch(), 3)
        c = (.1 + .1*torch.rand(1, 1, 3, 3, 3, dtype=torch.float64)).requires_grad_()
        target = torch.randn(stencil.projection_shape, dtype=c.dtype, requires_grad=True)
        prediction, gradient = op.residual_gradient(c, target, stencil)
        reference_prediction = op.project_normalized(c, stencil)
        reference = op.vjp(c, reference_prediction-target, stencil)
        torch.testing.assert_close(prediction, reference_prediction, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(gradient, reference, rtol=1e-13, atol=1e-13)
        probe = torch.randn_like(c)
        actual_derivatives = torch.autograd.grad((gradient*probe).sum(), (c, target), create_graph=True)
        expected_derivatives = torch.autograd.grad((reference*probe).sum(), (c, target), create_graph=True)
        for actual, expected in zip(actual_derivatives, expected_derivatives):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.autograd.gradgradcheck(
            lambda x, y: op.residual_gradient(x, y, stencil)[1], (c, target),
            eps=1e-6, atol=1e-6, rtol=1e-5))
        with torch.no_grad():
            _, inference_gradient = op.residual_gradient(c, target, stencil)
            torch.testing.assert_close(inference_gradient, reference)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA cache needs a CUDA device')
    def test_cuda_geometry_cache_bound_invalidation_and_dtype(self):
        batch = asymmetric_batch(torch.float32)
        op = ExplicitRayOperator(samples_per_ray=7, ray_chunk=2, device_cache_budget_bytes=1_000_000)
        stencil = op.prepare(batch, 5)
        c = torch.rand(1, 1, 5, 5, 5, device='cuda', dtype=torch.float32)*.1
        expected = ExplicitRayOperator(samples_per_ray=7, ray_chunk=2, device_cache_budget_bytes=0)(c, batch)
        torch.testing.assert_close(op(c, stencil), expected)
        first = op._resident(stencil.stations[0][2], c.device, c.dtype)
        self.assertIs(first, op._resident(stencil.stations[0][2], c.device, c.dtype))
        self.assertIs(first, op._resident(stencil.stations[0][2], torch.device('cuda'), c.dtype))
        double = op._resident(stencil.stations[0][2], c.device, torch.float64)
        self.assertIsNot(first, double)
        self.assertEqual(double.weights.dtype, torch.float64)
        changed = copy.deepcopy(batch)
        changed['input_t_near'] += .0001
        different = op.prepare(changed, 5)
        self.assertIsNot(first, op._resident(different.stations[0][2], c.device, c.dtype))
        self.assertLessEqual(op.device_cache_bytes, op.device_cache_budget_bytes)
        # Force eviction, while a previously prepared graph still owns its data.
        op.clear_cache()
        op.device_cache_budget_bytes = first.nbytes
        for _, _, station in stencil.stations:
            op._resident(station, c.device, c.dtype)
        self.assertEqual(len(op._device_cache), 1)
        self.assertLessEqual(op.device_cache_bytes, op.device_cache_budget_bytes)
        op.clear_cache()
        self.assertEqual(op.device_cache_bytes, 0)

    def test_coverage_dense_linear_analytic_and_per_case_normalization(self):
        # A single edge-parallel ray samples x=.15,.45. For N=3 its
        # integration coefficients are (.25,.5,.25)*normalized path length.
        batch = dict(input_station_origin_m=torch.zeros(2, 2, 3, dtype=torch.float64),
                     input_ray_direction=torch.tensor([1., 0., 0.], dtype=torch.float64).expand(2, 2, 1, 1, 3).clone(),
                     input_t_near=torch.zeros(2, 2, 1, 1, 1, dtype=torch.float64),
                     input_path_length_m=torch.full((2, 2, 1, 1, 1), .6, dtype=torch.float64),
                     input_valid_ray=torch.ones(2, 2, 1, 1, 1, dtype=torch.bool),
                     view_mask=torch.tensor([[True, False], [True, True]]))
        op = ExplicitRayOperator(samples_per_ray=2, ray_chunk=1)
        stencil = op.prepare(batch, 3)
        first, second = op.coverage(stencil)
        normalized_length = .6e6/op.projection_scale_ppm_m
        t = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float64)
        t[0, 0, :, 0, 0] = torch.tensor([.25, .5, .25], dtype=torch.float64)*normalized_length
        torch.testing.assert_close(first[0:1], t)
        torch.testing.assert_close(first[1:2], 2*t)
        torch.testing.assert_close(second[0:1], t*normalized_length)
        torch.testing.assert_close(second[1:2], 2*t*normalized_length)
        d0 = op.geometry_preconditioner(stencil)
        torch.testing.assert_close(d0[0], d0[1])
        torch.testing.assert_close(d0[:1], t*normalized_length*(28.965/16.043)**2)
        # At C=0, J=d_max*T. The row-sum diagonal controls its Gram.
        matrix = (t.reshape(-1)*(28.965/16.043))[None]
        gram = matrix.T @ matrix
        majorizer = torch.diag(d0[0].reshape(-1))
        self.assertGreaterEqual(torch.linalg.eigvalsh(majorizer-gram).min().item(), -1e-12)

    def test_padding_invalid_nan_masking_and_cache(self):
        op = ExplicitRayOperator(samples_per_ray=11, ray_chunk=2, cache_budget_bytes=100_000)
        batch = asymmetric_batch()
        c = torch.rand(1, 1, 5, 5, 5, dtype=torch.float64)*.1
        a = op.prepare(batch, 5)
        b = op.prepare(batch, 5)
        self.assertIs(a.stations[0][2], b.stations[0][2])
        self.assertLessEqual(op.cache_bytes, op.cache_budget_bytes)
        for _, _, station in a.stations:
            self.assertEqual(station.indices.device.type, 'cpu')
            self.assertEqual(station.weights.device.type, 'cpu')
            self.assertTrue((station.weights >= 0).all())
            torch.testing.assert_close(station.weights.sum(-1), torch.ones_like(station.factors[:, None]).expand(-1, 11))
        padded = copy.deepcopy(batch)
        for key in padded:
            value = padded[key]
            pad = torch.full_like(value[:, :1], False if value.dtype == torch.bool else float('nan'))
            padded[key] = torch.cat([value, pad], 1)
        result = op(c, padded)
        torch.testing.assert_close(result[:, :2], op(c, batch))
        self.assertEqual(result[:, 2].count_nonzero().item(), 0)
        # Invalid individual rays can carry arbitrary NaN metadata and duals.
        padded['input_valid_ray'][0, 0, 0, 0, 1] = False
        padded['input_ray_direction'][0, 0, 0, 1] = float('nan')
        padded['input_t_near'][0, 0, 0, 0, 1] = float('nan')
        padded['input_path_length_m'][0, 0, 0, 0, 1] = float('nan')
        st = op.prepare(padded, 5)
        u = torch.ones_like(result)
        u[~st.valid_ray] = float('nan')
        self.assertTrue(torch.isfinite(op(c, st)).all())
        self.assertTrue(torch.isfinite(op.vjp(c, u, st)).all())
        fused_prediction, fused_gradient = op.residual_gradient(c, u, st)
        self.assertTrue(torch.isfinite(fused_prediction).all())
        self.assertTrue(torch.isfinite(fused_gradient).all())
        changed = copy.deepcopy(batch)
        changed['input_t_near'] += .0001
        different = op.prepare(changed, 5)
        self.assertIsNot(a.stations[0][2], different.stations[0][2])
        op.clear_cache()
        self.assertEqual(op.cache_bytes, 0)

    def test_empty_case_and_mismatched_configuration_rejected(self):
        batch = asymmetric_batch()
        batch['view_mask'].zero_()
        with self.assertRaisesRegex(ValueError, 'at least one valid ray'):
            ExplicitRayOperator().prepare(batch, 5)
        batch = asymmetric_batch()
        c = torch.rand(1, 1, 5, 5, 5, dtype=torch.float64)
        op = ExplicitRayOperator(samples_per_ray=7)
        stencil = op.prepare(batch, 5)
        with self.assertRaisesRegex(ValueError, 'different operator'):
            ExplicitRayOperator(samples_per_ray=8)(c, stencil)


if __name__ == '__main__':
    unittest.main()
