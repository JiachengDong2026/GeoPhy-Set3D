import unittest
import numpy as np
from scipy.integrate import quad
import torch

from geophy_set3d.geometry.tdlas import ray_box_intersection
from geophy_set3d.physics.projectors import differentiable_project, mass_to_mole


def asymmetric_rays():
    origins = np.array([[-.2,.11,.07], [.12,-.2,.19]], dtype=np.float64)
    directions = np.array([[[[1,.21,.39], [1,.38,.16], [1,-.1,.43]]],
                           [[[.18,1,.31], [.41,1,-.12], [.07,1,.47]]]], dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    near, far, length, valid = ray_box_intersection(origins[:, None, None], directions)
    return dict(station_origin_m=origins, ray_direction=directions,
                t_near=near, t_far=far, path_length_m=length, valid_ray=valid)


class ProjectorResolutionTests(unittest.TestCase):
    def test_constant_64_and_128_pic_and_invalid_ray(self):
        geometry = asymmetric_rays()
        geometry['valid_ray'][0, 0, 2] = False
        geometry['path_length_m'][0, 0, 2] = 0
        for dtype, tolerance in [(torch.float64, 1e-12), (torch.float32, 3e-7)]:
            for constant in (0., .001, .5, 1.):
                outputs = []
                for resolution in (64, 128):
                    volume = torch.full((resolution,)*3, constant, dtype=dtype)
                    result = differentiable_project(volume, geometry, 96).numpy()
                    actual_constant = volume[0, 0, 0].item()
                    expected = 1e6 * mass_to_mole(actual_constant) * geometry['path_length_m']
                    expected[~geometry['valid_ray']] = 0
                    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=1e-9)
                    outputs.append(result)
                np.testing.assert_allclose(*outputs, rtol=tolerance, atol=1e-9)

    def test_asymmetric_affine_xyz_independent_quadrature_and_autograd(self):
        geometry = asymmetric_rays()
        coefficients = np.array([.019, .051, .087])
        expected = np.zeros(geometry['valid_ray'].shape)
        midpoint = np.zeros_like(expected)
        quadrature_bound = np.zeros_like(expected)
        for view, row, column in np.argwhere(geometry['valid_ray']):
            origin = geometry['station_origin_m'][view]
            direction = geometry['ray_direction'][view, row, column]
            near, far = geometry['t_near'][view, row, column], geometry['t_far'][view, row, column]
            expected[view, row, column] = 1e6 * quad(
                lambda t: mass_to_mole(.013 + np.dot(coefficients, origin+t*direction)),
                near, far, epsabs=1e-13)[0]
            positions = near + (np.arange(96)+.5)/96 * (far-near)
            c = .013 + (origin[None] + positions[:, None]*direction) @ coefficients
            midpoint[view, row, column] = 1e6 * mass_to_mole(c).mean() * (far-near)
            # CH4 mass->mole is nonlinear even for an affine mass-fraction field.
            # Composite-midpoint error <= L^3 max|f''|/(24 K^2).
            a, b = 1/16.043, 1/28.965
            minimum_c = min(.013 + np.dot(coefficients, origin+near*direction),
                            .013 + np.dot(coefficients, origin+far*direction))
            second_derivative = (2*a*b*(a-b)*np.dot(coefficients, direction)**2
                                 / (b+(a-b)*minimum_c)**3)
            quadrature_bound[view, row, column] = 1e6*(far-near)**3*second_derivative/(24*96**2)
        outputs = []
        for resolution in (64, 128):
            axis = np.linspace(0, .6, resolution)
            x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
            field = .013 + coefficients[0]*x + coefficients[1]*y + coefficients[2]*z
            volume = torch.tensor(field, dtype=torch.float64, requires_grad=True)
            projection = differentiable_project(volume, geometry, 96)
            np.testing.assert_allclose(projection.detach().numpy(), midpoint, rtol=1e-12, atol=1e-8)
            self.assertTrue(np.all(np.abs(projection.detach().numpy()-expected)
                                   <= quadrature_bound*1.001+1e-8))
            projection.sum().backward()
            self.assertTrue(torch.isfinite(volume.grad).all())
            self.assertGreater(volume.grad.abs().sum().item(), 0)
            outputs.append(projection.detach().numpy())
        np.testing.assert_allclose(*outputs, rtol=1e-12, atol=1e-8)
