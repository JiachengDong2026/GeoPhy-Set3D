import unittest
import torch
from geophy_set3d.models.ray_relational_interaction import (
    RayGeometryDescriptor, LocalObservabilityDescriptor, ObservabilityAwareRayInteraction)


class RayRelationalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(81)
        self.f = torch.randn(1, 4, 32, 2, 2, 2)
        self.g = torch.randn(1, 4, 4, 2, 2, 2)
        self.visible = torch.ones(1, 4, 2, 2, 2, dtype=torch.bool)
        self.vm = torch.ones(1, 4, dtype=torch.bool)
        self.xyz = torch.rand(8, 3)*.6
        self.origins = torch.tensor([[[-6., 0., 0.], [0., -6., 1.], [6., 0., -1.], [1., 6., 2.]]])
        self.azi = torch.tensor([[0., 45., 90., 135.]])
        self.rays = RayGeometryDescriptor(); self.obs = LocalObservabilityDescriptor()
        self.model = ObservabilityAwareRayInteraction()
        self.d, self.r, self.o = self.descriptors()

    def descriptors(self, visible=None, vm=None):
        visible = self.visible if visible is None else visible
        vm = self.vm if vm is None else vm
        valid = visible & vm[:, :, None, None, None]
        d, r = self.rays(self.origins, self.xyz, self.azi, valid)
        mask = valid.flatten(2).transpose(1, 2).reshape(-1, 4)
        o = self.obs(d, mask, vm.sum(1).expand(8))
        return d, r, o

    def run_block(self, f=None, g=None, visible=None, vm=None, r=None, o=None):
        return self.model(self.f if f is None else f, self.g if g is None else g,
            self.visible if visible is None else visible, self.vm if vm is None else vm,
            self.r if r is None else r, self.o if o is None else o, True)

    def test_ray_direction_from_station_to_voxel(self):
        expected = torch.nn.functional.normalize(self.xyz[0]-self.origins[0], dim=-1)
        torch.testing.assert_close(self.d[0], expected)
        self.assertEqual(self.d.shape, (8, 4, 3))

    def test_pairwise_angle_range(self):
        self.assertTrue(((self.r >= 0) & (self.r <= 1)).all())
        angle = torch.acos(self.r[..., 0])/(torch.pi/2)
        self.assertTrue(((angle >= 0) & (angle <= 1)).all())
        torch.testing.assert_close(self.r[:, torch.arange(4), torch.arange(4)],
                                  torch.tensor([1., 0., 0.]).expand(8, 4, 3))

    def test_pairwise_geometry_symmetry(self):
        torch.testing.assert_close(self.r, self.r.transpose(1, 2), atol=0, rtol=0)

    def test_pairwise_geometry_permutation_consistency(self):
        p = torch.tensor([2, 0, 3, 1])
        d, r = self.rays(self.origins[:, p], self.xyz, self.azi[:, p], self.visible[:, p])
        torch.testing.assert_close(d, self.d[:, p])
        torch.testing.assert_close(r, self.r[:, p][:, :, p])

    def test_observability_tensor_symmetric(self):
        _, s = self.obs(self.d, torch.ones(8, 4, dtype=torch.bool), torch.full((8,), 4), True)
        torch.testing.assert_close(s, s.transpose(-1, -2), atol=0, rtol=0)
        torch.testing.assert_close(s.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(8))

    def test_observability_eigenvalues_finite(self):
        self.assertTrue(torch.isfinite(self.o).all())
        self.assertTrue((self.o[:, :3] >= 0).all())
        self.assertTrue((self.o[:, :2] >= self.o[:, 1:3]).all())
        torch.testing.assert_close(self.o[:, :3].sum(-1), torch.ones(8))

    def test_observability_permutation_invariance(self):
        p = torch.tensor([2, 0, 3, 1])
        other = self.obs(self.d[:, p], torch.ones(8, 4, dtype=torch.bool), torch.full((8,), 4))
        torch.testing.assert_close(self.o, other, atol=3e-7, rtol=0)

    def test_observability_all_invalid_zero(self):
        o, s = self.obs(torch.full_like(self.d, float('nan')),
            torch.zeros(8, 4, dtype=torch.bool), torch.zeros(8), True)
        self.assertTrue((o == 0).all() and (s == 0).all())

    def test_synthetic_geometry_sanity(self):
        # Noncoplanar 3D rays; coplanar sets may have lambda3=0 even when spread.
        desc = []
        for angles in ([0., 5., 10., 15.], [0., 45., 90., 135.]):
            a = torch.deg2rad(torch.tensor(angles))
            d = torch.stack((a.cos(), a.sin(), torch.full_like(a, .2)), -1)
            d = torch.nn.functional.normalize(d, dim=-1)[None]
            desc.append(self.obs(d, torch.ones(1, 4, dtype=torch.bool), torch.tensor([4])))
        self.assertTrue(all(torch.isfinite(o).all() for o in desc))
        self.assertGreater(float((desc[0]-desc[1]).abs().max()), .01)

    def test_oari_permutation_equivariance(self):
        p = torch.tensor([2, 0, 3, 1]); c, a, bias = self.run_block()
        other, ap, bp = self.run_block(self.f[:, p], self.g[:, p], self.visible[:, p],
            self.vm[:, p], self.r[:, p][:, :, p])
        torch.testing.assert_close(c[:, p], other, atol=2e-6, rtol=0)
        torch.testing.assert_close(a[:, :, p][:, :, :, p], ap, atol=3e-7, rtol=0)
        torch.testing.assert_close(bias[:, :, p][:, :, :, p], bp, atol=0, rtol=0)

    def test_oari_padding_invariance(self):
        c, a, _ = self.run_block()
        f = torch.cat((self.f, torch.full_like(self.f[:, :1], float('nan'))), 1)
        g = torch.cat((self.g, torch.full_like(self.g[:, :1], float('nan'))), 1)
        visible = torch.cat((self.visible, torch.ones_like(self.visible[:, :1])), 1)
        vm = torch.cat((self.vm, torch.zeros_like(self.vm[:, :1])), 1)
        r = torch.full((8, 5, 5, 3), float('nan')); r[:, :4, :4] = self.r
        other, ap, _ = self.run_block(f, g, visible, vm, r)
        torch.testing.assert_close(c, other[:, :4], atol=2e-6, rtol=0)
        torch.testing.assert_close(a, ap[:, :, :4, :4], atol=3e-7, rtol=0)
        self.assertTrue((other[:, 4] == 0).all())

    def test_oari_all_invalid_safe(self):
        f = torch.full_like(self.f, float('nan')); g = torch.full_like(self.g, float('nan'))
        c, a, bias = self.run_block(f, g, torch.zeros_like(self.visible),
            r=torch.full_like(self.r, float('nan')), o=torch.full_like(self.o, float('nan')))
        self.assertTrue((c == 0).all() and (a == 0).all() and (bias == 0).all())
        c.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_invalid_view_does_not_change_valid_context(self):
        vm = self.vm.clone(); vm[:, -1] = False
        _, r, o = self.descriptors(vm=vm)
        normal, _, _ = self.run_block(vm=vm, r=r, o=o)
        f, g = self.f.clone(), self.g.clone(); f[:, -1] = float('nan'); g[:, -1] = float('nan')
        r[:, -1] = float('nan'); r[:, :, -1] = float('nan')
        other, _, _ = self.run_block(f, g, vm=vm, r=r, o=o)
        torch.testing.assert_close(normal, other, atol=0, rtol=0)

    def test_invalid_pairwise_bias_masked(self):
        torch.nn.init.normal_(self.model.pairwise_bias[-1].weight)
        vm = self.vm.clone(); vm[:, -1] = False
        _, a, bias = self.run_block(vm=vm)
        self.assertTrue((bias[:, :, -1] == 0).all() and (bias[:, :, :, -1] == 0).all())
        self.assertTrue((a[:, :, -1] == 0).all() and (a[:, :, :, -1] == 0).all())

    def test_single_valid_view_safe(self):
        vm = torch.tensor([[True, False, False, False]])
        _, r, o = self.descriptors(vm=vm)
        torch.testing.assert_close(o[:, :3], torch.tensor([1., 0., 0.]).expand(8, 3), atol=3e-7, rtol=0)
        c, a, _ = self.run_block(vm=vm, r=r, o=o)
        self.assertTrue(torch.isfinite(c).all() and torch.isfinite(a).all())
        self.assertTrue((a[:, :, 0, 0] == 1).all() and (c[:, 1:] == 0).all())

    def test_visible_fraction_excludes_padding(self):
        visible = self.visible.clone(); visible[:, -1] = False
        _, _, o = self.descriptors(visible=visible)
        torch.testing.assert_close(o[:, -1], torch.full((8,), .75))
        vm = self.vm.clone(); vm[:, -1] = False
        _, _, o = self.descriptors(visible=visible, vm=vm)
        torch.testing.assert_close(o[:, -1], torch.ones(8))

    def test_pairwise_bias_zero_init(self):
        _, _, bias = self.run_block()
        self.assertTrue((bias == 0).all())

    def test_pairwise_bias_changes_attention(self):
        # Deliberately activate the initially-zero final layer to isolate its
        # connection to logits, using identical features and geometry tokens.
        with torch.no_grad():
            self.model.pairwise_bias[0].weight.zero_(); self.model.pairwise_bias[0].bias.zero_()
            self.model.pairwise_bias[0].weight[0, 0] = 1
            self.model.pairwise_bias[-1].weight.zero_(); self.model.pairwise_bias[-1].bias.zero_()
            self.model.pairwise_bias[-1].weight[:, 0] = 1
        f, g = self.f[:, :1].expand_as(self.f), self.g[:, :1].expand_as(self.g)
        _, a, _ = self.run_block(f, g)
        _, other, _ = self.run_block(f, g, r=torch.zeros_like(self.r))
        self.assertGreater(float((a-other).abs().max()), .01)

    def test_observability_changes_interaction(self):
        # Fixed features, per-view descriptors and relations; change ray-set
        # diversity only through the independently computed observability token.
        parallel = self.d[:, :1].expand_as(self.d)
        alternative = self.obs(parallel, torch.ones(8, 4, dtype=torch.bool), torch.full((8,), 4))
        c, a, _ = self.run_block()
        other, ap, _ = self.run_block(o=alternative)
        self.assertGreater(float((c-other).abs().max()), 1e-5)

    def test_pairwise_bias_gradient_nonzero(self):
        c, _, _ = self.run_block(); c.square().mean().backward()
        self.assertGreater(float(self.model.pairwise_bias[-1].weight.grad.abs().sum()), 0)
        # The hidden-layer gradient becomes active after the zero final layer's
        # first update, which is the expected zero-init behavior.
        torch.optim.SGD(self.model.parameters(), lr=.1).step(); self.model.zero_grad()
        c, _, _ = self.run_block(); c.square().mean().backward()
        self.assertGreater(float(self.model.pairwise_bias[0].weight.grad.abs().sum()), 0)

    def test_observability_embedding_gradient_nonzero(self):
        c, _, _ = self.run_block(); c.square().mean().backward()
        self.assertGreater(float(self.model.observability_embedding[0].weight.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()))
