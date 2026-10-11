import unittest
import torch
from geophy_set3d.models.attention_fusion import GeometryConditionedAttentionPooling, normalized_attention_entropy


class AttentionFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.fusion = GeometryConditionedAttentionPooling()
        self.features = torch.randn(2, 4, 32, 3, 3, 3, requires_grad=True)
        self.geometry = torch.randn(2, 4, 4, 3, 3, 3)
        self.visibility = torch.rand(2, 4, 3, 3, 3) > .25
        self.visibility[:, :, 0, 0, 0] = False
        self.view_mask = torch.tensor([[True, True, True, False], [True]*4])

    def test_shape_mask_and_all_invalid_backward(self):
        self.features.data[0, -1] = float('nan')
        self.geometry[0, -1] = float('nan')
        fused, weights = self.fusion(self.features, self.geometry, self.visibility, self.view_mask)
        valid = self.visibility & self.view_mask[:, :, None, None, None]
        self.assertEqual(fused.shape, (2, 32, 3, 3, 3))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue((weights[~valid] == 0).all())
        self.assertTrue((fused[:, :, 0, 0, 0] == 0).all())
        torch.testing.assert_close(weights.sum(1), valid.any(1).float())
        fused.square().mean().backward()
        for parameter in self.fusion.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(sum(p.grad.abs().sum().item() for p in self.fusion.parameters()), 0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in self.fusion.geometry_embedding.parameters()), 0)

    def test_permutation_padding_and_geometry_conditioning(self):
        fused, weights = self.fusion(self.features, self.geometry, self.visibility, self.view_mask)
        p = torch.tensor([3, 1, 0, 2])
        other, reordered = self.fusion(self.features[:, p], self.geometry[:, p], self.visibility[:, p], self.view_mask[:, p])
        torch.testing.assert_close(fused, other, atol=5e-7, rtol=1e-6)
        torch.testing.assert_close(weights[:, p], reordered)
        padded = self.fusion(torch.cat((self.features, torch.full_like(self.features[:, :1], float('nan'))), 1),
                            torch.cat((self.geometry, torch.full_like(self.geometry[:, :1], float('nan'))), 1),
                            torch.cat((self.visibility, torch.ones_like(self.visibility[:, :1])), 1),
                            torch.cat((self.view_mask, torch.zeros_like(self.view_mask[:, :1])), 1))
        torch.testing.assert_close(fused, padded[0], atol=5e-7, rtol=1e-6)
        self.assertTrue((padded[1][:, -1] == 0).all())
        _, changed = self.fusion(self.features, self.geometry.flip(1), self.visibility, self.view_mask)
        self.assertGreater((weights-changed).abs().max().item(), 1e-5)

    def test_uniform_scores_recover_mean(self):
        from geophy_set3d.models.fusion import masked_mean
        for parameter in self.fusion.parameters():
            parameter.data.zero_()
        fused, weights = self.fusion(self.features, self.geometry, self.visibility, self.view_mask)
        valid = self.visibility & self.view_mask[:, :, None, None, None]
        torch.testing.assert_close(fused, masked_mean(self.features, valid))
        self.assertAlmostEqual(normalized_attention_entropy(weights, valid).item(), 1., places=6)

    def test_whole_batch_invalid(self):
        self.visibility.zero_()
        fused, weights = self.fusion(self.features, self.geometry, self.visibility, self.view_mask)
        self.assertTrue((fused == 0).all() and (weights == 0).all())
        fused.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.fusion.parameters()))
