import unittest

import torch

from geophy_set3d.models.geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D
from geophy_set3d.models.geolift_oari_mrgr_c0_adapter import GeoLiftOARIMRGRC0Adapter, C0ObservabilityAdapter
from geophy_set3d.training.utils import permute_views
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.data.collate import projection_pool_collate
from pool_fixture import make_pool


class C0AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary, root, _, _ = make_pool()
        ds = MeasurementOnlyPoolDataset(root, split='train', min_input_views=3, max_input_views=3)
        cls.batch = projection_pool_collate([ds[0]])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def models(self):
        torch.manual_seed(31)
        base = GeoLiftOARIMRGRRes3D().eval()
        with torch.no_grad():
            base.residual_adapter[-1].weight.normal_(0, .02)
        candidate = GeoLiftOARIMRGRC0Adapter().eval()
        missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(set(missing), {f'c0_adapter.body.{i}.{k}' for i in (0, 2) for k in ('weight', 'bias')})
        self.assertEqual(unexpected, [])
        return base, candidate

    def test_zero_initialization_preserves_full_feedback_output(self):
        base, candidate = self.models()
        with torch.no_grad():
            a = base(self.batch, return_stages=True)
            b = candidate(self.batch, return_stages=True)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, msg=key)
        self.assertEqual(sum(p.numel() for p in candidate.parameters()) -
                         sum(p.numel() for p in base.parameters()), 1152)

    def test_active_adapter_preserves_permutation_and_padding(self):
        _, model = self.models()
        with torch.no_grad():
            model.c0_adapter.body[-1].weight.normal_(0, .01)
        padded = dict(self.batch)
        for k, v in self.batch.items():
            if (k.startswith('input_') or k == 'view_mask') and isinstance(v, torch.Tensor):
                padded[k] = torch.cat((v, torch.zeros_like(v[:, :1])), 1)
        padded['input_projection'][:, -1] = float('nan')
        padded['input_projection_raw'][:, -1] = float('nan')
        with torch.no_grad():
            a = model(self.batch, return_stages=True)
            b = model(permute_views(self.batch, torch.tensor([2, 0, 1])), return_stages=True)
            c = model(padded, return_stages=True)
        for key in ('C0', 'C1', 'c0_adapter_correction'):
            torch.testing.assert_close(a[key], b[key], atol=3e-6, rtol=0)
            torch.testing.assert_close(a[key], c[key], atol=3e-6, rtol=0)

    def test_no_visible_features_cannot_create_adapter_correction(self):
        module = C0ObservabilityAdapter()
        with torch.no_grad():
            module.body[-1].bias.fill_(1)
        fused = torch.zeros(1, 32, 2, 2, 2)
        descriptor = torch.full((1, 5, 2, 2, 2), float('nan'))
        result, correction = module(fused, descriptor, torch.zeros(1, 2, 2, 2, dtype=torch.bool))
        self.assertTrue(torch.equal(result, fused) and torch.equal(correction, fused))

    def test_feedback_trains_zero_initialized_adapter_without_input_gradients(self):
        _, model = self.models()
        batch = dict(self.batch)
        batch['input_projection_raw'] = batch['input_projection_raw'].clone().requires_grad_()
        model.train()
        stages = model(batch, return_stages=True)
        stages['C1'].square().mean().backward()
        grad = model.c0_adapter.body[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)
        self.assertIsNone(batch['input_projection_raw'].grad)

    def test_all_invalid_full_path_remains_finite(self):
        _, model = self.models()
        batch = dict(self.batch)
        batch['view_mask'] = torch.zeros_like(batch['view_mask'])
        batch['input_valid_ray'] = torch.zeros_like(batch['input_valid_ray'])
        batch['input_projection_raw'] = torch.full_like(batch['input_projection_raw'], float('nan'))
        with torch.no_grad():
            stages = model(batch, return_stages=True)
        self.assertTrue(torch.isfinite(stages['C1']).all())
        self.assertTrue((stages['c0_adapter_correction'] == 0).all())

