import unittest
import torch
from geophy_set3d.models import GeoLiftOARIRes3D, GeoLiftOARIMRGRRes3D, build_model
from geophy_set3d.models.geolift_oari_mrgr_res3d import Y_SCALE, PROJECTOR_SAMPLES
from geophy_set3d.training.utils import load_config, permute_views
from geophy_set3d.training.comparison import assert_fair_oari_mrgr
from geophy_set3d.data.projection_pool_dataset import ProjectionPoolDataset
from geophy_set3d.data.collate import projection_pool_collate
from pool_fixture import make_pool


class MRGRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary, root, _, _ = make_pool()
        cls.dataset = ProjectionPoolDataset(root, split='train', min_input_views=3, max_input_views=3)
        cls.batch = projection_pool_collate([cls.dataset[0]])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def model(self, activate=False):
        torch.manual_seed(25)
        model = GeoLiftOARIMRGRRes3D().eval()
        if activate:
            with torch.no_grad():
                model.residual_adapter[-1].weight.normal_(0, .02)
        return model

    def stages(self):
        with torch.no_grad():
            return self.model()(self.batch, return_stages=True)

    def test_mrgr_output_shape(self):
        s = self.stages()
        self.assertEqual(s['C1'].shape, (1, 1, 64, 64, 64))
        self.assertEqual(s['residual_features'].shape, (1, 3, 16, 64, 64))

    def test_mrgr_c0_c1_finite(self):
        s = self.stages()
        self.assertTrue(torch.isfinite(s['C0']).all() and torch.isfinite(s['C1']).all())

    def test_adapter_zero_init_makes_initial_c1_match_c0(self):
        s = self.stages()
        torch.testing.assert_close(s['C0'], s['C1'], atol=0, rtol=0)
        self.assertTrue((s['residual_context'] == 0).all())

    def test_residual_uses_raw_ppm_m(self):
        s = self.stages(); valid = self.batch['input_valid_ray'].bool()
        torch.testing.assert_close(s['measurement_residual'][valid],
            (self.batch['input_projection_raw']-s['measurement_prediction_Y0'])[valid], atol=0, rtol=0)
        self.assertTrue((s['measurement_residual'][valid] < 0).any())

    def test_residual_scaling_constant(self):
        self.assertEqual(Y_SCALE, 32284.894580078173)
        self.assertEqual(PROJECTOR_SAMPLES, 96)
        s = self.stages()
        torch.testing.assert_close(s['residual_input'][:, :, :1], s['measurement_residual']/Y_SCALE, atol=0, rtol=0)

    def test_residual_invalid_ray_zero(self):
        batch = dict(self.batch); valid = batch['input_valid_ray'].bool()
        batch['input_projection_raw'] = torch.where(valid, torch.zeros_like(batch['input_projection_raw']), float('nan'))
        with torch.no_grad():
            s = self.model()(batch, return_stages=True)
        self.assertTrue((s['measurement_residual'][~valid] == 0).all())
        self.assertTrue((s['residual_input'][:, :, 1:2][valid] == 1).all())
        torch.testing.assert_close(s['measurement_residual'][valid], -s['measurement_prediction_Y0'][valid])

    def test_residual_and_base_visibility_match(self):
        s = self.stages()
        self.assertTrue(torch.equal(s['visibility_base'], s['visibility_residual']))

    def test_mrgr_shared_initialization_matches_oari(self):
        torch.manual_seed(20261011); oari = GeoLiftOARIRes3D()
        torch.manual_seed(20261011); model = GeoLiftOARIMRGRRes3D()
        for key, value in oari.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)

    def test_mrgr_shared_modules_match_oari_architecture(self):
        model, oari = self.model(), GeoLiftOARIRes3D()
        for name in ('encoder', 'lifting', 'interaction', 'fusion', 'decoder'):
            self.assertEqual(repr(getattr(model, name)), repr(getattr(oari, name)))
        self.assertLess(sum(p.numel() for p in model.parameters()), 350000)
        a, b = (load_config('configs/train_supervised_'+n+'_300.yaml') for n in ('oari', 'oari_mrgr'))
        assert_fair_oari_mrgr(a, b)
        self.assertIsInstance(build_model(b), GeoLiftOARIMRGRRes3D)

    def test_residual_permutation_invariance(self):
        model = self.model(True)
        with torch.no_grad():
            p = model(self.batch)
            q = model(permute_views(self.batch, torch.tensor([2, 0, 1])))
        torch.testing.assert_close(p, q, atol=3e-6, rtol=0)

    def test_residual_padding_invariance(self):
        model = self.model(True); padded = dict(self.batch)
        for key, value in self.batch.items():
            if (key.startswith('input_') or key == 'view_mask') and isinstance(value, torch.Tensor):
                pad = torch.zeros_like(value[:, :1])
                padded[key] = torch.cat((value, pad), 1)
        padded['input_projection'][:, -1] = float('nan')
        padded['input_projection_raw'][:, -1] = float('nan')
        with torch.no_grad():
            p = model(self.batch, return_stages=True); q = model(padded, return_stages=True)
        for stage in ('C0', 'C1'):
            torch.testing.assert_close(p[stage], q[stage], atol=3e-6, rtol=0)
        self.assertTrue((q['residual_context'][:, -1] == 0).all())

    def test_residual_all_invalid_safe(self):
        batch = dict(self.batch); batch['view_mask'] = torch.zeros_like(batch['view_mask'])
        batch['input_valid_ray'] = torch.zeros_like(batch['input_valid_ray'])
        batch['input_projection_raw'] = torch.full_like(batch['input_projection_raw'], float('nan'))
        with torch.no_grad():
            s = self.model(True)(batch, return_stages=True)
        self.assertTrue(torch.isfinite(s['C1']).all())
        self.assertTrue((s['context1'] == 0).all())
        self.assertTrue((s['measurement_residual'] == 0).all())

    def test_residual_branch_gradient_nonzero(self):
        model = self.model(True); s = model(self.batch, return_stages=True)
        (s['C1']-self.batch['gt_volume_64']).square().mean().backward()
        for module in (model.residual_encoder, model.residual_adapter):
            grads = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_projector_to_refinement_gradient_finite(self):
        model = self.model(True); batch = dict(self.batch)
        for key in ('input_projection_raw', 'input_station_origin_m', 'input_path_length_m', 'input_ray_direction', 'input_t_near'):
            batch[key] = batch[key].clone().requires_grad_()
        s = model(batch, return_stages=True)
        # Isolate the feedback path: loss -> Y0 -> C0, not the shared context route.
        grad_y, = torch.autograd.grad(s['C1'].square().mean(), s['measurement_prediction_Y0'], retain_graph=True)
        grad_c0, = torch.autograd.grad(s['measurement_prediction_Y0'], s['C0'], grad_outputs=grad_y)
        self.assertTrue(torch.isfinite(grad_y).all() and torch.isfinite(grad_c0).all())
        self.assertGreater(float(grad_y.abs().sum()), 0)
        self.assertGreater(float(grad_c0.abs().sum()), 0)
        self.assertTrue(all(batch[k].grad is None for k in batch if isinstance(batch[k], torch.Tensor) and batch[k].requires_grad))

    def shortcut_diff(self, mode):
        model = self.model(True)
        with torch.no_grad():
            p = model(self.batch); q = model(self.batch, residual_mode=mode)
        self.assertGreater(float((p-q).abs().max()), 1e-7)

    def test_residual_sign_changes_refinement(self):
        self.shortcut_diff('signflip')

    def test_zero_residual_changes_refinement(self):
        self.shortcut_diff('zero')

    def test_wrong_view_residual_changes_refinement(self):
        self.shortcut_diff('wrong_pairing')

    def test_shortcuts_evaluation_only(self):
        with self.assertRaises(ValueError):
            self.model().train()(self.batch, residual_mode='zero')
