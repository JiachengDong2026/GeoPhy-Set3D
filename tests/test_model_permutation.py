import unittest
import torch
from geophy_set3d.models import GeoLiftMeanRes3D
from geophy_set3d.training.utils import permute_views
from model_fixture import model_batch


class ModelPermutationTests(unittest.TestCase):
    def test_permutation_and_padding(self):
        torch.set_num_threads(2); torch.manual_seed(6)
        model = GeoLiftMeanRes3D().eval(); batch = model_batch(1, 3)
        with torch.no_grad():
            output = model(batch)
            permuted = model(permute_views(batch, torch.tensor([2, 0, 1])))
            torch.testing.assert_close(output, permuted, atol=2e-6, rtol=0)
            padded = dict(batch)
            for key, value in batch.items():
                if key.startswith('input_') or key == 'view_mask':
                    padding = torch.full_like(value[:, :1], 900) if value.dtype != torch.bool else torch.zeros_like(value[:, :1])
                    padded[key] = torch.cat([value, padding], 1)
            padded['input_projection'][:, -1] = float('nan')
            padded_output = model(padded)
            torch.testing.assert_close(output, padded_output, atol=2e-6, rtol=0)
