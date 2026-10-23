"""Fresh same-seed controls; never convert or load a trained OARI checkpoint."""
import torch
from uniform_models import model_for as official_uniform
from geophy_set3d.models.set_interaction import GeometryConditionedSetInteraction
from geophy_set3d.models.geolift_plain_set_mrgr import GeoLiftPlainSetMRGR

PLAIN = 'uniform_plain_set_interaction'


def model_for(config, arm, device):
    if arm != PLAIN:
        return official_uniform(config, arm, device)
    model = official_uniform(config, 'uniform', 'cpu')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['training']['seed'] + 2)
        plain = GeometryConditionedSetInteraction(**config['model']['set_interaction'])
    # All compatible tensors come from a FRESH initialized control. The token
    # projection has a different input dimension and remains independently fresh.
    original = model.interaction.state_dict()
    with torch.no_grad():
        for key, value in plain.state_dict().items():
            if not key.startswith('token_projection.'):
                assert key in original and original[key].shape == value.shape
                value.copy_(original[key])
    model.interaction = plain
    del model.ray_geometry
    del model.observability
    model.__class__ = GeoLiftPlainSetMRGR
    return model.to(device)
