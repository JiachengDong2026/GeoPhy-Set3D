"""One-factor changes to the canonical multiscale MRGR with uniform fusion."""
from runtime import ARMS, load_side
from module_models_v2 import UniformFusion, model_for as module_model
from models import model_for as original_model
from geophy_set3d.models.geolift_oari_mrgr_zero_residual import GeoLiftOARIMRGRZeroResidualControl

def model_for(config, arm, device):
    if arm not in ARMS:
        raise ValueError(arm)
    if arm in ('uniform_no_feedback', 'uniform_single_scale'):
        model = original_model(config, arm.removeprefix('uniform_'), 'cpu')
    elif arm == 'uniform_no_ray_relations':
        model = module_model(config, 'no_ray_relations', 'cpu')
    else:
        model = original_model(config, 'multiscale', 'cpu')
    model.fusion = UniformFusion()
    if arm == 'uniform_zero_residual':
        model.__class__ = GeoLiftOARIMRGRZeroResidualControl
    if arm == 'uniform_feedback_oari':
        model = load_side().augment(model)
    return model.to(device)
