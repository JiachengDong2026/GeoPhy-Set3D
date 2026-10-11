from .geolift_mean_res3d import GeoLiftMeanRes3D
from .geolift_gca_res3d import GeoLiftGCARes3D
from .geolift_gcsi_res3d import GeoLiftGCSIRes3D
from .geolift_oari_res3d import GeoLiftOARIRes3D

from .geolift_oari_mrgr_res3d import GeoLiftOARIMRGRRes3D

from .geolift_oari_mrgr_zero_residual import GeoLiftOARIMRGRZeroResidualControl

__all__ = ['GeoLiftMeanRes3D', 'GeoLiftGCARes3D', 'GeoLiftGCSIRes3D', 'GeoLiftOARIRes3D', 'GeoLiftOARIMRGRRes3D', 'GeoLiftOARIMRGRZeroResidualControl', 'build_model']


def build_model(config):
    if config['model']['name'] == 'geolift_mean_res3d':
        return GeoLiftMeanRes3D()
    if config['model']['name'] == 'geolift_gca_res3d':
        return GeoLiftGCARes3D(**config['model']['gca'])
    if config['model']['name'] == 'geolift_gcsi_res3d':
        return GeoLiftGCSIRes3D(**config['model']['gca'], **config['model']['set_interaction'])
    if config['model']['name'] == 'geolift_oari_res3d':
        return GeoLiftOARIRes3D(**config['model']['gca'], **config['model']['set_interaction'])
    if config['model']['name'] == 'geolift_oari_mrgr_res3d':
        return GeoLiftOARIMRGRRes3D(**config['model']['gca'], **config['model']['set_interaction'])
    if config['model']['name'] == 'geolift_oari_mrgr_zero_residual_control':
        return GeoLiftOARIMRGRZeroResidualControl(**config['model']['gca'], **config['model']['set_interaction'])
    raise ValueError('Unknown model')
