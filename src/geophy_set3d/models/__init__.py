from .geolift_mean_res3d import GeoLiftMeanRes3D
from .geolift_gca_res3d import GeoLiftGCARes3D

__all__ = ['GeoLiftMeanRes3D', 'GeoLiftGCARes3D', 'build_model']


def build_model(config):
    if config['model']['name'] == 'geolift_mean_res3d':
        return GeoLiftMeanRes3D()
    if config['model']['name'] == 'geolift_gca_res3d':
        return GeoLiftGCARes3D(**config['model']['gca'])
    raise ValueError('Unknown model')
