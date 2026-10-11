"""Validate the frozen 300-epoch comparison without touching the legacy protocol."""
import copy
from pathlib import Path
import yaml


def validate_comparison_config(config):
    required = dict(batch_size=2, gradient_accumulation=1, max_epochs=300,
                    gradient_clip_norm=1.0, early_stopping_patience=None,
                    seed=20261011, precision='fp32')
    if any(config['training'].get(k) != v for k, v in required.items()):
        raise ValueError('Architecture comparison requires the frozen full 300-epoch protocol')
    expected = dict(
        train=dict(split='train', geometry_pool='train', min_input_views=3, max_input_views=6, hold_views=0),
        validation=dict(split='val', geometry_pool='train', input_views=4, hold_views=0),
        optimizer=dict(name='adamw', lr=2e-4, weight_decay=1e-4),
        scheduler=dict(name='cosine', T_max=300),
        loss=dict(smooth_l1=1., relative_l1=.1, relative_epsilon=1e-6),
        normalization=dict(method='linear_p99_5', statistics_file='train_statistics.json'))
    if any(config[k] != v for k, v in expected.items()):
        raise ValueError('Data/loss/optimizer/scheduler/normalization differs from frozen comparison')
    frozen = Path(config['dataset']['root'])
    output = Path(config['experiment']['output_root']).resolve()
    if output.is_relative_to(frozen.resolve()):
        raise ValueError('Output cannot be inside the frozen dataset')
    if config['model']['name'] in ('geolift_gca_res3d', 'geolift_gcsi_res3d', 'geolift_oari_res3d', 'geolift_oari_mrgr_res3d'):
        acquisition = yaml.safe_load((frozen/'generation_config.yaml').read_text())['geometry']
        scales = dict(distance_scale_m=float(acquisition['radius_m'][1]),
                      azimuth_scale_deg=float(acquisition['azimuth_deg'][1]))
        if config['model']['gca'] != scales:
            raise ValueError('Geometry scales must match frozen acquisition ranges')


def assert_fair_configs(b0, b1):
    for config in (b0, b1):
        validate_comparison_config(config)
    if b0['model']['name'] != 'geolift_mean_res3d' or b1['model']['name'] != 'geolift_gca_res3d':
        raise ValueError('Comparison requires B0 and B1')
    normalized = []
    for config in (b0, b1):
        other = copy.deepcopy(config)
        other.pop('experiment')
        other['training'].pop('device')
        other['model'].pop('name')
        other['model'].pop('gca', None)
        normalized.append(other)
    if normalized[0] != normalized[1]:
        raise ValueError('B0/B1 training protocols differ beyond fusion and output paths')


def assert_fair_b1_b2(b1, b2):
    for config in (b1, b2):
        validate_comparison_config(config)
    if b1['model']['name'] != 'geolift_gca_res3d' or b2['model']['name'] != 'geolift_gcsi_res3d':
        raise ValueError('Requires frozen B1 and new B2')
    normalized=[]
    for config in (b1, b2):
        other=copy.deepcopy(config)
        other.pop('experiment');other['training'].pop('device')
        other['model'].pop('name');other['model'].pop('set_interaction',None)
        normalized.append(other)
    if normalized[0] != normalized[1]:
        raise ValueError('B1/B2 protocols differ beyond set interaction and output paths')


def assert_fair_b2_oari(b2, oari):
    for config in (b2, oari):
        validate_comparison_config(config)
    if b2['model']['name'] != 'geolift_gcsi_res3d' or oari['model']['name'] != 'geolift_oari_res3d':
        raise ValueError('Requires frozen B2 and OARI')
    normalized = []
    for config in (b2, oari):
        other = copy.deepcopy(config)
        other.pop('experiment'); other['training'].pop('device'); other['model'].pop('name')
        normalized.append(other)
    if normalized[0] != normalized[1]:
        raise ValueError('B2/OARI protocols differ beyond interaction and output paths')


def assert_fair_oari_mrgr(oari, mrgr):
    for config in (oari, mrgr):
        validate_comparison_config(config)
    if (oari['model']['name'], mrgr['model']['name']) != ('geolift_oari_res3d', 'geolift_oari_mrgr_res3d'):
        raise ValueError('Requires frozen OARI and MRGR')
    normalized = []
    for config in (oari, mrgr):
        other = copy.deepcopy(config)
        other.pop('experiment'); other['training'].pop('device'); other['model'].pop('name')
        normalized.append(other)
    if normalized[0] != normalized[1]:
        raise ValueError('OARI/MRGR training protocols differ')
