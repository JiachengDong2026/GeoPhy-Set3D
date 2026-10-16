"""Explicit A2 budget variants. Frozen v2/v3 sources remain untouched.

K4, Q96, full concentration gradients, 32^3 lifting and 64^3 physics stay fixed.
Width and decoder changes are architectural ablations, not FP32 equivalence.
"""
import copy
from pathlib import Path
import torch
from torch import nn
import yaml

from plan_a_runtime_candidates import configure_backend, configure_model

PROTOCOL = 'scheme_a_budget_v4'
RUNTIME_BASE = dict(network_layout='contiguous', cudnn_autotune=True, compile_prior=True)
ROOT = Path(__file__).resolve().parents[1]


class LightDecoder3D(nn.Sequential):
    """Do residual processing at 32^3; retain one learned spatial conv at 64^3."""
    def __init__(self, output_resolution=64):
        from geophy_set3d.models.blocks import ResidualBlock
        super().__init__(
            nn.Conv3d(32, 16, 1), nn.GroupNorm(8, 16), nn.SiLU(),
            ResidualBlock(16, 3), ResidualBlock(16, 3),
            nn.Upsample(size=(output_resolution,) * 3, mode='trilinear', align_corners=True),
            nn.Conv3d(16, 8, 3, padding=1), nn.GroupNorm(8, 8), nn.SiLU(),
            nn.Conv3d(8, 1, 1), nn.Sigmoid())


class MultiScalePrior(nn.Module):
    """32^3 contextual prior plus a learned 64^3 local spatial correction.

    All nine input channels remain available to both paths. Only learned prior
    compute is pooled; physical states, projections and J^T r stay at 64^3.
    """
    def __init__(self, width=16, local_width=8):
        super().__init__()
        from geophy_set3d.models.set_unrolled3d import SharedVolumePrior
        self.coarse = SharedVolumePrior(width)
        self.local = nn.Sequential(nn.Conv3d(9, local_width, 3, padding=1),
                                   nn.GroupNorm(8, local_width), nn.SiLU(), nn.Conv3d(local_width, 1, 1))
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)

    def forward(self, x):
        coarse = self.coarse(torch.nn.functional.avg_pool3d(x, 2))
        coarse = torch.nn.functional.interpolate(coarse, size=x.shape[-3:],
                                                 mode='trilinear', align_corners=True)
        return coarse + self.local(x)


def validate_budget_config(config):
    from geophy_set3d.training.research_selfsup import validate_research_config
    if config.get('protocol') != PROTOCOL or config.get('status') not in ('budget_executable_v4', 'executable'):
        raise ValueError('Require a separately registered budget v4 config')
    model = config['model']
    if (model['variant'], model['steps'], model['samples'], model['lift_resolution'], model['output_resolution']) != ('A2', 4, 96, 32, 64):
        raise ValueError('Budget v4 preserves A2 K4/Q96/lift32/output64')
    if model['prior_width'] not in (16, 32) or model.get('checkpoint_prior') is not False:
        raise ValueError('Require width 16/32 without prior checkpointing')
    if config.get('architecture') not in ({'decoder': 'historical'}, {'decoder': 'light_v1'},
                                          {'decoder': 'historical', 'prior': 'multiscale_v1'},
                                          {'decoder': 'historical', 'prior': 'multiscale_wide_v1'},
                                          {'decoder': 'light_v1', 'prior': 'multiscale_v1'}):
        raise ValueError('Unknown budget decoder architecture')
    runtime = config.get('execution_runtime', {})
    if runtime not in [dict(RUNTIME_BASE, network_precision=p) for p in ('fp32', 'bf16')]:
        raise ValueError('Budget v4 only supports qualified runtime settings and FP32/BF16 networks')
    cache = config.get('geometry_cache', dict(cpu_budget_mib=512, cuda_budget_mib=512))
    if cache not in [dict(cpu_budget_mib=v, cuda_budget_mib=v) for v in (512, 1536)]:
        raise ValueError('Require explicit bounded 512/1536 MiB geometry caches')
    # Validate all existing data/loss/optimizer/epoch boundaries independently of
    # process-local protocol installation. Do not weaken the frozen validator.
    original = copy.deepcopy(config)
    import geophy_set3d.training.research_selfsup as training
    original.update(status='executable', protocol=training.PROTOCOL)
    validate_research_config(original)
    config = copy.deepcopy(config)
    config['status'] = 'executable'
    config['entrypoint_required'] = 'scripts/train_plan_a_budget.py'
    return config


def load_budget_config(path):
    return validate_budget_config(yaml.safe_load(Path(path).read_text()))


def build_budget_model(config, *, runtime=True):
    from geophy_set3d.models.set_unrolled3d import SetUnrolled3D
    settings = dict(config['model'])
    settings.pop('name')
    # Construct exactly the declared width; light decoder replacement is explicit
    # and repeatable. Historical A2 decoder/checkpoints are never overwritten.
    model = SetUnrolled3D(**settings)
    if config['architecture']['decoder'] == 'light_v1':
        model.initializer.decoder = LightDecoder3D(settings['output_resolution'])
    if config['architecture'].get('prior') == 'multiscale_v1':
        model.prior = MultiScalePrior(settings['prior_width'])
    if config['architecture'].get('prior') == 'multiscale_wide_v1':
        model.prior = MultiScalePrior(settings['prior_width'], local_width=32)
    cache = config.get('geometry_cache', dict(cpu_budget_mib=512, cuda_budget_mib=512))
    model.operator.cache_budget_bytes = cache['cpu_budget_mib'] * 1024**2
    model.operator.device_cache_budget_bytes = cache['cuda_budget_mib'] * 1024**2
    return configure_model(model, config['execution_runtime']) if runtime else model


def budget_provenance(config, train, val, original):
    from geophy_set3d.training.research_selfsup import _canonical, _sha
    value = original(config, train, val)
    for name in ('scripts/plan_a_budget_v4.py', 'scripts/train_plan_a_budget.py',
                 'scripts/plan_a_runtime_candidates.py'):
        value['code'][name] = _sha(ROOT / name)
    value.update(source_sha256=_canonical(value['code']), protocol=PROTOCOL,
                 architecture=copy.deepcopy(config['architecture']),
                 geometry_cache=copy.deepcopy(config.get('geometry_cache', dict(cpu_budget_mib=512, cuda_budget_mib=512))),
                 execution_runtime=copy.deepcopy(config['execution_runtime']))
    return value
