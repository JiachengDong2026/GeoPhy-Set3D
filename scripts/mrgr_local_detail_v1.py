"""Independent MRGR multiscale + narrow local-detail candidate; frozen src untouched."""
from pathlib import Path
import torch
from torch import nn
import probe_mrgr_multiscale_v1 as base

ROOT = base.ROOT


class LocalDetail(nn.Module):
    """32³ local spatial path: 32->8, depthwise3, pointwise8, zero output32."""
    def __init__(self, width=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(32, width, 1), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1, groups=width), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv3d(width, width, 1), nn.GroupNorm(4, width), nn.SiLU())
        self.output = nn.Conv3d(width, 32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(self.features(x))


class LocalContextDecoder(nn.Sequential):
    def __init__(self, original):
        super().__init__(*[v for k, v in original._modules.items() if k != 'context'])
        self.context = original.context
        self.local = LocalDetail()

    def forward(self, x):
        for name, module in self._modules.items():
            if name in ('context', 'local'):
                continue
            x = module(x)
            if name == '2':
                x = (x + self.context(x)) + self.local(x)
        return x


def model_for(config, arm='local', device='cpu'):
    if arm not in ('multiscale', 'local'):
        raise ValueError(arm)
    model = base.model_for(config, 'multiscale', 'cpu')
    state = torch.get_rng_state()
    if arm == 'local':
        torch.manual_seed(config['training']['seed'] + 2)
        model.decoder = LocalContextDecoder(model.decoder)
    torch.set_rng_state(state)
    return model.to(device)


def source_hashes():
    result = base.source_hashes()
    for name in ('mrgr_local_detail_v1.py', 'run_mrgr_local_detail_v1.py'):
        p = ROOT / 'scripts' / name
        result[str(p.relative_to(ROOT))] = base.digest(p)
    return result
