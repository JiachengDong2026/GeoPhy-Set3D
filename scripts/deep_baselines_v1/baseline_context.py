"""Independent ray-baseline dispatch; frozen data/loss/engine reused read-only."""
import copy
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts/plain_final_v1'))
import context as c
sys.path.insert(0, str(HERE))

ROOT = Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_v1')
ARMS = ('deepsets', 'senseiver', 'gnot')
SEEDS = (20261011, 20261019)
read, freeze, sha = c.read, c.freeze, c.sha


def model_config(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return read(PROJECT / 'configs/baselines' / arm / 'model_v1.json')


def build_model(arm, device='cpu', seed=20261011):
    from geophy_set3d.training.utils import seed_everything
    seed_everything(seed)
    m = model_config(arm)
    factory = m.get('factory', m.get('implementation')).replace(':', '.')
    module, name = factory.rsplit('.', 1)
    cls = getattr(importlib.import_module(module), name)
    return cls(**m.get('kwargs', m.get('settings', {}))).to(device)


def training_config(arm, seed, target):
    anchor = c.verified(ROOT / 'main_anchor.json')
    config = copy.deepcopy(anchor['model']['config'])
    config['model'] = dict(name='ray_' + arm + '_v1', baseline=model_config(arm))
    config['training']['seed'] = seed
    config['training']['device'] = 'cuda:0'
    config['experiment'] = dict(name=target.name, output_root=str(target))
    validate_config(config)
    return config


def validate_config(config, arm=None):
    """Preserve all frozen fields except identity, seed, output and device."""
    expected = copy.deepcopy(c.verified(ROOT / 'main_anchor.json')['model']['config'])
    actual = copy.deepcopy(config)
    assert actual['training']['seed'] in SEEDS
    assert actual['model']['baseline'] in [model_config(a) for a in ARMS]
    if arm is not None:
        assert actual['model'] == dict(name='ray_' + arm + '_v1', baseline=model_config(arm))
    for value in (expected, actual):
        value.pop('experiment', None)
        value.pop('model')
        value['training'].pop('seed')
        value['training'].pop('device')
    assert actual == expected, 'Non-model training setting changed'
    from geophy_set3d.training.selfsup import FROZEN_LOSS
    assert config['loss'] == FROZEN_LOSS
    assert config['training']['batch_size'] == 2
    assert config['training']['max_epochs'] == 300


def source_identity():
    paths = list((PROJECT / 'src').rglob('*.py'))
    paths += list((PROJECT / 'src').rglob('*.cpp'))
    paths += list(HERE.glob('*.py'))
    paths += list((PROJECT / 'configs/baselines').rglob('*.json'))
    paths += [PROJECT / 'scripts/probe_mrgr_multiscale_v1.py']
    paths += list((PROJECT / 'scripts/paper_final_v1').glob('*.py'))
    return {str(p): sha(p) for p in sorted(paths)}


def regression():
    """Bitwise CPU check of the untouched final model on its fixed val input."""
    import torch
    from geophy_set3d.models.ray_senseiver_ss import ray_tokens
    anchor = c.verified(ROOT / 'main_anchor.json')
    c.verify_hashes(anchor['training_sources'])
    c.verify_hashes(anchor['protocol_hashes'])
    for m in (anchor['model'], anchor['reference']):
        assert sha(m['checkpoint']) == m['checkpoint_sha256']
    m = anchor['model']
    ck = torch.load(m['checkpoint'], map_location='cpu', weights_only=False)
    model = c.model_for(m['config'], c.PLAIN, 'cpu').eval().requires_grad_(False)
    model.load_state_dict(ck['model'], strict=True)
    with c.AccessGuard(splits=['val'], output=ROOT) as guard, torch.inference_mode():
        ds = c.dataset('val', anchor['benchmark']['setting'])
        batch = c.batch_for(ds, 0)
        assert c.input_hash(batch) == anchor['benchmark']['input_sha256']
        out = model(batch, return_stages=True)
        hashes = {k: c.tensor_hash(out[k]) for k in ('C0', 'C1')}
        assert hashes == anchor['benchmark']['output_tensor_hashes']
        # Shared tokenization is fixed measurement/geometry preprocessing only.
        tokens, mask, counts = ray_tokens(batch)
        assert tokens.shape[-1] == 203 and int(counts[0]) > 0
    return dict(passed=True, bitwise_equal=True, output_hashes=hashes,
                anchor_sha256=sha(ROOT / 'main_anchor.json'), audit=guard.summary())
