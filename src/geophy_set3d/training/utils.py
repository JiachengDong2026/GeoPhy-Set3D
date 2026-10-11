"""Experiment provenance, deterministic setup, plots and checkpoint utilities."""
import copy
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def permute_views(batch, permutation, geometry_only=False):
    result = dict(batch)
    for key, value in batch.items():
        geometry = key.startswith('input_') and key not in ('input_projection', 'input_projection_raw')
        if (geometry if geometry_only else key.startswith('input_') or key == 'view_mask'):
            if isinstance(value, torch.Tensor):
                result[key] = value[:, permutation]
    return result


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    # Fail explicitly rather than quietly accepting forbidden/different model choices.
    expected = dict(name='geolift_mean_res3d', input_channels=3, feature_channels=32,
                    lift_resolution=32, output_resolution=64, output_activation='sigmoid')
    model = dict(config['model'])
    name = model.pop('name')
    gca = model.pop('gca', None)
    interaction = model.pop('set_interaction', None)
    if name not in ('geolift_mean_res3d', 'geolift_gca_res3d', 'geolift_gcsi_res3d', 'geolift_oari_res3d', 'geolift_oari_mrgr_res3d', 'geolift_oari_mrgr_zero_residual_control'):
        raise ValueError('Unknown supervised architecture')
    if model != {k: v for k, v in expected.items() if k != 'name'} or config['training']['precision'] != 'fp32':
        raise ValueError('Supervised comparison requires the fixed encoder/lifting/decoder in FP32')
    if name == 'geolift_mean_res3d' and gca is not None:
        raise ValueError('B0 cannot contain attention settings')
    if name in ('geolift_gca_res3d', 'geolift_gcsi_res3d', 'geolift_oari_res3d', 'geolift_oari_mrgr_res3d', 'geolift_oari_mrgr_zero_residual_control') and gca != dict(distance_scale_m=6.5, azimuth_scale_deg=180.0):
        raise ValueError('B1/B2 require the frozen acquisition geometry scales')
    if name in ('geolift_gcsi_res3d', 'geolift_oari_res3d', 'geolift_oari_mrgr_res3d', 'geolift_oari_mrgr_zero_residual_control'):
        if interaction != dict(d_model=32, num_heads=4, num_blocks=1, ffn_channels=64):
            raise ValueError('B2 v1 fixes a single lightweight view attention block')
    elif interaction is not None:
        raise ValueError('Only B2 can contain set interaction settings')
    if config['train']['split'] != 'train' or config['train']['geometry_pool'] != 'train':
        raise ValueError('Training requires train/Pool A')
    if config['validation']['split'] != 'val' or config['validation']['geometry_pool'] != 'train':
        raise ValueError('Model selection requires val/Pool A')
    if config['training']['gradient_accumulation'] < 1:
        raise ValueError('gradient_accumulation must be positive')
    return config


def directories(config, suffix=''):
    root = Path(config['experiment']['output_root'])
    name = config['experiment']['name'] + suffix
    result = {key: root/key/name for key in ('logs', 'figures', 'checkpoints')}
    for p in result.values():
        p.mkdir(parents=True, exist_ok=True)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def environment(config):
    return dict(torch_version=str(torch.__version__), numpy_version=np.__version__,
                cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(config['training']['device'])
                if torch.cuda.is_available() else None,
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                seed=config['training']['seed'], precision='fp32', test_used=False)


def save_slices(path, target, prediction, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g, p = target.detach().cpu().numpy().squeeze(), prediction.detach().cpu().numpy().squeeze()
    error = np.abs(p-g)
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)
    maximum = max(float(g.max()), float(p.max()), 1e-9)
    for row, (name, axis) in enumerate((('XY (z=32)', 2), ('XZ (y=32)', 1), ('YZ (x=32)', 0))):
        for col, (label, volume) in enumerate((('GT', g), ('Prediction', p), ('Absolute error', error))):
            img = axes[row, col].imshow(np.take(volume, 32, axis=axis).T, origin='lower',
                                       extent=[0, .6, 0, .6], vmin=0, vmax=maximum, cmap='magma')
            axes[row, col].set_title(f'{label}: {name}')
            fig.colorbar(img, ax=axes[row, col], shrink=.7)
    fig.suptitle(title+'; CH4 mass fraction')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_curves(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for split in ('train', 'val'):
        available = [r for r in rows if split in r]
        if available:
            axes[0].plot([r['epoch'] for r in available], [r[split]['loss']['total'] for r in available], label=split)
            axes[1].plot([r['epoch'] for r in available], [r[split]['metrics']['mean_per_case']['relative_l1'] for r in available], label=split)
    for ax, title in zip(axes, ('Supervised loss', 'Mean per-case relative L1')):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_yscale('log'); ax.legend()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu()); generator.set_state(state['loader'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
