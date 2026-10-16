#!/usr/bin/env python3
"""Measurement-only reconstruction for independently frozen budget v4 models."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from plan_a_budget_v4 import build_budget_model,configure_backend,validate_budget_config


def install_budget_inference():
    import geophy_set3d.models.set_unrolled3d as models
    def build(config):
        config=validate_budget_config(config)
        configure_backend(config['execution_runtime'])
        return build_budget_model(config)
    models.build_set_unrolled_model=build


def main():
    if '--device' not in sys.argv:
        sys.argv.extend(['--device','cuda:0'])
    if not sys.argv[sys.argv.index('--device')+1].startswith('cuda'):
        raise ValueError('The qualified BF16 budget inference requires CUDA')
    install_budget_inference()
    from reconstruct_plan_a import main as reconstruct
    reconstruct()


if __name__=='__main__':
    main()
