#!/usr/bin/env python3
"""Reuse frozen validation-only prediction/GT boundaries for budget v4 models.

This entry does not permit evaluation of engineering smoke/gate checkpoints;
all existing completed-300-epoch and saved-prediction guards remain in force.
"""
import sys
from reconstruct_plan_a_budget import install_budget_inference


def main():
    if 'predict' in sys.argv:
        if '--device' not in sys.argv:
            sys.argv.extend(['--device','cuda:0'])
        if not sys.argv[sys.argv.index('--device')+1].startswith('cuda'):
            raise ValueError('The qualified BF16 budget inference requires CUDA')
    install_budget_inference()
    from evaluate_research_selfsup import main as evaluate
    evaluate()


if __name__=='__main__':
    main()
