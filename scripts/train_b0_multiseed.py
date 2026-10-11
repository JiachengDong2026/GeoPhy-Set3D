#!/usr/bin/env python3
"""Train only the two preregistered additional B0 seeds; reference seed is reused."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from geophy_set3d.evaluation.training_runs import run_training

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',default='configs/evaluate_b0_generalization_v1.yaml')
    p.add_argument('--seed',type=int,required=True,choices=[20261012,20261013])
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();run_training(a.protocol,a.seed,resume=a.resume)
