#!/usr/bin/env python3
"""Option A: fresh seed20261011 diagnostic with cosine T_max=220, preserving the reference."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from geophy_set3d.evaluation.training_runs import run_training

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',default='configs/evaluate_b0_generalization_v1.yaml')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();run_training(a.protocol,20261011,horizon=True,resume=a.resume)
