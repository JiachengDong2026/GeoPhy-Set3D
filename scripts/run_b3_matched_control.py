#!/usr/bin/env python3
"""Add only the matched lambda=0 control; never launch formal B3 training."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from geophy_set3d.training.utils import load_config
from geophy_set3d.training.matched_control import train_control
from geophy_set3d.evaluation.matched_control import prepare,evaluate_control,append_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','train','evaluate','report'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); repo=Path(__file__).resolve().parents[1]; torch.set_num_threads(2)
    if args.stage=='prepare': prepare(repo)
    elif args.stage=='train': train_control(load_config(repo/'configs/train_b3_gate_control.yaml'),repo,args.resume)
    elif args.stage=='evaluate': evaluate_control(repo)
    else: append_report(repo)


if __name__=='__main__':
    main()
