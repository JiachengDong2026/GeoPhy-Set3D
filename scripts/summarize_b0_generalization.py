#!/usr/bin/env python3
"""Evaluate completed seeds/horizon, then verify integrity and write the final summary."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from geophy_set3d.evaluation.protocols import load_protocol,require_reference,freeze_reference
from geophy_set3d.evaluation.summary import summarize_training,final_summary
from geophy_set3d.evaluation.report import write_report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default='configs/evaluate_b0_generalization_v1.yaml')
    a=p.parse_args();torch.set_num_threads(2);protocol,config,root=load_protocol(a.protocol);require_reference(root)
    freeze_reference(protocol,config,root)
    seeds,shortcuts,horizon=summarize_training(protocol,config,root)
    summary=final_summary(protocol,config,root,seeds,shortcuts,horizon)
    print(write_report(protocol,config,root,summary),flush=True)
    print(summary['b1_recommendation'],flush=True)
