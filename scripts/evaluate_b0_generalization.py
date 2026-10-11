#!/usr/bin/env python3
"""Reference reproduction first, then frozen-weight generalization experiments on val only."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from geophy_set3d.evaluation.protocols import load_protocol,reproduce_reference
from geophy_set3d.evaluation.experiments import geometry_generalization,view_count,angular_coverage,condition_analysis

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default='configs/evaluate_b0_generalization_v1.yaml')
    p.add_argument('--stage',choices=['reference','geometry','views','coverage','conditions','all'],default='all')
    a=p.parse_args();torch.set_num_threads(2);protocol,config,root=load_protocol(a.protocol)
    model,reference=reproduce_reference(protocol,config,root)
    if a.stage in ('geometry','all'):geometry_generalization(protocol,config,root,model,reference)
    if a.stage in ('views','all'):view_count(protocol,config,root,model)
    if a.stage in ('coverage','all'):angular_coverage(protocol,config,root,model)
    if a.stage in ('conditions','all'):condition_analysis(protocol,config,root,model)
