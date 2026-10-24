"""Second registered candidate: same v2 initialization, baseline-only loss adapter."""
from repair_context import *
import argparse
import pilot
from geophy_set3d.training.baseline_consistency_repair_v3 import homogeneous_consistency

base_config=pilot.config

def adapted_config(arm,target):
    value=base_config(arm,target)
    value['objective_adapter']=dict(name='homogeneous_consistency_v3',
        implementation='geophy_set3d.training.baseline_consistency_repair_v3.homogeneous_consistency',
        change='replace absolute epsilon denominator by exact homogeneous ratio with FP64 reductions; zero/zero=0',
        weights='original hold1 cross0.5 observed0.1 consistency0.05; measurement discrepancy unchanged')
    return value


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['register','run']);a.add_argument('--arm',required=True,choices=ARMS);args=a.parse_args()
    pilot.config=adapted_config
    # Process-local override only for this version: no shared source edits and
    # no change to any other running pilot or frozen primary-model process.
    ss.consistency_discrepancy=homogeneous_consistency
    getattr(pilot,args.action)(args.arm,'v3')


if __name__=='__main__':main()
