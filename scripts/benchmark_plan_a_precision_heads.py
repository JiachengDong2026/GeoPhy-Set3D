#!/usr/bin/env python3
"""BF16 network experiment retaining FP32 final concentration/correction heads.

Run either existing speed benchmark or bounded learning probe. No training
source or qualified runtime implementation is modified by this experiment.
"""
import argparse
import functools
import json
from pathlib import Path
import sys
import torch
import plan_a_runtime_candidates as runtime
from benchmark_plan_a_speed import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--mode',choices=('benchmark','probe'),default='benchmark')
    mode,remaining=parser.parse_known_args()
    if '--network-precision' not in remaining or remaining[remaining.index('--network-precision')+1]!='bf16':
        raise ValueError('FP32 output-head experiment requires BF16 network flag')
    original_configure=runtime.configure_model

    def configure(model,settings):
        for head in (model.initializer.decoder[-2],model.prior[-1]):
            original=head.forward

            @functools.wraps(original)
            def forward(x,_original=original):
                with torch.autocast('cuda',enabled=False):
                    return _original(x.float())

            head.forward=forward
        return original_configure(model,settings)

    runtime.configure_model=configure
    sys.argv=[sys.argv[0]]+remaining
    if mode.mode=='benchmark':
        from benchmark_plan_a_speed import main as run
    else:
        from probe_plan_a_runtime import main as run
    result=run()
    path=Path(remaining[remaining.index('--output')+1])
    record=json.loads(path.read_text())
    record['runtime']['fp32_output_heads']=True
    record['precision']='fp32_physics_bf16_body_fp32_heads'
    record['head_experiment_script_sha256']=sha(Path(__file__))
    path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    raise SystemExit(main())
