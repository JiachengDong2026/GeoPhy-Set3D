"""Engineering-only weak taps: avoid retaining checkpoint graphs through closures.

Original registered numerical files and all thresholds remain intact. Each tap
calls the same original function and returns its output object unchanged.
"""
from diag_context import *
import argparse
import gc
import importlib
import weakref
import trace as taps

class WeakTrace(taps.Trace):
    def wrap(self,obj,name,label):
        original=getattr(obj,name);owner=weakref.ref(self)
        def tapped(*args,**kwargs):
            out=original(*args,**kwargs)
            target=owner()
            if target is not None:target.save(label,out)
            return out
        setattr(obj,name,tapped);self.restores.append((obj,name,original))

def install():
    taps.Trace=WeakTrace
    # Clear only unreachable graphs between independent diagnostics. No tensor,
    # loss, mask, ray, query, model mode, or gradient formula is changed.
    for name in ('layer_gradients','finite_difference'):
        original=getattr(taps,name)
        def compact(*args,_function=original,**kwargs):
            gc.collect()
            result=_function(*args,**kwargs)
            gc.collect()
            return result
        setattr(taps,name,compact)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--script',choices=['experiment_c','train'],required=True)
    ap.add_argument('arguments',nargs=argparse.REMAINDER);args=ap.parse_args()
    amendment=verified('engineering_memory_v2.json');assert c.sha(__file__)==amendment['adapter_sha256']
    install();module=importlib.import_module(args.script)
    values=args.arguments[1:] if args.arguments and args.arguments[0]=='--' else args.arguments
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True)
    if args.script=='train':
        parser.add_argument('--experiment',choices=['A','B'],required=True);parser.add_argument('--version',choices=['v2','v3'],required=True)
        parser.add_argument('--n',type=int,choices=[1,2,4],required=True)
    a=parser.parse_args(values)
    if args.script=='train':module.run(a.experiment,a.arm,a.version,a.n)
    else:module.run(a.arm)

if __name__=='__main__':main()
