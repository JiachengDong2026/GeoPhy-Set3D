"""Repeat the complete independent audit into a NEW directory; never overwrite evidence.

Usage: PYTHONPATH=src .../python scripts/deep_baselines_latent_readout_v1/recheck_latent.py
Optional --name LABEL selects a new folder below scratch/rechecks/.
"""
from latent_context import *
import argparse
import review_latent_fp64 as review

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S'));args=ap.parse_args()
    assert Path(args.name).name==args.name and args.name not in ('.','..')
    audit=ROOT/'rechecks'/args.name;audit.mkdir(parents=True,exist_ok=False)
    class RepeatAccess(Access):
        def __exit__(self,*exc):
            self.active=False;freeze(audit/'access.json',dict(phase=self.phase,reads=self.reads,denied=self.denied))
    def newfreeze(p,value):
        assert Path(p).name=='independent_review.json';freeze(audit/Path(p).name,value)
    review.Access=RepeatAccess;review.freeze=newfreeze
    review.main();print('Recheck audit:',audit)
if __name__=='__main__':main()
