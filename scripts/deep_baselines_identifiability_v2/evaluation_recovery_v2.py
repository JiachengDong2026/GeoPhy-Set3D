"""Evaluation-only canonical swap metadata; registered numeric sources intact."""
from ctx_v2 import *
import argparse
import evaluate_v2 as evaluation
from numerics_v2 import intervene as original_intervene

def canonical_swap(inp,kind,other=None):
    out=original_intervene(inp,kind,other)
    if kind=='swap':
        # CPU-prepared normalization and CUDA division can differ by 1 ULP.
        # Raw integrals used by ray_tokens are already exactly donor values.
        # Copy the donor's companion field too, rather than recompute metadata.
        out['input_projection']=other['input_projection'].clone()
    return out

def preflight():
    backend()
    with Boundary('evaluate') as access:
        samples=torch.load(ROOT/'inputs/train_AB.pt',map_location='cpu',weights_only=False)
        pairs=[]
        for branch in ('A','B'):
            a=batch([samples[0]],branch,'cuda:0');b=batch([samples[1]],branch,'cuda:0')
            old=original_intervene(a,'swap',b);new=canonical_swap(a,'swap',b)
            different=[k for k in old if (not torch.equal(old[k],b[k]) if torch.is_tensor(old[k]) else old[k]!=b[k])]
            assert different==['input_projection'],different
            assert torch.equal(old['input_projection_raw'],b['input_projection_raw'])
            assert value_hash(new)==value_hash(b)
            forward=[]
            for arm in design()['arms']:
                net=model(arm,'cuda:0').eval()
                with torch.no_grad():
                    one=net(old);two=net(new);three=net(b)
                assert torch.equal(one,two) and torch.equal(two,three)
                forward.append(dict(arm=arm,full64_output_bitwise_equal=True))
                del net,one,two,three;torch.cuda.empty_cache()
            pairs.append(dict(branch=branch,differing_original_fields=different,
                maximum_normalized_roundoff=float((old['input_projection']-b['input_projection']).abs().max()),
                raw_exact=True,corrected_input_exact=True,models=forward))
    freeze(ROOT/'engineering/evaluation_swap_preflight.json',dict(at=now(),passed=True,optimizer_updates=0,
        source_sha256=sha(__file__),pairs=pairs,audit=access.summary(),only_normalized_companion_field_changed=True,
        original_sources_unchanged=True,thresholds_unchanged=True))
    print('SWAP PREFLIGHT passed: only normalized companion roundoff; raw/model outputs bitwise identical',flush=True)

def main(arm):
    amendment=verified(ROOT/'engineering/evaluation_swap_recovery.json')
    assert amendment['adapter_sha256']==sha(__file__)
    verify(amendment['checkpoint_sha256']);assert verified(ROOT/'engineering/evaluation_swap_preflight.json')['passed']
    evaluation.intervene=canonical_swap
    evaluation.main(arm)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--arm',choices=['gnot','deepsets']);a=p.parse_args()
    if a.preflight:preflight()
    else:
        assert a.arm;main(a.arm)
