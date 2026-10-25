"""Final engineering seal, before the first optimizer update."""
from ctx_v2 import *

def main():
    pre=verified(ROOT/'preregistration.json');verify(pre['historical_tracked']);verify(pre['frozen_main_checkpoints'])
    amendment=verified(ROOT/'engineering/preupdate_final_corrections.json')
    for path,expected in pre['sources'].items():
        if sha(path)!=expected:
            change=amendment['changes'][path]
            assert change['before_sha256']==expected and sha(change['preserved_original'])==expected
            assert change['after_sha256']==sha(path)
    inp=verified(ROOT/'input_registration.json');verify(inp['inputs'])
    assert verified(ROOT/'engineering_tests.json')['passed']
    smoke={arm:verified(ROOT/'smoke'/arm/'result.json') for arm in design()['arms']}
    assert all(s['passed'] and s['optimizer_updates']==0 for s in smoke.values())
    hashes={arm:sha(ROOT/'smoke'/arm/'roundtrip.pt') for arm in design()['arms']}
    freeze(ROOT/'registration.json',dict(at=now(),sources=sources(),inputs=inp['inputs'],initial_states=inp['initial_states'],
        smoke_checkpoint_sha256=hashes,CPU_tests_sha256=sha(ROOT/'engineering_tests.json'),
        protocol_sha256=sha(PROJECT/'docs/deep_baselines_identifiability_v2_protocol_zh.md'),
        preregistration_sha256=sha(ROOT/'preregistration.json'),updates_before_seal=0,new_budget=400,prior_updates=2000,cumulative_budget=2500))
    print('SEALED protocol,sources,inputs,initialization;0 updates before seal',flush=True)

if __name__=='__main__':main()
