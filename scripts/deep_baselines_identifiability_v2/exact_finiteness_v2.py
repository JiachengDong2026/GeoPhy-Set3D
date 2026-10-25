"""Zero-update recheck using integer finite counts, not CUDA floating means."""
from ctx_v2 import *
import numerics_v2 as numerical
import gc

original_stats=numerical.stats
def counted_stats(x):
    old=original_stats(x);mask=torch.isfinite(x)
    total=x.numel();finite=int(mask.sum().item());all_finite=bool(mask.all().item())
    assert all_finite==(finite==total)
    return dict(old,finite_fraction=finite/total,finite_count=finite,total_count=total,all_finite=all_finite)

def main():
    plan=verified(ROOT/'engineering/exact_finiteness_plan.json');assert plan['source_sha256']==sha(__file__)
    verify(verified(ROOT/'registration.json')['sources']);backend();numerical.stats=counted_stats
    output=[];roundoff_fields=0
    with Boundary('evaluate') as access:
        samples=torch.load(ROOT/'inputs/train_AB.pt',map_location='cpu',weights_only=False)+torch.load(ROOT/'inputs/unseen_AB.pt',map_location='cpu',weights_only=False)
        for arm in design()['arms']:
            net=model(arm,'cuda:0')
            for update in design()['evaluation_updates']:
                path=ROOT/'runs'/arm/f'checkpoint_{update:03d}.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
                net.load_state_dict(ck['model'],strict=True);before=state_hash(net.state_dict());del ck
                prior=verified(ROOT/'runs'/arm/f'evaluation_{update:03d}.json')
                for index,sample in enumerate(samples):
                    for branch in ('A','B'):
                        gradient=numerical.sensitivity(net,batch([sample],branch,'cuda:0'))
                        old=prior['cases'][index]['branches'][branch]['gradient'];rounding=[]
                        for key in ('measurement_channel','geometry_channels','readout_weight','readout_bias'):
                            assert gradient[key]['all_finite']
                            # Confirm recorded fractions describe exactly the
                            # same integer count; one missing finite element
                            # would differ by at least 1/N, not a float ULP.
                            assert round(old[key]['finite_fraction']*gradient[key]['total_count'])==gradient[key]['finite_count']
                            if old[key]['finite_fraction']!=gradient[key]['finite_fraction']:
                                rounding.append(key);roundoff_fields+=1
                        output.append(dict(arm=arm,update=update,index=index,branch=branch,gradient=gradient,
                            original_fraction_roundoff_fields=rounding,checkpoint_sha256=sha(path),optimizer_updates=0))
                        gc.collect()
                assert state_hash(net.state_dict())==before
                print('INTEGER FINITE AUDIT',arm,update,'all16case-branches finite',flush=True)
            del net;torch.cuda.empty_cache()
    assert len(output)==224
    freeze(ROOT/'exact_finiteness_audit.json',dict(at=now(),passed=True,optimizer_updates=0,case_branches=224,
        recorded_roundoff_fields=roundoff_fields,all_actual_gradients_finite=True,original_sources_and_evidence_unchanged=True,
        records=output,audit=access.summary(),thresholds_unchanged=True))

if __name__=='__main__':main()
