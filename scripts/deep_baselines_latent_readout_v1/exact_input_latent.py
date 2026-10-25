"""Supplement exact raw inputs: ray token back-scaling has FP32 round-trip error.

No network calls; preserve original extraction and all its hashes. Save the raw
valid-ray slice and spatial validity mask directly, without normalization.
"""
from latent_context import *

def main():
    verify_seal();hashes={};checks=[]
    with Access('extract_raw'):
        ss=samples()
        for arm in EXPECTED:
            assert (ROOT/f'extraction_{arm}.json').exists()
            for b in ('A','B'):
                for kind in ('normal','zero'):
                    for i in (range(8) if kind=='normal' else [0]):
                        inp=old.batch([ss[i]],b);inp=zero(inp) if kind=='zero' else inp
                        valid=inp['input_valid_ray'][:,:,0].bool() & inp['view_mask'][:,:,None,None]
                        exact=inp['input_projection_raw'][:,:,0][valid].numpy().reshape(1,-1,1)
                        base=feature_path(arm,b,kind,i).parent
                        arrays={'exact_raw_measurement':exact,'spatial_valid_mask':valid.numpy()}
                        for name,x in arrays.items():
                            p=base/f'{name}.npy';assert not p.exists();np.save(p,x);hashes[str(p)]=sha(p)
                        back=np.load(base/'raw_measurement.npy');assert back.shape==exact.shape
                        checks.append(dict(arm=arm,branch=b,kind=kind,case=i,maximum_roundtrip_error=float(np.abs(back-exact).max()),exact_raw_saved=True))
    freeze(ROOT/'exact_raw_supplement.json',dict(at=now(),source_sha256=sha(__file__),reason='Retain back-scaled token channel and add exact raw slice/spatial mask without changing historical or captured results',checks=checks,arrays_sha256=hashes,neural_forward_calls=0,neural_updates=0))
if __name__=='__main__':main()
