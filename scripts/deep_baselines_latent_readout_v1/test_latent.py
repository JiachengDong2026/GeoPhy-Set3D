"""Meaningful engineering tests: estimator, plume exclusion, capture transparency, boundary."""
from latent_context import *
from ridge_latent import sufficient,solve
from extract_latent import Capture
import unittest
from geophy_set3d.models.baseline_deepsets_v1 import RayDeepSetsV1
from geophy_set3d.models.baseline_gnot_v1 import RayGNOTV1

class Tests(unittest.TestCase):
    def test_ridge_affine_and_fold(self):
        rng=np.random.default_rng(24);xs=[rng.normal(size=(45,4)) for i in range(6)];w=np.array([.1,-.2,.4,.7]);ys=[x@w+.02 for x in xs]
        ss=[sufficient(x,y,13) for x,y in zip(xs,ys)]
        fit=solve(ss,1e-10);np.testing.assert_allclose(fit['coef'],w,atol=1e-8);self.assertLess(abs(fit['intercept']-.02),1e-8)
        clean=solve(ss[:5],1e-4);ys[5][:]=1e9;ss[5]=sufficient(xs[5],ys[5]);again=solve(ss[:5],1e-4)
        np.testing.assert_array_equal(clean['coef'],again['coef'])
    def test_stage_denial(self):
        access=Access('fit');access.active=True
        with self.assertRaises(PermissionError):access.audit('open',(str(OLD/'inputs/truth_6.npy'),'r',0))
        with self.assertRaises(PermissionError):access.audit('open',('/data/forbidden_GT.npy','r',0))
        with self.assertRaises(PermissionError):access.audit('open',(str(checkpoint('gnot')),'w',0))
        access.audit('open',(str(OLD/'inputs/truth_0.npy'),'r',0))
        self.assertEqual(len(access.denied),3)
    def test_hooks_cpu(self):
        old.backend();torch.manual_seed(2);ss=torch.load(OLD/'inputs/train_AB.pt',weights_only=False);inp=old.batch([ss[0]],'A')
        for arm,net in [('deepsets',RayDeepSetsV1(hidden_dim=128)),('gnot',RayGNOTV1(width=64,layers=2,query_chunk_size=16))]:
            net.eval().requires_grad_(False)
            with torch.inference_mode():
                def f():
                    if arm=='deepsets':return net.decode_queries(net.encode(inp),net.query_coordinates_m[:32])
                    from geophy_set3d.models.ray_senseiver_ss import ray_tokens
                    tokens,excluded,counts=ray_tokens(inp)
                    return net.forward_tokens(tokens,excluded,counts,coordinates_m=net.query_coordinates_m[:32])
                off=f()
                with Capture(net,arm) as cap:on=f()
                self.assertTrue(torch.equal(off,on));arrays=cap.arrays();self.assertEqual(arrays['prehead'].shape[1],32)
                self.assertTrue(all(np.isfinite(x).all() for x in arrays.values()))
            self.assertTrue(all(p.grad is None for p in net.parameters()))
if __name__=='__main__':unittest.main()
