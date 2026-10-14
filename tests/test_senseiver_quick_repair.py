import unittest
import torch
from geophy_set3d.training.senseiver_readout import standardize,ridge,ScalarQueryReadout


class ReadoutTests(unittest.TestCase):
    def test_fit_only_statistics_and_degenerate_rejection(self):
        gen=torch.Generator().manual_seed(3);x=torch.randn(16,64,64,generator=gen);y=torch.arange(16.)
        z,t,s=standardize(x,y);xx=x.clone();xx[8:]*=1000;yy=y.clone();yy[8:]*=1000
        zz,tt,ss=standardize(xx,yy)
        for k in s:torch.testing.assert_close(s[k],ss[k],rtol=0,atol=0)
        torch.testing.assert_close(z[:8],zz[:8]);torch.testing.assert_close(t[:8],tt[:8])
        with self.assertRaises(ValueError):standardize(torch.ones_like(x),y)
        with self.assertRaises(ValueError):standardize(x,torch.ones_like(y))

    def test_ridge_uses_fit_only_and_fixed_regularization(self):
        gen=torch.Generator().manual_seed(4);z=torch.randn(16,64,64,generator=gen,dtype=torch.float64);y=z[:,0,0].clone()
        p,s=ridge(z,y);y[8:]+=1000;q,ss=ridge(z,y)
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertEqual(s,ss);self.assertAlmostEqual(s['lambda_value'],float(.001*(z[:8].flatten(1)@z[:8].flatten(1).T).trace()/8))
        self.assertLess(s['solve_relative_residual'],1e-12)

    def test_query_linear_output_and_frozen_latent(self):
        torch.set_num_threads(2);head=ScalarQueryReadout();x=torch.randn(8,64,64)
        y=head(x);self.assertEqual(tuple(y.shape),(8,));y.square().mean().backward()
        self.assertIsNone(x.grad);self.assertTrue(all(p.grad is not None for p in head.parameters()))
        self.assertEqual(tuple(head.center_query.shape),(1,1,96))


if __name__=='__main__':unittest.main()
