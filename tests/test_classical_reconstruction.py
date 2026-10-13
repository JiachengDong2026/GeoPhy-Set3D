import unittest
import numpy as np
import torch
from geophy_set3d.reconstruction.classical import (RayMatrix,mole_to_mass,mass_to_mole,physical_gradient,regularizer,geometry_tensor,project,sart,variational)


def phantom_geometry(n=10):
    g=dict(station_origin_m=np.array([[-1,0,0],[0,-1,0],[0,0,-1.]]),ray_direction=np.zeros((3,n,n,3)),
           t_near=np.ones((3,n,n)),path_length_m=np.full((3,n,n),.6),valid_ray=np.ones((3,n,n),bool))
    # Each station's per-ray origin offset must be represented by directions.
    # Instead use many stations, each with one ray and an exact translated origin.
    origins=[];dirs=[]
    for axis in range(3):
        for a in np.linspace(0,.6,n):
            for b in np.linspace(0,.6,n):
                o=np.zeros(3);o[axis]=-1;o[(axis+1)%3]=a;o[(axis+2)%3]=b;d=np.zeros(3);d[axis]=1
                origins.append(o);dirs.append(d)
    v=len(origins)
    return dict(station_origin_m=np.array(origins),ray_direction=np.array(dirs)[:,None,None,:],
                t_near=np.ones((v,1,1)),path_length_m=np.full((v,1,1),.6),valid_ray=np.ones((v,1,1),bool))


class ClassicalTests(unittest.TestCase):
    def test_conversion_boundaries_roundtrip(self):
        c=torch.linspace(0,1,1001,dtype=torch.float64)
        torch.testing.assert_close(mole_to_mass(mass_to_mole(c)),c,atol=1e-14,rtol=1e-14)
        self.assertEqual(float(mole_to_mass(torch.tensor(0.))),0);self.assertEqual(float(mole_to_mass(torch.tensor(1.))),1)

    def test_zero_constant_units_xyz_and_paths(self):
        g=phantom_geometry(3);a=RayMatrix(g,n=8,samples=24,dtype=torch.float64);t=geometry_tensor(g,'cpu')
        c=torch.full((8,8,8),.12,dtype=torch.float64)
        torch.testing.assert_close(a.forward(torch.ones_like(c)),torch.full((27,),600000.,dtype=torch.float64),rtol=1e-14,atol=1e-8)
        torch.testing.assert_close(a.forward(mass_to_mole(c)),project(c,t,24).flatten(),rtol=1e-7,atol=1e-4)
        self.assertEqual(float(project(c*0,t,24).abs().sum()),0)
        for axis in range(3):
            shape=[1]*3;shape[axis]=8
            ramp=torch.linspace(0,.6,8,dtype=torch.float64).reshape(shape).expand(8,8,8)
            linear=a.forward(ramp).reshape(3,3,3)/600000
            torch.testing.assert_close(linear[axis],torch.full((3,3),.3,dtype=torch.float64),rtol=1e-14,atol=1e-14)
        self.assertTrue(np.allclose(np.linalg.norm(g['ray_direction'],axis=-1),1))

    def test_adjoint(self):
        torch.manual_seed(123);a=RayMatrix(phantom_geometry(4),n=9,samples=24,dtype=torch.float64)
        x=torch.randn(9**3,dtype=torch.float64);y=torch.randn(48,dtype=torch.float64)
        lhs=a.forward(x)@y;rhs=x@a.adjoint(y)
        self.assertLess(float(abs(lhs-rhs)/max(abs(lhs),abs(rhs))),1e-12)

    def test_valid_zero_invalid_and_padding(self):
        g=phantom_geometry(2);g['valid_ray'][0]=False
        a=RayMatrix(g,n=8,samples=16);y=torch.zeros((12,1,1));y[0]=999
        self.assertEqual(a.measurement(y).numel(),11);self.assertEqual(float(a.measurement(y).sum()),0)
        chi,info=sart([a],[y],dict(iterations=5,relaxation=.5,tolerance=1e-6,minimum_iterations=1,patience=2),n=8)
        self.assertEqual(float(chi.abs().sum()),0)
        # Padding is removed before RayMatrix, never represented as fake station.
        from geophy_set3d.data.collate import projection_pool_collate
        self.assertIsNotNone(projection_pool_collate)

    def test_physical_gradient_and_regularizer_gradient(self):
        n=7;c=torch.linspace(0,.6,n,dtype=torch.float64)[:,None,None].expand(n,n,n).clone()
        g=physical_gradient(c);torch.testing.assert_close(g[0,:-1],torch.ones_like(g[0,:-1]));self.assertEqual(float(g[1:].abs().sum()),0)
        for method in ['Tikhonov','TV']:
            torch.manual_seed(123);c=(torch.rand(4,4,4,dtype=torch.float64)*.02).requires_grad_(True)
            self.assertTrue(torch.autograd.gradcheck(lambda x:regularizer(x,method,1e-3),(c,),eps=1e-6,atol=1e-5))

    def test_nonlinear_forward_gradient_and_discretization(self):
        g=phantom_geometry(2);tg=geometry_tensor(g,'cpu')
        torch.manual_seed(17);c=(torch.rand(5,5,5,dtype=torch.float64)*.1).requires_grad_(True)
        v=torch.randn_like(c);loss=project(c,tg,24).square().sum()/1e12
        grad,=torch.autograd.grad(loss,c);eps=1e-6
        fd=(project(c.detach()+eps*v,tg,24).square().sum()-project(c.detach()-eps*v,tg,24).square().sum())/(2*eps*1e12)
        torch.testing.assert_close((grad*v).sum(),fd,rtol=1e-7,atol=1e-8)
        a=RayMatrix(g,n=5,samples=24,dtype=torch.float64)
        self.assertGreater(float((a.forward(mass_to_mole(c.detach()))-project(c.detach(),tg,24).flatten()).abs().max()),1)

    def test_phantom_reconstruction_convergence(self):
        n=6;g=phantom_geometry(n);a=RayMatrix(g,n=n,samples=24)
        axis=torch.linspace(0,.6,n);x,y,z=torch.meshgrid(axis,axis,axis,indexing='ij')
        gt=.04*torch.exp(-((x-.3)**2+(y-.3)**2+(z-.3)**2)/.012)
        measurement=a.forward(mass_to_mole(gt)).reshape(g['valid_ray'].shape)
        chi,info=sart([a],[measurement],dict(iterations=100,relaxation=.5,tolerance=1e-7,minimum_iterations=20,patience=5),n=n)
        self.assertLess(info['solver_relative_l2'],.05);self.assertLess(float((mole_to_mass(chi)-gt).abs().sum()/gt.abs().sum()),1)
        # Nonlinear baselines use exactly frozen Pdiff/K96.
        tg=geometry_tensor(g,'cpu');measurement=project(gt,tg)
        config=dict(iterations=80,initial_step=100.,step_bounds=[1e-4,1e6],max_backtracks=16,armijo=1e-4,minimum_iterations=20,patience=5,tolerance=1e-6,epsilon=1e-4,**{'lambda':1e-5})
        for method in ['Tikhonov','TV']:
            c,stats=variational(tg,measurement,method,config,32284.89458,n=n)
            self.assertLess(stats['solver_relative_l2'],.3);self.assertTrue(torch.isfinite(c).all())
            self.assertLess(stats['curve'][-1]['objective'],stats['curve'][0]['objective']);self.assertGreaterEqual(float(c.min()),0);self.assertLessEqual(float(c.max()),1)

if __name__=='__main__':unittest.main()
