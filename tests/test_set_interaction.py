import unittest
import torch
from geophy_set3d.models.set_interaction import GeometryConditionedSetInteraction
from geophy_set3d.models.attention_fusion import GeometryConditionedAttentionPooling


class SetInteractionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(23)
        self.block=GeometryConditionedSetInteraction()
        self.features=torch.randn(2,4,32,2,3,2,requires_grad=True)
        self.geometry=torch.randn(2,4,4,2,3,2)
        self.visible=torch.rand(2,4,2,3,2)>.2
        self.visible[:,: ,0,0,0]=False
        self.views=torch.tensor([[True,True,True,False],[True]*4])

    def forward(self, features=None, geometry=None, visible=None, views=None):
        return self.block(self.features if features is None else features,
                          self.geometry if geometry is None else geometry,
                          self.visible if visible is None else visible,
                          self.views if views is None else views,True)

    def test_set_interaction_shapes_and_finite(self):
        context,weights=self.forward()
        self.assertEqual(context.shape,self.features.shape)
        self.assertEqual(weights.shape,(2,4,4,4,2,3,2))
        self.assertTrue(torch.isfinite(context).all() and torch.isfinite(weights).all())
        valid=self.visible & self.views[:,:,None,None,None]
        pair=valid[:,None,:,None]&valid[:,None,None,:]
        self.assertTrue((weights[~pair.expand_as(weights)]==0).all())
        torch.testing.assert_close(weights.sum(3),valid[:,None].expand(-1,4,-1,-1,-1,-1).float())
        self.assertTrue((context[~valid[:,:,None].expand_as(context)]==0).all())

    def test_set_interaction_gradients(self):
        context,_=self.forward();context.square().mean().backward()
        for name,p in self.block.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        for module in (self.block.qkv,self.block.geometry_embedding):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()),0.)

    def test_set_interaction_permutation_equivariance(self):
        context,weights=self.forward();p=torch.tensor([3,1,0,2])
        other,matrices=self.forward(self.features[:,p],self.geometry[:,p],self.visible[:,p],self.views[:,p])
        torch.testing.assert_close(context[:,p],other,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(weights[:,:,p][:,:,:,p],matrices,atol=1e-6,rtol=1e-6)

    def test_invalid_views_do_not_affect_valid_views(self):
        context,weights=self.forward();valid=self.visible & self.views[:,:,None,None,None]
        bad_features=torch.where(valid[:,:,None],self.features,float('nan'))
        bad_geometry=torch.where(valid[:,:,None],self.geometry,float('nan'))
        other,matrices=self.forward(bad_features,bad_geometry)
        torch.testing.assert_close(context,other,atol=0,rtol=0)
        torch.testing.assert_close(weights,matrices,atol=0,rtol=0)
        f=torch.cat((self.features,torch.full_like(self.features[:,:1],float('nan'))),1)
        g=torch.cat((self.geometry,torch.full_like(self.geometry[:,:1],float('nan'))),1)
        m=torch.cat((self.visible,torch.ones_like(self.visible[:,:1])),1)
        vm=torch.cat((self.views,torch.zeros_like(self.views[:,:1])),1)
        padded,w=self.forward(f,g,m,vm)
        torch.testing.assert_close(context,padded[:,:4],atol=1e-6,rtol=1e-6)
        self.assertTrue((padded[:,-1]==0).all() and (w[:,:,-1]==0).all() and (w[:,:,:,-1]==0).all())

    def test_all_invalid_voxel_and_batch(self):
        self.visible.zero_();context,weights=self.forward()
        self.assertTrue((context==0).all() and (weights==0).all())
        fused,w=GeometryConditionedAttentionPooling()(context,self.geometry,self.visible,self.views)
        self.assertTrue((fused==0).all() and (w==0).all())
        fused.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.block.parameters()))

    def test_context_and_geometry_reach_other_views(self):
        self.visible.fill_(True);self.views.fill_(True)
        context,weights=self.forward()
        changed=self.features.detach().clone();changed[:,1]+=torch.randn_like(changed[:,1])
        other,_=self.forward(changed)
        self.assertGreater((context[:,0]-other[:,0]).abs().max().item(),1e-4)
        geometry=self.geometry.clone();geometry[:,1]+=3
        other,matrices=self.forward(geometry=geometry)
        self.assertGreater((context[:,0]-other[:,0]).abs().max().item(),1e-4)
        self.assertGreater((weights-matrices).abs().max().item(),1e-5)

    def test_no_positional_view_index_dependency(self):
        f=self.features[:1,:1].expand(1,4,-1,-1,-1,-1)
        g=self.geometry[:1,:1].expand(1,4,-1,-1,-1,-1)
        m=torch.ones(1,4,2,3,2,dtype=torch.bool);vm=torch.ones(1,4,dtype=torch.bool)
        context,_=self.forward(f,g,m,vm)
        for i in range(1,4):torch.testing.assert_close(context[:,0],context[:,i],atol=0,rtol=0)
