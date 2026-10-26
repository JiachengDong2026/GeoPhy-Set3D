"""Classical 3D U-Net on the frozen FNO measurement/geometry backprojection.

Five resolution levels, double Conv3d/GroupNorm/ReLU, MaxPool, trilinear
upsampling, and concatenated skips. No learned ray mapper or attention.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .baseline_fno_v1 import FixedRayBackprojection

class DoubleConv3D(nn.Sequential):
 def __init__(self,inputs,outputs):
  super().__init__(nn.Conv3d(inputs,outputs,3,padding=1,bias=False),nn.GroupNorm(8,outputs,eps=1e-5),nn.ReLU(),
                   nn.Conv3d(outputs,outputs,3,padding=1,bias=False),nn.GroupNorm(8,outputs,eps=1e-5),nn.ReLU())

class UNet3DV1(nn.Module):
 def __init__(self,channels=(16,32,64,128,256),activation_checkpoint=True):
  super().__init__()
  if tuple(channels)!=(16,32,64,128,256):raise ValueError('Fixed preregistered five-level U-Net')
  self.activation_checkpoint=bool(activation_checkpoint)
  self.adapter=FixedRayBackprojection()
  self.encoder=nn.ModuleList([DoubleConv3D(i,o) for i,o in zip((6,*channels[:-1]),channels)])
  self.pool=nn.MaxPool3d(2)
  self.up=nn.ModuleList([nn.Conv3d(i,o,1) for i,o in zip(channels[:0:-1],channels[-2::-1])])
  self.decoder=nn.ModuleList([DoubleConv3D(2*o,o) for o in channels[-2::-1]])
  self.readout=nn.Conv3d(channels[0],1,1)
  axis=torch.linspace(-1,1,64)
  self.register_buffer('coordinates',torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),0)[None],persistent=False)
 def _block(self,block,x):
  return checkpoint(block,x,use_reentrant=False) if self.activation_checkpoint and self.training and torch.is_grad_enabled() else block(x)
 def input_features(self,batch):
  inputs=self.adapter(batch)
  return torch.cat((inputs,self.coordinates.expand(len(inputs),-1,-1,-1,-1)),1)
 def forward(self,batch,return_logits=False):
  x=self.input_features(batch);skips=[]
  for i,block in enumerate(self.encoder):
   if i:x=self.pool(x)
   x=self._block(block,x);skips.append(x)
  for up,block,skip in zip(self.up,self.decoder,reversed(skips[:-1])):
   x=F.interpolate(x,size=skip.shape[-3:],mode='trilinear',align_corners=False)
   x=up(x);x=self._block(block,torch.cat((skip,x),1))
  logits=self.readout(x)
  return (logits.sigmoid(),logits) if return_logits else logits.sigmoid()
