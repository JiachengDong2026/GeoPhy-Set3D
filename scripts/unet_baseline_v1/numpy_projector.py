"""Independent NumPy trilinear K96 ray integration of saved full XYZ fields."""
import numpy as np

def project(volume,origin,direction,near,length,mask):
 out=np.zeros(mask.shape,dtype=np.float64);flat=volume.reshape(64,64,64).astype(np.float64)
 for v in range(mask.shape[1]):
  ids=np.flatnonzero(mask[0,v,0]);d=direction[0,v].reshape(-1,3)[ids].astype('float32')
  n=near[0,v,0].reshape(-1)[ids].astype('float32');ell=length[0,v,0].reshape(-1)[ids].astype('float32')
  for start in range(0,len(ids),256):
   sl=slice(start,start+256);fraction=(np.arange(96,dtype='float32')+.5)/96
   t=n[sl,None]+ell[sl,None]*fraction
   points=origin[0,v].astype('float32')+d[sl,None,:]*t[...,None]
   # Reproduce physical coordinate convention; interpolation and mass conversion in FP64.
   grid=(points/np.float32(.6)*np.float32(2)-np.float32(1)).astype('float64')
   xyz=np.clip((grid+1)*31.5,0,63);low=np.floor(xyz).astype('int64');high=np.minimum(low+1,63);w=xyz-low
   c=np.zeros(t.shape,dtype='float64')
   for ix in (0,1):
    for iy in (0,1):
     for iz in (0,1):
      x=(high if ix else low)[...,0];y=(high if iy else low)[...,1];z=(high if iz else low)[...,2]
      weight=(w[...,0] if ix else 1-w[...,0])*(w[...,1] if iy else 1-w[...,1])*(w[...,2] if iz else 1-w[...,2])
      c+=flat[x,y,z]*weight
   c=np.clip(c,0,1);chi=(c/16.043)/(c/16.043+(1-c)/28.965)
   out[0,v,0].reshape(-1)[ids[sl]]=chi.mean(-1)*ell[sl].astype('float64')*1e6
 return out
