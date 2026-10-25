"""Independent FP64 analytic fields and continuous Gauss-Legendre reference."""
from ctx_v2 import *

def analytic(points,spec):
    points=np.asarray(points,dtype=np.float64);out=np.zeros(points.shape[:-1],dtype=np.float64)
    for comp in spec['components']:
        d=points-np.asarray(comp['center']);cs,sn=np.cos(comp['angle']),np.sin(comp['angle'])
        rotated=np.stack((cs*d[...,0]+sn*d[...,1],-sn*d[...,0]+cs*d[...,1],d[...,2]),-1)
        out+=comp['peak']*np.exp(-.5*np.sum((rotated/np.asarray(comp['sigma']))**2,axis=-1))
    assert np.isfinite(out).all() and out.min()>=0 and out.max()<1
    return out

def reference(inp,spec,order):
    nodes,weights=np.polynomial.legendre.leggauss(order);nodes=(nodes+1)*.5;weights=weights*.5
    mask=inp['input_valid_ray'][:,0].numpy().astype(bool)
    direction=inp['input_ray_direction'].numpy().astype(np.float64)
    origin=np.broadcast_to(inp['input_station_origin_m'][:,None,None,:].numpy(),direction.shape)[mask].astype(np.float64)
    direction=direction[mask];near=inp['input_t_near'][:,0].numpy()[mask].astype(np.float64)
    length=inp['input_path_length_m'][:,0].numpy()[mask].astype(np.float64);values=[]
    for start in range(0,len(length),512):
        sl=slice(start,start+512)
        points=origin[sl,None,:]+direction[sl,None,:]*(near[sl,None]+length[sl,None]*nodes)[...,None]
        mass=analytic(points,spec);mole=(mass/16.043)/(mass/16.043+(1-mass)/28.965)
        values.append((mole@weights)*length[sl]*1e6)
    result=np.zeros(mask.shape,dtype=np.float64);result[mask]=np.concatenate(values)
    return result[:,None]
