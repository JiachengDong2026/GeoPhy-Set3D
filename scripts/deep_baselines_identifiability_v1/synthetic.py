"""Independent FP64 analytic plume quadrature; synthetic truth is evaluation-only."""
from diagnostics import *

def analytic(points,spec):
    points=np.asarray(points,dtype=np.float64);out=np.zeros(points.shape[:-1],dtype=np.float64)
    for comp in spec['components']:
        d=points-np.array(comp['center']);angle=comp['angle'];cs,sn=np.cos(angle),np.sin(angle)
        rotated=np.stack((cs*d[...,0]+sn*d[...,1],-sn*d[...,0]+cs*d[...,1],d[...,2]),-1)
        out+=comp['peak']*np.exp(-.5*np.sum((rotated/np.array(comp['sigma']))**2,axis=-1))
    assert np.isfinite(out).all() and out.min()>=0 and out.max()<1
    return out

def reference(inp,spec,order):
    nodes,weights=np.polynomial.legendre.leggauss(order);nodes=(nodes+1)*.5;weights=weights*.5
    mask=inp['input_valid_ray'][:,0].numpy().astype(bool)
    origin=inp['input_station_origin_m'][:,None,None,:].numpy()
    direction=inp['input_ray_direction'].numpy();near=inp['input_t_near'][:,0].numpy();length=inp['input_path_length_m'][:,0].numpy()
    origins=np.broadcast_to(origin,direction.shape)[mask].astype(float);directions=direction[mask].astype(float)
    near=near[mask].astype(float);length=length[mask].astype(float);result=np.zeros(mask.shape,dtype=np.float64)
    values=[]
    for start in range(0,len(length),512):
        sl=slice(start,start+512)
        points=origins[sl,None,:]+directions[sl,None,:]*(near[sl,None]+length[sl,None]*nodes)[...,None]
        mass=analytic(points,spec)
        # Independent explicit physical conversion, not the torch projector.
        mole=(mass/16.043)/(mass/16.043+(1-mass)/28.965)
        values.append((mole@weights)*length[sl]*1e6)
    result[mask]=np.concatenate(values)
    return result[:,None]

def generate(ds):
    outputs=[];rows=[]
    for index,spec in enumerate(design()['synthetic_cases']):
        sample=selected_sample(ds,index,('A','B','H'))
        for branch,inp in sample.items():
            low=reference(inp,spec,128);high=reference(inp,spec,256)
            rows.append(dict(index=index,case=spec['name'],branch=branch,reference_orders=[128,256],
                max_abs_reference_difference_ppm_m=float(np.max(np.abs(high-low))),
                reference_relative_l1_difference=float(np.sum(np.abs(high-low))/np.sum(np.abs(high)))))
            inp['input_projection_raw']=torch.from_numpy(high.astype(np.float32))
            inp['input_projection']=inp['input_projection_raw']/Y_SCALE
            inp['case_id']='synthetic_'+spec['name'];inp['sample_id']='synthetic_'+spec['name'];inp['split']='synthetic'
        axis=np.linspace(0,.6,64);xyz=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1)
        truth=analytic(xyz,spec).astype(np.float32)
        np.save(ROOT/f'inputs/synthetic_truth_{index}.npy',truth,allow_pickle=False)
        # Diagnostic target geometry identical across v2/v3, all valid rays retained.
        for branch in ('A','B','H'):
            target=batch([sample],branch,'cpu')
            with torch.no_grad():projected=project_input(torch.from_numpy(truth)[None,None],target)
            row=next(r for r in rows if r['index']==index and r['branch']==branch)
            row['grid64_K96_vs_reference_relative_l1']=scores(projected,target)[0]['relative_l1']
        atomic(ROOT/f'inputs/synthetic_H_{index}.pt',sample.pop('H'))
        outputs.append(sample)
    return outputs,rows
