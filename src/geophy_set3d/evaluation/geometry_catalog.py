"""Pool A geometry-only descriptors and strata, no GT/measurements/model interface."""
import numpy as np
from geophy_set3d.evaluation.clustered_one_swap import directions


class GeometryCatalog:
    def __init__(self, geometry, coverage_resolution=16, samples=96):
        self.g=geometry;self.n=coverage_resolution;self.cover=[];self.rays=[]
        for i,origin in enumerate(geometry['station_origin_m']):
            valid=geometry['valid_ray'][i].astype(bool)
            d=geometry['ray_direction'][i][valid].astype(np.float64)
            near=geometry['t_near'][i][valid];length=geometry['path_length_m'][i][valid]
            t=near[:,None]+length[:,None]*(np.arange(samples)+.5)/samples
            xyz=origin+d[:,None,:]*t[:,:,None]
            p=np.clip(xyz/.6*(self.n-1),0,self.n-1)
            lo=np.minimum(np.floor(p).astype(int),self.n-2);frac=p-lo;hit=np.zeros((self.n,)*3,bool)
            for a in (0,1):
                for b in (0,1):
                    for c in (0,1):
                        q=lo+np.array([a,b,c]);w=np.prod(np.where(np.array([a,b,c]),frac,1-frac),-1)>0
                        q=q[w];hit[q[:,0],q[:,1],q[:,2]]=True
            self.cover.append(hit)
            self.rays.append(d)
        self.center_dirs=directions(geometry['station_origin_m'])

    def describe(self, ids):
        ids=list(map(int,ids))
        if len(ids)<2 or len(set(ids))!=len(ids):raise ValueError('Distinct real stations required')
        origin=self.g['station_origin_m'][ids].astype(float);center=self.center_dirs[ids]
        dot=np.abs(center@center.T)[np.triu_indices(len(ids),1)]
        d=np.concatenate([self.rays[i] for i in ids]);q=d.T@d/len(d)
        sample=d[np.linspace(0,len(d)-1,min(len(d),512)).astype(int)]
        rd=np.abs(sample@sample.T)[np.triu_indices(len(sample),1)]
        cover=np.stack([self.cover[i] for i in ids]).sum(0)
        return dict(V=len(ids),station_ids=ids,center_direction_score=float(np.mean(1-dot)),
            actual_ray_direction_score_sampled=float(np.mean(1-rd)),actual_ray_direction_tensor_eigenvalues=np.linalg.eigvalsh(q).tolist(),
            valid_ray_count=len(d),coverage_resolution=self.n,coverage_fraction=float((cover>0).mean()),
            coverage_at_least_two_stations=float((cover>=2).mean()),coverage_by_xyz_plane=[(cover>0).mean(tuple(j for j in range(3) if j!=i)).tolist() for i in range(3)],
            station_height_m=origin[:,2].tolist(),station_distance_to_center_m=np.linalg.norm(origin-.3,axis=1).tolist(),
            station_origins_m=origin.tolist(),descriptor_status='geometry description only; not observability or condition number')
