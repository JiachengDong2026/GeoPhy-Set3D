"""Meaningful isolation, geometry, quadrature and physical backward checks."""
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]/'scripts/deep_baselines_identifiability_v2'
sys.path.insert(0,str(HERE))
from ctx_v2 import Boundary,geometry_only,batch,nonmeasurement,value_hash,ROOT,GEOMETRY,design
from synthetic_v2 import analytic,reference
from numerics_v2 import objective,intervene,valid
from geophy_set3d.models.baseline_measurement_init_v2 import RayDeepSetsMeasurementInitV2,RayGNOTMeasurementInitV2
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset

class DiagnosticsV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with Boundary('test') as g:cls.geometry=geometry_only()
        assert list(g.opens)==[str(GEOMETRY.resolve())]

    def test_geometry_only_never_calls_projection_getters(self):
        with patch.object(MeasurementOnlyPoolDataset,'_projections',side_effect=AssertionError('forbidden')),patch.object(MeasurementOnlyPoolDataset,'_measurement_sample',side_effect=AssertionError('forbidden')):
            with Boundary('test') as guard:other=geometry_only()
        self.assertEqual(value_hash(other['A']),value_hash(self.geometry['A']))
        self.assertEqual(set(guard.opens),{str(GEOMETRY.resolve())})
        groups=[set(v['input_station_indices'].tolist()) for v in other.values()]
        self.assertTrue(all(not a&b for i,a in enumerate(groups) for b in groups[i+1:]))

    def test_worker_file_access_denial(self):
        for phase in ('train','smoke'):
            with Boundary(phase) as guard:
                for file in [ROOT/'inputs/H_0.pt',ROOT/'inputs/truth_0.npy',ROOT/'inputs/unseen_AB.pt',GEOMETRY,GEOMETRY.parent.parent/'train_statistics.json']:
                    with self.assertRaises(PermissionError):file.open('rb')
            self.assertEqual(len(guard.denied),5)

    def test_measurements_distinct_geometry_and_swaps_exact(self):
        source=copy.deepcopy(self.geometry);target=copy.deepcopy(self.geometry)
        for b in ('A','B'):
            mask=source[b]['input_valid_ray'];source[b]['input_projection_raw'][mask]=1.;target[b]['input_projection_raw'][mask]=2.
            source[b]['input_projection']=source[b]['input_projection_raw']/32284.894580078173
            target[b]['input_projection']=target[b]['input_projection_raw']/32284.894580078173
        a=batch([source],'A');b=batch([target],'A')
        swapped=intervene(a,'swap',b)
        self.assertEqual(value_hash(swapped),value_hash(b))
        for kind in ('zero','permuted'):
            v=intervene(a,kind);self.assertEqual(value_hash(nonmeasurement(v)),value_hash(nonmeasurement(a)))
            self.assertTrue(torch.equal(v['input_projection_raw'][~valid(a)],a['input_projection_raw'][~valid(a)]))

    def test_independent_reference_and_sampling(self):
        spec=design()['synthetic_cases'][0];a=self.geometry['A']
        lo=reference(a,spec,128);hi=reference(a,spec,256)
        self.assertLess(np.abs(lo-hi).sum()/hi.sum(),1e-8)
        self.assertGreater(hi.sum(),0)
        self.assertEqual(analytic(np.array([[.2,.3,.25]]),spec).dtype,np.float64)

    def test_full_grid_physical_objective_backward_and_roundtrip(self):
        samples=copy.deepcopy(self.geometry)
        for b in ('A','B'):
            y=reference(samples[b],design()['synthetic_cases'][0],256)
            samples[b]['input_projection_raw']=torch.from_numpy(y.astype(np.float32))
            samples[b]['input_projection']=samples[b]['input_projection_raw']/32284.894580078173
        ab={k:batch([samples],k) for k in ('A','B')}
        for cls,kwargs in [(RayDeepSetsMeasurementInitV2,dict(hidden_dim=8)),(RayGNOTMeasurementInitV2,dict(width=8,layers=1,heads=2,experts=1,inner_multiplier=1))]:
            net=cls(mass_fraction=design()['initial_mass_fraction'],**kwargs)
            loss,parts=objective(net,ab);loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
            checkpoint=io.BytesIO();torch.save(net.state_dict(),checkpoint);checkpoint.seek(0)
            fresh=cls(mass_fraction=design()['initial_mass_fraction'],**kwargs);fresh.load_state_dict(torch.load(checkpoint,weights_only=True))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(net.parameters(),fresh.parameters())))
        with self.assertRaises(ValueError):objective(net,dict(ab,H=ab['A']))

if __name__=='__main__':unittest.main()
