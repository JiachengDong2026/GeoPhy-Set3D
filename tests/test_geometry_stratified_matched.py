import copy,json,unittest
from pathlib import Path
import numpy as np
import torch
import yaml
from geophy_set3d.evaluation.geometry_stratified_sampler import StratifiedABHSampler
from geophy_set3d.evaluation.geometry_strata import membership
from geophy_set3d.data.measurement_only import DisjointABHSampler
from geophy_set3d.models import build_model
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.selfsup import FROZEN_LOSS,selection_metric
REPO=Path(__file__).resolve().parents[1];OUT=Path('/data/djc/GeoPhy-Set3D/outputs/experiments/geometry_stratified_selfsup_matched_v1')

class MatchedSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=yaml.safe_load((REPO/'configs/train_selfsup_oari_mrgr_v1.yaml').read_text())
        cls.g=dict(np.load('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1/geometry/train_geometry.npz'))
        b=json.loads((REPO/'docs/reports/artifacts/geometry_strata_v2/boundaries.json').read_text())['V']
        cls.bounds={v:b[v]['boundaries'] for v in ('3','4')};cls.bounds['5']=[.32,.42]
        cls.s=StratifiedABHSampler(cls.g['station_origin_m'],cls.bounds)
        cls.plans={a:json.loads((OUT/f'{a}_station_plan.json').read_text()) for a in ('Control','Treatment')} if (OUT/'protocol.json').exists() else None
    def test_control_treatment_initial_state_identical(self):
        torch.manual_seed(20261011);a=build_model(self.cfg);torch.manual_seed(20261011);b=build_model(self.cfg)
        self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()));self.assertEqual(sum(p.numel() for p in a.parameters()),332070)
    def test_control_treatment_same_case_order(self):
        gen1=torch.Generator().manual_seed(20261011);gen2=torch.Generator().manual_seed(20261011)
        for _ in range(300):self.assertEqual(torch.randperm(166,generator=gen1).tolist(),torch.randperm(166,generator=gen2).tolist())
    def test_control_treatment_same_optimizer_steps(self):self.assertEqual(300*((166+1)//2),24900)
    def test_control_treatment_same_loss(self):self.assertEqual(self.cfg['loss'],FROZEN_LOSS)
    def test_control_treatment_same_view_count_distribution(self):
        original=DisjointABHSampler()
        for i in range(60):
            a,b,_=original.sample(64,epoch=i//10,sample_index=i,case_id=str(i));ta,tb,_,_=self.s.sample(64,epoch=i//10,sample_index=i,case_id=str(i))
            self.assertEqual((len(a),len(b)),(len(ta),len(tb)))
        if self.plans:
            for a,b in zip(self.plans['Control'],self.plans['Treatment']):
                for ca,tb in zip(a,b):self.assertEqual((len(ca['A']),len(ca['B']),len(ca['H'])),(len(tb['A']),len(tb['B']),len(tb['H'])))
    def test_sampler_is_only_training_difference(self):
        if (OUT/'protocol.json').exists():
            p=json.loads((OUT/'protocol.json').read_text());a,b=[copy.deepcopy(p['configs'][k]) for k in ('Control','Treatment')]
            for cfg in (a,b):cfg['experiment'].pop('name');cfg['training'].pop('device')
            self.assertEqual(a,b)
    def test_A_B_H_disjoint_in_stratified_sampler(self):
        for i in range(60):
            a,b,h,_=self.s.sample(64,case_id=str(i));self.assertFalse(set(a)&set(b));self.assertFalse(set(a)&set(h));self.assertFalse(set(b)&set(h))
    def test_A_B_same_target_stratum(self):
        for i in range(60):
            a,b,_,q=self.s.sample(64,case_id=str(i));self.assertEqual(membership(self.s.score(a),self.bounds[str(len(a))]),q);self.assertEqual(membership(self.s.score(b),self.bounds[str(len(b))]),q)
        if self.plans:
            p=json.loads((OUT/'protocol.json').read_text());v5=json.loads((OUT/'V5_boundaries.json').read_text());bounds=dict(self.bounds);bounds['5']=v5['boundaries']
            s=StratifiedABHSampler(self.g['station_origin_m'],bounds)
            for epoch in self.plans['Treatment']:
                for r in epoch:
                    for k in ('A','B'):self.assertEqual(membership(s.score(np.array(r[k])),bounds[str(len(r[k]))]),r['target_stratum'])
    def test_extreme_clustered_not_training_stratum(self):
        for i in range(60):self.assertIn(self.s.sample(64,case_id=str(i))[-1],('low','middle','high'))
    def test_checkpoint_selector_identical(self):
        x={'measurement':{'heldout':{'relative_l1':.123}}};self.assertEqual(selection_metric(x),.123)
        self.assertEqual(self.cfg['validation'],dict(split='val',geometry_pool='train',a_views=[4,4],b_views=[4,4],heldout_views=4))

if __name__=='__main__':unittest.main()
