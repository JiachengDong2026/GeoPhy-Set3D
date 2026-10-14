import unittest
import numpy as np
import pandas as pd
from geophy_set3d.evaluation.deep_comparison import case_means,paired_bootstrap,pair_reference,statistics
from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import METRICS,GROUPS


class ComparisonStatisticsTests(unittest.TestCase):
    def test_draws_average_before_bootstrap(self):
        rows=[]
        for i in range(36):
            for g in GROUPS:
                for d in range(5):
                    row=dict(case_id=str(i),case_index=i,group=g,draw=d)
                    for m in METRICS:row.update({'Ours_'+m:i+d,'DeepSets_'+m:i+d+1,'delta_'+m:-1})
                    rows.append(row)
        means=case_means(pd.DataFrame(rows));self.assertEqual(len(means),180)
        self.assertEqual(float(means[means.case_id=='0']['Ours_relative_l1'].iloc[0]),2)
        c=dict(comparison=dict(bootstrap_seed=2026101801,bootstrap_samples=10000,tie_epsilon=1e-8))
        stats,tails,contrasts=statistics(means,c)
        self.assertEqual(len(stats),50);self.assertTrue((stats.ours_wins==36).all());self.assertTrue((stats.ci_low==-1).all())
        with self.assertRaises(ValueError):case_means(pd.DataFrame(rows[:-1]))
        with self.assertRaises(ValueError):case_means(pd.concat([pd.DataFrame(rows),pd.DataFrame(rows[:1])]))

    def test_difference_sign_and_case_resampling(self):
        a=np.array([.1,.5,.3]);b=np.array([.2,.4,.3]);indices=np.array([[0,0,0],[1,1,1],[2,2,2]])
        r=paired_bootstrap(a,b,indices,1e-8)
        self.assertEqual((r['ours_wins'],r['deepsets_wins'],r['ties']),(1,1,1))
        self.assertLess(r['ci_low'],0);self.assertGreater(r['ci_high'],0)

    def test_pairing_rejects_wrong_input_order_and_hash(self):
        records=[];br=[];ours=[]
        for i in range(36):
            for g in GROUPS:
                for d in range(5):
                    r=dict(case_id=str(i),case_index=i,group=g,draw=d,geometry_id=g+str(d),station_ids=[0,1,2,3],H_station_ids=[4,5,6,7]);records.append(r)
                    x=dict(r,input_identity_sha256='input',input_mask_sha256='mask',H_mask_sha256='hmask',checkpoint_sha256='checkpoint',prediction_sha256='prediction')
                    br.append(dict(x,**{m:1. for m in METRICS}));ours.append(dict(x,**{'C1_'+m:.5 for m in METRICS}))
        b=pd.DataFrame(br);o=pd.DataFrame(ours);p=pair_reference(b,o,records);self.assertTrue((p.delta_relative_l1==-.5).all())
        bad=o.copy();bad.at[0,'station_ids']=[1,0,2,3]
        with self.assertRaises(ValueError):pair_reference(b,bad,records)
        bad=o.copy();bad.at[0,'H_mask_sha256']='wrong'
        with self.assertRaises(ValueError):pair_reference(b,bad,records)
