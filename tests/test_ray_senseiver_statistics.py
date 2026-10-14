import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from geophy_set3d.evaluation.ray_senseiver import pair_three,case_means,statistics
from geophy_set3d.training.ray_senseiver import load_comparison_config


class ThreeMethodStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.c=load_comparison_config('configs/train_ray_senseiver_selfsup_v1.yaml')
        self.root=Path(self.c['comparison']['deepsets_artifact'])
        self.deep=pd.read_csv(self.root/'deepsets_900.csv');self.old=pd.read_csv(self.root/'paired_900.csv')
        self.records=json.loads((self.root/'station_subsets.json').read_text())
        self.new=self.deep.copy();self.new['input_ray_view_sha256']='fixture'

    def test_complete_pair_and_case_bootstrap(self):
        pair=pair_three(self.new,self.old,self.deep,self.records);case=case_means(pair);stats,tails,contrasts=statistics(case,self.c)
        self.assertEqual((len(pair),len(case),len(stats),len(tails),len(contrasts)),(900,180,100,150,60))
        ref=pd.read_csv(self.root/'paired_statistics.csv')
        for _,s in stats[stats.family=='primary'].iterrows():
            old=ref[(ref.group==s.group)&(ref.metric==s.metric)].iloc[0]
            np.testing.assert_allclose([s.mean_delta,s.ci_low,s.ci_high],[old.mean_delta,old.ci_low,old.ci_high],rtol=1e-11,atol=1e-14)
        self.assertTrue((stats[stats.family=='auxiliary'][['mean_delta','ci_low','ci_high']].to_numpy()==0).all())

    def test_zero_contrast_denominator(self):
        from geophy_set3d.evaluation.ray_senseiver import summarize
        value=summarize(np.ones(36),np.zeros(36),np.arange(36)[None],1e-8)
        self.assertIsNone(value['relative_reduction']);self.assertEqual(value['mean_delta'],1.)

    def test_reject_physical_mutation_and_missing_draw(self):
        for field in ['complete_input_sha256','complete_H_sha256','input_identity_sha256','H_mask_sha256']:
            bad=self.new.copy();bad.loc[0,field]='changed'
            with self.assertRaises(ValueError):pair_three(bad,self.old,self.deep,self.records)
        with self.assertRaises(ValueError):pair_three(self.new.iloc[:-1],self.old,self.deep,self.records)
        bad=self.new.copy();bad.loc[0,'station_ids']='[1,2,3,4]'
        with self.assertRaises(ValueError):pair_three(bad,self.old,self.deep,self.records)

if __name__=='__main__':unittest.main()
