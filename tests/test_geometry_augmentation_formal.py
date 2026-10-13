import unittest
import numpy as np
import torch
from geophy_set3d.evaluation.geometry_augmentation_formal_stats import paired,decide,holm,worst
from geophy_set3d.training.geometry_augmentation_formal import tree_hash

class FormalStatisticsTests(unittest.TestCase):
    def test_paired_case_bootstrap_direction_and_identity(self):
        c=np.arange(1.,37.);t=c*.8
        x=paired(c,t);self.assertEqual(x['n_cases'],36);self.assertEqual(x['Treatment_wins'],36)
        self.assertLess(x['delta_ci95'][1],0);self.assertAlmostEqual(x['ratio'],.8)
        self.assertEqual(x,paired(c,t));self.assertEqual(paired(c,c)['ties'],36)
    def test_decision_boundary_and_natural_harm(self):
        g={'delta_ci95':[-.1,-.01],'improvement':.15}
        self.assertEqual(decide(g,{'ratio':1.03}),'GO')
        self.assertEqual(decide(g,{'ratio':1.04}),'PARTIAL')
        self.assertEqual(decide(g,{'ratio':1.11}),'NO-GO')
        g['delta_ci95']=[-.1,.001]
        self.assertEqual(decide(g,{'ratio':1.02}),'PARTIAL')
        g['improvement']=.099
        self.assertEqual(decide(g,{'ratio':1.02}),'NO-GO')
    def test_holm_and_case_worst(self):
        np.testing.assert_allclose(holm([.04,.001,.03]),[.06,.003,.06])
        x=worst([1,2,2],['a','b','c']);self.assertEqual(x['worst_case_ids'],['b','c'])
    def test_optimizer_hash_is_order_invariant_and_detects_tensor_change(self):
        a={'z':torch.tensor([1.,2.]),'a':{'step':torch.tensor(1)}}
        b={'a':a['a'],'z':a['z'].clone()};self.assertEqual(tree_hash(a),tree_hash(b))
        b['z'][0]+=1;self.assertNotEqual(tree_hash(a),tree_hash(b))
    def test_pairing_rejects_misaligned_vectors(self):
        with self.assertRaises(ValueError):paired([1,2],[1,2,3])
if __name__=='__main__':unittest.main()
