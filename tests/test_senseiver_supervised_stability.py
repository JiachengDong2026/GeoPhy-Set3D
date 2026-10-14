import unittest
from geophy_set3d.training.senseiver_supervised_stability import joint_pass,stability_exit,formal_exit


class StabilityTests(unittest.TestCase):
    def good(self):return dict(mean=dict(nmse=.1,relative_l1=.3),interventions=[dict(relative_MA=.01,wrong_minus_normal_nmse=.2) for _ in range(3)])
    def test_three_joint_pass_and_wrong_input_requirement(self):
        e=self.good();s={}
        for i in [1800,2000,2200]:s=stability_exit(i,[.1]*i,e,.7,s,10)
        self.assertEqual(s['status'],'PASSED');self.assertEqual(s['pass_streak'],3)
        e['interventions'][0]['wrong_minus_normal_nmse']=0
        self.assertFalse(joint_pass(e,.7))
        self.assertEqual(stability_exit(4000,[.1]*4000,e,.7,{},10)['status'],'UNRESOLVED_STABILITY')
    def test_plateau_minimum_and_budget(self):
        e=self.good();e['mean']['relative_l1']=.8
        s=stability_exit(800,[.5]*800,e,.7,{},0);self.assertFalse(s['plateau_condition'])
        s=stability_exit(1000,[.5]*1000,e,.7,s,0);self.assertIsNone(s['status'])
        self.assertEqual(stability_exit(1200,[.5]*1200,e,.7,s,0)['status'],'UNRESOLVED_STABILITY')
        e['mean']['nmse']=1.;e['interventions']=[]
        s=stability_exit(1000,[1.]*1000,e,.7,{},0)
        self.assertEqual(stability_exit(1200,[1.]*1200,e,.7,s,0)['status'],'FAILED_TO_LEARN')
    def test_significance_counter_distinct_from_raw_best_and_minimum50(self):
        rows=[];s={};best=100.
        for epoch in range(1,61):
            value=1.-epoch*.00001;rows.append(dict(train=dict(mean=dict(nmse=.2)),val=dict(mean=dict(nmse=value))))
            oldbest=best;best=min(best,value)
            self.assertLess(best,oldbest)
            s=formal_exit(epoch,rows,None,s)
            if epoch<50:self.assertIsNone(s['reason'])
        self.assertEqual(s['reason'],'VAL_EARLY_STOP')
        self.assertEqual(s['nonsignificant_epochs'],59)
    def test_nolearning_windows_and_positive_learning(self):
        rows=[dict(train=dict(mean=dict(nmse=1.)),val=dict(mean=dict(nmse=1.))) for _ in range(30)]
        h=dict(epoch=30,interventions=[dict(relative_MA=0.)]*3)
        self.assertEqual(formal_exit(30,rows,h,{})['reason'],'FAILED_TO_LEARN')
        rows[-1]['val']['mean']['nmse']=.5
        self.assertIsNone(formal_exit(30,rows,h,{})['reason'])
if __name__=='__main__':unittest.main()
