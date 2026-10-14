import copy
import json
from pathlib import Path
import sys
import unittest
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from geophy_set3d.training.selfsup import make_dataset,move_dual_batch,measurement_discrepancy,objective as old_objective
from geophy_set3d.data.measurement_only import dual_measurement_collate
from geophy_set3d.training.senseiver_relative_probe import validate,objective,from_projections,components,exit_event,verify_resume
from diagnose_senseiver_relative_probe import gradient_split,grouped_constant_contributions,validate_group_sum


class RelativeProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.c=yaml.safe_load(Path('configs/ray_senseiver_relative_probe_v1.yaml').read_text())

    def test_original_sum_denominator_and_valid_zero(self):
        # All targets share the per-case sum denominator: y=0 contributes nonzero gradient.
        t=dict(input_valid_ray=torch.tensor([[[[[True,True,False]]]]]),view_mask=torch.tensor([[True]]),input_projection_raw=torch.tensor([[[[[0.,2.,100.]]]]]))
        p=torch.tensor([[[[[1.,3.,999.]]]]],requires_grad=True)
        s,r=components(p,t);on=measurement_discrepancy(p,t,relative_l1=.1);off=measurement_discrepancy(p,t,relative_l1=0.)
        torch.testing.assert_close(r,torch.tensor(1.))
        torch.testing.assert_close(on-off,.1*r,atol=1e-7,rtol=1e-6)
        g=torch.autograd.grad(r,p)[0];torch.testing.assert_close(g,torch.tensor([[[[[.5,.5,0.]]]]]))
        self.assertGreater(float(torch.autograd.grad(s,p)[0].flatten()[0]),0.)

    def test_full_control_equivalence_and_output_gradient_difference(self):
        ds=make_dataset(self.c);b=move_dual_batch(dual_measurement_collate([ds[0]]),'cpu')
        a=torch.full((1,1,64,64,64),.0015,requires_grad=True);other=torch.full_like(a,.0016,requires_grad=True)
        class Cached:
            def __call__(self,x):return a if x is b['A'] else other
        on=objective(Cached(),b,.1);old=old_objective(Cached(),b)
        torch.testing.assert_close(on[0],old[0],rtol=0,atol=0)
        for k in on[1]:torch.testing.assert_close(on[1][k],old[1][k],rtol=0,atol=0)
        split=gradient_split(None,on,b,check_reference=True)
        self.assertLess(split['only_difference']['output_gradient_remainder_l2'],1e-5)
        self.assertTrue(split['equivalence']['loss_bitwise_equal'])
        # autograd branch decomposition uses true scalar->projector->loss chain.
        c=torch.tensor(1e-7,requires_grad=True)
        class Constant:
            def __call__(self,x):return c.expand(1,1,64,64,64)
        result=objective(Constant(),b,.1);split=gradient_split(None,result,b,scalar=c)
        groups=grouped_constant_contributions(result[3],b,c);validate_group_sum(split,groups)
        self.assertGreater(split['branches']['measurement_relative_weighted']['dL_dc_constant'],0.)
        self.assertLess(split['branches']['measurement_smooth']['dL_dc_constant'],0.)

    def test_treatment_training_gradient_excludes_relative(self):
        ds=make_dataset(self.c);b=move_dual_batch(dual_measurement_collate([ds[0]]),'cpu')
        a=torch.full((1,1,64,64,64),.001,requires_grad=True)
        class Constant:
            def __call__(self,x):return a
        result=objective(Constant(),b,0.)
        smooth=result[4]['measurement_smooth'];cons=.05*result[1]['consistency']
        ga=torch.autograd.grad(result[0],a,retain_graph=True)[0];gb=torch.autograd.grad(smooth+cons,a)[0]
        torch.testing.assert_close(ga,gb,atol=1e-7,rtol=1e-5)
        self.assertGreater(float(result[4]['measurement_relative_raw'].detach()),0.)

    def test_exit_requires_both_metrics_and_missing_dependence(self):
        def history(l1=1.,old_l2=1.,new_l2=1.):return [dict(eval=dict(measurement=dict(heldout=dict(relative_l1=l1,relative_l2=old_l2 if i<25 else new_l2)))) for i in range(40)]
        def health(depend=False):return dict(full_fixed_subset=dict(measurement=dict(heldout=dict(relative_l1=1.)),output=dict(mean=.001)),logits=dict(fraction_below_minus12=0.),diagnostic_cases=[dict(measurement_dependence=dict(relative_MA=.002 if depend else 0.)) for i in range(3)])
        state=exit_event(20,history()[:20],health(),.001,{})
        self.assertEqual(state['failure_streak'],0)
        state=exit_event(25,history()[:25],health(),.001,state);self.assertEqual(state['failure_streak'],1)
        state=exit_event(30,history()[:30],health(),.001,state);self.assertEqual(state['reason'],'FAILED_TO_LEARN')
        self.assertFalse(exit_event(30,history(old_l2=1.,new_l2=.8)[:30],health(),.001,{})['failure_condition'])
        self.assertFalse(exit_event(30,history()[:30],health(True),.001,{})['failure_condition'])
        self.assertFalse(exit_event(30,history(l1=.8)[:30],health(),.001,{})['failure_condition'])
        h=health();h['logits']['fraction_below_minus12']=.99;h['full_fixed_subset']['output']['mean']=1e-8
        a=exit_event(5,history()[:5],h,.001,{});self.assertEqual(a['collapse_streak'],0)
        a=exit_event(10,history()[:10],h,.001,a);a=exit_event(15,history()[:15],h,.001,a);self.assertEqual(a['reason'],'NEAR_ZERO_COLLAPSE')

    def test_strict_configuration_and_resume_identity(self):
        validate(self.c)
        for section,key,value in [('loss','observed',0.),('loss','consistency',0.),('intervention','control_relative_l1',.05),('training','seed',2)]:
            c=copy.deepcopy(self.c);c[section][key]=value
            with self.assertRaises(ValueError):validate(c)
        saved=dict(config=self.c,arm='treatment',relative_l1=0.,initial_hash='i',registration_sha256='r',source_manifest_sha256='s',epoch=10,updates=40,scheduler=dict(last_epoch=10),exit_state=dict(failure_streak=1))
        verify_resume(saved,self.c,'treatment','i','r','s');self.assertEqual(saved['exit_state']['failure_streak'],1)
        for key,value in [('relative_l1',.1),('arm','control'),('updates',41),('exit_state',dict(reason='FAILED_TO_LEARN'))]:
            ck=copy.deepcopy(saved);ck[key]=value
            with self.assertRaises(ValueError):verify_resume(ck,self.c,'treatment','i','r','s')


if __name__=='__main__':unittest.main()
