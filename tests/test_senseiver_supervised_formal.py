import json,tempfile,unittest
from pathlib import Path
import torch
from geophy_set3d.training.senseiver_supervised_formal import FormalGT64Guard,validate_last,batch
from geophy_set3d.training.senseiver_supervised_probe import normalized_mse,fresh
from geophy_set3d.training.utils import rng_state,restore_rng

class FormalTests(unittest.TestCase):
    def test_original_model_and_supervised_scale(self):
        m=fresh(dict(bias_logit=-6.))
        self.assertEqual(sum(p.numel() for p in m.parameters()),254913)
        g=torch.tensor([1.,2.]).reshape(2,1,1,1,1)
        self.assertEqual(float(normalized_mse(torch.zeros_like(g),g)),1.)
        p=g.clone().requires_grad_();normalized_mse(p,g).backward();self.assertTrue(torch.equal(p.grad,torch.zeros_like(g)))
    def test_access_white_list_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';root.mkdir();own=Path(tmp)/'own';own.mkdir()
            rows=[dict(split=s,train_projection_path=f'{s}/p.npy',gt_volume_64_path=f'{s}/g.npy') for s in ['train','val','test']]
            (root/'source_index.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            guard=FormalGT64Guard(root,own);guard.active=True
            for p in ['train/g.npy','val/g.npy','train/p.npy']:guard._audit('open',(root/p,'r',0))
            for p in ['test/g.npy','GT128.npy','geometry/ood_geometry.npz']:
                with self.assertRaises(PermissionError):guard._audit('open',(root/p,'r',0))
            with self.assertRaises(PermissionError):guard._audit('open',(root/'train/g.npy','r+',2))
            with self.assertRaises(PermissionError):guard._audit('open',(Path(tmp)/'other.pt','rb',0))
            guard._audit('open',(own/'last.pt','rb',0))
    def test_resume_full_state_and_scientific_stop(self):
        c={'test':True};s=dict(config=c,registration_sha256='x',exit_state={},optimizer={},scheduler=dict(last_epoch=2),rng={},epoch=2,position=17,order=[],train_rows=[],history=[],updates=183,elapsed_seconds=3,best=.5,best_epoch=1)
        validate_last(s,c,'x');s['exit_state']['reason']='VAL_EARLY_STOP'
        with self.assertRaises(ValueError):validate_last(s,c,'x')
        s['exit_state']={};s['scheduler']['last_epoch']=3
        with self.assertRaises(ValueError):validate_last(s,c,'x')
    def test_cosine_epoch_and_rng_resume(self):
        p=torch.nn.Parameter(torch.tensor(1.));o=torch.optim.AdamW([p],lr=2e-4);sc=torch.optim.lr_scheduler.CosineAnnealingLR(o,300,eta_min=2e-5)
        for _ in range(3):o.step();sc.step()
        q=torch.nn.Parameter(torch.tensor(1.));o2=torch.optim.AdamW([q],lr=2e-4);sc2=torch.optim.lr_scheduler.CosineAnnealingLR(o2,300,eta_min=2e-5);o2.load_state_dict(o.state_dict());sc2.load_state_dict(sc.state_dict())
        o.step();sc.step();o2.step();sc2.step();self.assertEqual(sc.get_last_lr(),sc2.get_last_lr())
        gen=torch.Generator().manual_seed(13);s=rng_state(gen);a=torch.randperm(166,generator=gen);restore_rng(s,gen);self.assertTrue(torch.equal(a,torch.randperm(166,generator=gen)))
if __name__=='__main__':unittest.main()
