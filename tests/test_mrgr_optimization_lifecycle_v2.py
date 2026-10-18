"""CPU transaction tests, including recovery after the final epoch commit."""
import contextlib
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import train_mrgr_optimization_formal_v2 as run
import evaluate_mrgr_optimization_frozen_v2 as evaluation


class Dataset:
    def __init__(self, validation=False):
        self.validation=validation
        self.epoch=0
    def set_epoch(self,epoch): self.epoch=epoch


class Guard:
    def __init__(self,*args): pass
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def summary(self):
        return dict(gt_open_attempts=0,forbidden_npy_attempts=0,supervised_checkpoint_open_attempts=0)


def model_for(config,candidate,device):
    random.seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    torch.manual_seed(config['training']['seed'])
    return torch.nn.Linear(2,1)


def loader_for(ds,config,shuffle=False,generator=None):
    return SimpleNamespace(dataset=ds,generator=generator,num_workers=0,persistent_workers=False)


def fake_epoch(model,loader,config,optimizer=None):
    # Exercise Python/NumPy/global torch/loader RNG and real AdamW state without CUDA.
    x=torch.randn(2)+random.random()+float(np.random.rand())
    if loader.generator is not None:
        x=x+torch.rand(2,generator=loader.generator)
    pred=model(x).squeeze()
    loss=pred.square()+.01
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    value=float(loss.detach())
    return dict(loss={k:value for k in ('total','hold','cross','observed','consistency')},
        measurement=dict(heldout=dict(relative_l1=value,relative_l2=value)),
        sample_count=36 if loader.dataset.validation else 166)


class LifecycleTest(unittest.TestCase):
    def test_resume_committed_epoch_and_final_epoch(self):
        config=dict(training=dict(seed=20261011,device='cpu'),dataset=dict(root='synthetic-test'),
            optimizer=dict(lr=2e-4,weight_decay=1e-4),loss=copy.deepcopy(run.base.selfsup.FROZEN_LOSS))
        initial=run.base.state_hash(model_for(config,'baseline','cpu').state_dict())
        protocol=dict(config=config,candidate='baseline',seed=20261011,initial_model_sha256=initial)
        original_write=run.old.write_metrics
        original_objective=run.base.selfsup.objective
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            patches=[patch.object(run,'verify',return_value=protocol),patch.object(run,'model_for',side_effect=model_for),
                patch.object(run.old,'verify_data'),patch.object(run.old,'plot'),
                patch.object(run.base,'GTReadGuard',Guard),patch.object(run.base,'environment',return_value={}),
                patch.object(run.base.selfsup,'make_dataset',side_effect=lambda c,validation=False:Dataset(validation)),
                patch.object(run.base.selfsup,'make_loader',side_effect=loader_for),
                patch.object(run.base.selfsup,'run_epoch',side_effect=fake_epoch),
                patch.object(torch.cuda,'is_available',return_value=False),
                patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'reset_peak_memory_stats'),
                patch.object(torch.cuda,'max_memory_allocated',return_value=0),
                patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(uuid='cpu-test'))]
            for p in patches:stack.enter_context(p)
            self.addCleanup(setattr,run.base.selfsup,'objective',original_objective)
            results=[]
            for cut in (None,7,300):
                out=Path(temporary)/f'cut{cut}'
                (out/'checkpoints').mkdir(parents=True)
                (out/'protocol.json').write_text(json.dumps(protocol))
                def interrupt_writer(path,values):
                    original_write(path,values)
                    if len(values)==cut:raise RuntimeError('simulated process loss after epoch commit')
                if cut is None:
                    with patch('builtins.print'):
                        run.train(out)
                else:
                    with patch.object(run.old,'write_metrics',side_effect=interrupt_writer),patch('builtins.print'):
                        with self.assertRaisesRegex(RuntimeError,'simulated process loss'):
                            run.train(out)
                    self.assertFalse((out/'summary.json').exists())
                    with patch('builtins.print'):
                        run.train(out,resume=True)
                state=torch.load(out/'checkpoints/last.pt',map_location='cpu',weights_only=False)
                summary=json.loads((out/'summary.json').read_text())
                self.assertEqual((summary['epochs'],summary['updates']),(300,24900))
                self.assertEqual(len(state['metrics']),300)
                results.append((state,summary))
            reference,ref_summary=results[0]
            for actual,summary in results[1:]:
                self.assertEqual(run.base.state_hash(actual['model']),run.base.state_hash(reference['model']))
                self.assertEqual(run.base.state_hash(actual['best_model']),run.base.state_hash(reference['best_model']))
                self.assertEqual(actual['scheduler'],reference['scheduler'])
                self.assertEqual(summary['best_epoch'],ref_summary['best_epoch'])
                self.assertEqual(summary['final_val_H'],ref_summary['final_val_H'])
                self.assertTrue(torch.equal(actual['rng']['torch'],reference['rng']['torch']))
                self.assertTrue(torch.equal(actual['rng']['loader'],reference['rng']['loader']))
                for a,b in zip(actual['metrics'],reference['metrics']):
                    self.assertEqual(a['train'],b['train'])
                    self.assertEqual(a['val'],b['val'])
                    self.assertEqual(a['learning_rate'],b['learning_rate'])

    def test_manifest_rejects_layout_substitution_and_duplicate(self):
        old=json.loads((evaluation.REFERENCE/'registration.json').read_text())
        records=[dict(condition=s['condition'],draw=s['draw'],case_id=c,
            input_ids=s['input_ids'][c],heldout_ids=s['heldout_ids'][c]) for s in old['settings'] for c in old['cases']]
        evaluation.verify_prediction_rows(old,records)
        for change in ('duplicate','layout','holdout'):
            bad=copy.deepcopy(records)
            if change=='duplicate':bad[-1]=bad[0]
            elif change=='layout':bad[0]['input_ids']=list(reversed(bad[0]['input_ids']))
            else:bad[0]['heldout_ids']=bad[0]['input_ids']
            with self.assertRaises(AssertionError):evaluation.verify_prediction_rows(old,bad)

    def test_metric_records_reject_duplicate_stage_and_layout(self):
        old=json.loads((evaluation.REFERENCE/'registration.json').read_text())
        records=[json.loads(x) for x in (evaluation.REFERENCE/'per_case_metrics.jsonl').read_text().splitlines()]
        evaluation.verify_metric_rows(old,records)
        for change in ('duplicate','stage','layout'):
            bad=copy.deepcopy(records)
            if change=='duplicate':bad[-1]=bad[0]
            elif change=='stage':bad[0]['stage']='C_feedback1'
            else:bad[0]['condition']='unregistered-layout'
            with self.assertRaises(AssertionError):evaluation.verify_metric_rows(old,bad)


if __name__=='__main__':unittest.main()
