"""Verify exact diagnostic paths despite a cached historical train module."""
from pathlib import Path
import sys
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/deep_baselines_identifiability_v1'))
from engineering_dispatch_v3 import load_training,HERE

def test_exact_dispatch_ignores_generic_cached_train_module(monkeypatch):
    foreign=types.ModuleType('train');foreign.marker='historical'
    monkeypatch.setitem(sys.modules,'train',foreign)
    worker=load_training()
    assert Path(worker.__file__)==HERE/'train.py' and callable(worker.run)
    assert Path(worker.evaluate.__code__.co_filename)==HERE/'evaluate.py'
    assert sys.modules['train'] is foreign
