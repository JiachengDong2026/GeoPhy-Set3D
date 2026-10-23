"""Runtime open audit: only Pool A train/val projection NPY reads are permitted."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys


class GTReadGuard:
    def __init__(self, dataset_root, checkpoint_root):
        self.root=Path(dataset_root).absolute()
        self.checkpoint_root=Path(checkpoint_root).absolute()
        rows=[json.loads(l) for l in (self.root/'source_index.jsonl').read_text().splitlines() if l.strip()]
        self.gt64={os.path.abspath(self.root/r['gt_volume_64_path']) for r in rows}
        self.gt128={os.path.abspath(r['source_target_path']) for r in rows}
        allowed=[os.path.abspath(self.root/r['train_projection_path']) for r in rows if r['split'] in ('train','val')]
        self.allowed={p:i for i,p in enumerate(allowed)}
        self.touched=mp.Array('b',len(allowed))
        self.gt_attempts=mp.Value('i',0); self.other_npy_attempts=mp.Value('i',0)
        self.projection_opens=mp.Value('i',0); self.supervised_checkpoint_attempts=mp.Value('i',0)
        self.active=False

    def _audit(self,event,args):
        if not self.active or event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=os.path.abspath(os.fsdecode(args[0]))
        if path in self.gt64 or path in self.gt128:
            with self.gt_attempts.get_lock(): self.gt_attempts.value+=1
            raise PermissionError('Volumetric GT access is forbidden in the measurement-only process')
        mode=args[1]; flags=args[2]
        reading=('r' in mode or '+' in mode) if isinstance(mode,str) else not bool(flags & os.O_WRONLY)
        if not reading:
            return
        if path.endswith('.npy'):
            if path not in self.allowed:
                with self.other_npy_attempts.get_lock(): self.other_npy_attempts.value+=1
                raise PermissionError('Only Pool A train/val projection NPY is readable during selfsup training')
            with self.projection_opens.get_lock(): self.projection_opens.value+=1
            self.touched[self.allowed[path]]=1
        if path.endswith('ood_geometry.npz'):
            raise PermissionError('Pool B geometry is forbidden in training/selection')
        if path.endswith(('.pt','.pth','.ckpt')) and not Path(path).is_relative_to(self.checkpoint_root):
            with self.supervised_checkpoint_attempts.get_lock(): self.supervised_checkpoint_attempts.value+=1
            raise PermissionError('Selfsup cannot read supervised/foreign checkpoints')

    def __enter__(self):
        self.active=True
        # The hook and multiprocessing counters are inherited by forked loader workers.
        sys.addaudithook(self._audit)
        return self

    def __exit__(self,*args):
        self.active=False

    def summary(self):
        return dict(gt_file_opened=self.gt_attempts.value>0,gt_open_attempts=self.gt_attempts.value,
                    forbidden_npy_attempts=self.other_npy_attempts.value,
                    supervised_checkpoint_open_attempts=self.supervised_checkpoint_attempts.value,
                    projection_open_count=self.projection_opens.value,
                    unique_pool_a_projection_files=sum(self.touched),
                    policy='Only Pool A train/val projection NPY; GT64/GT128 and foreign checkpoints denied',
                    audit_hook_inherited_by_workers=True)
