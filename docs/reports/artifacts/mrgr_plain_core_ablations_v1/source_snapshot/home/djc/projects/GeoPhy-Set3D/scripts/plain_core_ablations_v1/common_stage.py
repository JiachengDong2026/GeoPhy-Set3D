"""Six registered one-factor ablations; historical numerical sources are immutable."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT/'scripts/oari_targeted_v1'))
import stage as prior
from stage import *
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
import copy
import argparse
import fcntl
import time
import traceback

PRIOR = prior.BASE
BASE = PRIOR.parent/'mrgr_plain_core_ablations_v1'
ART = PROJECT/'docs/reports/artifacts'/BASE.name
REPORT = PROJECT/'docs/reports/mrgr_plain_core_ablations_v1_zh.md'
TRACKER = PROJECT/'docs/reports/mrgr_plain_final_experiment_tracking_v1_zh.md'
GEOMETRY = PLAIN+'_no_geometry_lifting'
SINGLE = PLAIN+'_single_scale_mrgr'
NO_PRF = PLAIN+'_no_prf'
TASKS = tuple((arm,seed) for arm in (GEOMETRY,SINGLE,NO_PRF) for seed in (20261011,20261019))
SEED = 20261026
EXPECTED = {20261011:'0ccea9b24a77e9d5fac025c626a98b167c3ab57fa202688e3e4acc75261bc18c',
            20261019:'fe19095926c6fad4194cfd6addcf96fed523b94631e754d021600f1b7313525d'}
legacy_lifting = load_unique('plain_core_formal_lifting',PROJECT/'scripts/no_spatial_lifting_v1/runtime.py')
from models import SingleScaleContext
from geophy_set3d.models.geolift_plain_set_mrgr import PlainSetStage0, GeoLiftPlainSetMRGR


class PlainNoPRF(GeoLiftPlainSetMRGR):
    def forward(self,batch,return_stages=False):
        c=PlainSetStage0.forward(self,batch)
        return dict(C0=c,C1=c) if return_stages else c


def model_for(config,arm,device):
    assert arm in (PLAIN,GEOMETRY,SINGLE,NO_PRF),arm
    m=prior.model_for(config,PLAIN,'cpu')
    if arm==GEOMETRY:m.lifting.__class__=legacy_lifting.PlanarExtrusionLifting
    elif arm==SINGLE:m.decoder.context.__class__=SingleScaleContext
    elif arm==NO_PRF:
        del m.residual_encoder;del m.residual_adapter;m.__class__=PlainNoPRF
    return m.to(device)


def training_sources():
    d=prior.training_sources()
    d.update({str(p):sha(p) for p in HERE.glob('*.py')})
    d[str(PROJECT/'scripts/no_spatial_lifting_v1/runtime.py')]=sha(PROJECT/'scripts/no_spatial_lifting_v1/runtime.py')
    d[str(PROJECT/'docs/mrgr_plain_core_ablations_v1_protocol_zh.md')]=sha(PROJECT/'docs/mrgr_plain_core_ablations_v1_protocol_zh.md')
    return d


class TrainGuard(AccessGuard):
    def __init__(self):super().__init__(splits=['train','val'],output=BASE)
    def audit(self,event,args):
        if self.active and event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            # Training may not consult any historical test result or prediction.
            if any(p.is_relative_to(root) for root in (PRIOR/'evaluation',BASE/'evaluation',HISTORY/'evaluation/test')) or p.name in ('metrics.csv','statistics.json','summary.csv'):
                self.denied.append(dict(path=str(p),reason='test evidence forbidden during training'))
                raise PermissionError('test evidence forbidden during training: '+str(p))
        super().audit(event,args)


def scheduler():
    s=prior.scheduler();s.BASE=BASE;s.GPU_LOCKS=BASE.parent/'gpu_locks';s.progress=lambda:None
    return s


def command(script,*args):return [sys.executable,str(HERE/script),*map(str,args)]


def validate_training(target,load_checkpoint=True):
    target=Path(target);p=verified(target/'protocol.json');s=read(target/'summary.json')
    assert s['completed'] and s['epochs']==300 and s['updates']==24900
    assert all(s['gt_audit'][k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    assert not read(target/'boundary_audit.json')['denied']
    assert read(target/'environment.json')['initial_model_sha256']==p['initial_model_sha256']
    log=rows(target/'metrics.jsonl');assert len(log)==300 and [r['epoch'] for r in log]==list(range(1,301))
    chosen=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])
    assert s['best_epoch']==chosen['epoch'] and s['best_val_H']==chosen['val']['measurement']['heldout']['relative_l1']
    for name in ('best','last'):assert sha(target/f'checkpoints/{name}.pt')==s[f'{name}_checkpoint_sha256']
    if load_checkpoint:
        ck=torch.load(target/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==300 and ck['updates']==24900 and ck['metrics']==log and ck['config']==p['config']
        assert ck['protocol_sha256']==sha(target/'protocol.json') and not ck['volumetric_gt_used']
    return dict(key=target.name,arm=p['arm'],seed=p['seed'],config=p['config'],parameters=p['parameters'],
        checkpoint=str(target/'checkpoints/best.pt'),checkpoint_sha256=s['best_checkpoint_sha256'],best_epoch=s['best_epoch'],run=str(target))


def methods():return [validate_training(BASE/'training'/f'{a}_seed{s}') for a,s in TASKS]


def update_tracker():
    lines=['# 普通集合交互最终模型实验追踪表','',f'更新时间：{now()}。本轮仅登记六个消融，结果为既有35案例上的探索性分析。','',
        '| 实验 | 种子 | 状态 | epoch / 更新 | 参数 | best SHA256 |','|---|---:|---|---|---:|---|']
    for m in verified(BASE/'references.json'):
        lines.append(f"| {m['key']} | {m['seed']} | 已核验，复用 | 300 / 24900 | {m['parameters']} | `{m['checkpoint_sha256']}` |")
    for a,s in TASKS:
        root=BASE/'training'/f'{a}_seed{s}';p=verified(root/'protocol.json')
        st=read(root/'status.json') if (root/'status.json').exists() else {}
        lines.append(f"| {root.name} | {s} | {st.get('state','已登记，等待资源')} | {st.get('epoch',0)} / {st.get('updates',0)} | {p['parameters']} | {st.get('best_checkpoint_sha256','待完成')} |")
    lines+=['',f'完整状态：`{BASE}/status.json`。日志、GPU快照和异常记录：`{BASE}/scheduling/`。',
        '', '完整模型不重训。OARI历史消融只复用定义，不能代替普通交互消融。所有六项300轮及全局预测封存完成后才允许本轮GT64评分，GT128禁止。']
    TRACKER.write_text('\n'.join(lines)+'\n')

# Historical imports prepend their own directories. Restore this stage first so
# CLI helpers cannot accidentally import another stage's same-named script.
sys.path.insert(0,str(HERE))
