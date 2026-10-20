"""Reconstruct a registered measurement-only sample with the frozen main model."""
from runtime import *
import argparse
import numpy as np
import torch
from evaluate import TestGuard
from uniform_models import model_for
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case-index',type=int,default=0)
    parser.add_argument('--layout-index',type=int,default=0)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    o=parser.parse_args();backend()
    verified(BASE/'final_lock.json')
    method=next(m for m in verified(BASE/'reuse_methods.json') if m['key']=='uniform_fusion_seed20261011')
    assert method['checkpoint_sha256']==verified(BASE/'design.json')['main_inference_checkpoint']
    setting=verified(BASE/'layouts.json')['test'][o.layout_index]
    output=Path(o.output).resolve()
    assert output.is_relative_to(BASE),'Keep reproduced arrays in experiment output root'
    with TestGuard(splits=['test'],checkpoints=[method['checkpoint']]) as guard:
        assert sha(method['checkpoint'])==method['checkpoint_sha256']
        ck=torch.load(method['checkpoint'],map_location='cpu',weights_only=False)
        model=model_for(ck['config'],'uniform',o.device).eval();model.load_state_dict(ck['model'],strict=True)
        batch=move_batch(projection_pool_collate([dataset('test',setting)[o.case_index]]),o.device)
        inputs={k:v for k,v in batch.items() if not k.startswith('hold_')}
        with torch.no_grad():stages=model(inputs,return_stages=True)
        output.parent.mkdir(parents=True,exist_ok=True)
        assert not output.exists(),'Preserve existing predictions'
        np.savez_compressed(output,**{k:stages[k][0,0].cpu().numpy() for k in ('C0','C1')})
        write(output.with_suffix('.audit.json'),dict(model_id='MRGR-UniformFusion',checkpoint_sha256=sha(method['checkpoint']),
            output_sha256=sha(output),case_index=o.case_index,layout_index=o.layout_index,audit=guard.summary()))
    print(output)

if __name__=='__main__':main()
