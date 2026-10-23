"""Read-only invariant anchor for the current plain final model, before baseline work."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plain_final_v1'))
from context import *
ROOT=BASE.parent/'deep_baselines_v1'

def main():
    backend();ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'main_anchor.json').exists()
    prior=verified(BASE/'model_manifest.json');m=prior['models'][0]
    assert m['key']==MAIN and m['checkpoint_sha256']=='0ccea9b24a77e9d5fac025c626a98b167c3ab57fa202688e3e4acc75261bc18c'
    for item in prior['models']:assert sha(item['checkpoint'])==item['checkpoint_sha256']
    verify_hashes(verified(BASE/'protocol.json')['sources'])
    p=verified(Path(m['run'])/'protocol.json');verify_hashes(p['sources'])
    ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
    assert ck['epoch']==298 and ck['config']==m['config'] and ck['protocol_sha256']==sha(Path(m['run'])/'protocol.json') and not ck['volumetric_gt_used']
    model=model_for(m['config'],PLAIN,'cpu').eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==342209
    setting=verified(HISTORY/'layouts.json')['val'][0]
    with AccessGuard(splits=['val'],output=ROOT) as guard,torch.inference_mode():
        ds=dataset('val',setting);b=batch_for(ds,0);out=model(b,return_stages=True)
        path=ROOT/'main_anchor_prediction.npz';assert not path.exists();np.savez_compressed(path,**{k:out[k][0,0].numpy() for k in ('C0','C1')})
        check_fields(path,['C0','C1'])
    freeze(ROOT/'main_anchor.json',dict(at=now(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        source_worktree_clean=not subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT,text=True).strip(),
        note='only new freeze script untracked at audit; no pre-existing user edits',
        model=m,reference=prior['models'][1],training_sources=p['sources'],
        protocol_hashes={str(q):sha(q) for q in [Path(m['run'])/'protocol.json',BASE/'protocol.json',HISTORY/'final_lock.json',HISTORY/'layouts.json',HISTORY/'registration.json',DATA/'source_index.jsonl',DATA/'train_statistics.json']},
        benchmark=dict(device='cpu',split='val',case_id=ds.records[0]['case_id'],setting=setting,input_sha256=input_hash(b),prediction=str(path),prediction_sha256=sha(path),
            output_tensor_hashes={k:tensor_hash(out[k]) for k in ('C0','C1')},model_state_sha256=canonical_worker.probe.state_hash(model.state_dict())),
        resource_snapshot=resources(),audit=guard.summary(),no_training=True,GT64_read=False,GT128_read=False))
    print('MAIN_FROZEN',m['key'],m['checkpoint_sha256'])

if __name__=='__main__':main()
