"""Read-only exact CPU resume and stop-policy checks before continuation."""
from continuation_context import *
from evaluation import evaluate
from train import stop_reason


def main():
    setup();p=verify_all()
    ck=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert ck['steps']==100 and ck['scheduler']['last_epoch']==6
    assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={100}
    model=fresh();model.load_state_dict(ck['model'])
    payload=torch.load(read(OLD_ART/'prepared.json')['cases'][0]['path'],map_location='cpu',weights_only=False)
    directory=ROOT/'preflight';directory.mkdir(exist_ok=True)
    row=evaluate(model,payload['batch'],payload['extra'],directory,0,False)
    r=read(OLD_ART/'case0_original_refinement.json')['final']
    np.testing.assert_allclose([row['loss'],row['extra_E_H']],[r['loss'],r['extra_E_H']],rtol=1e-6,atol=1e-7)
    import copy
    stable=[copy.deepcopy(row) for _ in range(9)]
    for i,v in enumerate(stable):v['step']=25*i
    assert stop_reason(stable)=='loss_and_H_plateau_two_100step_windows'
    assert stop_reason([row]) is None
    bad=copy.deepcopy(stable[:4])
    for v in bad[1:]:v['extra_E_H']=1.3*row['extra_E_H']
    assert stop_reason(bad)=='unseen_E_persistently_worse_25percent'
    bad=copy.deepcopy(row);bad['fields'][0]['mean']=0
    assert stop_reason([bad])=='output_collapse'
    write(ART/'preflight.json',dict(passed=True,optimizer_step=100,scheduler_last_epoch=6,CPU_loss=row['loss'],E=row['extra_E_H'],new_optimizer_updates=0,source_sha256=sha(SOURCE)))
    print('Read-only CPU resume and stopping policy checks passed')


if __name__=='__main__':main()
