"""Read-only train-only endpoint audit of normalized teacher MSE and output-bias pressure."""
from feasible_context import *


def main():
    setup();verify();registration=read(ART/'preheat_outlier_registration.json')
    for path,digest in registration['sources'].items():assert sha(PROJECT/path)==digest,path
    checkpoint=ROOT/'unet/last_preheat.pt';assert sha(checkpoint)==registration['checkpoint_sha256']
    model=build('unet');ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['stage']=='preheat' and ck['epoch']==60 and ck['optimizer_steps']==4980
    model.load_state_dict(ck['model']);model.eval();del ck
    teachers=read(ART/'teachers.json')['cases'];ds=make_dataset(config());ds.set_epoch(0)
    guard=GTReadGuard(DATA,ROOT);rows=[]
    with guard,torch.no_grad():
        for r in teachers:
            target=torch.load(r['path'],map_location='cpu',weights_only=False)['concentration'].double()
            batch=dual_measurement_collate([ds[r['index']]])
            a,b=[model(batch[branch]).double() for branch in ['A','B']]
            scale=target.square().mean().clamp_min(1e-20)
            raw=.5*((a-target).square().mean()+(b-target).square().mean())
            # For this standard sigmoid head, dc/db = c*(1-c).
            bias=((a-target)*a*(1-a)).mean()+((b-target)*b*(1-b)).mean()
            row=dict(index=r['index'],case_id=r['case_id'],normalized_MSE=float(raw/scale),raw_MSE=float(raw),
                head_bias_gradient=float(bias/scale),predicted_mean=float(.5*(a.mean()+b.mean())),teacher_mean=float(target.mean()),
                abnormal_after_one_iteration=r['solver_status'].startswith('ABNORMAL') and r['accepted_iterations']==1)
            rows.append(row)
            if len(rows)%20==0:print('PREHEAT_OUTLIER_AUDIT',len(rows),'/166',flush=True)
    groups={}
    for flag,name in [(True,'abnormal4'),(False,'other162')]:
        group=[r for r in rows if r['abnormal_after_one_iteration']==flag]
        groups[name]=dict(cases=len(group),mean_normalized_MSE=float(np.mean([r['normalized_MSE'] for r in group])),
            sum_head_bias_gradient=float(sum(r['head_bias_gradient'] for r in group)),sum_normalized_MSE=float(sum(r['normalized_MSE'] for r in group)))
    fraction=groups['abnormal4']['sum_normalized_MSE']/sum(r['normalized_MSE'] for r in rows)
    write(ART/'preheat_outlier_audit.json',dict(checkpoint_sha256=sha(checkpoint),epoch=60,sampling_epoch=0,
        groups=groups,abnormal4_fraction_of_total_normalized_MSE=fraction,records=rows,access=guard.summary(),
        new_optimizer_updates=0,GT_used=False,not_training_epoch_average=True))
    print('PREHEAT_OUTLIER_FRACTION',fraction,'groups',groups,flush=True)
    verify()


if __name__=='__main__':main()
