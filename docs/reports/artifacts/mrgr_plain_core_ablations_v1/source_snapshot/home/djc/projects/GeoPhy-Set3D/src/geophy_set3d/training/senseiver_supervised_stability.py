"""Policies unique to the newly authorized supervised stability/full baseline."""


def joint_pass(evaluation, template_nmse):
    m=evaluation['mean'];dep=evaluation.get('interventions',[])
    return (m['nmse']<=.2 and m['relative_l1']<=.4 and m['nmse']<=.7*template_nmse
            and sum(r['relative_MA']>=.001 for r in dep)>=2
            and len(dep)==3 and all(r['wrong_minus_normal_nmse']>0 for r in dep))


def stability_exit(updates,losses,e,template,previous,elapsed):
    good=joint_pass(e,template);passed=previous.get('pass_streak',0)+1 if good else 0
    recent=older=improvement=None
    if len(losses)>=400:
        recent=sum(losses[-200:])/200;older=sum(losses[-400:-200])/200
        improvement=(older-recent)/max(older,1e-12)
    plateau=updates>=1000 and improvement is not None and improvement<.02 and not good
    bad=previous.get('plateau_streak',0)+1 if plateau else 0
    why='STABLE_PASSED' if passed>=3 else 'PLATEAU' if bad>=2 else 'MAX_UPDATES' if updates>=4000 else 'WALLTIME' if elapsed>=1200 else None
    learning=e['mean']['nmse']<.98 and sum(r['relative_MA']>=.001 for r in e.get('interventions',[]))>=2
    status='PASSED' if why=='STABLE_PASSED' else ('FAILED_TO_LEARN' if why=='PLATEAU' and not learning and e['mean']['nmse']>=.98 else 'UNRESOLVED_STABILITY') if why else None
    return dict(updates=updates,pass_condition=good,pass_streak=passed,plateau_condition=plateau,
        plateau_streak=bad,recent200=recent,previous200=older,relative_improvement=improvement,reason=why,status=status)


def formal_exit(epoch,rows,health,previous):
    """Raw-min selector lives outside this significant-improvement state machine."""
    value=rows[-1]['val']['mean']['nmse']
    anchor=previous.get('significant_anchor')
    improved=anchor is None or value<=anchor*(1-.005)
    anchor=value if improved else anchor
    bad=0 if improved else previous.get('nonsignificant_epochs',0)+1
    nolearn=False
    if epoch>=30:
        checks=[]
        for key in ['train','val']:
            a=[r[key]['mean']['nmse'] for r in rows]
            windows=[sum(a[-30:-20])/10,sum(a[-20:-10])/10,sum(a[-10:])/10]
            checks.append(windows[-1]>=.98 and all((x-y)/max(x,1e-12)<.02 for x,y in zip(windows,windows[1:])))
        nolearn=all(checks) and health is not None and sum(r['relative_MA']>=.001 for r in health['interventions'])<2
    stalled=previous.get('no_dependency_stall_streak',0)
    if health is not None and health.get('epoch')==epoch:
        no_dep=sum(r['relative_MA']>=.001 for r in health['interventions'])<2
        prior=previous.get('health_val_nmse')
        stalled=stalled+1 if prior is not None and no_dep and (prior-value)/max(prior,1e-12)<.02 else 0
    why=('FAILED_TO_LEARN' if nolearn else 'STALLED_WITHOUT_DEPENDENCE' if epoch>=40 and stalled>=2 else
         'VAL_EARLY_STOP' if epoch>=50 and bad>=30 else 'MAX_EPOCHS' if epoch>=300 else None)
    return dict(epoch=epoch,significant_anchor=anchor,nonsignificant_epochs=bad,
        significant_improved=improved,no_learning=nolearn,no_dependency_stall_streak=stalled,
        health_val_nmse=value if health is not None and health.get('epoch')==epoch else previous.get('health_val_nmse'),reason=why)
