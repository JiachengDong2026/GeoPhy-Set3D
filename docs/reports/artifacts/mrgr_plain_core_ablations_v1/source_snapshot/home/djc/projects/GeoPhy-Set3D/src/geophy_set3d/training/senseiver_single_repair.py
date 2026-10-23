"""New 80-epoch boundary; historical exit functions and objectives unchanged."""
from geophy_set3d.training.senseiver_relative_probe import exit_event as historical_exit
import copy
from geophy_set3d.training.senseiver_consistency_probe import validate as validate_source,INTERVENTION as OLD_INTERVENTION

NAME='ray_senseiver_quick_repair_v1'
REPAIR=dict(direction='decoder',kind='global_latent_residual',channels=64,bias=False,initialization='zeros',relative_l1=0.,consistency=0.)


def validate(c):
    base=copy.deepcopy(c)
    if base['experiment']['name']!=NAME or base.pop('intervention')!=REPAIR or base['training']['max_epochs']!=80 or base['loss']['consistency']!=0.:
        raise ValueError('Only registered single decoder repair and 80 epoch budget allowed')
    base['experiment']['name']='ray_senseiver_consistency_probe_v1'
    base['intervention']=OLD_INTERVENTION
    base['training']['max_epochs']=40;base['loss']['consistency']=.05
    validate_source(base)
    return c


def exit_event(epoch,history,health,c_init,previous):
    event=historical_exit(epoch,history,health,c_init,previous)
    # Historical function's scientific conditions/counters are reused exactly;
    # only this newly registered experiment has an 80-epoch ceiling.
    if event['reason']=='BUDGET_COMPLETE':event['reason']=None
    if not event['reason'] and epoch==80:event['reason']='BUDGET_COMPLETE'
    return event


def verify_resume(saved,c,arm,initial,registration_hash,source_hash):
    if saved['config']!=c or saved['arm']!=arm or saved['initial_hash']!=initial or saved['registration_sha256']!=registration_hash or saved['source_manifest_sha256']!=source_hash:
        raise ValueError('Own arm/config/initial/registration identity mismatch')
    if saved['relative_l1']!=0. or saved['consistency_weight']!=0.:
        raise ValueError('Loss identity mismatch')
    if saved['exit_state'].get('reason'):raise ValueError('Scientifically stopped arm cannot resume')
    if saved['updates']!=4*saved['epoch'] or saved['scheduler']['last_epoch']!=saved['epoch']:
        raise ValueError('Incomplete update/scheduler state')
