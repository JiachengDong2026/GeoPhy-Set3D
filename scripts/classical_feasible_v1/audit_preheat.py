"""Read-only train-case scale audit of an immutable live checkpoint snapshot."""
from feasible_context import *
import shutil


def main():
    setup();verify();folder=ROOT/'preheat_scale_audit';folder.mkdir(exist_ok=False)
    snapshot=folder/'snapshot.pt';shutil.copyfile(ROOT/'unet/last_preheat.pt',snapshot)
    ck=torch.load(snapshot,map_location='cpu',weights_only=False);epoch=ck['epoch'];model=build('unet');model.load_state_dict(ck['model']);model.eval();del ck
    ds=make_dataset(config());teachers=read(ART/'teachers.json')['cases'];records=[]
    guard=GTReadGuard(DATA,ROOT)
    with guard,torch.no_grad():
        for index in [0,1,2,7,10,77,82,126,140,165]:
            batch=dual_measurement_collate([ds[index]]);target=torch.load(teachers[index]['path'],map_location='cpu',weights_only=False)['concentration']
            values=[];energy=target.double().square().mean().item()
            for branch in ['A','B']:
                pred=model(batch[branch]);values.append(dict(MSE=(pred-target).double().square().mean().item(),mean=pred.double().mean().item(),max=pred.max().item()))
            records.append(dict(index=index,teacher_energy=energy,normalized_MSE=float(np.mean([v['MSE'] for v in values]))/energy,
                mean=float(np.mean([v['mean'] for v in values])),max=float(np.mean([v['max'] for v in values])),teacher_mean=target.mean().item(),teacher_max=target.max().item()))
    write(ART/'preheat_scale_audit.json',dict(epoch=epoch,snapshot=str(snapshot),snapshot_sha256=sha(snapshot),records=records,
        new_optimizer_updates=0,GT_used=False,access=guard.summary()))
    for r in records:print(r,flush=True)


if __name__=='__main__':main()
