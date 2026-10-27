"""Passively preserve upcoming29/30 checkpoint identities; never pause training."""
from feasible_context import *


def main():
    verify();deadline=time.monotonic()+8*3600;seen=None;saved={}
    path=ROOT/'pilot_gnot/last.pt'
    while time.monotonic()<deadline:
        if path.exists() and path.stat().st_ino!=seen:
            with path.open('rb') as stream:
                inode=os.fstat(stream.fileno()).st_ino
                ck=torch.load(stream,map_location='cpu',weights_only=False)
            seen=inode;epoch=ck['epoch'];stage=ck['stage'];updates=ck['optimizer_steps'];del ck
            if stage=='physical' and epoch in [29,30] and str(epoch) not in saved:
                assert updates==epoch*83
                destination=ROOT/f'pilot_gnot/gate_checkpoint_epoch{epoch}.pt';os.link(path,destination)
                assert destination.stat().st_ino==inode,'Checkpoint changed during passive capture; preserve evidence'
                saved[str(epoch)]=dict(path=str(destination),sha256=sha(destination),updates=updates)
                write(ART/'gnot_gate_checkpoints.json',dict(state='monitoring',saved=saved,new_updates=0,training_interrupted=False))
                print('PRESERVED_GNOT_GATE',epoch,updates,flush=True)
        task=read(ROOT/'scheduling/pilot_gnot_resume.json')
        if task['state'] in ['complete','failed','interrupted']:
            write(ART/'gnot_gate_checkpoints.json',dict(state='complete' if task['state']=='complete' else 'engine_failed',
                saved=saved,new_updates=0,training_interrupted=False,missing_epochs=[e for e in [29,30] if str(e) not in saved]));return
        time.sleep(1)
    raise TimeoutError('Passive checkpoint observation wait limit')


if __name__=='__main__':main()
