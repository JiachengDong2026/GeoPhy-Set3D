"""Join real scheduler and worker measurements into one reviewable cost ledger."""
from common import *
import csv
import numpy as np


def main():
    records=[];cache_path=OUT/'task_peak_cache.json';cache=read(cache_path) if cache_path.exists() else {}
    for directory in sorted((OUT/'scheduling').iterdir()):
        if not directory.is_dir():continue
        stored=cache.get(directory.name,dict(offset=0,peaks={}));peaks={int(k):v for k,v in stored['peaks'].items()}
        snapshots=directory/'resources.jsonl'
        if snapshots.exists():
            with snapshots.open() as handle:
                handle.seek(stored['offset'])
                while True:
                    line=handle.readline()
                    if not line.endswith('\n'):break
                    snap=json.loads(line)
                    for process_line in snap['gpu_processes'].splitlines():
                        cols=[x.strip() for x in process_line.split(',')]
                        if len(cols)==4:
                            try:peaks[int(cols[1])]=max(peaks.get(int(cols[1]),0),int(cols[3]))
                            except ValueError:pass
                    stored['offset']=handle.tell()
            cache[directory.name]=dict(offset=stored['offset'],peaks=peaks)
        for p in directory.glob('*.json'):
            r=read(p)
            if 'pid' not in r:continue
            records.append(dict(phase=directory.name,key=r['key'],gpu_uuid=r['gpu_uuid'],pid=r['pid'],started_at=r['started_at'],ended_at=r.get('ended_at'),seconds=r.get('seconds'),exit_code=r.get('exit_code'),log=r['log'],peak_sampled_GPU_process_MiB=peaks.get(r['pid']),sampling_note='15s nvidia-smi samples; exact Torch peak is in per-case/per-epoch records'))
    write(OUT/'task_costs.json',dict(at=now(),records=records))
    write(cache_path,cache)
    if records:
        with (OUT/'task_costs.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return records

if __name__=='__main__':main()
