"""Verify separately registered extra probes and seal the combined delivery."""
from probe_context import *
import seal as base_seal
import gzip
import shutil


def main():
    base_seal.main()
    for name in ['bounded_solver_registration.json','network_solver_registration.json','teacher_probe_registration.json']:
        r=read(ART/name)
        assert sha(PROJECT/r['source'])==r['source_sha256']
    teacher=read(ART/'case0_teacher_probe.json')
    directory=ROOT/'case0_teacher_probe'
    journal=[json.loads(s) for s in (directory/'updates.jsonl').read_text().splitlines()]
    assert len(journal)==teacher['actual_updates']==200
    assert [v['update'] for v in journal]==list(range(1,201))
    ck=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
    assert ck['steps']==200 and ck['scheduler']['last_epoch']==12
    assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={200}
    assert ck['rng'] and all(torch.isfinite(v).all() for v in ck['model'].values())
    del ck
    assert sha(directory/'last.pt')==teacher['checkpoint_sha256']
    total_field_count=sum(read(ART/name)['full_fields'] for name in ['independent_review.json','bounded_independent_review.json','network_independent_review.json','teacher_independent_review.json'])
    assert total_field_count==40
    accepted_voxel=0;voxel_evaluations=0
    checkpoints=[]
    for index in [0,82]:
        result=read(ART/f'case{index}_bounded_solver.json')
        accepted_voxel+=result['accepted_iterations'];voxel_evaluations+=result['function_evaluations']
        p=ROOT/f'case{index}_bounded_solver/final_field.pt'
        assert sha(p)==result['final_checkpoint_sha256']
        checkpoints.append(dict(path=str(p),sha256=sha(p)))
    network=read(ART/'case0_unet_lbfgs.json')
    assert network['accepted_iterations']<=20 and network['closure_evaluations']<=60
    p=ROOT/'case0_unet_lbfgs/final.pt'
    assert sha(p)==network['checkpoint_sha256']
    checkpoints.append(dict(path=str(p),sha256=sha(p)))
    checkpoints.append(dict(path=str(directory/'last.pt'),sha256=sha(directory/'last.pt')))
    for record in [teacher,network]+[read(ART/f'case{i}_bounded_solver.json') for i in [0,82]]:
        assert not any(record['access'][k] for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
    for directory in [ROOT/'case0_teacher_probe',ROOT/'case0_unet_lbfgs',ROOT/'case0_bounded_solver',ROOT/'case82_bounded_solver']:
        for p in sorted(directory.glob('*.jsonl')):
            with p.open('rb') as source,gzip.open(ART/(directory.name+'_'+p.name+'.gz'),'wb') as target:
                shutil.copyfileobj(source,target)
    write(ART/'extra_checkpoint_manifest.json',checkpoints)
    write(ART/'combined_integrity.json',dict(original_Adam_updates=1200,teacher_Adam_updates=200,
        bounded_voxel_accepted_iterations=accepted_voxel,bounded_voxel_function_evaluations=voxel_evaluations,
        network_LBFGS_accepted_iterations=network['accepted_iterations'],network_LBFGS_closures=network['closure_evaluations'],
        independent_full_fields=total_field_count,physical_integrals=total_field_count*4,
        old_sources_unchanged=147,old_inputs_unchanged=207,CPU_only=True,maximum_task_threads=4,
        GT64_reads=0,GT128_reads=0,PoolB_reads=0,final_test_reads=0,formal_training=False,main_merge=False,
        conclusion='Original physical objective has effective case-specific field fits; conditional neural adaptation not declared repaired'))
    manifest={str(p.relative_to(PROJECT)):sha(p) for p in sorted(ART.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'}
    write(ART/'delivery_manifest.json',manifest)
    for name,digest in manifest.items():assert sha(PROJECT/name)==digest,name
    print('Combined integrity verified; Adam 1200+200; bounded optimizer counts separate; 40 fields / 160 integrals')


if __name__=='__main__':main()
