"""Seal all completed registered diagnostics; no training or new search."""
from probe_context import *
import seal_all
import gzip
import shutil


def main():
    seal_all.main()
    for name,key in [('extra_analysis_registration.json','source_sha256'),('final_analysis_registration.json','source_sha256')]:
        for source,digest in read(ART/name)[key].items():assert sha(PROJECT/source)==digest,source
    r=read(ART/'case0_original_refinement.json')
    review=read(ART/'refinement_independent_review.json')
    assert review['passed'] and review['actual_updates']==r['actual_updates']==100
    source=ROOT/'case0_teacher_probe/last.pt'
    assert sha(source)==r['source_sha256']==read(ART/'refinement_registration.json')['initial_checkpoint_sha256']
    directory=ROOT/'case0_original_refinement'
    checkpoint=directory/'last.pt'
    assert sha(checkpoint)==r['checkpoint_sha256']
    with (directory/'updates.jsonl').open('rb') as source_stream,gzip.open(ART/'case0_original_refinement_updates.jsonl.gz','wb') as target:
        shutil.copyfileobj(source_stream,target)
    write(ART/'refinement_checkpoint_manifest.json',dict(path=str(checkpoint),sha256=sha(checkpoint),actual_updates=100,complete_fields=[m['arrays'] for m in r['metrics'] if 'arrays' in m]))
    names=['independent_review.json','bounded_independent_review.json','network_independent_review.json','teacher_independent_review.json','refinement_independent_review.json']
    assert all(read(ART/n)['passed'] for n in names)
    assert sum(read(ART/n)['full_fields'] for n in names)==44
    assert sum(read(ART/n)['physical_integrals'] for n in names)==176
    combined=read(ART/'combined_integrity.json')
    combined.update(refinement_Adam_updates=100,total_Adam_updates=1500,independent_full_fields=44,physical_integrals=176,
        final_targeted_check_complete=True,automatic_further_search=False,
        conclusion='Original objective has effective selected-case field fits; same U-Net learns a self-generated field, then original-loss refinement succeeds locally; cross-case classical baselines not declared repaired')
    write(ART/'combined_integrity.json',combined)
    publication=read(ART/'publication.json')
    publication.update(original_Adam_updates=1200,teacher_Adam_updates=200,refinement_Adam_updates=100,actual_updates=1500,
        independent_full_fields=44,physical_integrals=176,conclusion=combined['conclusion'])
    write(ART/'publication.json',publication)
    manifest={str(p.relative_to(PROJECT)):sha(p) for p in sorted(ART.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'}
    write(ART/'delivery_manifest.json',manifest)
    for name,digest in manifest.items():assert sha(PROJECT/name)==digest,name
    assert sha(PROJECT/publication['report'])==publication['report_sha256']
    print('FINAL integrity: 1200+200+100 AdamW updates; 44 fields / 176 physical integrations; protected history unchanged')


if __name__=='__main__':main()
