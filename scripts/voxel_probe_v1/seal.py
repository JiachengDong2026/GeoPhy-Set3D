"""Seal final diagnostic source/report hashes after all log writers finish."""
from probe_context import *


def main():
    verify()
    amendment=read(ART/'engineering_amendment_1.json')
    for name,digest in amendment['new_source_sha256'].items():
        assert sha(PROJECT/name)==digest,name
    for record,key in [('post_analysis_registration.json','sources_sha256'),
                       ('review_registration.json','review_source_sha256')]:
        for name,digest in read(ART/record)[key].items():
            assert sha(PROJECT/name)==digest,name
    assert read(ART/'execution_integrity.json')['actual_updates']==1200
    sources={str(p.relative_to(PROJECT)):sha(p) for p in sorted((PROJECT/'scripts/voxel_probe_v1').glob('*.py'))}
    write(ART/'analysis_source_manifest.json',sources)
    report=PROJECT/'docs/reports/voxel_network_probe_v1_zh.md'
    write(ART/'publication.json',dict(report=str(report.relative_to(PROJECT)),report_sha256=sha(report),
          source_manifest_sha256=sha(ART/'analysis_source_manifest.json'),conclusion='Mechanism diagnosis only; final per-arm results in summary.csv',
          actual_updates=1200,formal_training=False,main_merge=False))
    manifest={str(p.relative_to(PROJECT)):sha(p) for p in sorted(ART.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'}
    write(ART/'delivery_manifest.json',manifest)
    for name,digest in manifest.items():
        assert sha(PROJECT/name)==digest,name
    print('Final publication sealed',len(manifest),'evidence files; original scientific sources unchanged')


if __name__=='__main__':
    main()
