"""Seal the completed audit delivery, without accessing measurements or weights."""
from audit import *

def main():
    review=read(ART/'independent_review.json')
    assert review['all_passed'] and review['gates']['outcome']=='NO_GO_EXISTING_SAME_PROTOCOL_FAILED'
    files=[*Path(__file__).parent.glob('*.py'),PROJECT/'docs/senseiver_selfsup_decision_v1_protocol_zh.md',
        PROJECT/'docs/reports/senseiver_selfsup_decision_v1_zh.md',*ART.iterdir()]
    target=ART/'delivery_manifest.json'
    manifest={str(p.relative_to(PROJECT)):sha(p) for p in files if p.is_file() and p!=target}
    history=read(ART/'history_audit.json')
    assert all(sha(p)==h for p,h in history['evidence_sha256'].items())
    assert all(sha(PROJECT/p)==h for p,h in read(ROOT/'protected_tracked_files.json').items())
    git_evidence=subprocess.check_output(['git','log','--all','--format=%H %as %s','--',
        'docs/ray_senseiver_selfsup_v1_protocol_zh.md','docs/reports/ray_senseiver_learning_diagnostic_zh.md'],cwd=PROJECT,text=True)
    write(target,dict(status='SEALED_NO_GO',new_training_updates=0,files_sha256=manifest,
        raw_arrays_sha256=review['raw_arrays_sha256'],preregistration_commit='2bb3d2c',
        historical_git_evidence=git_evidence.splitlines(),
        original_training_checkpoints=history['checkpoints'],
        historical_files_preserved=True,
        scope='Git contains audit code/protocol/config identity/scalars/curves/report; large predictions and original weights remain at hashed local paths'))
    print(json.dumps(dict(status='SEALED_NO_GO',files=len(manifest))))

if __name__=='__main__':
    main()
