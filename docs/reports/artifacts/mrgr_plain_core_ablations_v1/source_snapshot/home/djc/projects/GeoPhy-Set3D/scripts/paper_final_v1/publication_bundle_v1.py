"""Select reviewable paper evidence for Git; keep datasets/weights/arrays local."""
from common import *
import argparse


def select():
    accepted=read(OUT/'final_acceptance.json');assert accepted['completed']
    lock=verified(OUT/'final_lock.json');verify_hashes(lock['sources']);verify_hashes(lock['evidence'])
    selected=set()
    def add(p):
        p=Path(p)
        if p.exists() and p.is_file():
            assert p.stat().st_size<50*1024**2,p
            selected.add(p)
            side=p.with_suffix(p.suffix+'.sha256')
            if side.exists():selected.add(side)
    add(OUT/'README.md')
    for name in ('registration.json','layouts.json','model_manifest.json','final_method_choices_closed.json','final_lock.json',
                 'final_acceptance.json','pipeline_status.json','pipeline_events.jsonl','task_costs.csv','task_costs.json',
                 'numerical_checks.json','boundary_checks.json','loss_checks.json','training_resume_verification.json','architecture_costs.json','development_methods.json'):
        add(OUT/name)
    add(OUT/'github_regression_checks.json')
    add(OUT/'audit_public_summary.json')
    for folder in ('module_ablations_v2','module_ablations_v2/scheduling_overlap_v3','sart','recovery/seal_checkpoint_allowlist_v2'):
        for p in (OUT/folder).glob('*.json'):
            if p.name not in ('resources.json','pipeline_process.json','addition_execution_audit.json'):add(p)
    for name in ('search.json','selected.json','calibration_seal.json','calibration_metrics.jsonl','implementation_amendment_v2.json'):
        add(OUT/'classical'/name)
    for folder in ('classical/source_v1','sart/source_before_cache_fix_v2','module_ablations_v2/scheduling_overlap_v3'):
        for p in (OUT/folder).glob('*.py'):add(p)
    for d in (OUT/'training').iterdir():
        if d.is_dir():
            for name in ('protocol.json','summary.json','metrics.jsonl','status.json','gt_access_audit.json'):add(d/name)
    for scope in ('classical_dev','sart_dev','val','test','classical_slices_v2'):
        d=OUT/'evaluation'/scope
        names=('summary.json','per_case_metrics.csv','paired_comparisons.csv','worst_cases.csv','ablation_paired.json',
               'independent_verification.json','all_predictions_sealed.json','gt_access.json','figure_audit.json')
        for name in names:add(d/name)
        if scope=='classical_slices_v2':
            for name in ('visualization_predictions_sealed.json','slice_amplitude_metadata.jsonl'):add(d/name)
        if scope=='test':add(d/'classical_test35_figure_audit.json')
        if scope=='test':add(d/'metrics.jsonl')
        for ext in ('*.png','*.pdf'):
            for p in d.glob(ext):add(p)
    # Manifests contain identities and runtime only, never the prediction arrays.
    for m in lock['methods']:
        from evaluation import target_for
        for split in ('val','test'):
            d=target_for(m,split)
            for name in ('manifest.jsonl','complete.json'):add(d/name)
    for p in (OUT/'stages').glob('*.json'):add(p)
    for name in ('final_all_seal.log','final_all_seal_recovery_v2.log','final_independent_verification.log','final_acceptance.log'):add(OUT/'stages'/name)
    assert all(p.suffix not in ('.pt','.pth','.ckpt','.npz','.npy') for p in selected)
    assert all('resources' not in p.name and 'predictions' not in p.parts and 'scheduling' not in p.parts for p in selected)
    manifest=OUT/'github_publication_manifest.json'
    freeze(manifest,dict(at=now(),purpose='Reviewable scalar, identity and figure evidence for completed final35 evaluation',
        final_acceptance_sha256=sha(OUT/'final_acceptance.json'),files={str(p.relative_to(ROOT)):sha(p) for p in sorted(selected)},
        total_bytes=sum(p.stat().st_size for p in selected),excluded=['datasets','weights','prediction arrays','recovery tensors','raw resource/process snapshots'],
        reproduction='Original absolute run/data paths and source hashes are retained as execution provenance; supply data and weights locally to rerun.'))
    selected.update((manifest,manifest.with_suffix('.json.sha256')))
    filelist=OUT/'github_publication_paths.txt'
    filelist.write_text(''.join(str(p.relative_to(ROOT))+'\n' for p in sorted(selected)))
    print(f'Selected {len(selected)} files, {sum(p.stat().st_size for p in selected)/1024**2:.2f} MiB; list: {filelist}')


if __name__=='__main__':select()
