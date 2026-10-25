"""Final independent input alignment, traceability and export verification."""
from ctx_v2 import *
import re

def main():
    publication=verified(ART/'publication_manifest.json');verify({str(PROJECT/p):h for p,h in publication['files_sha256'].items()})
    for sidecar in ART.rglob('*.sha256'):
        assert sha(sidecar.with_suffix(''))==sidecar.read_text().split()[0],sidecar
    report=PROJECT/'docs/reports/deep_baselines_identifiability_v2_zh.md'
    for link in re.findall(r'\]\(([^)]+)\)',report.read_text()):assert (report.parent/link).exists(),link
    assert not [p for p in ART.rglob('*') if p.suffix in ('.pt','.pth','.ckpt','.npy','.npz','.h5','.hdf5')]
    totals=0
    with Boundary('audit') as access:
        samples=torch.load(ROOT/'inputs/train_AB.pt',map_location='cpu',weights_only=False)+torch.load(ROOT/'inputs/unseen_AB.pt',map_location='cpu',weights_only=False)
        held=[torch.load(ROOT/f'inputs/H_{i}.pt',map_location='cpu',weights_only=False) for i in range(8)]
        for arm in design()['arms']:
            run=ROOT/'runs'/arm
            for update in design()['evaluation_updates']:
                for i,sample in enumerate(samples):
                    targets=dict(sample,H=held[i])
                    with np.load(run/f'raw/update{update:03d}_case{i}.npz',allow_pickle=False) as arrays:
                        for b,t in targets.items():
                            source=batch([{b:t}],b)
                            mask=source['input_valid_ray'].bool()&source['view_mask'][:,:,None,None,None]
                            assert np.array_equal(arrays['y_'+b],source['input_projection_raw'].numpy())
                            assert np.array_equal(arrays['mask_'+b],mask.numpy());totals+=1
            ck=torch.load(run/'checkpoint_200.pt',map_location='cpu',weights_only=False)
            training=verified(run/'training_result.json')
            assert state_hash(ck['model'])==training['final_state_sha256']
            assert len(rows(run/'updates.jsonl'))==200
    pre=verified(ROOT/'preregistration.json');verify(pre['historical_tracked']);verify(pre['frozen_main_checkpoints'])
    assert read(ROOT.parent/'deep_baselines_v1/queue_status.json')['state']=='paused_by_user'
    assert totals==336
    result=dict(at=now(),passed=True,source_sha256=sha(__file__),sealed_publication_files=len(publication['files_sha256']),
        raw_target_and_mask_groups_exactly_aligned=totals,final_states_match_training=True,all_manifest_hashes_and_report_links_valid=True,
        exported_restricted_arrays=False,historical_and_frozen_main_hashes_unchanged=True,old_queue_paused=True,
        new_optimizer_updates=0,cumulative_optimizer_updates=2400,audit=access.summary())
    freeze(ROOT/'delivery_verification.json',result);freeze(ART/'delivery_verification.json',result)
    print('DELIVERY VERIFIED',totals,'raw target/mask groups; hashes,links,exports,final states',flush=True)

if __name__=='__main__':main()
