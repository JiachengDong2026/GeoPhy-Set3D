# Paper MRGR GeometryLifting mapping diagnostic

Frozen epoch278 / 344678 parameters / checkpoint SHA256
`650b4f2c0f0ab345b5fd99a7d629e0db388cede0937acb7e3191462cd57898d7`.

The official evaluation factory and strict checkpoint loading are reused.
Only the sampling grid in the first lifting forward is flipped; angular_grid
and the second lifting's original forward remain unchanged. Read-only hooks
verify descriptors, relations, observability and masks on every forward.

Five historical draw0 layouts, all35 used-test cases, three fixed flips:
525 formal forwards plus12 separately recorded engineering forwards. Historical
175 baseline predictions are reused, with no full baseline inference rerun.
GT64 access is guarded until all formal predictions are globally sealed;
GT128, training, backward, optimizers and model selection are forbidden.

Run from the repository with its verified gas-torch Python environment:

```bash
python scripts/lifting_mapping_v1/diagnostic.py register
python scripts/lifting_mapping_v1/diagnostic.py controller
```

Registration discovers model/data/prediction paths through the official
manifests and checks their existence and hashes. The scratch parent is checked
before creating a new directory. Existing experiments are never overwritten;
failed stages retain logs and block downstream stages without automatic retry.
The existing UUID scheduler supplies locks, two15-second idle checks and
inherited lock descriptors. Inference runs sequentially on an idle GPU.

Analysis includes all cases and intervention directions, case-based paired
20000-sample bootstrap (seed20261022), independent metric formulas and an
independent CSV aggregation/weighted-bootstrap verifier. The primary summary
averages flips within case/layout, then five layouts within case, using35 cases.
This is a mechanism diagnostic on a previously used test set, not independent
confirmation or evidence by itself of generalization to new geometry.
