# Uniform-fusion feedback OARI research v1

Independent research branch; not part of the frozen MRGR-UniformFusion model.
The feedback path is copied from the already audited pilot implementation.
`augment` attaches an independent deepcopy of the initial first-stage OARI to
an otherwise unchanged uniform-fusion backbone without consuming RNG.
Training/evaluation orchestration: `scripts/uniform_final_v1/` in the main worktree.
No descriptor removal, loss changes, or multiscale changes accompany OARI.
