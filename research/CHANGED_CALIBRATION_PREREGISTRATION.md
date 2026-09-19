# Changed-regime calibration split

Date: 2026-09-19. This is a development transfer audit over already-completed
matched branches. It is not a new promotion claim: the underlying root
outcomes predate this split declaration.

## Question

After calibration on a small amount of changed-regime data, can the existing
numerical response model rank the two fixed actions on a complete held-out
seed across FineWeb-Edu 139M, TinyStories 85M, and TinyStories 139M?

## Frozen data and split

Use only these locked middle-timing campaigns:

- `runs/shadow-gate-fineweb139-s9-11-v1`;
- `runs/shadow-gate-tinystories85-s9-11-v1`;
- `runs/shadow-gate-tinystories139-s9-11-v1`.

Build one nine-group atlas with actions `{noop, trajectory_shadow_stop}`.
Run three leave-one-seed-out fits. For each fit, hold out every root whose
seed is the selected seed across all three strata; train on the other six
groups. Do not mix the FineWeb 85M timing roots into this calibration audit.

Use the existing 128-hidden, three-member, 400-epoch ensemble, fixed feature
representation, support quantile, risk quantile, and no-op fallback. Do not
tune them per seed. Report immediate, recovery, and final prediction/ranking,
support, abstention, and action-only prior results separately.

## Interpretation

This audit is informative if the model has support on held-out changed roots
and improves final action ranking over the action-only prior without unsafe
recovery selection. It does not establish transfer to unseen architecture or
data, and it does not authorize a planner, online learner, or larger steerer.
If support remains zero or recovery ranking regresses, collect more causal
calibration roots before increasing model capacity.
