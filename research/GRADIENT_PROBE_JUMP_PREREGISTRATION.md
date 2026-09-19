# One-gradient 32-step probe-jump preregistration

Date: 2026-09-20. This is a new causal skip hypothesis. It is not evidence
until a matched real-data pilot completes.

## Question

Can one observed AdamW update provide a locally smooth direction that safely
replaces a short sequence of future updates when the data cursor and optimizer
clock are advanced together?

## Fixed action

From an exact AdamW parent at global step 1280:

1. consume and train on exactly one ordinary next batch;
2. record the full parameter update `u` after that AdamW step;
3. apply `theta <- theta + 0.5 * 31 * u`;
4. advance the data cursor by the remaining 31 batches;
5. decay each AdamW `exp_avg` and `exp_avg_sq` by `beta_1**31` and
   `beta_2**31`, and advance the AdamW step counter by 31;
6. continue ordinary AdamW.

The matched branches are `noop` and `probe_jump_32_half`. No future
checkpoint, shadow average, stored moment direction, role-specific tuning, or
learned selector is used. The first gradient is real; the remaining 31
updates are an extrapolation. The action must record physical gradient tokens,
exposed/cursor tokens, skipped tokens, optimizer-state policy, and provenance.

## Pilot

Use the existing 85M decoder loop, AdamW, FineWeb-Edu train/validation files,
parent step 1280, and fresh seeds 30--32. Each branch is evaluated at the
matched parent+32 immediate point, after a further 32 ordinary recovery
steps, and after a further 128 ordinary final steps. This is a compact
response screen; it does not satisfy the robust 10x promotion roster.

Report two ledgers:

- physical: actually executed gradient, maneuver, evaluation, and wall/FLOP
  work;
- conservative: physical work plus the 31 skipped baseline-equivalent steps.

No speedup claim is allowed unless the quality comparison is durable at all
three horizons and the ledger states which accounting is being used. The
pilot clears this action family only if at least 2/3 roots have final delta
at most `-1e-3`, maximum immediate/recovery/final delta at most `+1e-3`, and
strictly lower physical cost than matched no-op. Otherwise close the family;
do not fit a selector or extend the skip horizon.

## Reproduction

```bash
uv run python -m driver.history_jump --mode self-check

CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.history_jump --mode collect \
  --output runs/gradient-probe-fineweb85-s30-32-trace-v1 \
  --landscape phase_switch --seed 30 --seed 31 --seed 32 \
  --width 768 --layers 12 --heads 12 --vocab-size 256 --context 128 \
  --batch-size 128 --steps 1280 --checkpoint-every 1280 \
  --train-file data/raw/FineWeb-Edu-train-40m.txt \
  --validation-file data/raw/FineWeb-Edu-valid-8m.txt --device cuda

CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.history_jump --mode benchmark \
  --source runs/gradient-probe-fineweb85-s30-32-trace-v1 \
  --output runs/gradient-probe-fineweb85-s30-32-v1 \
  --landscape phase_switch --seed 30 --seed 31 --seed 32 \
  --horizon 32 --blend 0.5 --action probe_jump --max-parents 1 --device cuda
```
