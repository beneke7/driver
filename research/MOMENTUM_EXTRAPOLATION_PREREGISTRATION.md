# AdamW momentum-extrapolation probe preregistration

Date: 2026-09-19. This is an equal-token response-atlas experiment. It is
not a speedup or driver-promotion claim.

## Question

Does the AdamW state at a matched checkpoint contain an immediately useful
update-space direction that survives ordinary continuation? A positive result
would justify a later, separately charged skip experiment; a negative result
closes this direction before any learned selector is trained.

## Actions

From the exact parent checkpoint, compare:

- `noop`: ordinary AdamW continuation;
- `momentum_extrapolate_small`: apply `alpha=0.25` of one AdamW-equivalent
  update computed from the current bias-corrected `exp_avg` and `exp_avg_sq`;
- `momentum_extrapolate_medium`: the same operation with `alpha=0.50`.

The displacement includes proportionally scaled decoupled weight decay. The
AdamW moment tensors, optimizer step, RNG state, and data cursor remain
unchanged. No training data is skipped and no optimizer time is advanced. The
extra parameter arithmetic is recorded as maneuver FLOPs, and the additional
post-maneuver validation is charged. This action is therefore a causal,
equal-token probe of update geometry, not a free extrapolation or a speed
claim.

## Pilot and stopping rule

Use the existing 85M decoder loop, AdamW, parent step 1280, and horizons 128,
512, and 768. Run three fresh FineWeb-Edu roots and three fresh TinyStories
roots, with seeds selected before opening outcomes. The fixed data files,
validation stream, target configuration, code hash, checkpoint, and complete
branch ledger are recorded by `decoder_campaign.py`.

For each non-noop action, report immediate, recovery, final, threshold timing,
cost, and failure status. A branch is durable-safe only when all three loss
deltas versus its matched no-op are at most `1e-3`. The family clears this
pilot only if one fixed alpha has durable-safe improvement on at least two of
three roots in both data regimes and does not regress the other regime's mean
final loss. Otherwise close it and do not train a selector or add a skip.

This is a small oracle ceiling, not a claim that the state is transferable.
Even a cleared family must be held out by complete roots and tested at a
changed width before a data-skipping action is considered.

## Reproduction

```bash
uv run python -m driver.decoder_campaign --self-check

CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.decoder_campaign \
  --output runs/momentum-extrapolation-fineweb85-s27-29-v2 \
  --landscape phase_switch --seed 27 --seed 28 --seed 29 \
  --strategy noop \
  --strategy momentum_extrapolate_small \
  --strategy momentum_extrapolate_medium \
  --width 768 --layers 12 --heads 12 --vocab-size 256 --context 128 \
  --batch-size 128 --prefix-steps 1280 --immediate-steps 128 \
  --recovery-steps 512 --final-steps 768 \
  --train-file data/raw/FineWeb-Edu-train-40m.txt \
  --validation-file data/raw/FineWeb-Edu-valid-8m.txt --device cuda

CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.decoder_campaign \
  --output runs/momentum-extrapolation-tinystories85-s12-14-v2 \
  --landscape phase_switch --seed 12 --seed 13 --seed 14 \
  --strategy noop \
  --strategy momentum_extrapolate_small \
  --strategy momentum_extrapolate_medium \
  --width 768 --layers 12 --heads 12 --vocab-size 256 --context 128 \
  --batch-size 128 --prefix-steps 1280 --immediate-steps 128 \
  --recovery-steps 512 --final-steps 768 \
  --train-file data/raw/TinyStories-train-prefix.txt \
  --validation-file data/raw/TinyStories-valid.txt --device cuda
```
