# Matched shadow timing-grid preregistration

Date: 2026-09-19. This is an atlas-collection experiment, not a promotion
claim. The fixed action was already frozen in
[`SHADOW_GATE_PREREGISTRATION.md`](SHADOW_GATE_PREREGISTRATION.md); this grid
adds parent-timing variation before fitting a selector.

## Question

Does the shadow-stop macro-action have a repeatable action-conditioned response
at more than one parent time, or is the observed 1280-step result a narrow
timing artifact?

## Frozen roster

Run two fresh FineWeb-Edu 85M campaigns, each with seeds 24, 25, and 26:

| Grid | Parent step | No-op endpoint | Shadow endpoint |
| --- | ---: | ---: | ---: |
| early | 768 | 1536 | 1280 |
| late | 1792 | 2560 | 2304 |

The existing exact 1280-step roots (seeds 21--23) are the middle timing
condition. They remain a separate already-completed roster and are not opened
or re-tuned for this grid.

Every campaign contains both `noop` and `trajectory_shadow_stop`, so each pair
shares one parent checkpoint, AdamW state, RNG state, data cursor, data/config
hashes, code hash, and objective. The shadow branch uses immediate 128,
recovery 384, final 512, trajectory interval 128, window 2, and alpha 0.25.
The no-op uses immediate 128, recovery 512, final 768. Prefix, target,
evaluation, snapshot, merge, and failure costs are charged. Existing relative
thresholds are recorded but are not assumed to discriminate.

## Measurements and gate

For every pair record immediate, recovery, raw pre-average final, post-average
final, wall, FLOPs, tokens, failure, and full provenance. A durable endpoint
pass is post-average candidate final loss no higher than matched no-op final
loss. The timing condition is descriptive unless it is supported by at least
2/3 endpoint passes with zero failures; no timing rule will be selected from
these results after the fact.

The result is sufficient to fit the smallest response model only if both new
timing conditions complete and at least one differs from the other in endpoint
pass rate or response magnitude. A model must then be evaluated by complete
seed holdout, with no-op fallback outside support. If both conditions are
indistinguishable, retain the fixed 1280 rule as the simpler baseline and stop
adding selector capacity.

This grid does not authorize PPO/SAC, imagined rollouts, online core updates,
or a pretrained controller. It is six new roots / twelve branches, with a
single-GPU serial execution budget and no parallel timing claims.

## Reproduction commands

```bash
CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.decoder_campaign \
  --output runs/shadow-timing-grid-fineweb85-early-s24-26-v1 \
  --landscape phase_switch --seed 24 --seed 25 --seed 26 \
  --strategy noop --strategy trajectory_shadow_stop \
  --width 768 --layers 12 --heads 12 --vocab-size 256 --context 128 \
  --batch-size 128 --prefix-steps 768 --immediate-steps 128 \
  --recovery-steps 512 --final-steps 768 \
  --trajectory-interval 128 --trajectory-window 2 --trajectory-alpha 0.25 \
  --train-file data/raw/FineWeb-Edu-train-40m.txt \
  --validation-file data/raw/FineWeb-Edu-valid-8m.txt --device cuda

CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.decoder_campaign \
  --output runs/shadow-timing-grid-fineweb85-late-s24-26-v1 \
  --landscape phase_switch --seed 24 --seed 25 --seed 26 \
  --strategy noop --strategy trajectory_shadow_stop \
  --width 768 --layers 12 --heads 12 --vocab-size 256 --context 128 \
  --batch-size 128 --prefix-steps 1792 --immediate-steps 128 \
  --recovery-steps 512 --final-steps 768 \
  --trajectory-interval 128 --trajectory-window 2 --trajectory-alpha 0.25 \
  --train-file data/raw/FineWeb-Edu-train-40m.txt \
  --validation-file data/raw/FineWeb-Edu-valid-8m.txt --device cuda
```
