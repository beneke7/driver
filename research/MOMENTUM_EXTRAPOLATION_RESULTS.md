# AdamW momentum-extrapolation probe results

Date: 2026-09-20. This is a completed equal-token oracle pilot, not a
speedup or driver-promotion result.

## Coverage and validity

The corrected FineWeb-Edu 85M campaign ran three fresh roots at parent step
1280 with AdamW and the preregistered actions `noop`,
`momentum_extrapolate_small` (`alpha=0.25`), and
`momentum_extrapolate_medium` (`alpha=0.50`). All 9/9 branches completed with
zero failures. The failed v1 launch is retained as an engineering artifact;
its six failures were caused by nonnumeric provenance strings accidentally
being placed in the numeric action schema and are not included here.

The intervention used the current bias-corrected AdamW moments, preserved the
optimizer state and data cursor, consumed the same `12,582,912` tokens as
no-op, and charged `684,380,160` maneuver FLOPs plus the extra validation.
It therefore tests update geometry only.

## Matched response

Loss deltas are intervention minus its matched no-op; negative is better.
The three entries in each row are immediate, recovery, and final.

| Seed | Small α=.25 | Medium α=.50 |
| ---: | --- | --- |
| 27 | `+0.000152, +0.000125, -0.000091` | `+0.000271, +0.002929, +0.000017` |
| 28 | `-0.001102, -0.000087, -0.000950` | `-0.001714, -0.001435, -0.000098` |
| 29 | `-0.000234, +0.000627, +0.000412` | `-0.000446, -0.000800, -0.001209` |
| Mean | `-0.000395, +0.000222, -0.000210` | `-0.000630, +0.000231, -0.000430` |

The small action is durable-safe on 3/3 roots under the `+1e-3` all-horizon
tolerance, but it has 0/3 final improvements of at least `-1e-3`. The medium
action is durable-safe on 2/3 roots and reaches the `-1e-3` final improvement
on only seed 29; seed 27 has a recovery regression of `+0.002929`.

The candidate branches consumed equal tokens. Their measured wall time was
about `0.2%` above no-op and their charged FLOPs about `0.04%` above no-op,
because this is a probe rather than a skipped continuation.

## Decision

The preregistered family gate is not met. Close the equal-token momentum
extrapolation family as a standalone driver primitive; do not train a
selector, relax the recovery tolerance, or run the planned TinyStories
transfer for this family. A later skip experiment would be a new hypothesis,
not a continuation of this result.

The result does not show that optimizer state is useless. It shows that a
current AdamW moment direction, applied without a fresh gradient, is not a
robust durable improvement at this scale. The next admissible test is a
one-real-gradient, explicitly charged skip that advances data and optimizer
state together; it must be preregistered separately.

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
```
