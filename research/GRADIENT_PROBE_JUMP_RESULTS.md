# One-gradient 32-step probe-jump results

Date: 2026-09-20. This is a completed causal skip pilot, not a promotion
result.

## Coverage and accounting

The preregistered FineWeb-Edu 85M pilot used three fresh seeds (30--32), an
exact AdamW parent at step 1280, and matched `noop`/`probe_jump_32_half`
branches. All 6/6 branches completed with zero failures. The candidate took
one real gradient step, extrapolated its full parameter update over the
remaining 31 steps, advanced the data cursor, decayed AdamW moments, and
advanced the optimizer step counter.

The candidate physically trained `2,637,824` target tokens versus
`3,145,728` for no-op, while exposing the same cursor horizon. Its physical
FLOP ratio was `1.1852x` and its per-branch wall ratio was approximately
`1.18x`. The conservative ratio after charging the 31 skipped baseline steps
was `0.9948x` FLOPs. The result therefore cannot be called a speedup unless
the quality contract is first met; it was not.

## Matched response

Loss deltas are candidate minus matched no-op; negative is better. The three
entries are immediate, recovery, and final.

| Seed | Δloss | Durable-safe? | Final pass at `-1e-3`? |
| ---: | --- | :---: | :---: |
| 30 | `+1.428644, +0.075042, +0.015167` | no | no |
| 31 | `+1.850775, +0.114576, +0.023333` | no | no |
| 32 | `+1.660633, +0.100399, +0.022560` | no | no |
| Mean | `+1.646684, +0.096672, +0.020353` | 0/3 | 0/3 |

The candidate's large immediate loss spike is not recovered by ordinary
AdamW. The data cursor and optimizer clock were advanced consistently, so this
is not an uncharged duplicate-data artifact. It is evidence against using one
local update as a 32-step extrapolation at this parent regime.

## Decision

The preregistered family gate fails on every root. Close the one-gradient
probe-jump family at horizon 32 and alpha 0.5. Do not tune the blend, train a
selector, or extend the skip horizon based on this result. Smaller blends or
different horizons would be new preregistered action families, not favorable
interpretations of this pilot.

The strongest current conclusion is now narrower and useful: the fixed shadow
endpoint gives a roughly 1.13x mechanism baseline in selected regimes, but
both a stored-moment equal-token extrapolation and a real-gradient 32-step
skip fail to preserve durable capability. Further progress requires a new
action family or a better observable state/action geometry, not a larger
steerer around these jumps.

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
