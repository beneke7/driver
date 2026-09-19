# Cost-aware shadow-stop gate preregistration

Date: 2026-09-19

This is a fixed-mechanism control and a gate calibration screen. It is not a
learned driver and cannot support the 10x claim by itself.

## Hypothesis

At a parent checkpoint at global step 1280, a two-window trajectory shadow
average can reach the matched full AdamW/no-op endpoint quality after a
512-step continuation, while the baseline uses 768 steps. The rule is
timing-conditioned and abstains outside its declared timing support:

```text
if parent_step == 1280: shadow_average, stop at step 1792
otherwise:              AdamW/no-op continuation to step 2048
```

The timing rule is fixed before reading the fresh results. Earlier completed
controls showed regression at step 768 and durable improvement at steps 1280
and 1792; those controls are development evidence, not part of this fresh
roster.

## Matched branches

For each root, run two separate campaigns from the same deterministic parent
configuration and verify the parent checkpoint hash, data hash, config hash,
and code revision before comparison:

- `noop`: prefix 1280, immediate 128, recovery 512, final 768;
- `shadow_stop_512`: prefix 1280, immediate 128, recovery 384, final 512,
  with trajectory interval 128, window 2, and alpha 0.25.

Both branches preserve the full AdamW parent state, RNG, and data cursor. The
candidate's earlier endpoint is real early stopping, not hidden data skipping.
Charge prefix work, target work, shadow snapshot/merge work, evaluations, and
all consumed tokens. Compare the candidate's final validation loss at step
1792 with the matched no-op's final loss at step 2048. A candidate is a
durable pass only if it is finite, has no unsafe observed phase, and is at or
below the matched no-op final loss.

The report must include per-root immediate/recovery/final values, first stable
threshold reach for the existing `0.995/0.99/0.98` parent-relative thresholds,
wall and FLOP ratios including the shared prefix, token exposure, failure
rate, geometric mean/median, worst case, and the cost-to-quality prediction
versus reality gap. No result is pooled across roots before those rows are
visible.

## Fresh roster

Use three fresh seeds per stratum:

| Data | Width | Seeds |
| --- | ---: | --- |
| FineWeb-Edu | 85M | 21, 22, 23 |
| FineWeb-Edu | 139M | 9, 10, 11 |
| TinyStories | 85M | 9, 10, 11 |
| TinyStories | 139M | 9, 10, 11 |

Promotion of this fixed baseline requires at least 2/3 durable passes in
every stratum, no catastrophic failures, and a positive conservative wall and
FLOP advantage on the successful branches. Failure closes the fixed gate as a
general mechanism; it does not justify a larger selector.

## Reproduction commands

Replace `<data>` and the width-specific model arguments as shown below. The
full and stop campaigns must be run into separate empty output directories:

```bash
uv run python -m driver.decoder_campaign \
  --output runs/shadow-gate-<data>-<width>-full-s9-11 \
  --landscape phase_switch --seed 9 --seed 10 --seed 11 \
  --strategy noop --prefix-steps 1280 --immediate-steps 128 \
  --recovery-steps 512 --final-steps 768 \
  --trajectory-interval 128 --trajectory-window 2 --trajectory-alpha 0.25 \
  --train-file <train-file> --validation-file <validation-file>

uv run python -m driver.decoder_campaign \
  --output runs/shadow-gate-<data>-<width>-stop-s9-11 \
  --landscape phase_switch --seed 9 --seed 10 --seed 11 \
  --strategy trajectory_shadow_average --prefix-steps 1280 \
  --immediate-steps 128 --recovery-steps 384 --final-steps 512 \
  --trajectory-interval 128 --trajectory-window 2 --trajectory-alpha 0.25 \
  --train-file <train-file> --validation-file <validation-file>
```

The 85M configuration is width 768 / 12 layers / 12 heads; the 139M
configuration is width 1024 / 11 layers / 16 heads. The data files and seed
roster are recorded in each manifest; the commands above are templates, not a
claim that an unrecorded run has completed.
