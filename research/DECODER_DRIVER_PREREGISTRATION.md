# Decoder landscape-driver preregistration

Status: preregistered before the next decisive atlas or policy experiment.

This document fixes the claim boundary for the decoder optimization track. It
does not retroactively promote the existing Muon/shadow screens. Those screens
remain mechanism evidence and development data.

## Primary question

Can a driver compress a decoder training state into a transferable
action-response representation and reach the same held-out capability as a
matched AdamW/no-op continuation at no more than one tenth of its valid total
deployment cost?

The initial target optimizer is AdamW. Muon and other optimizers are secondary
baselines or separate optimization-track controls; their results cannot be
used to claim a driver beat AdamW.

## Unit of evidence

One case is a complete target run identified by landscape, seed, architecture,
data version, objective, and code revision. One decision is a branch from an
immutable parent checkpoint. The parent must preserve model parameters,
optimizer/scheduler/scaler state, Python/CPU/CUDA RNG, data cursor,
configuration, dataset, objective, and code hashes.

The canonical response is recorded at three predeclared horizons:

```text
R(s,a) = (immediate, recovery, final, cost, risk)
```

Each loss response is reported both absolutely and as a delta from the matched
no-op branch. A branch is unsafe when it fails, becomes non-finite, or has a
positive maximum loss delta across immediate, recovery, and final horizons.
Transient improvement is never sufficient evidence.

## Canonical actions

The first atlas roster is deliberately finite:

1. `noop`: ordinary AdamW continuation;
2. `role_pulse_small`, `role_pulse_medium`, `role_pulse_large`: bounded
   learning-rate/update pulses with explicit attention, MLP, embedding, norm,
   and head multipliers;
3. `momentum_extrapolate_small` and `momentum_extrapolate_medium`:
   update-space directions from initialized AdamW moments;
4. `damping`: bounded cooling of the base update;
5. `momentum_state_blend`: a declared moment interpolation, never an implicit
   reset;
6. `low_rank_probe`: a bounded structured direction whose probe and recovery
   work are charged;
7. `shadow_average`: endpoint-only trajectory rectification; it is not a live
   training skip and must not be mixed with live macro-actions in analysis;
8. `jump_or_skip`: a separate hypothesis that must account for data exposure,
   cursor movement, state handling, and all recovery work.

The first implementation need not execute every action in one campaign. An
action is admitted to a policy comparison only after its branch implementation
has a self-check and complete cost/provenance record. Unsupported actions stay
out of the deployment claim rather than being assigned a synthetic outcome.

## Atlas and model split

The response atlas is built from real matched branches. Development roots may
fit normalization, predictors, and action priors. A held-out root is never
used to fit weights, calibrate an uncertainty radius, choose a stop horizon,
or select a fallback before its locked evaluation.

The first world model is a small numerical history model or ensemble. It must
predict action-conditioned immediate, recovery, and final deltas plus an
uncertainty or support estimate. Its first gate is predictive, not speed-based:

- beat a constant/action-only prior on held-out action ranking;
- report RMSE, rank regret, calibration, support/abstention rate, and the
  prediction-versus-reality gap;
- preserve `noop` as the fallback when support or risk is insufficient.

No PPO, SAC, CEM, beam search, imagined policy training, pretrained language
backbone, or recursive online core update is admissible before that gate is
met on complete held-out roots. Imagined transitions are proposals only and
never evidence.

### Actionability labels

Before fitting a productive-region model, derive labels only from matched
causal atlas rows. Use an absolute validation-loss tolerance of `1e-3`, fixed
before label generation. For a non-noop action with finite observations at all
three horizons:

- `dangerous`: the branch fails/non-finite or its final delta versus noop is
  greater than `1e-3`;
- `jumpable`: it is `productive` and reaches the noop final loss by the
  immediate or recovery horizon with lower charged wall time than the noop
  final endpoint;
- `productive`: its final delta is at most `-1e-3` and no observed horizon
  delta is greater than `1e-3`;
- `recoverable`: its final delta is at most `1e-3`, but an earlier observed
  delta is greater than `1e-3`;
- `neutral`: finite, non-noop evidence that matches none of the above;
- `stalled`: no finite final outcome is available without a declared branch
  failure. A failed branch remains `dangerous` rather than being relabeled
  as a benign stall.

`noop_baseline` is a separate label for the ordinary continuation and is not
treated as a positive action. `jumpable` requires an observed, charged
early-to-final comparison; a hindsight checkpoint or an uncharged data skip
cannot receive that label. The labeler emits `unresolved` metadata when a
required horizon is missing and never creates a counterfactual outcome.
Labels are development artifacts until an action-conditioned model beats the
action-only prior on complete held-out roots with support calibration.

### Trajectory-anchor initialization track

Knowledge transfer is evaluated separately from fixed-architecture optimizer
control. The primary anchor screen is a **disjoint-adaptation** experiment:
it initializes a target branch from a complete step-1536 checkpoint produced
on a different seed in the same data and architecture regime, then gives both
the target noop control and the anchor the same predeclared byte interval for
their continuation. The interval must be outside the source checkpoint's
consumed prefix; its offset, length, hash, and cursor are recorded. This
avoids treating nearby rotations of one byte file as independent fresh data.

The source checkpoint's model and AdamW state are retained, while the primary
run uses the target RNG after the transfer. A source-RNG variant is a control.
The target data cursor and target data hash are explicit. Source and target
seeds are paired before reading target outcomes. A zero-moment control resets
both AdamW moment tensors and the optimizer step clock; preserving source
moments preserves the clock.

The target noop control and anchor are evaluated at 128, 512, and 768 target
steps against three fixed thresholds (`0.995`, `0.99`, `0.98` times the target
parent validation loss). Report the first horizon that reaches and stays at
each threshold, source-prefix creation cost, target continuation cost,
checkpoint I/O, and ratios for fixed deployment counts `N = 1, 2, 4, 8, 16`:

```text
C_anchor(N) = C_source_prefix / N + C_target_continuation
S_anchor(N) = C_target_noop / C_anchor(N)
```

The source-prefix cost is never treated as free. A fixed cyclic seed pairing,
normal continuation, source optimizer-state preservation, target-RNG control,
and a zero-moment state control are development controls. This track cannot
claim optimization speedup for a single deployment; any positive result is a
knowledge-transfer result and must survive fresh target seeds, a changed data
regime, and the same amortization accounting.

### Same-corpus data-mixture oracle

The next allocation screen tests one fixed, non-learned action: on FineWeb-Edu
85M and 139M matched parents, replace exactly 25% of the next 768-step
continuation with blocks from a predeclared disjoint tail interval of the same
FineWeb training file. Blocks contain 16 optimizer steps (`16,512` stream
bytes at the current batch/context); four of every sixteen blocks are
alternate data and the remaining blocks preserve the ordinary target
continuation order. The source/no-op branch and candidate consume identical
tokens, optimizer steps, validation, and compute accounting. The alternate
interval, selected-byte count, candidate cursor, and hashes are recorded.

The baseline is a recomputed AdamW/no-op continuation from the immutable
parent. The candidate uses the same parent optimizer/RNG state and target
cursor; only the continuation bytes differ. This is a causal fixed-allocation
oracle, not a learned selector. Pilot coverage is three fresh roots at each
width. Promotion requires durable improvement on at least two of three roots
at both widths and a conservative threshold-cost advantage over the fixed
shadow control; otherwise close the allocation family. A coarse
FineWeb/TinyStories switch has already been tested separately and is not
reused as evidence for this mixture.

If the FineWeb oracle clears that equal-budget durability screen, run a
separate transfer check on TinyStories 85M seeds 3--5 with the same 25%
block schedule and a disjoint tail interval of the TinyStories training file.
This changed-data check is not pooled with the FineWeb result; it is evidence
about allocation transfer only.

### Pre-parent low-rank transport oracle

The next structured-transport ceiling uses only trajectory snapshots available
before the step-1536 parent. The nested basis roster is fixed as secants:

1. rank 1: `theta_1536 - theta_1408`;
2. rank 2: `theta_1408 - theta_1280`, `theta_1536 - theta_1408`;
3. rank 3: `theta_1280 - theta_1152`, `theta_1408 - theta_1280`,
   `theta_1536 - theta_1408`.

For each rank, a hindsight coefficient fit projects the step-2048 future
parameter delta into that pre-parent span. The projected state is applied at
the step-1536 parent with AdamW moments preserved, the data cursor advanced to
step 2048, and a 128-step recovery plus final step-2304 continuation is
measured. The projection fit, future target, and skipped exposure are charged
and marked oracle-only; future optimizer state is not assumed to be available.
The primary roster is three FineWeb-Edu 85M roots, three FineWeb-Edu 139M
roots, and three TinyStories 85M roots with complete provenance. Parent-cursor
and zero-moment/clock-reset controls are separate follow-ups, not silently
pooled with the primary result.

Report explained future-delta energy, residual, singular values, condition
number, immediate/recovery/final loss, fixed thresholds, failures, and
conservative cost. This oracle cannot support a driver claim; it only remains
open if a low-rank action shows durable improvement on at least two of three
fresh roots within a regime and a positive conservative threshold-cost
opportunity.

## Capability and cost contract

For every held-out case define three thresholds before reading candidate
results: the matched AdamW/no-op validation loss at the predeclared easy,
middle, and hard global checkpoints. The evaluator records the first point at
which the candidate reaches a threshold and remains at or below it through its
declared horizon. A run that never reaches a threshold is a visible failure.

Deployment cost includes, without exception:

- target training FLOPs and consumed tokens;
- driver inference and online-update work;
- probes, trajectory storage/merge, evaluation, and recovery;
- rejected or failed branches used by the deployed policy;
- wall-clock time from the common initialization, including the shared prefix.

One-time meta-training and search costs are separate ledger entries and are
amortized only over a stated deployment count. No data exposure, evaluation,
recovery, or failed run may be omitted to improve a ratio.

For threshold `Q*`, report

```text
S(Q*) = C_adamw_noop(Q*) / C_driver(Q*)
```

The robust 10x gate is eligible only when all of the following hold on the
locked promotion roster:

- at least five held-out landscapes, three fresh seeds per landscape, and
  three thresholds, or a preregistered equivalent;
- at least one changed data mixture or architecture condition;
- complete matched coverage and zero catastrophic driver failures;
- every per-case/per-threshold wall-cost ratio is at least 10.0;
- every landscape median and seed median is at least 10.0;
- geometric mean is at least 10.0 and the declared clustered 95% lower bound
  is at least 10.0;
- the same gates pass for any additionally declared gated metric.

The repository evaluator remains the source of truth for aggregation and
fail-closed behavior. If a future campaign changes thresholds or metrics, the
contract and this preregistration must be updated before candidate results are
opened.

## Stopping rules and next rung

The atlas rung stops when a held-out action-ranking experiment either meets
the predictive gate above or shows that the current telemetry/action set has
no useful support. A negative result means revise observations or actions;
it does not justify a larger steerer. Add a recurrent model only when an
explicit-history MLP loses at matched driver cost because of nonlocal memory.
Add imagined planning only after selected model actions remain calibrated on
fresh real branches. The 10x claim is not made if any gate fails; the report
must state the strongest durable ratio and its limiting factor.
