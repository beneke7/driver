# Long-horizon loss-landscape maneuver

The central hypothesis is stronger than “can a jump lower the next loss?”:

> Can a controlled high-energy maneuver move an overparameterized model into a
> more productive training regime, then preserve or improve held-out
> capability after ordinary training recovers?

The current 64-step byte-landscape screen cannot answer this. It is useful for
causal plumbing, but its short horizon and synthetic data make a transient
intervention look more important than it is. The next campaign must measure a
long continuation from a matched checkpoint.

## What “productive regime” means

For a branch beginning at the same checkpoint, call a maneuver productive only
if it improves at least one predeclared long-run quantity without an
unacceptable loss in the others:

- area under held-out loss after recovery;
- post-recovery loss slope over a fixed token window;
- tokens and wall time to each locked capability threshold;
- final held-out loss and a capability suite at the end of the budget;
- total target, probe, recovery, and driver cost.

The immediate result is diagnostic. A maneuver that briefly lowers loss but
returns to a worse slope is a failed maneuver. A branch that reaches a better
endpoint only after spending more recovery compute is not automatically a
speedup.

## Target and data ladder

Use the same target loop and evaluator while changing one axis at a time:

1. Keep the existing 85M decoder as the reproducible control.
2. Add a roughly 135M condition that fits the 5090 with full AdamW state and
   activations after profiling. Treat this as the overparameterized condition,
   not as a claim that larger is universally better.
3. Replace the byte landscapes with a versioned, non-overlapping language-data
   shard. Start with a small FineWeb/FineWeb-Edu or TinyStories slice whose
   tokenizer, checksum, order, and license metadata are recorded.
4. Keep a locked validation/capability stream unavailable to the maneuver
   selector. A changed-data holdout is a separate transfer test.

The first useful comparison is an 85M/135M pair at the same data source and
objective, with enough tokens that a recovery window is small relative to the
whole run. Do not call a model overparameterized from parameter count alone;
report parameters per training token and the same data/compute budget for both
conditions.

## Maneuver interface

Every maneuver starts from an immutable full checkpoint and consumes a known
data interval. Keep the action set small:

- `noop`: ordinary AdamW continuation;
- `lr_pulse`: a bounded temporary global or role-wise learning-rate increase;
- `structured_pulse`: a normalized role/block displacement with explicit
  energy and direction;
- `secant_jump`: a checkpoint-history extrapolation baseline;
- `recovery`: damping, clipping, or optimizer-state correction applied only
  under a declared failure rule.

Define energy in normalized parameter space, for example

\[
E(a)=\sum_b \frac{\|\Delta\theta_b(a)\|^2}
                         {\|\theta_b\|^2+\epsilon},
\]

and report it beside the ordinary update energy. Sweep a modest geometric grid
of energy levels before training a policy. The high-energy question is about a
controlled displacement, not an unbounded learning-rate explosion.

For every action, record optimizer-state handling, skipped/consumed tokens,
the exact data cursor, immediate outcome, recovery outcome, and long-run
continuation. A branch is invalid if it cannot be resumed from its checkpoint
with the same RNG, optimizer, scaler, and data state.

## Matched long-run experiment

At predeclared checkpoints in each target run, fork:

| Branch | Intervention | Continuation |
| --- | --- | --- |
| A | AdamW noop | ordinary AdamW for the full budget |
| B | low-energy maneuver | recovery, then ordinary AdamW |
| C | high-energy maneuver | recovery, then ordinary AdamW |
| D | high-energy maneuver plus declared state correction | recovery, then ordinary AdamW |

Measure immediately, after a short recovery window, after a medium window, and
at the locked endpoint. Continue the winning and no-op branches long enough to
separate a regime change from a transient. Use paired seeds and parent
checkpoints; one branch from one parent is not an independent replication.

The minimum promotion table is:

| Question | Required evidence |
| --- | --- |
| Did the maneuver change the regime? | post-recovery slope/AUC differs on a held-out continuation |
| Did it preserve capability? | locked endpoint capability is no worse within the preregistered margin |
| Did it accelerate learning? | cost-to-threshold improves after all probe/recovery/driver costs |
| Does overparameterization matter? | effect replicates in 135M but not merely in the 85M control, or the converse is reported |
| Is it a landscape effect? | effect transfers across seeds and at least one data/architecture holdout |

## Where the steerer enters

Do not train a large steerer before the open-loop maneuver has a long-run
signal. The architecture ladder is:

1. fixed energy/action grid;
2. retrieval and a case-level uncertainty gate;
3. small MLP over normalized state and action;
4. GRU/per-run memory if history improves action ranking;
5. numerical Transformer only if long history remains the measured bottleneck.

The model should first predict post-recovery and endpoint outcomes, not only
the next loss. A policy may choose a maneuver only when its conservative upper
bound predicts a nonnegative long-run benefit; otherwise it must choose
`noop`. Imagined rollouts are deferred until selected real branches remain
calibrated under fresh continuations.

## Failure interpretations

- No high-energy branch changes the post-recovery slope: the local landscape
  may not contain useful collapsible redundancy at this scale, or telemetry
  cannot expose it.
- High-energy branches improve loss but hurt locked capability: the maneuver is
  exploiting the monitored objective, not improving learning.
- The effect appears only in the synthetic byte landscapes: report a toy
  mechanism and stop short of an LM claim.
- The effect appears only after enormous search/meta-training cost: report
  amortized deployment count and do not call it a training speedup by default.

This protocol is the bridge from the current history-jump screen to the actual
loss-landscape question. It intentionally postpones architectural ambition
until a real long-horizon maneuver has something for a steerer to learn.
