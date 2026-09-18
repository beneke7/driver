# Initial research plan

## Hypothesis

Training trajectories contain observable, repeatable opportunities for a
bounded intervention to reduce cost to the same held-out capability. The first
test is not whether a large controller can imagine a training universe; it is
whether matched checkpoint branches expose a useful state-dependent choice.

## Staged route

1. Profile one target loop end to end: tokens/s, memory, optimizer time,
   checkpoint I/O, evaluation, telemetry, and driver-call overhead.
2. Build a modern baseline recipe and freeze the target architecture, data
   version, objective, evaluator, and branch protocol.
3. Compare `noop` with a bounded `role_pulse` from identical checkpoints.
4. Add fixed, random, task-greedy, and curiosity-only selector controls. Keep
   the observed-action oracle diagnostic only; it is not a deployable result.
5. Add the cheap online action-statistics baseline. Only after it has signal,
   replace it with a shared numerical history model and a small per-run
   adapter.
6. Test structured corrections first, then adaptive weight nowcasting, then
   data/work allocation. Keep the three routes separately attributable.
7. Add short imagined rollouts only when action-conditioned predictions are
   calibrated on fresh real branches.

## Branch contract

Every branch must preserve or identify:

- model weights and optimizer/scheduler/scaler state;
- random-generator state, data cursor, tokenizer and objective;
- code, configuration, and data hashes;
- starting checkpoint and parent transition;
- action, horizon, tokens, wall time, estimated FLOPs, losses, quality,
  recovery cost, failure status, and selector metadata.

`driver/checkpoints.py` now provides the matched-branch primitive, and its
torch self-check verifies model, optimizer, CPU/CUDA RNG, and hash validation.
The quadratic mechanism benchmark still uses synthetic initial states rather
than full target-model checkpoints; the decoder experiment must use this
primitive before any intervention result is considered causal.

An archive is valid only when parent transitions appear earlier in the same
causal run, IDs are unique, and failed branches remain visible in the cost
account. Split evaluation by complete runs, not adjacent windows. Anything
used for adaptation or selection is development data, not a locked test.

## Promotion rule

Do not promote a candidate because of one transient loss improvement. Require
preselected immediate, recovery, and longer-horizon measurements, then repeat
on fresh seeds. A useful first result is a mechanism-level gain; a broad
speedup requires held-out widths/depths or a changed data distribution and
whole-run accounting.

## First measured checkpoint

The initial CUDA mechanism run uses 16-dimensional diagonal, rotated, and
two-block positive-definite landscapes. It is useful evidence that local
curvature probes can create a large hard-threshold gain, but it does not meet
the full research objective: the easy threshold is below 10×, the target is
not a language model, and the current synthetic archive has no serialized
model/optimizer/data checkpoint. Keep this result as a mechanism screen and
do not promote it to a general training claim.

## Known ceilings

The current controller groups experience by action kind and ignores telemetry
geometry. That is intentional: it is a cheap control condition, not the
proposed deep driver. Upgrade it only when the fixed comparison shows that
state-dependent information is available and the baseline cannot use it.
