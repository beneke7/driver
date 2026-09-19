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

`driver/quadratic_benchmark.py` now supplies a contract-grade optimization
track for five synthetic positive-definite families. The serial evaluator and
`driver/contract.py` gate 15 held-out cases across three thresholds. The
decoder campaign now supplies the first 85M-parameter target-model campaign
for three local byte landscapes and three seeds. It compares six same-budget
policies from immutable matched parents and writes ranking/calibration and
capability-cost artifacts. Imagined branches remain disabled.

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

## First promotion checkpoint

The first optimization promotion gate passes on dimension-8, ill-conditioned positive-
definite landscapes: five families, three held-out seeds per family, and
relative targets `1e-4`, `3e-5`, and `1e-5`. The primary serial wall-time
geometric means are 26.36×, 29.52×, and 43.07×, with clustered 95% lower
bounds above 22× at every target. This is a valid optimization-track result
under the contract, not a general pretraining result. The decoder development
campaign is complete but does not pass or attempt the five-landscape
promotion gate; its first ranking model has 11.1% top-1 agreement. The next
tests are changed data/architecture transfer and a calibrated decoder driver.

## Known ceilings

The current controller groups experience by action kind and ignores telemetry
geometry. That is intentional: it is a cheap control condition, not the
proposed deep driver. Upgrade it only when the fixed comparison shows that
state-dependent information is available and the baseline cannot use it.

## Decoder driver preregistration

The decoder-specific claim, action roster, atlas split, capability thresholds,
cost ledger, and robust 10x gate are fixed in
[`DECODER_DRIVER_PREREGISTRATION.md`](DECODER_DRIVER_PREREGISTRATION.md).
The next implementation rung is the response atlas built from existing
matched campaign transitions, followed by a held-out action-conditioned
predictor. A policy is not promoted until it beats the action-only prior on
fresh roots; the fixed shadow baseline remains a mechanism control, not a
driver claim.

### Current atlas checkpoint (2026-09-19)

The first normalized AdamW atlas contains 61 matched parent groups and 213
observed action rows across the historical decoder campaigns. It preserves
immediate, recovery, and final outcomes, matched no-op deltas, risk flags,
provenance, and additive cost components. A history-conditioned selector was
trained on complete FineWeb roots and evaluated on four complete TinyStories
roots. Its raw final-loss RMSE was `0.0321`, but support calibration rejected
all four transfer deployments; the safe policy selected no-op every time.
The action-only prior was better than that abstaining policy on this small
holdout, so there is no justification yet for a planner or online learner.
The next atlas model must first beat that prior on held-out roots within a
single data regime, then survive the changed-data gate.

The new three-horizon ensemble was tested on the eight-root FineWeb atlas,
holding out one 139M and one 85M root. It supported one of the two roots and
selected the shadow action there at the final horizon, but its selected mean
final delta was `-0.0178` versus `-0.0351` for the action-only prior; it did
not beat the prior. On four complete TinyStories roots trained only from the
FineWeb atlas, support was `0.0`, final prediction RMSE was `0.0101`, and the
safe selector abstained everywhere. The raw final ranking happened to agree
with the realized best action, but the prediction/reality gap and zero support
make that diagnostic only. The predictive gate therefore remains closed:
there is no justified planner, dreaming stage, or online policy yet.
