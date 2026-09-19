# Ongoing experiment contract

This is an active research goal. Continue the experiment ladder until the user
explicitly asks to pause or stop. A quiet GPU, an interesting negative result,
or a nearly exhausted run is not a reason to stop; report blockers and move to
the next safe rung instead.

## Execution

- Keep one reproducible target loop and one locked evaluator as the source of
  truth.
- Run independent CPU work in parallel: trace inspection, manifests,
  predictors, analysis, and validation. Keep matched branches from one causal
  parent serial unless a profiling check proves parallel execution does not
  distort the timing comparison.
- Use the RTX 5090 continuously for the strongest currently valid workload.
  Increase batch size, target size, or independent queued work only after
  checking memory, throughput, temperature, and result integrity. Do not
  launch uncontrolled concurrent jobs that make wall-time claims invalid.
- Prefer fresh output directories and append-safe artifacts. Never discard a
  failed branch; failed work is part of the cost account.

## Experiment ladder

1. Validate trace storage and complete checkpoint restoration.
2. Measure fixed AdamW/noop versus simple extrapolation and momentum-jump
   controls.
3. Collect enough causal branches to train a numerical history predictor.
4. Compare snapshot, history, recurrent-memory, and online-adapter drivers.
5. Add repeated jump selection only after single-jump predictions calibrate on
   fresh branches.
6. Add imagined planning, external trajectory pretraining, stronger optimizer
   baselines, or changed target architectures only when the previous rung
   identifies a reason to do so.

Every rung must record immediate, recovery, and longer-horizon outcomes, total
target/driver/probe/evaluation/recovery work, failures, and the exact code,
configuration, data, RNG, optimizer, and data-cursor provenance.

## Evidence gates

Do not promote an intervention from one seed or one transient loss. Use whole
trajectory holdouts, fresh seeds, multiple quality thresholds, and at least one
changed data or architecture condition before broadening a claim. A negative
result narrows the next experiment; it does not get hidden or overwritten.

## Checkpointing and handoff

Commit and push after each meaningful setup or experiment checkpoint, with the
result path and validation command in the handoff. Keep the repository clean
before long campaigns so manifests identify the exact code revision. Pause only
for explicit user instruction, a hardware-safety issue, corrupted provenance,
or a genuinely missing authority that changes the experiment's meaning.

The decoder-specific robust 10x definition and preregistered action/atlas
boundary live in `research/DECODER_DRIVER_PREREGISTRATION.md`; do not infer a
decoder promotion claim from the synthetic quadratic contract.
