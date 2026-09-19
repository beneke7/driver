# Upper-layer work-allocation preregistration

Date: 2026-09-20

This is a causal oracle screen for a new training-system action family. It is
not a learned-driver or 10x claim. The fixed pulse, transport, skip, and data
mixture families remain closed under their existing decisions.

## Question

Can a target decoder spend part of a continuation updating only a shallow
prefix of its blocks, then return to the full model, without losing durable
quality? If yes, the action would reduce target compute without skipping data
or advancing the optimizer clock for inactive parameters.

## Frozen action

The target is the existing 85M decoder: width 768, 12 layers, 12 heads,
vocabulary 256, context 128, batch size 128, AdamW, and the versioned
FineWeb-Edu byte streams. Each branch starts from its own immutable parent.

The candidate action is exactly:

```text
depth_curriculum:
  branch steps 0..383: execute blocks 0..5 only (6 of 12)
  branch steps 384..767: execute all 12 blocks
```

The candidate and no-op consume the same 128 * 128 training tokens at every
branch step and use the same data cursor. The candidate retains the parent
optimizer state. During the shallow phase, inactive blocks receive no gradient
and AdamW does not update their moments or weight decay; this is the declared
state transform, not an omitted accounting detail. When full depth resumes,
those blocks re-enter from their parent state.

No layer count, duration, learning rate, batch size, data mixture, optimizer,
or action coefficient may be tuned after opening the fresh results. The action
is evaluated against a matched AdamW/no-op branch from the same parent.

## Horizons and measurements

Use parent steps 768, 1280, and 1792. For each timing point run fresh seeds
33, 34, and 35, with `immediate=128`, `recovery=512`, and `final=768` branch
steps. Record candidate-minus-no-op validation-loss deltas at all horizons,
raw loss, failure/risk flags, consumed tokens, measured end-to-end wall time,
and estimated FLOPs.

The estimator charges each training step as `6 * active_parameter_count *
training_tokens`; evaluation, checkpointing, and all other declared costs are
charged separately. Wall time is measured from the common parent. The full
model is used for validation. The branch manifest must retain the action
parameters, active-layer schedule, optimizer/data state, hashes, code commit,
and per-step active-layer telemetry.

## Pilot decision rule

A root is a durable-safe pass only if both branches are finite, the candidate
has no catastrophic failure, and its loss delta is at most `+1e-3` at
immediate, recovery, and final horizons. A timing stratum remains open only
when at least 2 of its 3 fresh roots pass and its geometric mean conservative
FLOP and measured wall ratios are greater than 1.0. No-op remains the default
outside an opened stratum; the timing result is diagnostic and does not select
a deployable policy.

If no timing stratum opens, close upper-layer skipping at this depth/duration
without adding a selector. If one opens, run a separately preregistered
changed-data or width holdout before fitting a gate. A positive pilot result is
at most a work-allocation mechanism result; it is not evidence for 10x.

## Reproduction commands

Run each command in a clean checkout at the committed preregistration code:

```bash
for parent in 768 1280 1792; do
  CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.decoder_campaign \
    --output runs/depth-allocation-fineweb85-p${parent}-s33-35-v1 \
    --landscape phase_switch --seed 33 --seed 34 --seed 35 \
    --strategy noop --strategy depth_curriculum \
    --width 768 --layers 12 --heads 12 --vocab-size 256 --context 128 \
    --batch-size 128 --prefix-steps "${parent}" \
    --immediate-steps 128 --recovery-steps 512 --final-steps 768 \
    --depth-curriculum-layers 6 --depth-curriculum-steps 384 \
    --train-file data/raw/FineWeb-Edu-train-40m.txt \
    --validation-file data/raw/FineWeb-Edu-valid-8m.txt --device cuda
done
```

The shell loop is a command template only; each parent point needs a fresh
output directory and must finish before its results are opened. Do not pool
timing strata before the per-root decision is visible.
