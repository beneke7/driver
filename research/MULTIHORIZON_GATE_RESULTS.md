# Multi-horizon cost-aware gate results

Date: 2026-09-19. This is a frozen-policy evaluation, not a new training
experiment or a promotion result.

## Evaluation

The gate reused the three timing-holdout response-model checkpoints and the
three changed-regime calibration checkpoints. It selected a shadow-stop action
only when the measured cost was no higher than no-op and the model's final,
recovery, and immediate predictions all passed the preregistered three-sigma
risk rule. Unsupported or uncertain groups fell back to no-op.

The evaluation was rerun after correcting the reporting code so that the
prediction/reality gap compares realized loss change with the ensemble mean,
not with its conservative risk margin. The decision rule and checkpoints were
not changed.

## Timing holdouts

The six groups are the FineWeb-Edu 85M seeds 24--26 at parent steps 768 and
1792. The model selected one shadow action, on seed 26 at step 1792, and
selected no-op on the other five groups. The selected branch improved final
loss but had a recovery regression of `+0.00267`, beyond the `1e-3` gate.

| Measure | Result |
| --- | ---: |
| Groups | 6 |
| Non-noop selections | 1/6 |
| Supported groups | 5/6 |
| Recovery violations | 1/6 |
| Final mean, selected | `-0.001656` |
| Final mean, action-only prior | `0.000000` |
| Geometric wall ratio | `1.0167x` |
| Geometric FLOP ratio | `1.0177x` |
| Geometric token ratio | `1.0177x` |
| Mean final prediction gap | `0.000492` |

This is not a deployable result. The risk gate removed the early harmful
selections and abstained on seeds 24 and 25, but it did not remove the late
recovery false positive. Its pooled speed ratio is small because the safe
fallback selected no-op on five of six groups.

## Changed-regime calibration

The nine groups are FineWeb-Edu 139M and TinyStories 85M/139M, with one seed
held out per calibration fit. Four groups received the shadow action. No
selected branch violated the recovery gate, but the selected final mean was
worse than the action-only prior.

| Measure | Result |
| --- | ---: |
| Groups | 9 |
| Non-noop selections | 4/9 |
| Supported groups | 7/9 |
| Recovery violations | 0/9 |
| Final mean, selected | `-0.012733` |
| Final mean, action-only prior | `-0.019329` |
| Geometric wall ratio | `1.0565x` |
| Geometric FLOP ratio | `1.0610x` |
| Geometric token ratio | `1.0611x` |
| Mean final prediction gap | `0.002221` |

The gate is safer here but not better. It does not earn a deployment or a
larger response model.

## Decision

The preregistered decision boundary is not met: timing transfer retains a
recovery failure, while changed-regime calibration trades away durable final
quality. The useful finding is narrower: an ensemble uncertainty margin can
turn a timing-conditioned shadow action into a mostly abstaining policy, but
the current state representation and calibration do not support safe,
beneficial selection across complete held-out roots.

Keep the gate, no-op, action-only prior, and fixed shadow action as controls.
Do not add PPO/SAC, dreaming, online core updates, or a larger steerer. Any
next experiment must introduce a new preregistered causal action family and
clear a small matched oracle ceiling before another selector is trained.

## Reproduction

```bash
uv run python -m driver.multihorizon_gate --self-check

uv run python -m driver.multihorizon_gate \
  --atlas runs/shadow-timing-grid-atlas-v1/atlas.jsonl \
  --model runs/shadow-timing-grid-model-seed24-v1/world_model.pt \
  --source-report runs/shadow-timing-grid-model-seed24-v1/report.json \
  --output runs/multihorizon-gate-timing-seed24-v1.json

# Substitute seeds 25 and 26, then repeat for the three changed-calibration
# model/report pairs using runs/changed-calibration-atlas-v1/atlas.jsonl.
```
