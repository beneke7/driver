# Matched shadow timing-grid results

Date: 2026-09-19. This is a response-atlas and selector pilot, not a
promotion result.

## Matched mechanism response

The preregistered early and late campaigns completed 6/6 branches with zero
failures. Combined with the exact middle condition from the fixed gate, the
atlas has 9 matched FineWeb-Edu 85M roots and 18 branches. All threshold
records are present, but the existing parent-relative thresholds again reach
at the first immediate point and do not measure cost-to-quality.

| Parent step | Seeds | Endpoint passes | Mean final Δloss | Mean recovery Δloss | Geo wall | Geo FLOPs | Geo tokens |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 768 | 24--26 | 0/3 | +0.05017 | +0.05388 | 1.1863x | 1.1995x | 1.2000x |
| 1280 | 21--23 | 3/3 | -0.00618 | -0.02036 | 1.1328x | 1.1426x | 1.1429x |
| 1792 | 24--26 | 3/3 | -0.01083 | +0.00187 | 1.1049x | 1.1109x | 1.1111x |

This is a sharp timing interaction. The shadow action is harmful at parent
step 768, useful at 1280, and useful at 1792. The late condition has small
positive recovery deltas on seeds 25 and 26 but durable final gains; a selector
must therefore model more than an immediate or recovery snapshot. The raw
pre-average endpoint remains worse than no-op on every root; the final
snapshot average is the operative macro-action.

The timing result closes any claim that the fixed action is generally safe. It
also establishes a real, low-dimensional state signal worth testing: parent
training position can separate a harmful from a productive action while the
cost advantage remains visible.

## Small model, complete-seed holdout

The existing numerical ensemble was trained three times. Each run held out
both timing roots for one fresh seed and trained on the other seven complete
roots. It used the same 128-hidden, three-member, 400-epoch model and the
support-gated no-op fallback.

| Held-out seed | Support | Final top-1 | Selected final Δloss | Action-only prior | Final prediction gap | Recovery selected Δloss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 2/2 | 2/2 | -0.00655 | 0.00000 | 0.00154 | 0.00000 |
| 25 | 1/2 | 2/2 | -0.00473 | 0.00000 | 0.00177 | +0.00306 |
| 26 | 2/2 | 2/2 | -0.00497 | 0.00000 | 0.00186 | +0.00134 |

The final-horizon selector chose no-op at 768 and shadow-stop at 1792 on all
three held-out seeds: 6/6 final action rankings were correct. It would have a
diagnostic geometric wall ratio of about `1.051x` over these six deployments
(three no-op early roots and three shadow late roots). This is not a 10x claim,
and the sample has one data source, one width, two action choices, and only two
held-out timing states.

Recovery ranking is not yet safe: the model selected the late shadow action on
two held-out seeds where its recovery delta was positive. The final selector
therefore demonstrates a promising timing-conditioned final response, not a
deployable multi-horizon policy. Add a conservative all-horizon risk gate or
collect more late-state responses before deployment.

The exact model artifacts are:

- `runs/shadow-timing-grid-model-seed24-v1/report.json` and `world_model.pt`;
- `runs/shadow-timing-grid-model-seed25-v1/report.json` and `world_model.pt`;
- `runs/shadow-timing-grid-model-seed26-v1/report.json` and `world_model.pt`.

## Decision

Go to the next small-model experiment: retain the two-action response atlas,
add a cost-aware multi-horizon risk rule, and validate it on changed data or
width. Do not scale the steerer. The timing grid has shown that a state-aware
gate may add value over the action-only prior, but the recovery failure and
single-regime scope keep the deployment gate closed.

## Reproduction

```bash
uv run python -m driver.shadow_gate_report --self-check
uv run python -m driver.shadow_gate_report \
  --results runs/shadow-timing-grid-fineweb85-early-s24-26-v1 \
  --results runs/shadow-timing-grid-fineweb85-late-s24-26-v1 \
  --results runs/shadow-gate-fineweb85-smoke-s21-v3 \
  --results runs/shadow-gate-fineweb85-s22-23-v2 \
  --output runs/shadow-timing-grid-report-v1.json

uv run python -m driver.response_atlas --require-matched \
  --results runs/shadow-gate-fineweb85-smoke-s21-v3 \
  --results runs/shadow-gate-fineweb85-s22-23-v2 \
  --results runs/shadow-timing-grid-fineweb85-early-s24-26-v1 \
  --results runs/shadow-timing-grid-fineweb85-late-s24-26-v1 \
  --output runs/shadow-timing-grid-atlas-v1
```
