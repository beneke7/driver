# Changed-regime calibration results

Date: 2026-09-19. This is a development transfer audit over locked matched
branches. It is not a contract-valid promotion result.

## Split and artifacts

The atlas contains 9 complete groups and 18 action rows from FineWeb-Edu 139M,
TinyStories 85M, and TinyStories 139M, all at the fixed 1280-step parent. Each
fit holds out all three width/data groups for one seed and trains on the other
six groups. The model and split were frozen in
[`CHANGED_CALIBRATION_PREREGISTRATION.md`](CHANGED_CALIBRATION_PREREGISTRATION.md).

Artifacts:

- `runs/changed-calibration-atlas-v1/atlas.jsonl`;
- `runs/changed-calibration-model-seed9-v1/report.json` and `world_model.pt`;
- `runs/changed-calibration-model-seed10-v1/report.json` and `world_model.pt`;
- `runs/changed-calibration-model-seed11-v1/report.json` and `world_model.pt`.

## Results

`selected` is the support/risk-gated policy; `prior` is the action-only prior
fit on the corresponding training roots. Negative loss delta is better.

| Held-out seed | Support | Final top-1 | Final selected Δ | Final prior Δ | Recovery selected Δ | Recovery prior Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 3/3 | 2/3 | -0.02079 | -0.02051 | -0.03814 | -0.03814 |
| 10 | 1/3 | 3/3 | -0.00812 | -0.01984 | -0.01618 | -0.03819 |
| 11 | 3/3 | 3/3 | -0.01764 | -0.01764 | -0.03640 | -0.03640 |

The model is safe on the selected recovery branches, but it does not beat the
action-only prior: seed 10's abstentions avoid some useful shadow actions as
well as the one FineWeb miss. Pooled across the nine held-out groups, the
selected final mean is approximately `-0.01552` versus `-0.01933` for the
prior. Final prediction/reality gaps are `0.00284`, `0.00274`, and `0.00371`.

## Decision

Within-regime calibration improves support and gives a conservative fallback,
but it does not establish a transferable selector. Do not increase model
capacity or add planning. The next useful work is more causal calibration in a
changed regime plus an explicitly cost-aware, all-horizon risk objective; if
that still cannot beat the prior, close this shadow-action selector route.

This remains a mechanism result around `1.1--1.2x` cost reduction, not evidence
for the 10x target.
