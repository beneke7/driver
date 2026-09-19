# Passive trajectory corpus and model

Status: representation-only rung, 2026-09-19. This work uses observed
AdamW/no-op trajectories only. It is not causal intervention evidence and
does not claim a training speedup.

## Corpus

[`driver/passive_trajectory.py`](../driver/passive_trajectory.py) converts
complete local decoder results into a normalized JSONL corpus. Each row keeps:

- complete-root identity, architecture, data/configuration/checkpoint hashes;
- global step and consumed-token position;
- raw scalar and tensor-role optimizer telemetry;
- protocol phase (`immediate`, `recovery`, or `late`);
- observed training-loss deltas at horizons 1, 8, 32, and 128 steps.

The collector does not synthesize action outcomes. Future model-only snapshot
paths are retained as provenance references, not as deployable state targets.

The local pilot contains 17 complete roots and 13,056 rows:

- FineWeb-Edu: ten 85M roots and three 139M roots;
- TinyStories: three 85M roots and one 139M root.

The external Pythia/PolyPythia/OLMo sources remain deferred. Their public
weights are useful for passive phase discovery, but missing optimizer state or
data-cursor provenance would not make them causal jump roots. The local
complete traces are therefore the authoritative first corpus.

Build command:

```bash
uv run python -m driver.passive_trajectory \
  --result runs/fineweb-edu-85m-seed3-late-action-set \
  --result runs/fineweb-edu-85m-seed4-late-action-set \
  --result runs/fineweb-edu-85m-seed5-late-action-set \
  --result runs/fineweb-edu-85m-seed9-late-action-set \
  --result runs/fineweb-edu-85m-seed10-late-action-set \
  --result runs/fineweb-edu-85m-seed12-history-action-set \
  --result runs/fineweb-edu-85m-seed13-history-action-set \
  --result runs/fineweb-edu-85m-seed14-history-action-set \
  --result runs/fineweb-edu-85m-seed15-history-action-set \
  --result runs/fineweb-edu-85m-seed16-history-action-set \
  --result runs/fineweb-edu-139m-seed2-late-action-set \
  --result runs/fineweb-edu-139m-seed5-history-action-set \
  --result runs/fineweb-edu-139m-seed6-history-action-set \
  --result runs/tinystories-85m-seed3-history-action-set \
  --result runs/tinystories-85m-seed4-history-action-set \
  --result runs/tinystories-85m-seed5-history-action-set \
  --result runs/tinystories-139m-seed3-history-action-set \
  --output runs/passive-trajectory-corpus-local-v1
```

## Passive model

[`driver/passive_trajectory_model.py`](../driver/passive_trajectory_model.py)
fits a small numerical MLP to future training-loss deltas and the protocol
phase. Splits are by complete root. The phase label is partly determined by
the recorded schedule and is only a plumbing target; its accuracy is not a
dynamics result.

| Split | Held-out roots | Phase accuracy | h1 RMSE (baseline/model) | h8 | h32 | h128 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mixed width/data | FineWeb 85M, FineWeb 139M, TinyStories 139M | `0.867` | `0.106 / 0.161` | `0.128 / 0.221` | `0.130 / 0.249` | `0.133 / 0.208` |
| FineWeb 85M | seed 16 | `0.998` | `0.126 / 0.097` | `0.146 / 0.099` | `0.146 / 0.101` | `0.150 / 0.100` |
| TinyStories 85M | seed 5 | `0.978` | `0.041 / 0.047` | `0.079 / 0.070` | `0.088 / 0.071` | `0.084 / 0.070` |

The within-regime results show useful passive signal, especially at longer
horizons. The mixed holdout is worse than the constant baseline at every
horizon. This is a transfer/calibration failure, not evidence that a larger
steerer is needed. Architecture and data metadata, support estimation, and
possibly regime-specific normalization must be resolved before this model can
serve an action gate.

The model artifacts are passive diagnostics only:

- `runs/passive-trajectory-corpus-local-v1/manifest.json`;
- `runs/passive-trajectory-model-local-v1/report.json`;
- `runs/passive-trajectory-model-fineweb85-v1/report.json`;
- `runs/passive-trajectory-model-tiny85-v1/report.json`.

## Decision

The passive rung is useful enough to retain a compact numerical history model,
but it does not justify a recurrent model, pretrained backbone, planner, or
online policy. The next model must consume action-conditioned real branches
and predict immediate/recovery/final response, support, and risk. Keep noop as
the fallback. If action-conditioned labels do not transfer within one regime,
move the next oracle screen to data/work allocation rather than increasing
model capacity.
