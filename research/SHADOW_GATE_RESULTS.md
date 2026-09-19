# Exact shadow-stop gate results

Date: 2026-09-19. This is a fixed-mechanism screen, not a learned-driver
result and not a contract-valid 10x claim.

## Protocol and provenance

Each case was run in one campaign containing both `noop` and
`trajectory_shadow_stop`. Both branches therefore fork the same immutable
parent. The report rejects a case if the source checkpoint, configuration, data,
or code hash differs between the pair. The candidate uses the declared
128/384/512 horizons, saves trajectory snapshots, and averages the final four
snapshots before its final validation measurement. The no-op uses 128/512/768.
The common 1280-step prefix is charged to both branches.

The complete machine-readable artifacts are:

- `runs/shadow-gate-report-v1.json` — paired outcomes, costs, thresholds, and
  aggregate statistics;
- `runs/shadow-gate-atlas-v1/atlas.jsonl` — 24 causal action rows in 12
  matched parent groups;
- each input run's `manifest.json` and `transitions.jsonl` — full checkpoint,
  RNG/data/config, code, and cost provenance;
- `runs/shadow-gate-model-fineweb139-v1/` — the smallest existing numerical
  response-model diagnostic and its held-out report.

The common code, lockfile, and objective hashes are respectively
`5801f79232c2cbda603df20ef4f95d2192d14d0329599d7726dfb27018759312`,
`967392c443cc7128484640d1cbec15a34cd0f3a3e24b4d969cac265999c2e399`, and
`6486c6ca14b9316a67b345b70fa8f7b2dbd056343985ff1e9bb7ee613999f338`.
The smoke root was recorded at commit `658af79`; the remaining roots at
`50c95ab`. The code hash is identical. The report contains the full
per-case `source_checkpoint_sha256`, `config_sha256`, and `data_sha256`.

The separate baseline-only run
`runs/shadow-gate-fineweb85-full-s21-23` is excluded because it has no matched
candidate. The unblocked combined attempt
`runs/shadow-gate-fineweb85-s21-23-v2` is excluded because the host process
segfaulted before producing a valid paired archive. Neither is scientific
evidence.

## Fresh roster and result

All 24 branches completed without a recorded failure. The endpoint pass is the
preregistered durable test: candidate final validation loss no higher than the
matched no-op final loss.

| Stratum | Roots | Endpoint passes | Mean final Δloss | Geo wall | Geo FLOPs | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FineWeb-Edu, 85M | 3 | 3/3 | -0.00618 | 1.1328x | 1.1426x | 1.1429x |
| FineWeb-Edu, 139M | 3 | 2/3 | -0.00286 | 1.1318x | 1.1426x | 1.1429x |
| TinyStories, 85M | 3 | 3/3 | -0.02787 | 1.1312x | 1.1426x | 1.1429x |
| TinyStories, 139M | 3 | 3/3 | -0.02726 | 1.1316x | 1.1426x | 1.1429x |
| **All** | **12** | **11/12** | **-0.01604** | **1.1319x** | **1.1426x** | **1.1429x** |

The one miss is FineWeb-Edu 139M seed 9 (`Δloss=+0.000848`). Immediate
mean Δloss is `+0.000369`; recovery mean is `-0.03327`; final median Δloss is
`-0.01676`. Thus the durable effect is not an immediate-loss spike.

The raw candidate endpoint before snapshot averaging is worse than no-op on
all 12 roots: mean Δloss `+0.06072`. The final four-snapshot average moves
the result by mean `-0.07676`, leaving the reported post-merge mean at
`-0.01604`. Snapshot averaging is therefore the operative mechanism in this
screen; “early stopping alone” would be an inaccurate description.

The cost ratios include the shared prefix, target work, snapshot/merge work,
and evaluation. The token and FLOP savings are the planned 768-versus-512
continuation difference; the measured wall ratio is slightly lower than the
linear planned ratio, with mean prediction/reality gap `-0.0110x`.

The existing `0.995`, `0.99`, and `0.98` parent-relative thresholds are not
discriminating in this roster: all 36 baseline/candidate threshold pairs are
reached at the same first immediate point. Consequently this screen reports
endpoint equivalence and total cost, not a cost-to-threshold speedup.

## Model diagnostic

The existing small numerical ensemble was trained on nine complete roots and
held out on all three FineWeb-Edu 139M roots. It predicted the shadow action
as the raw final best action on 2/3 held-out roots, but its support gate marked
0/3 roots supported, so the deployable policy abstained on all three. The safe
selected mean was `0.0` versus the action-only prior `-0.00286`; final
prediction/reality RMSE was `0.00512` on the six final-horizon action rows.
This is a useful fail-closed result, not a successful selector.

## Decision

The fixed gate passes its local preregistered promotion screen (at least 2/3
in every stratum, zero failures, and positive conservative cost advantage), so
`trajectory_shadow_stop` is now an audited baseline/action family. It is only a
rough `1.13x` wall and `1.14x` FLOP/token mechanism, nowhere near 10x, and it
is not evidence that a learned driver can choose the action.

The next rung is a small cost-aware action gate over `noop`, full continuation,
and this shadow-stop macro-action, with parent timing varied before fitting.
Keep complete-root holdouts and the support-gated no-op fallback. Do not add a
larger steerer, planner, dreaming simulator, or RL until that model beats the
action-only prior on held-out action ranking and durable cost-to-quality.

## Reproduction

```bash
uv run python -m driver.shadow_gate_report --self-check
uv run python -m driver.shadow_gate_report \
  --results runs/shadow-gate-fineweb85-smoke-s21-v3 \
  --results runs/shadow-gate-fineweb85-s22-23-v2 \
  --results runs/shadow-gate-tinystories85-s9-11-v1 \
  --results runs/shadow-gate-tinystories139-s9-11-v1 \
  --results runs/shadow-gate-fineweb139-s9-11-v1 \
  --output runs/shadow-gate-report-v1.json

uv run python -m driver.response_atlas --self-check
uv run python -m driver.response_atlas --require-matched \
  --results runs/shadow-gate-fineweb85-smoke-s21-v3 \
  --results runs/shadow-gate-fineweb85-s22-23-v2 \
  --results runs/shadow-gate-tinystories85-s9-11-v1 \
  --results runs/shadow-gate-tinystories139-s9-11-v1 \
  --results runs/shadow-gate-fineweb139-s9-11-v1 \
  --output runs/shadow-gate-atlas-v1
```
