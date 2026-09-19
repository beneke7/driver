# Pre-parent low-rank transport oracle

Date: 2026-09-19

This is a diagnostic oracle-ceiling experiment, not a deployable driver
result. It asks whether a future parameter displacement is already captured by
the recent, pre-parent trajectory directions. The coefficients are fit with the
step-2048 future checkpoint, so they are hindsight and cannot be used by a
real policy.

## Protocol

Each branch starts from an exact AdamW parent at step 1536. The basis uses only
snapshots before that parent and is made from parent-centered secants:

```text
rank 1: theta_1536 - theta_1408
rank 2: theta_1408 - theta_1280, theta_1536 - theta_1408
rank 3: theta_1280 - theta_1152, theta_1408 - theta_1280,
        theta_1536 - theta_1408
```

In FP32, the hindsight future delta `theta_2048 - theta_1536` is projected into
each span with an SVD pseudoinverse. The projected parameters replace the
parent parameters; parent AdamW state is preserved. The data cursor is advanced
to step 2048, followed by 128 recovery steps and a step-2304 endpoint. The
projection work and the declared 512-step skipped exposure are recorded. All
branches retain oracle-only markers because future coefficients and future
parameters are used.

The primary roster is nine complete roots: three FineWeb-Edu 85M, three
FineWeb-Edu 139M, and three TinyStories 85M. The first aggregate rank-3 launch
hit a host-memory failure before producing a manifest; it is not counted as a
scientific result. After releasing CPU snapshots before the GPU branch, the
three strata completed independently without branch failures.

## Results

Positive final deltas mean the transported branch is worse than its matched
AdamW/no-op continuation.

| Basis | Roots | Mean residual future-delta energy | Mean final delta | Durable endpoint passes | Ideal FLOP ratio | Conservative FLOP ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rank 1 | 9 | `0.98245` | `+0.04784` | `0/9` | `1.2856x` | `1.0000x` |
| Rank 2 | 9 | `0.93519` | `+0.04309` | `0/9` | `1.2856x` | `1.0000x` |
| Rank 3 | 9 | `0.91867` | `+0.04620` | `0/9` | `1.2856x` | `1.0000x` |

Rank 3 remained negative in every stratum: mean final deltas were `+0.04620`
for FineWeb 85M, `+0.05132` for FineWeb 139M, and `+0.02706` for TinyStories
85M. The mean ideal wall ratio was about `1.22--1.28x`, but that apparent gain
comes from skipping 512 optimizer steps. Once the skipped work is charged,
the ratio is effectively `1.00x`, before assigning any real policy the cost of
learning or selecting the coefficients.

The residuals are also informative: even the rank-3 hindsight span leaves
about 92% of the future displacement energy outside the basis on average. More
importantly, fitting the best coefficients in hindsight still produces a
durable regression. This is not a representation-capacity result—the action
was given the future target—but evidence that recent raw update geometry is a
poor transport direction when the parent optimizer state and data exposure
are held as in this test.

## Decision

Close the pre-parent low-rank basis family as a promotion route. Do not train a
learned coefficient predictor, planner, dreaming simulator, or larger steerer
on top of this action. The result does not prove that every structured action
is futile; it says that this cheap hindsight ceiling is not large enough to
justify more model capacity.

The next admissible work must introduce a new predeclared causal action with a
common task objective and a conservative accounting advantage, or collect a
new action-conditioned atlas that changes the state/action hypothesis. Any
future transport attempt must model optimizer-state and data-state response
explicitly rather than extrapolating parameters alone.

## Reproduction artifacts

- [`driver/trajectory_low_rank_oracle.py`](../driver/trajectory_low_rank_oracle.py)
- [`runs/trajectory-low-rank-oracle-balanced-r1-v1/manifest.json`](../runs/trajectory-low-rank-oracle-balanced-r1-v1/manifest.json)
- [`runs/trajectory-low-rank-oracle-balanced-r2-v1/manifest.json`](../runs/trajectory-low-rank-oracle-balanced-r2-v1/manifest.json)
- [`runs/trajectory-low-rank-oracle-fineweb85-r3-v1/manifest.json`](../runs/trajectory-low-rank-oracle-fineweb85-r3-v1/manifest.json)
- [`runs/trajectory-low-rank-oracle-fineweb139-r3-v1/manifest.json`](../runs/trajectory-low-rank-oracle-fineweb139-r3-v1/manifest.json)
- [`runs/trajectory-low-rank-oracle-tinystories85-r3-v1/manifest.json`](../runs/trajectory-low-rank-oracle-tinystories85-r3-v1/manifest.json)

The self-check is:

```bash
uv run python -m driver.trajectory_low_rank_oracle --self-check
```
