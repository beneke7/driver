# Disjoint trajectory-anchor initialization screen

Date: 2026-09-19

This is an oracle/knowledge-transfer diagnostic. It is not an optimizer-speed
claim and is not eligible for the 10x promotion gate.

## Hypothesis and controls

The anchor imports a complete step-1536 model and AdamW state from a different
seed in the same 85M architecture/data regime. The target noop control and
anchor then consume the same predeclared byte interval, beginning at byte
offset `26,000,000`. That interval is outside the source checkpoint's
consumed prefix and is recorded with a hash in each manifest. This is
same-corpus reusable initialization, not fresh-data transfer: the source and
target prefixes are historical rotations of the same corpus files.

The primary condition restores the target RNG after loading the source
checkpoint. The controls use the source RNG and reset AdamW moments plus the
optimizer step clock. All branches use the same target continuation bytes,
validation data, horizons (`128/512/768`), and fixed thresholds (`0.995`,
`0.99`, `0.98` times the target-parent validation loss).

## Balanced preserve-state result

Run: [`manifest.json`](../runs/trajectory-anchor-adaptation-85-balanced-v1/manifest.json)

Six cyclic source/target pairs completed without execution failures:

| Regime | Endpoint wins | Mean final anchor-minus-noop loss | Median hard-threshold wall ratio, N=1 |
| --- | ---: | ---: | ---: |
| FineWeb-Edu, 3 pairs | 1/3 | `+0.000637` | `1.10x` |
| TinyStories, 3 pairs | 1/3 | `-0.000176` | `1.00x` |
| Combined | 2/6 | `+0.000230` | `1.05x` |

The observed final deltas range from `-0.01031` to `+0.00786`. The anchor
does not consistently reach the fixed thresholds earlier. The single-pair
wall ratios near `1x` are expected: both branches pay the full target
continuation and the source prefix is charged rather than treated as free.
Larger hypothetical deployment counts can make the amortization curve look
large, but they are not measured deployments and cannot establish the
contract gate.

## State and RNG controls

| Control | Artifact | Pairs | Endpoint wins | Interpretation |
| --- | --- | ---: | ---: | --- |
| Zero moments and reset step clock, target RNG | [`manifest.json`](../runs/trajectory-anchor-adaptation-85-zero-moments-seed3-v1/manifest.json) | 2 | 2 | Small seed-3 diagnostic; not a replication of the preserve-state effect |
| Preserve state, source RNG | [`manifest.json`](../runs/trajectory-anchor-adaptation-85-source-rng-seed3-v1/manifest.json) | 2 | 2 | RNG policy changes the observed deltas; no isolated RNG conclusion |

On the shared seed-3 pairs, preserve-state/target-RNG deltas were `-0.00697`
(FineWeb) and `-0.01031` (TinyStories). The zero-moment control produced
`-0.00835` and `-0.00317`; the source-RNG control produced `-0.00669` and
`-0.00989`. These are too few and too noisy to identify a useful transferred
state component.

## Accounting and limitations

The evaluator charges target/source prefix training FLOPs, continuation work,
evaluation, adaptation-data loading, and deployment checkpoint loads. The
historical source `prefix_seconds` field does not include the original parent
checkpoint write, so the manifest marks that missing quantity explicitly and
the wall ratios are lower bounds. No ratio is promoted as contract-valid.

The route therefore does not expose a conservative, transferable 10x ceiling:
it has only 2/6 balanced endpoint wins, approximately neutral single-use cost,
and no demonstrated threshold advantage that survives the controls. Do not
add a larger initialization steerer or planner on this evidence. The next
admissible work must use a new causal action family or collect fresh,
disjoint-data source/target deployments with complete checkpoint-I/O
accounting.

## Reproduction

```bash
uv run python -m driver.trajectory_anchor_adaptation \
  --source-manifest runs/fineweb-edu-85m-seed3-late-action-set/manifest.json \
  --source-manifest runs/fineweb-edu-85m-seed4-late-action-set/manifest.json \
  --source-manifest runs/fineweb-edu-85m-seed5-late-action-set/manifest.json \
  --source-manifest runs/tinystories-85m-seed3-history-action-set/manifest.json \
  --source-manifest runs/tinystories-85m-seed4-history-action-set/manifest.json \
  --source-manifest runs/tinystories-85m-seed5-history-action-set/manifest.json \
  --output runs/trajectory-anchor-adaptation-85-balanced-v1 \
  --optimizer-state-policy preserve --rng-policy target --device cuda
```

