# Data allocation oracle results

Status: diagnostic oracle screen, 2026-09-19. This run tests only a complete
switch between the two local byte corpora. It is not a learned allocator and
does not close every possible within-corpus mixture or example-selection
policy.

## Protocol

From the exact step-1536 AdamW parent, run two matched 768-step branches:

1. recomputed no-op on the parent corpus;
2. the same optimizer and token budget, but with the other versioned corpus
   starting at its deterministic seed offset.

Both branches evaluate the original corpus's validation split at steps 128,
512, and 768. The branch consumes `12,582,912` target tokens. The checkpoint
data hash is validated against the source manifest before either branch runs;
the switched stream receives its own combined train/validation hash. The
manifest records the parent checkpoint, source hashes, cursors, all three
horizons, shared-prefix cost, target cost, evaluation cost, and wall time.

The harness is [`driver/data_allocation_oracle.py`](../driver/data_allocation_oracle.py).
The pilot used the balanced FineWeb-Edu 85M/139M and TinyStories 85M/139M
roots:

```bash
CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.data_allocation_oracle \
  --source-manifest runs/fineweb-edu-85m-seed5-late-action-set/manifest.json \
  --source-manifest runs/fineweb-edu-139m-seed5-history-action-set/manifest.json \
  --source-manifest runs/tinystories-85m-seed5-history-action-set/manifest.json \
  --source-manifest runs/tinystories-139m-seed3-history-action-set/manifest.json \
  --output runs/data-allocation-oracle-balanced-v2 --device cuda
```

## Results

Candidate deltas are switched-corpus loss minus the recomputed source-corpus
no-op loss. Negative is better. All four candidate branches used the same
token count and had a compute ratio of exactly `1.00x`; this is a capability
screen, not a cost reduction.

| Parent regime | Immediate delta | Recovery delta | Final delta | Final pass |
| --- | ---: | ---: | ---: | ---: |
| FineWeb-Edu 85M | `+0.45149` | `+0.60862` | `+0.61058` | no |
| FineWeb-Edu 139M | `+0.45698` | `+0.65187` | `+0.62973` | no |
| TinyStories 85M | `+0.53960` | `+0.57434` | `+0.63262` | no |
| TinyStories 139M | `+0.55236` | `+0.60112` | `+0.63668` | no |
| **median** | **`+0.49829`** | **`+0.60487`** | **`+0.63117`** | **0/4** |

The recomputed no-op final loss differed from the recorded source no-op by at
most `0.00339`, while the switched-data regressions were about `+0.61`. This
separates ordinary rerun noise from the intervention effect. There were no
failed branches or uncharged skipped tokens.

## Decision

Close the naive complete cross-corpus switch. A data allocator cannot treat
FineWeb-Edu and TinyStories as interchangeable future work under the current
validation objectives. This does not establish that a domain-balanced mixture,
within-corpus example utility policy, or context/work schedule has no value;
those are separate actions with a common task-aligned validation objective.
They should be tested only if a predeclared mixture or utility signal is
available. Do not train a data selector on this negative switch result.

Manifest: `runs/data-allocation-oracle-balanced-v2/manifest.json`.
