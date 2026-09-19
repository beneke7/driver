# Trajectory transport oracle results

Status: diagnostic oracle ceiling, 2026-09-19. These runs are not a driver
claim. The future model state is selected from a recorded AdamW/no-op
trajectory with hindsight, so it is not available to a deployed policy.

## Question

Can a model be moved from a matched parent at global step 1536 to the recorded
future parameters at step 2048, then reach the recorded step-2304 capability
with only 256 real continuation steps? The branch preserves the parent
checkpoint's AdamW moments unless the explicit zero-moment control is selected.

The pilot uses raw snapshots from the existing
`trajectory_shadow_average` branches. Those snapshots are taken before the
endpoint merge and follow the ordinary no-op schedule. Each source manifest
and checkpoint retains its original data/configuration hashes.

## Protocol

For each case:

1. Load the immutable parent checkpoint at step 1536.
2. Replace parameters with the hindsight snapshot at step 2048.
3. Either preserve the parent data cursor or advance it by the declared
   512-step exposure.
4. Either preserve AdamW moments or zero all moments as a state-consistency
   control.
5. Measure transport, recovery after 128 further steps, and final loss after
   256 further steps.
6. Compare with the source no-op snapshots at steps 2048, 2176, and 2304.

The harness is [`driver/trajectory_transport_oracle.py`](../driver/trajectory_transport_oracle.py).
It validates the reconstructed data hash, records checkpoint and snapshot
hashes, and writes actual and conservative costs. The conservative ledger
charges the skipped 512-step exposure at ordinary target FLOPs; this is a
deliberate ceiling against hidden data skipping.

Commands used:

```bash
CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.trajectory_transport_oracle \
  --source-manifest runs/fineweb-edu-85m-seed3-late-action-set/manifest.json \
  --source-manifest runs/fineweb-edu-85m-seed4-late-action-set/manifest.json \
  --source-manifest runs/fineweb-edu-85m-seed5-late-action-set/manifest.json \
  --output runs/trajectory-transport-oracle-fineweb85-seed3-5 \
  --future-step 2048 --recovery-after 128 --device cuda

CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.trajectory_transport_oracle \
  --source-manifest runs/fineweb-edu-139m-seed2-late-action-set/manifest.json \
  --source-manifest runs/fineweb-edu-139m-seed5-history-action-set/manifest.json \
  --source-manifest runs/fineweb-edu-139m-seed6-history-action-set/manifest.json \
  --output runs/trajectory-transport-oracle-fineweb139-seed2-5-6 \
  --future-step 2048 --recovery-after 128 --cursor-policy skip \
  --optimizer-state-policy preserve --device cuda

CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.trajectory_transport_oracle \
  --source-manifest runs/tinystories-85m-seed3-history-action-set/manifest.json \
  --source-manifest runs/tinystories-85m-seed4-history-action-set/manifest.json \
  --source-manifest runs/tinystories-85m-seed5-history-action-set/manifest.json \
  --output runs/trajectory-transport-oracle-tiny85-seed3-5 \
  --future-step 2048 --recovery-after 128 --cursor-policy skip \
  --optimizer-state-policy preserve --device cuda
```

The first command also ran the preserve-cursor control. A separate command
ran the zero-moment control on the same three FineWeb 85M roots:

```bash
CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.trajectory_transport_oracle \
  --source-manifest runs/fineweb-edu-85m-seed3-late-action-set/manifest.json \
  --source-manifest runs/fineweb-edu-85m-seed4-late-action-set/manifest.json \
  --source-manifest runs/fineweb-edu-85m-seed5-late-action-set/manifest.json \
  --output runs/trajectory-transport-oracle-fineweb85-seed3-5-zero \
  --future-step 2048 --recovery-after 128 --cursor-policy skip \
  --optimizer-state-policy zero_moments --device cuda
```

## Results

The primary surviving variant is `cursor=skip, optimizer_state=preserve`.
Loss deltas are candidate final loss minus the matched no-op final loss; a
negative value is better.

| Target/data | Seeds | Passes | Final deltas | Mean wall speedup | Compute-only speedup | Conservative speedup |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 85M FineWeb-Edu | 3--5 | 2/3 | `-0.00314, -0.00234, +0.00040` | `1.21x` | `1.29x` | `1.00x` |
| 139M FineWeb-Edu | 2,5,6 | 2/3 | `-0.00367, -0.00358, +0.00117` | `1.24x` | `1.29x` | `1.00x` |
| 85M TinyStories | 3--5 | 3/3 | `-0.00362, -0.00341, -0.00238` | `1.21x` | `1.29x` | `1.00x` |
| **all** | **9** | **7/9** | median **`-0.00314`** | **`1.22x`** | **`1.29x`** | **`1.00x`** |

The source is nine roots, not a promotion roster. The results are a measured
oracle ceiling for this exact macro-action and horizon, not a robust training
efficiency result.

### State and cursor controls

On the 85M FineWeb roots, replaying the parent data cursor after copying the
future weights failed on all three roots, with final regressions of
`+0.06310`, `+0.05778`, and `+0.07189`. Advancing the cursor is therefore not
an accounting detail: the transported parameters expect the later data
exposure.

Zeroing AdamW moments while advancing the cursor failed catastrophically on
all three roots. Final regressions were `+0.82023`, `+0.18403`, and
`+1.02574`. Preserving the parent moments was the only viable tested state
policy, but it is not a principled future-state transform. A useful learned
transport must predict or safely transform optimizer state as well as
parameters.

## Decision

This oracle does not justify a large policy, RL, or a pretrained meta-brain.
It establishes three narrower facts:

- a future-state transport opportunity exists at this horizon, but its
  durable benefit is only a few thousandths of validation loss;
- data cursor and optimizer state are causally coupled to the parameter
  transport;
- after conservative accounting for the 512 skipped exposure steps, this
  action has no compute-efficiency gain.

The direct full-weight teleportation route is therefore closed as a
10x mechanism. The next admissible rung is a small learned structured
transport model that predicts role-wise parameter and moment deltas from a
short calibration history, with no skip or planner claim until it beats this
oracle's real-branch baseline under the same ledger. If it cannot reproduce
the measured action ranking and state consistency, stop transport work and
pivot to data/work allocation or initialization.

## Learned role-wise probe

The first non-oracle action used the only basis available from a short
calibration history: the parameter difference between steps 1408 and 1536.
It learned one scalar coefficient per tensor role from training roots and
applied that structured delta at step 1536. The candidate then advanced the
data cursor to step 2048, preserved AdamW moments, and continued for 256
steps. The future checkpoint was never loaded by the candidate.

The FineWeb-85M training coefficients were:

```text
embedding 2.5959   attention 0.1368   mlp 0.3499   norm 3.2107   head 0.7382
```

Held-out results:

| Training roots | Held-out roots | Passes | Final deltas |
| --- | --- | ---: | --- |
| FineWeb 85M seeds 3--5 | TinyStories 85M seeds 3--5 | 0/3 | `+0.03197, +0.03227, +0.03058` |
| FineWeb 85M seeds 3--4 | FineWeb 85M seed 5 | 0/1 | `+0.05432` |

The hindsight future-weight oracle passed 3/3 of those TinyStories roots, so
this is a failure of the available action basis/policy, not evidence that the
TinyStories target is intrinsically untransportable. The layer-wise audit
explains the gap: a single recent basis captures much of embedding/norm future
energy but only about 4--13% of attention/MLP future energy. Adding one scalar
per layer raises MLP explained energy to only about 5--6% on the tested roots;
attention remains near 1%. The parent AdamW preconditioned moment direction
adds only about 1--3% explained attention/MLP energy.

This closes the single-recent-delta role-wise policy as a promotion path. Do
not scale its model. A richer action basis would need explicitly measured
layer/gradient/curvature directions and a new costed probe; otherwise the
next efficient hypothesis is data/work allocation rather than parameter
transport.

The generated manifests are intentionally ignored run artifacts:

- `runs/trajectory-transport-oracle-fineweb85-seed3-5/manifest.json`
- `runs/trajectory-transport-oracle-fineweb85-seed3-5-zero/manifest.json`
- `runs/trajectory-transport-oracle-fineweb139-seed2-5-6/manifest.json`
- `runs/trajectory-transport-oracle-tiny85-seed3-5/manifest.json`
