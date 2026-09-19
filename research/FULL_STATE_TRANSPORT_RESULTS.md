# Complete-state transport sanity results

Status: diagnostic state-consistency control, 2026-09-19. This is not a
driver or speedup result.

## Question

Does the repository's complete checkpoint primitive preserve enough state to
resume ordinary AdamW training from a future point? The earlier transport
oracle copied only future model weights and therefore had an explicit
`future_optimizer_state_unavailable` violation.

## Protocol

For each balanced root, the harness:

1. loads the immutable step-1536 parent;
2. runs ordinary AdamW/no-op to step 2048;
3. saves a complete checkpoint with model, optimizer, RNG, data cursor,
   configuration/data hashes, code revision, and parent hash;
4. continues directly to step 2304;
5. loads the complete step-2048 checkpoint into a fresh model/optimizer and
   resumes the same 256 steps;
6. compares future, recovery, and final validation losses and records ideal
   versus conservative FLOP ratios.

The harness is [`driver/full_state_transport_sanity.py`](../driver/full_state_transport_sanity.py).
It uses the existing `save_checkpoint`/`load_checkpoint` contract and writes
one ignored complete checkpoint per root.

```bash
CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.full_state_transport_sanity \
  --source-manifest runs/fineweb-edu-85m-seed5-late-action-set/manifest.json \
  --source-manifest runs/fineweb-edu-139m-seed5-history-action-set/manifest.json \
  --source-manifest runs/tinystories-85m-seed5-history-action-set/manifest.json \
  --source-manifest runs/tinystories-139m-seed3-history-action-set/manifest.json \
  --output runs/full-state-transport-sanity-balanced-v1 --device cuda
```

## Results

All four roots completed without execution failures. The resumed future loss
matched the direct future loss exactly in the recorded precision on all four
roots. Subsequent differences were small CUDA/AMP replay drift:

| Root | Maximum absolute loss gap | Ideal compute-only ratio | Conservative ratio |
| --- | ---: | ---: | ---: |
| FineWeb-Edu 85M | `0.000214` | `1.286x` | `1.000145x` |
| FineWeb-Edu 139M | `0.000891` | `1.286x` | `1.000145x` |
| TinyStories 85M | `0.000069` | `1.286x` | `1.000145x` |
| TinyStories 139M | `0.000318` | `1.286x` | `1.000145x` |

The harness's deliberately strict `1e-5` loss-replay check therefore reports
`0/4`; that is a numerical reproducibility threshold, not a checkpoint-load
failure. The independent recomputed-noop control in the data-allocation pilot
varied from its recorded endpoint by up to `0.003392`, larger than every
complete-state resume gap. Checkpoint metadata and future data cursors were
present and validated for all four roots.

The ideal ratio assumes the future complete state is free. It is not a valid
deployment ratio: creating that state required the 512 preceding training
steps. Once those steps are conservatively charged, the ratio is approximately
`1.00x`. A complete future state reproduces ordinary continuation; it does not
create a cheap transport opportunity.

Manifest: `runs/full-state-transport-sanity-balanced-v1/manifest.json`.

## Decision

The earlier optimizer-state mismatch is now resolved as a provenance issue,
not a hidden source of leverage. Future model weights plus parent moments were
an invalid partial-state action; complete future state is consistent but
cost-neutral under the contract. Do not promote or train a larger transport
operator on this result. Any learned transport must predict a causal,
cost-justified state transition without assuming the future checkpoint was
already produced.
