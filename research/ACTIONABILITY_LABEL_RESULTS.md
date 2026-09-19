# Actionability label results

Status: label-corpus diagnostic, 2026-09-19. These labels summarize observed
matched branches; they are not predictions and do not promote an action.

The label definitions and the `1e-3` loss tolerance were committed in
[`DECODER_DRIVER_PREREGISTRATION.md`](DECODER_DRIVER_PREREGISTRATION.md)
before this corpus was generated. The labeler is
[`driver/actionability.py`](../driver/actionability.py). It keeps `noop` as a
separate baseline, marks a branch `jumpable` only when an observed early
horizon reaches the noop final loss at lower charged wall time, and never
fills missing outcomes.

## Existing atlas distributions

| Atlas | Groups | Rows | Noop | Productive | Jumpable | Recoverable | Dangerous | Neutral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW mixed expanded, 15 roots | 15 | 102 | 15 | 4 | 0 | 45 | 31 | 7 |
| FineWeb timing, 9 roots | 9 | 18 | 9 | 8 | 0 | 1 | 0 | 0 |
| FineWeb/Tiny timing, 12 roots | 12 | 24 | 12 | 11 | 0 | 1 | 0 | 0 |
| phase-switch long, 6 roots | 6 | 18 | 6 | 0 | 2 | 5 | 5 | 0 |

The two `jumpable` rows are the medium and large pulse on phase-switch seed
12. They are concentrated in one seed and one synthetic regime. The timing
atlases are intentionally not evidence of a transferable jump classifier:
their positive rows are the same fixed shadow mechanism evaluated across
roots.

Representative commands:

```bash
uv run python -m driver.actionability \
  --atlas runs/response-atlas-adamw-pilot-expanded-v2/atlas.jsonl \
  --output runs/actionability-adamw-expanded-v2

uv run python -m driver.actionability \
  --atlas runs/response-atlas-phase-switch-long-v2/atlas.jsonl \
  --output runs/actionability-phase-switch-long-v1
```

## Decision

Do not fit or deploy a productive-region classifier from these rows yet. The
mixed atlas has too few productive examples, and the apparent jumpable signal
does not transfer across roots. The labels are still useful as supervised
targets for the next causal collection: collect more matched branches within
one data regime, hold out complete roots, and require the action-conditioned
model to beat the action-only prior at the final horizon. No planner, RL, or
online adaptation is justified by this corpus.

The generated label manifests retain the input atlas checksums:

- `runs/actionability-adamw-expanded-v2/manifest.json`
- `runs/actionability-fineweb-timing-v1/manifest.json`
- `runs/actionability-fineweb-tiny-timing-v1/manifest.json`
- `runs/actionability-phase-switch-long-v1/manifest.json`
