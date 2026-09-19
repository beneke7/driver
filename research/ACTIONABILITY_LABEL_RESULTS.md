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

As a direct model gate, the existing numerical ensemble was retrained on
phase-switch seeds 9--12 and evaluated on complete roots 13--14. Both held-out
roots were inside the calibrated support radius, but final raw action top-1
agreement was `0/2`; the safe policy selected noop on both, with final
prediction/reality gap `0.0101`. This is a useful safe abstention, not a
successful selector. Reproduction:

```bash
uv run python -m driver.response_atlas_model \
  --atlas runs/response-atlas-phase-switch-long-v2/atlas.jsonl \
  --holdout-root 'phase_switch-13:5e5a6a3822f34e458d9dd909a8bc118a425742f9ff8649dd22849fd95fe6cfa6:fa4d8223f432b92f9587d79131446318cbdca44ffdabd32f6a712644a85a5185' \
  --holdout-root 'phase_switch-14:a626f127f872950cbfd50ba6bc3be2820ecb094cb6e588cbf15fc1129a83a183:8631083a79db0c6e29877c06b5aea445d47892e896245679991665fd7103941d' \
  --output runs/actionability-model-phase-switch-long-v1 --device cpu
```

The generated label manifests retain the input atlas checksums:

- `runs/actionability-adamw-expanded-v2/manifest.json`
- `runs/actionability-fineweb-timing-v1/manifest.json`
- `runs/actionability-fineweb-tiny-timing-v1/manifest.json`
- `runs/actionability-phase-switch-long-v1/manifest.json`
