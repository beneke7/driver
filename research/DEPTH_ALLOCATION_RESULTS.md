# Upper-layer work-allocation results

Date: 2026-09-20. This closes the preregistered fixed half-depth action as a
promotion route. It is not a driver or 10x result.

## Result

The candidate ran the first 384 of 768 continuation steps through 6 of 12
transformer blocks, then restored all 12 blocks. It consumed exactly the same
training tokens as the matched AdamW/no-op branch and preserved the parent
optimizer/data state. All 18 branches across the three timing strata finished
without failure.

| Parent step | Immediate mean delta | Recovery mean delta | Final mean delta | Durable-safe roots | End-to-end wall geo | End-to-end FLOP geo |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 768 | +0.143624 | +0.074385 | +0.056448 | 0/3 | 1.1277× | 1.1418× |
| 1280 | +0.167858 | +0.040126 | +0.032213 | 0/3 | 1.0928× | 1.1027× |
| 1792 | +0.208221 | +0.036517 | +0.023216 | 0/3 | 1.0728× | 1.0805× |

The per-root final deltas were:

```text
parent 768:  +0.059516  +0.060463  +0.049366
parent 1280: +0.031790  +0.034460  +0.030389
parent 1792: +0.022519  +0.023640  +0.023487
```

The branch-only measurements showed approximately 1.29× wall and 1.33× FLOP
ratios. Including the shared prefix, as required by the contract, reduced the
ratios to the table above. Token exposure was exactly equal in every pair.
The action was therefore a real compute-saving mechanism, but it was durably
worse at every timing point and failed the `+1e-3` all-horizon safety rule on
all 9 candidate roots.

## Decision

Close this fixed `6/12 for 384 steps` depth-curriculum action. Do not fit a
selector or claim a speedup. The result is consistent with an upper-layer
state/re-entry mismatch: later full-depth training reduced, but did not remove,
the regression. The broader work-allocation idea is not disproven, but any
future depth schedule needs a new preregistration that changes the state
handling or objective rather than tuning this result post hoc.

## Reproduction

The preregistration is
[`DEPTH_ALLOCATION_PREREGISTRATION.md`](DEPTH_ALLOCATION_PREREGISTRATION.md).
The three manifests are:

- `runs/depth-allocation-fineweb85-p768-s33-35-v1/manifest.json`
- `runs/depth-allocation-fineweb85-p1280-s33-35-v1/manifest.json`
- `runs/depth-allocation-fineweb85-p1792-s33-35-v1/manifest.json`

Each contains matched transitions, full parent checkpoints, action parameters,
per-step active-layer telemetry, code/config/data hashes, and explicit cost
components.
