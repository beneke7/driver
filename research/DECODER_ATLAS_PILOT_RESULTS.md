# Decoder response-atlas pilot

Date: 2026-09-19. The runs below are development evidence, not a promotion
claim. The contract-valid 10x gate remains closed.

## Coverage

All branches start from immutable matched checkpoints and report immediate,
recovery, and final horizons. The full pilot used the 85M decoder configuration
(`width=768`, `layers=12`, `context=128`, `batch_size=128`, 1536 prefix steps,
64 immediate, 256 recovery, 512 final) with AdamW. The three complete regimes
have three seeds each:

| Regime | Roots | Rows | Actions per root | Failures |
| --- | ---: | ---: | ---: | ---: |
| FineWeb-Edu | 3 | 24 | 8 | 0 |
| TinyStories | 3 | 24 | 8 | 0 |
| delayed-copy | 3 | 24 | 8 | 0 |
| FineWeb expansion | 3 | 15 | 5 | 0 |

The expanded atlas contains 12 roots and 87 rows. Every row has matched no-op
outcomes and an additive cost ledger. Pulse multipliers, vocabulary size, data,
configuration, source checkpoint, code, and seed are recorded in provenance.

## Final response deltas

`Δloss` is the action final validation loss minus the matched no-op final loss;
negative is better. These are equal-token branches, so they measure response,
not a speedup.

| Regime | small pulse | medium pulse | large pulse | damping |
| --- | ---: | ---: | ---: | ---: |
| FineWeb, 6 roots | -0.00018 (4/6 wins) | +0.00007 (2/6) | -0.00090 (5/6) | +0.00016 (2/6) |
| TinyStories, 3 roots | -0.00003 (2/3) | -0.00018 (2/3) | -0.00014 (1/3) | +0.00076 (1/3) |
| delayed-copy, 3 roots | +0.00816 (1/3) | +0.00675 (2/3) | +0.00802 (1/3) | +0.00083 (1/3) |

The apparent FineWeb pulse effect is small and not transferable: all pulse
families are immediately worse on average, with large-pulse immediate deltas
of `+0.0295` on FineWeb and `+0.0185` on TinyStories. Trajectory averaging is
worse on both language regimes; optimizer-state-reset extrapolation is harmful
on every language root and every delayed-copy root in this pilot.

## Atlas-model gate

The model is a small numerical ensemble with a conservative support gate. It is
not a planner and no imagined transition is counted as evidence.

| Split | Support | Final prediction RMSE | Selected final delta | Action-only prior | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb seeds 7–8 held out from six roots | 2/2 | 0.00578 | +0.00168 | +0.00112 | -0.00059 |
| FineWeb → TinyStories | 0/3 | 0.11386 | 0.00000 (abstain) | -0.00014 | -0.00050 |
| delayed-copy-only, seed 5 held out from two roots | 0/1 | 0.01678 | 0.00000 (abstain) | +0.02455 | -0.00067 |

The FineWeb split is the decisive no-go for a selector: being inside the
support radius did not make final-horizon action ranking better than the
action-only prior. The safe fallback correctly abstained on the unsupported
changed-regime and delayed-copy cases. No MPC, dreaming, online learning, or
RL rung is justified yet.

The model representation was then expanded with telemetry already present in
the archive: loss slope, batch context, role-specific gradient/update norms,
and the role-specific histories. On the held-out FineWeb split this reduced
the selected final delta from `+0.00112` to `+0.00038`, but on held-out
TinyStories the safe policy still abstained while the action-only prior was
`-0.00161`. The feature change is retained as a diagnostic, not as evidence
of a transferable selector.

## Reproduction

The core artifacts are ignored run outputs so large checkpoints do not enter
Git. Recreate the normalized atlas with:

```bash
.venv/bin/python -m driver.response_atlas \
  --require-matched \
  --results runs/fineweb-edu-85m-adamw-atlas-pilot-seed3-5 \
  --results runs/tinystories-85m-adamw-atlas-pilot-seed3-5 \
  --results runs/delayed-copy-85m-adamw-atlas-pilot-seed3-5 \
  --results runs/fineweb-edu-85m-adamw-pulse-expansion-seed6-8 \
  --output runs/response-atlas-adamw-pilot-expanded-v1
```

The next rung is more roots and better final-horizon representation/action
ranking. Do not turn the current small FineWeb pulse effect into a speedup
claim or deploy the selector until it beats the action-only prior on fresh
roots with durable outcomes and charged costs.

## Phase-switch extension

The existing synthetic `phase_switch` landscape was run with the same 85M
AdamW target and six fresh roots, using noop, small/medium/large role pulses,
and damping. Every branch completed. Across the six roots, the medium pulse
improved all six final branches (mean final delta `-0.00434`, zero durable
regressions); the large pulse improved five of six (mean `-0.00581`, one
durable regression). Both also improved immediate and recovery loss on
average. This is a durable equal-token response, not a speedup.

A model trained on phase roots 3--6 and tested on roots 7--8 selected the
large pulse at all three horizons and matched the realized best action on both
roots. It matched the fixed action prior, so this is transfer of a stable
macro-action rather than evidence of state-dependent gain.

Three fresh long-horizon roots (seeds 9--11; 128 immediate, 512 recovery,
2048 final) give a useful speed diagnostic. The medium pulse reaches the
matched noop branch's final validation loss by recovery on all three roots and
remains at or below it at final. Including the common prefix, the per-root
noop-final to medium-recovery ratios are approximately `1.75x` by both FLOPs
and wall time. This is one landscape and one threshold, not a contract-valid
promotion result. The large pulse is unsafe for this horizon: it improves
seeds 9 and 11 but regresses seed 10 by `+0.281` final loss, while the medium
pulse improves seed 10 by `-0.450`.

Adding long-horizon seeds 12--14 closes the apparent speed result. Across six
long roots, medium wins 4/6 final branches with mean delta `-0.0757` but has
two durable regressions; large wins 3/6 with mean delta `+0.0387` and three
durable regressions. The fresh seeds 13--14 regress under both pulse sizes.
The horizon-matched model trained on seeds 9--12 and held out on 13--14 has
support on both roots but final raw top-1 agreement `0` and prediction/reality
gap `0.0474`; its safe selected mean is `+0.00065`, versus `+0.00226` for the
action prior and `0` for the noop oracle. The fixed-pulse speed route is a
no-go until a different state/action representation explains these failures.

### Position observability ablation (2026-09-19)

The atlas state was missing the parent checkpoint's absolute `step` and
`tokens`. Adding those already-recorded fields is a minimal observability fix
for phase-dependent dynamics; it does not add a new sensor or action. On the
same long-horizon split (train roots 9--12, held-out roots 13--14), final
prediction RMSE fell from `0.0560` to `0.01496`. However, raw final action
top-1 agreement remained `0/2`, and the uncertainty-gated selector chose
no-op on both held-out roots. Its safe selected mean was `0`, versus `0` for
the noop oracle and `+0.00226` for the fixed action prior. The feature fixes a
real state omission, but does not justify a planner or reopen the pulse route.

An action-size grid on the two failure roots confirms the ceiling. On seed 13,
small/medium/large pulses have immediate deltas `-0.00128/-0.00583/-0.01089`
but final deltas `+0.00025/+0.00173/+0.00264`; on seed 14 the corresponding
final deltas are `+0.00234/+0.00403/+0.00364`. Every pulse is a durable
regression on both roots despite an initially favorable response. Immediate
loss is therefore not a safe proxy for long-horizon action value in this
landscape.
