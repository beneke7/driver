# Oracle-ceiling decision

Date: 2026-09-19. This is the decision point required before scaling a
transport driver. The figures below are diagnostic ceilings or mechanism
controls; none is a contract-valid 10x deployment result.

## Evidence matrix

| Route | Coverage | Durable result | Valid/conservative cost signal | Decision |
| --- | --- | --- | --- | --- |
| Hindsight future weights, parent moments | 85M/139M FineWeb-Edu and 85M TinyStories, 9 roots | 7/9 endpoint passes; median final delta `-0.00314` | `1.22x` wall / `1.29x` compute-only, `1.00x` after skipped exposure | No 10x ceiling |
| Role-wise hindsight weights | 4 balanced roots × 5 roles | 0/20 endpoint passes; attention/MLP only transiently better | `1.22x` only from declared skip; `1.00x` conservative | Close role-copy route |
| Learned one-basis role transport | FineWeb-to-Tiny and leave-one-root-out | 0/3 and 0/1 endpoint passes | Equal-token branches; no durable gain | Close single-basis policy |
| Complete future checkpoint | 4 balanced roots | Resume loss within `0.000891` of direct continuation; strict bitwise-style check 0/4 | Ideal `1.286x`, conservative `1.000145x` | State consistency only |
| Complete cross-corpus switch | 4 balanced roots | 0/4; median final delta `+0.63117` | Equal tokens and exact `1.00x` cost | Close coarse data switch |
| Fixed shadow stop | 85M/139M and two data sources | Durable mechanism at tested roots | About `1.13x` wall / `1.08x` charged FLOP speedup in prior report | Baseline only, not driver |
| Action-conditioned atlas ensemble | Phase-switch seeds 13--14 | Support 2/2, final top-1 `0/2`, safe noop both | Final prediction/reality gap `0.0101` | Gate closed |

The shadow row is retained as a useful fixed baseline; its exact charged
ratios and roster are in
[`PLAN.md`](PLAN.md) and the timing atlas reports. It is not combined with
the transport oracle rows.

## Decision

The current action space does not contain evidence for a cheap transferable
10x mechanism. A future state can be copied only with hindsight, partial
state copies are not durable, complete state copies are cost-neutral once the
state-creation work is charged, and a coarse data-domain switch is harmful.
The safe action-conditioned model abstains when its final-horizon predictions
are not reliable, including states that were inside its geometric support.

Therefore:

- do not train a larger steerer, pretrained meta-brain, planner, dreaming
  simulator, PPO/SAC policy, or online core updater;
- keep the existing noop, action-only, fixed-shadow, and numerical response
  models as controls;
- do not report any ideal/free-state ratio as a deployment gain;
- retain the 10x contract unchanged.

## Next-rung rule

Continue only with a new, predeclared causal action family that has a common
task objective and a plausible conservative ceiling: for example an
in-domain data/work schedule or a few-shot initialization/seed-screening
action. Run a small matched oracle first. If it does not show a durable
improvement on at least two of three fresh roots and a conservative
cost-to-threshold opportunity materially above the fixed-shadow baseline,
close that family before adding model capacity. If it does show a ceiling,
train the smallest action-conditioned model and evaluate complete held-out
roots before any planner or RL.

This is a no-go for the current transport hypothesis as a 10x mechanism, not
a claim that every possible initialization or task-aligned allocation has
been exhausted.

## Reproduction artifacts

- `runs/trajectory-transport-oracle-fineweb85-seed3-5/manifest.json`
- `runs/trajectory-transport-oracle-fineweb139-seed2-5-6/manifest.json`
- `runs/trajectory-transport-oracle-tiny85-seed3-5/manifest.json`
- `runs/trajectory-transport-role-oracle-balanced-v2/manifest.json`
- `runs/full-state-transport-sanity-balanced-v1/manifest.json`
- `runs/data-allocation-oracle-balanced-v2/manifest.json`
- `runs/actionability-model-phase-switch-long-v1/report.json`
