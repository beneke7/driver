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
| Pre-parent low-rank hindsight transport | 9 roots × ranks 1/2/3 | 0/27 endpoint passes; rank-3 residual energy `0.91867`, final delta `+0.04620` | `1.22--1.28x` ideal wall / `1.2856x` ideal FLOPs; `1.00x` after skipped exposure | Close pre-parent basis family |
| Complete future checkpoint | 4 balanced roots | Resume loss within `0.000891` of direct continuation; strict bitwise-style check 0/4 | Ideal `1.286x`, conservative `1.000145x` | State consistency only |
| Complete cross-corpus switch | 4 balanced roots | 0/4; median final delta `+0.63117` | Equal tokens and exact `1.00x` cost | Close coarse data switch |
| Disjoint trajectory-anchor initialization | 85M FineWeb-Edu/TinyStories, 6 preserve pairs | 2/6 endpoint wins; combined mean final delta `+0.00023` | Median hard-threshold wall ratio `1.05x` at one deployment; source checkpoint-write time unavailable | Close current same-corpus anchor screen |
| Same-corpus 25% block mixture | FineWeb 85M/139M, 6 roots | 6/6 durable equal-budget wins; final mean delta `-0.01252` | Exact `1.00x` FLOPs and approximately `1.00x` wall; TinyStories transfer 0/3 final wins | Fixed FineWeb control only; close as general driver route |
| Fixed shadow stop | Exact matched 85M/139M FineWeb-Edu and TinyStories screen, 12 roots | 11/12 durable endpoint passes; 0 failures; 4/4 strata meet 2/3 rule | `1.1319x` geometric wall / `1.1426x` FLOP / `1.1429x` token ratio including prefix; thresholds non-discriminating | Audited fixed baseline/action family only |
| Action-conditioned atlas ensemble | Phase-switch seeds 13--14 | Support 2/2, final top-1 `0/2`, safe noop both | Final prediction/reality gap `0.0101` | Gate closed |
| Timing-conditioned shadow gate pilot | FineWeb-Edu 85M, parent steps 768/1280/1792, seeds 21--26 | Complete-seed final holdouts select noop early and shadow late `6/6`; recovery selection regresses on 2/3 fresh seeds | Diagnostic selected wall ratio `1.051x`; final prediction gaps `0.0015--0.0019` | Promising small gate; multi-horizon deployment closed |
| Changed-regime timing transfer | FineWeb 85M train; FineWeb 139M/TinyStories holdout, 9 groups | Support `0/9`, safe noop `9/9`, raw final top-1 `4/9` | Final prediction/reality gap `0.00431`; oracle mean `-0.01942` | Geometry does not transfer; collect calibration data |
| Changed-regime calibration | FineWeb 139M/TinyStories, 9 groups; leave-one-seed-out | Final top-1 `2/3`, `3/3`, `3/3`; pooled selected mean `-0.01552` vs prior `-0.01933` | Recovery selected branches safe; prediction gaps `0.0027--0.0037` | No selector gain; do not add capacity |

The shadow row is retained as a useful fixed baseline; its exact charged
ratios and roster are in
[`PLAN.md`](PLAN.md) and the timing atlas reports. It is not combined with
the transport oracle rows.

## Decision

The current action space does not contain evidence for a cheap transferable
10x mechanism. A future state can be copied only with hindsight, partial
state copies are not durable, the pre-parent low-rank hindsight span leaves
most of the future displacement unexplained and still regresses, complete
state copies are cost-neutral once the state-creation work is charged, and a
coarse data-domain switch is harmful.
The disjoint same-corpus anchor screen adds only mixed, near-neutral evidence:
2/6 preserve-state endpoint wins with a combined mean final delta of `+0.00023`
and a median hard-threshold wall ratio of `1.05x` for one deployment. The
rank-1/2/3 low-rank screen adds 0/27 durable endpoint passes across nine roots;
its apparent `1.2856x` FLOP ratio becomes `1.00x` when the skipped exposure is
charged.
The fixed 25% same-corpus mixture is a useful FineWeb equal-budget control:
6/6 roots improve durably, but it has exact equal cost and its TinyStories
transfer regresses at the final horizon on 3/3 roots. It therefore does not
provide a portable action ceiling. The safe action-conditioned model abstains
when its final-horizon predictions
are not reliable, including states that were inside its geometric support.

The exact shadow-stop gate is the strongest current fixed mechanism. Its
candidate branch runs 512 continuation steps and averages the final four
trajectory snapshots, versus 768 no-op steps from the same 1280-step parent.
It passes 11/12 fresh endpoint comparisons across two widths and two data
sources, but the raw pre-average candidate is worse on all 12 roots. The
reported durable result is therefore a trajectory-averaging macro-action, not
free early stopping. The existing three parent-relative thresholds all reach
at the first immediate point for both branches, so no threshold-cost claim is
made. The full accounting and paired hashes are in
[`SHADOW_GATE_RESULTS.md`](SHADOW_GATE_RESULTS.md).

Adding parent timing creates the first useful state-dependent separation: the
same action loses at 768 steps and wins at 1280 and 1792. A small response
model reproduces that final-horizon ranking on complete-seed holdouts, but it
selects a late action with positive recovery loss on two of three fresh seeds.
The model therefore earns a conservative risk-gate experiment, not a deployed
driver or a larger steerer. Results are in
[`SHADOW_TIMING_GRID_RESULTS.md`](SHADOW_TIMING_GRID_RESULTS.md).

The changed-regime transfer check is closed for deployment. Training on the
nine FineWeb 85M timing roots produced zero supported groups on the held-out
FineWeb 139M and TinyStories roots, so the safe policy abstained everywhere;
raw final ranking was only 4/9. This validates the support gate but does not
validate transfer. A small changed-regime calibration set is required before
retesting the selector.

The changed-regime calibration audit supplies a small within-regime fit but
still loses to the action-only prior when pooled. It is a useful safety result,
not a driver result; a future attempt must improve the cost-aware, all-horizon
objective before model capacity is increased.

Therefore:

- do not train a larger steerer, pretrained meta-brain, planner, dreaming
  simulator, PPO/SAC policy, or online core updater;
- keep the existing noop, action-only, fixed-shadow, and numerical response
  models as controls;
- do not report any ideal/free-state ratio as a deployment gain;
- retain the 10x contract unchanged.

## Next-rung rule

Continue only with a new, predeclared causal action family that has a common
task objective and a plausible conservative ceiling, or with fresh disjoint
data if revisiting initialization. The current anchor screen did not show a
durable improvement on at least two of three roots within either data regime
and did not materially exceed the fixed-shadow baseline. Close this
same-corpus anchor and pre-parent low-rank families before adding model
capacity. If a future action family does show a ceiling, train the smallest
action-conditioned model and evaluate complete held-out roots before any
planner or RL.

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
- `runs/trajectory-anchor-adaptation-85-balanced-v1/manifest.json`
- `runs/trajectory-anchor-adaptation-85-zero-moments-seed3-v1/manifest.json`
- `runs/trajectory-anchor-adaptation-85-source-rng-seed3-v1/manifest.json`
- `runs/data-mixture-oracle-fineweb-balanced-v1/manifest.json`
- `runs/data-mixture-oracle-tinystories85-transfer-v1/manifest.json`
- `runs/actionability-model-phase-switch-long-v1/report.json`
- `runs/trajectory-low-rank-oracle-balanced-r1-v1/manifest.json`
- `runs/trajectory-low-rank-oracle-balanced-r2-v1/manifest.json`
- `runs/trajectory-low-rank-oracle-fineweb85-r3-v1/manifest.json`
- `runs/trajectory-low-rank-oracle-fineweb139-r3-v1/manifest.json`
- `runs/trajectory-low-rank-oracle-tinystories85-r3-v1/manifest.json`
- `runs/shadow-gate-report-v1.json`
- `runs/shadow-gate-atlas-v1/manifest.json`
- `runs/shadow-timing-grid-report-v1.json`
- `runs/shadow-timing-grid-atlas-v1/manifest.json`
- `runs/shadow-timing-transfer-atlas-v1/manifest.json`
- `runs/shadow-timing-transfer-model-v1/report.json`
- `runs/changed-calibration-model-seed9-v1/report.json`
- `runs/changed-calibration-model-seed10-v1/report.json`
- `runs/changed-calibration-model-seed11-v1/report.json`
