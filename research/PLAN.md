# Initial research plan

## Hypothesis

Training trajectories contain observable, repeatable opportunities for a
bounded intervention to reduce cost to the same held-out capability. The first
test is not whether a large controller can imagine a training universe; it is
whether matched checkpoint branches expose a useful state-dependent choice.

## Staged route

1. Profile one target loop end to end: tokens/s, memory, optimizer time,
   checkpoint I/O, evaluation, telemetry, and driver-call overhead.
2. Build a modern baseline recipe and freeze the target architecture, data
   version, objective, evaluator, and branch protocol.
3. Compare `noop` with a bounded `role_pulse` from identical checkpoints.
4. Add fixed, random, task-greedy, and curiosity-only selector controls. Keep
   the observed-action oracle diagnostic only; it is not a deployable result.
5. Add the cheap online action-statistics baseline. Only after it has signal,
   replace it with a shared numerical history model and a small per-run
   adapter.
6. Test structured corrections first, then adaptive weight nowcasting, then
   data/work allocation. Keep the three routes separately attributable.
7. Add short imagined rollouts only when action-conditioned predictions are
   calibrated on fresh real branches.

## Branch contract

Every branch must preserve or identify:

- model weights and optimizer/scheduler/scaler state;
- random-generator state, data cursor, tokenizer and objective;
- code, configuration, and data hashes;
- starting checkpoint and parent transition;
- action, horizon, tokens, wall time, estimated FLOPs, losses, quality,
  recovery cost, failure status, and selector metadata.

`driver/checkpoints.py` now provides the matched-branch primitive, and its
torch self-check verifies model, optimizer, CPU/CUDA RNG, and hash validation.
The quadratic mechanism benchmark still uses synthetic initial states rather
than full target-model checkpoints; the decoder experiment must use this
primitive before any intervention result is considered causal.

`driver/quadratic_benchmark.py` now supplies a contract-grade optimization
track for five synthetic positive-definite families. The serial evaluator and
`driver/contract.py` gate 15 held-out cases across three thresholds. The
decoder campaign now supplies the first 85M-parameter target-model campaign
for three local byte landscapes and three seeds. It compares six same-budget
policies from immutable matched parents and writes ranking/calibration and
capability-cost artifacts. Imagined branches remain disabled.

An archive is valid only when parent transitions appear earlier in the same
causal run, IDs are unique, and failed branches remain visible in the cost
account. Split evaluation by complete runs, not adjacent windows. Anything
used for adaptation or selection is development data, not a locked test.

## Promotion rule

Do not promote a candidate because of one transient loss improvement. Require
preselected immediate, recovery, and longer-horizon measurements, then repeat
on fresh seeds. A useful first result is a mechanism-level gain; a broad
speedup requires held-out widths/depths or a changed data distribution and
whole-run accounting.

## First promotion checkpoint

The first optimization promotion gate passes on dimension-8, ill-conditioned positive-
definite landscapes: five families, three held-out seeds per family, and
relative targets `1e-4`, `3e-5`, and `1e-5`. The primary serial wall-time
geometric means are 26.36×, 29.52×, and 43.07×, with clustered 95% lower
bounds above 22× at every target. This is a valid optimization-track result
under the contract, not a general pretraining result. The decoder development
campaign is complete but does not pass or attempt the five-landscape
promotion gate; its first ranking model has 11.1% top-1 agreement. The next
tests are changed data/architecture transfer and a calibrated decoder driver.

## Known ceilings

The current controller groups experience by action kind and ignores telemetry
geometry. That is intentional: it is a cheap control condition, not the
proposed deep driver. Upgrade it only when the fixed comparison shows that
state-dependent information is available and the baseline cannot use it.

## Decoder driver preregistration

The decoder-specific claim, action roster, atlas split, capability thresholds,
cost ledger, and robust 10x gate are fixed in
[`DECODER_DRIVER_PREREGISTRATION.md`](DECODER_DRIVER_PREREGISTRATION.md).
The next implementation rung is the response atlas built from existing
matched campaign transitions, followed by a held-out action-conditioned
predictor. A policy is not promoted until it beats the action-only prior on
fresh roots; the fixed shadow baseline remains a mechanism control, not a
driver claim.

### Current atlas checkpoint (2026-09-19)

The first normalized AdamW atlas contains 61 matched parent groups and 213
observed action rows across the historical decoder campaigns. It preserves
immediate, recovery, and final outcomes, matched no-op deltas, risk flags,
provenance, and additive cost components. A history-conditioned selector was
trained on complete FineWeb roots and evaluated on four complete TinyStories
roots. Its raw final-loss RMSE was `0.0321`, but support calibration rejected
all four transfer deployments; the safe policy selected no-op every time.
The action-only prior was better than that abstaining policy on this small
holdout, so there is no justification yet for a planner or online learner.
The next atlas model must first beat that prior on held-out roots within a
single data regime, then survive the changed-data gate.

The new three-horizon ensemble was tested on the eight-root FineWeb atlas,
holding out one 139M and one 85M root. It supported one of the two roots and
selected the shadow action there at the final horizon, but its selected mean
final delta was `-0.0178` versus `-0.0351` for the action-only prior; it did
not beat the prior. On four complete TinyStories roots trained only from the
FineWeb atlas, support was `0.0`, final prediction RMSE was `0.0101`, and the
safe selector abstained everywhere. The raw final ranking happened to agree
with the realized best action, but the prediction/reality gap and zero support
make that diagnostic only. The predictive gate therefore remains closed:
there is no justified planner, dreaming stage, or online policy yet.

The clean pilot at commit `259dc9f` expands the same action atlas to six roots:
three FineWeb-Edu and three TinyStories roots, each with no-op, small/medium/
large role pulses, damping, trajectory averaging, trajectory extrapolation,
and optimizer-state-reset extrapolation. It contains 48 matched rows across
three horizons with zero branch failures. On FineWeb, the large pulse improves
final loss on all three seeds (mean delta `-0.00227`) but is immediately worse
by `+0.0321`; on TinyStories it improves only one of three seeds (mean delta
`-0.00014`). The medium pulse is the best TinyStories pulse on average (mean
delta `-0.00018`), still far below a meaningful speedup. Averaging regresses
by `+0.00404` on FineWeb and `+0.00580` on TinyStories; extrapolation is near
zero on FineWeb (`-0.00047`) and regresses on TinyStories (`+0.00127`); reset
extrapolation regresses by roughly `+0.24` on both data regimes.

The FineWeb-to-TinyStories response model has zero calibrated support: its
held-out state distances are about `452` against a training support radius of
`19.2`. Its final prediction RMSE is `0.157`, and the safe policy abstains on
all three held-out roots. This is a useful falsification of a first shared
geometry, not evidence for a planner. The next rung is more controlled atlas
coverage and representation diagnosis; do not add MPC, dreaming, or online
RL until an action-conditioned model beats the action-only prior on held-out
roots within support.

The first representation diagnosis found that raw train-root standard
deviations made near-identical FineWeb trajectories over-penalize a changed
data regime. A fixed feature-scale floor of `0.1` reduced FineWeb-to-TinyStories
final prediction RMSE from `0.157` to `0.114`, reduced the prediction/reality
gap from `0.131` to `0.025`, and reduced held-out distances from roughly
`452` to `28`. The calibrated support radius is still only `6.7`, so the
policy continues to abstain. On the FineWeb-only holdout, final RMSE fell from
`1.36` to `0.19`, but support remains `0` with only two training roots. This
is a calibration improvement with a known ceiling, not permission to deploy a
planner; more complete roots are required before changing the support gate.

The third-regime extension adds three delayed-copy roots, bringing the clean
pilot to nine roots and 72 matched action rows. Delayed-copy does not share a
stable winner with the language regimes: its large pulse regresses by `+0.0080`
on average, while damping, medium pulse, and small pulse are all close to zero
and seed-dependent. Reset extrapolation remains consistently harmful (`+0.0196`
final delta on delayed-copy), although less catastrophic than on the language
corpora. The atlas now records `vocab_size` explicitly as architecture
provenance and a model feature.

When delayed-copy roots are held out after training on the two language
regimes, the model correctly marks the architecture/data shift unsupported
(support distance about `1377` versus radius `5.6`) and selects no-op. A
delayed-copy-only model trained on two roots has final RMSE `0.0168` and perfect
raw top-1 agreement on the one held-out root, but support is still zero and the
safe policy abstains. This is a useful ranking diagnostic, not deployment
evidence; the atlas needs more roots before its support calibration can be
relaxed.

The FineWeb expansion adds pulse-only roots 6--8, giving six FineWeb roots for
the within-regime check. Large pulses win 5/6 final branches with mean delta
`-0.00090`, and small pulses win 4/6 with mean delta `-0.00018`, but both remain
immediate-loss regressions and one large-pulse branch regresses durably. A model
trained on four roots and tested on seeds 7--8 has support on both roots, yet its
selected final delta is `+0.00168` versus `+0.00112` for the action-only prior
and `-0.00059` for the oracle. Support is therefore necessary but not
sufficient; the final-horizon selection gate remains closed.

### Phase-switch response and long-horizon checkpoint (2026-09-19)

The phase-switch extension provides the first stronger mechanism signal. Six
roots with the five-action pulse roster show mean final deltas of `-0.00434`
for the medium pulse (6/6 wins, zero durable regressions) and `-0.00581` for
the large pulse (5/6 wins, one durable regression). A phase-only model trained
on roots 3--6 and held out on roots 7--8 ranks the large pulse correctly at
immediate, recovery, and final horizons, but only matches the action prior.

The longer phase-switch diagnostic uses fresh seeds 9--11 and a 2048-step
final horizon. The medium pulse reaches the matched noop branch's final loss
by the 512-step recovery point on all three roots and remains no worse at the
final point. After charging the common 1536-step prefix, this is about `1.75x`
lower FLOP and wall cost to that one baseline-derived threshold. It is not the
10x gate: one landscape, one threshold, equal-action costs, and no policy
selection were involved. The large pulse is not safe at this horizon because
seed 10 regresses by `+0.281` final loss; the medium pulse improves that seed
by `-0.450`. The next experiment is a horizon-matched long atlas with more
roots, followed by held-out selection; do not transfer the short-horizon
large-pulse policy to this setting.

The horizon-matched expansion to seeds 12--14 closes that speed hypothesis.
Across six long roots, medium wins 4/6 and has two durable regressions, while
large wins 3/6 and has three. A model trained on long roots 9--12 and held out
on 13--14 is supported on both roots but has final raw top-1 agreement `0` and
prediction/reality gap `0.0474`; its safe selected mean delta `+0.00065` is
better than the action prior `+0.00226` but worse than noop `0`. The fixed
pulse route is therefore a no-go for promotion. The next rung should change
the observed/action geometry or move to a separately audited jump/data-
allocation hypothesis, not add more pulse-selector variants.

The final action-size check on phase seeds 13--14 confirms the failure mode:
small, medium, and large pulses all improve immediate loss but all regress at
the 2048-step final horizon. The route is therefore closed as a fixed pulse
mechanism. Any future intervention must model long-horizon/recovery response
directly and reject actions whose transient gain is not durable; the next
workload moves to an explicitly charged jump/nowcasting or data-allocation
hypothesis.

The atlas state then gained the parent checkpoint's absolute `step` and
`tokens`, which were already present in provenance but omitted from the model
input. On the same roots 9--12 to 13--14 holdout, final prediction RMSE
improved from `0.0560` to `0.01496`. Final raw top-1 agreement remained `0/2`,
and the safe selector chose no-op on both held-out roots, so this is an
observability/calibration improvement with a known ceiling, not evidence for
state-dependent control. Keep the support gate and do not add more pulse
variants.

### Timing-conditioned shadow early-stop screen (2026-09-19)

The existing two-window shadow merge was measured at parent steps 768, 1280,
and 1792 on fresh 85M FineWeb-Edu seeds 18--20. Mean equal-budget final
deltas were `-0.02837`, `-0.05492`, and `-0.02809`, respectively; the middle
parent is the strongest but all nine branches were positive. A separately
charged 512-step shadow continuation from parent step 1280 reached the matched
768-step noop endpoint on all three FineWeb seeds, with about `11.6--11.8%`
lower end-to-end wall time and `7.7%` lower end-to-end estimated FLOPs.

The same protocol transferred to TinyStories seeds 6--8: equal-budget final
deltas were `-0.05063`, `-0.04803`, and `-0.05095`, and the 512-step candidate
beat the full noop endpoint by `0.0234--0.0277` with about `11.9%` lower wall
and `7.7%` lower charged FLOPs. This is a fixed-mechanism baseline at roughly
`1.13x` wall efficiency, not a contract-valid 10x claim: it uses one target
width, one threshold, two data sources, and a fixed timing rule.

The timing atlas model ranked the final shadow action correctly on all three
leave-one-seed-out FineWeb splits. FineWeb-to-TinyStories ranking was also
correct, but the support gate abstained on all three changed-data roots, so
the learned state geometry has not transferred. Promote the fixed rule only as
an audited baseline; next test the same early-stop rule at the 139M width
holdout, then add duration/cost to the action-conditioned gate. Do not add a
larger steerer yet.

The 139M TinyStories width holdout is now complete. Equal-budget shadow deltas
on seeds 6--8 were `-0.05543`, `-0.05427`, and `-0.05203`; the 512-step
candidate beat the full 768-step noop by `0.02098--0.03169` on all three,
with about `11.3--11.4%` lower end-to-end wall and `7.7%` lower charged FLOPs.
The fixed shadow-stop mechanism therefore transfers across the two tested
widths and data sources. It remains a baseline result at roughly `1.13x`, not
the 10x contract gate and not a learned policy. The next rung is a minimal
cost-aware action gate over `{noop, shadow-stop-512, shadow-full-768}`; train
it only after the duration is represented in the response atlas, and preserve
no-op abstention on unsupported roots.

## Direction amendment (2026-09-19): learned trajectory transport

The fixed pulse route is now closed as a promotion path. The next research
question is whether a short calibration trajectory can identify a mesoscopic
productive region and generate a structured macro-action that transports the
model toward a future capability-equivalent state.

See [TRAJECTORY_TRANSPORT_DIRECTION.md](TRAJECTORY_TRANSPORT_DIRECTION.md) for
the amended contract boundary, passive public trajectory sources, productive-
region classifier, transport-action definition, oracle-ceiling ladder, and
few-shot initialization route.

The robust 10x gate is unchanged. Passive public checkpoints, hindsight
future-state oracles, and shadow endpoints are diagnostic evidence only; they
cannot be reported as deployed driver gains. The next implementation must
separate:

1. passive trajectory representation pretraining;
2. oracle ceilings for future-state transport and data utility;
3. learned low-rank/role-wise transport;
4. cheap productive-region gating and few-shot initialization;
5. planning or RL only after selected real actions are calibrated.

Do not add more pulse variants, a large pretrained meta-brain, MPC, dreaming,
or online RL before a transport or productive-region baseline beats noop and
the action-only prior on fresh complete roots.

### Hindsight trajectory transport oracle (2026-09-19)

The first oracle-ceiling rung is complete. A standalone harness loads exact
AdamW parent checkpoints at step 1536, applies recorded future parameters from
step 2048, and measures a 128-step recovery plus a step-2304 endpoint. It
validates source data/configuration hashes and records both actual and
conservative cost; the latter charges the declared 512-step skipped exposure.
See [TRAJECTORY_TRANSPORT_RESULTS.md](TRAJECTORY_TRANSPORT_RESULTS.md).

The surviving variant (`cursor=skip`, parent AdamW moments preserved) passed
7/9 roots across 85M and 139M FineWeb-Edu plus 85M TinyStories. Its median
final delta was `-0.00314`, with approximately `1.22x` wall and `1.29x`
compute-only speedup. Conservative charging of the skipped exposure reduced
the ratio to `1.00x`. Replaying the parent cursor failed 3/3 85M FineWeb
roots; zeroing moments failed catastrophically on all three. Parameter and
data state are therefore coupled, and future optimizer-state handling is an
unresolved part of transport.

This closes direct hindsight full-weight teleportation as a 10x mechanism. It
does not close learned structured transport, but the next model must predict
role-wise parameter and optimizer-state deltas from short calibration history
and must be evaluated on real branches. No planner, RL, or larger steerer is
justified until that model beats noop and the action-only baseline under the
same cost ledger.

The first learned role-wise baseline is now also negative. FineWeb-trained
coefficients applied to held-out TinyStories roots failed 3/3 by about `+0.031`
final loss; a FineWeb leave-one-root-out test failed by `+0.054`. Offline
projections show that a recent parameter or parent AdamW-moment direction
contains little future attention/MLP direction. The single-basis transport
policy is closed. A richer basis must earn a new probe, otherwise move the
next oracle screen to data/work allocation rather than increasing steerer
capacity.

The role-wise hindsight oracle screen is also closed. On four balanced roots
(FineWeb-Edu 85M/139M and TinyStories 85M/139M), copying only one of
`embedding`, `attention`, `mlp`, `norm`, or `head` from the recorded future
checkpoint produced 0/20 durable endpoint passes. Attention and MLP copies
had transient median improvements of `-0.01477` and `-0.01105`, but their
recovery medians were `+0.01897` and `+0.02875`, and final medians were
`+0.01659` and `+0.02063`. The other roles were worse immediately and at the
endpoint. The wall ratio was about `1.22x` only because the candidate declared
the 512-step exposure skipped; conservative cost was `1.00x` for every role.
This is evidence for long-horizon action labels and state coupling, not a
reason to scale a role-wise steerer. See
[`TRAJECTORY_TRANSPORT_RESULTS.md`](TRAJECTORY_TRANSPORT_RESULTS.md).

The next admissible rung is a cheap actionability/data-work oracle using real
branches and explicit cost, or a causal transport probe that predicts both
parameter and optimizer-state changes. Do not add planner, dreaming, PPO/SAC,
or a larger pretrained controller until one of those baselines demonstrates
held-out durable action ranking within support.

### Naive data-allocation oracle (2026-09-19)

The first data/work screen is complete. From four exact step-1536 parents,
each candidate consumed the same `12,582,912` tokens as a recomputed AdamW
no-op but switched the entire future training stream between the local
FineWeb-Edu and TinyStories byte corpora. There were 0/4 final passes and no
branch failures. Median loss deltas were `+0.49829` immediate, `+0.60487`
recovery, and `+0.63117` final; the cost ratio was `1.00x` by construction.
The recomputed no-op matched the recorded endpoint within `0.00339`, far below
the switched-data regression. See
[`DATA_ALLOCATION_RESULTS.md`](DATA_ALLOCATION_RESULTS.md).

This closes complete cross-corpus switching under the current validation
objectives, not all data allocation. Do not train a selector on it. A further
allocation experiment needs a predeclared common task objective and a smaller
action such as an in-domain mixture or work/context schedule; otherwise the
remaining cheap route is initialization or a rigorously causal action atlas.

The next small oracle is now preregistered as a 25% same-corpus FineWeb block
mixture. It keeps 75% of the target continuation in order, substitutes four
of every sixteen 16,512-byte optimizer-step blocks from a disjoint tail
interval, and holds the parent, optimizer, cursor, validation, and token
budget fixed. Test three fresh 85M and three fresh 139M roots; do not train a
selector unless at least two of three roots at both widths improve durably and
threshold cost materially beats the fixed shadow control.

### Complete-state transport sanity control (2026-09-19)

The state-consistency gap in the earlier future-weight oracle is now measured.
On the same four balanced roots, a complete checkpoint saved at step 2048
(parameters, AdamW state, RNG, cursor, hashes, and parent provenance) resumed
the remaining 256 steps with maximum loss gaps of `0.000214`, `0.000891`,
`0.000069`, and `0.000318`. The strict `1e-5` replay check was 0/4 because
CUDA/AMP replay is not bitwise deterministic, but every gap was below the
independent recomputed-noop variation of `0.003392`, and all checkpoint loads
validated. The ideal free-state ratio is `1.286x`; conservative accounting of
the 512 steps needed to create the state is `1.000145x`.

This closes the optimizer-state provenance gap without reopening the transport
promotion path. See
[`FULL_STATE_TRANSPORT_RESULTS.md`](FULL_STATE_TRANSPORT_RESULTS.md).

### Actionability label corpus (2026-09-19)

The preregistered causal labeler now converts observed matched atlas rows into
`productive`, `jumpable`, `recoverable`, `dangerous`, `stalled`, `neutral`, or
the separate `noop_baseline` class. On the mixed 15-root atlas it found only
4 productive non-noop rows against 31 dangerous and 45 recoverable rows. The
only two jumpable rows came from one phase-switch seed; timing-shadow atlases
are mostly productive because they represent one fixed mechanism, not a
transferable selector. The action-conditioned classifier gate therefore
remains closed. See
[`ACTIONABILITY_LABEL_RESULTS.md`](ACTIONABILITY_LABEL_RESULTS.md).

The next causal collection should increase within-regime action coverage and
hold out complete roots before fitting a classifier. Do not use the current
label imbalance to justify a larger model or a planner.

### Oracle ceiling decision (2026-09-19)

The transport/data oracle ladder has now been consolidated in
[`ORACLE_CEILING_DECISION.md`](ORACLE_CEILING_DECISION.md). No tested causal
or diagnostic transport action has a large conservative ceiling: complete
future state is approximately `1.000145x` after charging state-creation work,
role-wise future copies are 0/20 durable, the coarse data switch is 0/4, and
the learned action model remains at 0/2 final top-1 agreement on its supported
holdout. The fixed shadow stop remains a roughly `1.13x` mechanism baseline,
not a learned driver.

This is a no-go for the current transport hypothesis as a 10x mechanism. Do
not scale the steerer or add planning/RL. Any continuation must introduce a
new predeclared causal action family—task-aligned in-domain allocation or
few-shot initialization/seed screening—and first clear the small oracle
ceiling rule in the decision document.

### Disjoint trajectory-anchor initialization screen (2026-09-19)

The first implementation of this idea was rejected before use because nearby
seed rotations reused target bytes inside the source prefix. The corrected
screen gives the target noop and source-initialized anchor the same byte
interval outside the source prefix, charges source-prefix and deployment
costs, and records target-RNG, source-RNG, and zero-moment controls. Six
preserve-state pairs completed without failures, but only 2/6 anchors beat
the matched endpoint; the combined mean final delta was `+0.00023`, and the
median single-deployment hard-threshold wall ratio was `1.05x`. The zero-
moment and source-RNG seed-3 controls changed individual deltas without
providing a reproducible advantage. This closes the current same-corpus
anchor route as a driver promotion path. See
[`TRAJECTORY_ANCHOR_RESULTS.md`](TRAJECTORY_ANCHOR_RESULTS.md).

### Passive trajectory corpus and phase model (2026-09-19)

The first normalized passive corpus now contains 17 complete local AdamW roots
and 13,056 rows with step/token position, role-wise optimizer telemetry,
protocol phase, and future training-loss targets at horizons 1/8/32/128. The
collector preserves source manifest, transition, data, configuration, and
parent-checkpoint hashes; it explicitly marks the data as passive no-op
evidence. See [PASSIVE_TRAJECTORY_RESULTS.md](PASSIVE_TRAJECTORY_RESULTS.md).

A small root-held-out MLP improves future-loss prediction within FineWeb 85M
(`h128` RMSE `0.100` versus constant `0.150`) and mostly within TinyStories
(`0.070` versus `0.084`), but loses to the constant baseline on a mixed
width/data holdout at every horizon. Phase accuracy is largely a schedule
label and is not a dynamics claim. The result supports compact numerical
history features while closing the shortcut of scaling the steerer before
regime support and action-conditioned targets are available.
