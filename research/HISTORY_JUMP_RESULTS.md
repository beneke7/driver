# History-jump screen: first RTX 5090 result

Date: 2026-09-18. Code revision: `8780c62`.

The first collection campaign ran three independent 85,350,912-parameter
decoder jobs concurrently on the RTX 5090. Each used AdamW, two seeds, 64
teacher steps, and complete checkpoints every eight steps. All six teacher
trajectories completed; collection took 15–23 seconds per case. The ignored
artifacts are:

```text
runs/history-jump-traces-{delayed,phase,text}/
```

The branch screen then ran serially per landscape to keep wall-time comparisons
interpretable. It evaluated two parents per seed, horizons 8/16/32, `noop`, and
momentum jumps with blend `0.5` or `1.0`: 108 branches, zero failures.

| Landscape | Jump setting | Mean final-loss delta vs matched noop | Mean wall ratio | Interpretation |
| --- | ---: | ---: | ---: | --- |
| delayed copy | 32, 0.5 | about `-0.0007` | `0.84×` | Small jumps are plausible in this smooth regime. |
| delayed copy | 32, 1.0 | about `-0.0017` | `0.84×` | Stronger pulse is still close, but the margin is tiny. |
| phase switch | 32, 0.5 | about `+0.174` | `0.84×` | The same pulse is usually harmful across a regime change. |
| phase switch | 32, 1.0 | about `+0.051` | `0.84×` | One local branch improved, but the aggregate remains worse. |
| text shard | 32, 0.5 | about `+0.093` | `0.84×` | Jumping is not broadly safe on this target. |
| text shard | 32, 1.0 | about `+0.084` | `0.84×` | More aggressive scaling does not fix the mismatch. |

The wall ratio is branch wall time relative to noop, not a deployment speedup:
the jump branch still performs recovery and final training. Estimated target
FLOPs fall by roughly one sixth because one six-horizon segment is replaced by
a parameter update. Immediate loss is often worse even where final loss
recovers, so a useful controller needs a calibrated recovery-aware gate.

The result rejects a global momentum pulse as the driver. It supports the more
interesting hypothesis that action value is state- and landscape-dependent:
the controller must recognize when local smoothness makes a jump safe and when
a phase/data change makes it dangerous. The next experiment is therefore a
small supervised selector using the saved history features, with `noop` always
available and a rejected-jump path explicitly charged.

This is a mechanism screen, not a promotion result. It has only two seeds,
three local landscapes, two parent checkpoints per seed, and no external data
or architecture holdout. The full branch artifacts and exact commands remain
ignored local files; do not report this screen as a general pretraining gain.

## Selector screen: architecture is not the bottleneck yet

The richer telemetry collection (`b20e69a`) was used to fit the first offline
selector (v1): a small MLP predicts the final loss delta for each jump and keeps
`noop` available at zero predicted delta. The split was by complete cases, not
individual rows. On the seed-1 holdout, the selector chose a jump for 20% of
groups and produced a mean final delta of `+0.0117`; the matched no-op and the
fixed-action baseline were both `0.0`, while the hindsight oracle was only
`-0.00066`. On the text-shard landscape holdout it selected a jump for every
group and produced `+0.1417`, with top-1 action accuracy `0.0833`.

An initial risk gate calibrated to the 90th percentile training residual did
not change either decision pattern. A manually wider `0.05` gate reduced the
seed-1 jump rate to `13.3%` and the loss delta to `+0.0093`, but remained worse
than no-op; it did nothing on the text holdout. This is a deliberate negative
result: same-distribution residual calibration is not an uncertainty estimate
under landscape shift.

The next selector comparison is therefore retrieval or a small ensemble with
case-level holdouts and an abstain-on-disagreement rule. A recurrent or
Transformer steerer is deferred until that cheap control either shows a
capacity-limited action-ranking error or the added telemetry makes a compact
model underfit. Fresh seed-2 branch screens are being collected before that
comparison.

The first case-bootstrap ensemble (five MLPs) sharpened this result. On
delayed-copy seed 2 it abstained on all 12 decision groups and therefore
avoided harm, but the fixed action baseline was better (`-0.00121` versus
`0.0`). On the text holdout, ensemble disagreement alone still selected every
bad jump: the members shared the same extrapolation bias. A leave-one-case-out
nearest-neighbor support gate fixed that failure, rejecting all text cases;
its default 95th-percentile support radius also rejected the delayed seed-2
case. Widening the radius to `12` admitted delayed seed 2 but still selected
no jumps. This is useful safety machinery, not evidence of a steering gain.

## Fresh seed-2 causal holdout

The same 85M AdamW harness was extended to four parent checkpoints per seed
for the third seed: 36 branches per landscape, 108 branches total, all
successful. Relative to matched no-op branches, the mean final-loss deltas
were:

| Landscape | Blend 0.5 | Blend 1.0 | Immediate/recovery warning |
| --- | ---: | ---: | --- |
| delayed copy | `-0.00044` | `-0.00121` | immediate `+0.142/+0.379`; recovery `+0.004/+0.012` |
| phase switch | `+0.137` | `+0.206` | immediate is lower, but recovery/final lose |
| text shard | `+0.109` | `+0.155` | immediate `+1.285/+2.402`; recovery `+0.463/+0.581` |

The jump FLOP ratio was about `0.835×` and the wall ratio about `0.84–0.90×`,
but this is not a speedup claim because the jump still paid for recovery and
final evaluation. Every tested jump had a positive maximum over immediate,
recovery, and final deltas. A final-only objective would therefore reward a
transiently attractive but operationally unsafe action.

Training on seeds 0/1 and holding out all seed-2 cases produced 72 test jump
examples. With support disabled, the single MLP selected 16.7% of groups and
caused `+0.0167` mean final-loss damage. The five-model case-bootstrap
ensemble abstained on all groups; the default leave-one-case-out support gate
also abstained on all groups.

Selector v2 now trains on
`safe_delta = max(immediate_delta, recovery_delta, final_delta)`, with `noop`
at zero. On the same seed-2 holdout, the permissive single model still selected
16.7% of groups, with `+0.0202` mean selected safe delta; the support-gated
ensemble selected none. This is the correct conservative behavior, but still
not a useful steering result. More architecture is deferred until a
controller can identify a positive safe action in held-out data.

### Action-size and optimizer-state ablations

On delayed-copy seed 2, reducing the original pulse to blend `0.125` or
`0.25` did not produce a safe branch. Blend `0.25` was nearly neutral in final
loss (`+0.00016`) but still had positive immediate and recovery deltas
(`+0.0217` and `+0.00005`). An explicit `momentum_jump_decay` variant that also
decayed AdamW moments and advanced the optimizer step through the skipped
horizon was worse at the same blend: final `+0.00046`, recovery `+0.00173`.
The optimizer-state rule is therefore not promoted; the variant remains in the
archive as a causal negative control.

A checkpoint-history secant/nowcasting baseline was then tested on the same
delayed-copy seed-2 parents at blend `0.25`. It was closer to neutral than the
momentum pulse in final loss (`+0.000045`) but still worsened immediate loss
(`+0.0292`) and recovery (`+0.00099`), with a `0.871×` wall ratio and no safe
branches. This is a useful NiNo-style open-loop control, not evidence that
history alone can collapse the language-model landscape.

These short screens now have a clear ceiling: they test only 64-step synthetic
trajectories and a transient endpoint. The next required experiment is the
long-horizon maneuver protocol in
[`research/LONG_HORIZON_MANEUVER.md`](LONG_HORIZON_MANEUVER.md), using a
versioned language-data shard, an overparameterized target condition, and
post-recovery slope/capability measurements.

### First real-data long-horizon trajectory screen

The first real-data maneuver used raw UTF-8 bytes from the versioned TinyStories
shard, a 139M-parameter decoder, 256 prefix steps, and a 2,048-step
continuation. The trajectory actions used eight snapshots at 32-step spacing.
Relative to the matched no-op (`0.85126` final validation loss), recent-window
averaging ended at `0.85506` and two-window extrapolation at `0.85638`; both
also had worse immediate and recovery losses. The extrapolation intervention
had normalized energy `0.625` and added about `2.23e9` estimated FLOPs, so the
negative result includes the actual maneuver overhead.

To test the late-stage premise, an 85M control used a 2,048-step prefix and
eight snapshots at 256-step spacing. No-op finished at `0.77207`. Preserved
state averaging briefly improved the intervention loss (`0.89561` versus
parent `0.92123`) but finished at `0.77819`; extrapolation with alpha `1.0`
finished at `0.79516`. Zeroing AdamW moments after the maneuver was not a
solution: averaging-reset finished at `0.87358`, while extrapolation-reset
finished at `0.79397`. A reused-parent alpha scan for preserved-state
extrapolation found final losses `0.77834`, `0.77849`, `0.77722`, and `0.77867`
for alpha `0.0625`, `0.125`, `0.25`, and `0.5`, respectively. None beat the
matched no-op, although alpha `0.25` was the least harmful extrapolation.

Interpretation: this two-window action is not yet a productive regime change
under continued AdamW. The late-stage action can lower validation loss at the
intervention point, but its benefit is lost during ordinary continuation. The
next discriminating control is a cautious recovery schedule or an
optimizer-state interpolation, not a larger steerer. The interrupted 139M
late-prefix run is excluded because it never produced a complete parent and
manifest.

### Post-hoc trajectory rectification

The same 85M late parent was continued once with ordinary AdamW for 2,048
steps, saving eight continuation snapshots. Evaluating the raw endpoint gave
`0.77263`; without any additional gradient step, averaging the most recent
four snapshots gave `0.75750`, and two-window extrapolation with alpha `0.25`
gave `0.75704`. Alpha `0.5` gave `0.76140`, while alpha `1.0` overshot to
`0.78990`. The merge computation was about `1.37e9` estimated FLOPs, much less
than the continuation, though its validation evaluation and storage must still
be charged.

This is a useful real-data landscape signal and a new baseline: maintain a
shadow trajectory buffer and use a validated merged endpoint. It is not a
training speedup yet. The next test is whether the same rectification reaches
the raw endpoint's quality at an earlier checkpoint, and whether a driver can
choose when to expose the merged shadow weights without damaging the live
AdamW state.

The matched 139M campaign initially used a four-checkpoint shadow window and
was negative (`0.85449` merged versus `0.84933` raw). A saved-snapshot window
scan found that two checkpoints were better (`0.84181` versus `0.84933`). The
first complete harness run with `trajectory_window=2` reproduced that direction:
the raw shadow branch ended at `0.84763`, its merged endpoint at `0.84090`, and
the matched no-op at `0.84843`. The shadow branch took `257.56s` versus
`253.32s` for no-op and added about `4.56e9` estimated FLOPs. This is the
current strongest result, but it remains one seed, one byte-level corpus, and
an endpoint-quality gain rather than a demonstrated skip.

The window-2 progress curve did not yet show a cost-to-target win. At global
steps `1024`, `1280`, `1536`, `1792`, `2048`, and `2304`, the shadow losses
were `1.03294`, `0.98532`, `0.91146`, `0.89249`, `0.90203`, and `0.84090`;
the matched noop endpoint was `0.84843`. Thus the shadow was better than raw
at most late checkpoints and better than the final noop only at the endpoint.
That rules out calling this a skip until an earlier merged checkpoint reaches
a matched target on an independent continuation.

A fresh 139M seed-1 replication with the same window-2 protocol also passed:
noop ended at `0.85290`, the shadow branch's raw endpoint was `0.85342`, and
its merged endpoint was `0.84475`. The shadow branch took `257.34s` versus
`253.66s` for noop. The matched endpoint improvements are therefore `0.00673`
and `0.00867` on seeds 0 and 1, respectively. This is strong enough to
promote window-2 shadow averaging as a baseline mechanism, but not enough to
claim transfer until seed 2, changed data, and earlier cost-to-quality are
measured.

Seed 2 completed the same protocol with noop `0.84711`, shadow raw `0.84942`,
and merged `0.83947`. Across seeds 0–2, matched noop-to-shadow endpoint
improvements are `0.00753`, `0.00815`, and `0.00764`; the extra wall time was
approximately four seconds per branch. This is now a reproducible same-data
mechanism baseline. It still does not establish a training skip, and the
TinyStories byte representation is a deliberately narrow transfer test.

### Changed-data transfer and timing control

To test whether the endpoint effect was only a TinyStories artifact, the same
139M AdamW/window-2 shadow protocol was run on a versioned 40 MiB
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) byte
slice with a disjoint 8 MiB validation slice. Seeds 0 and 1 completed with no
failures:

| Seed | No-op final | Shadow raw | Shadow merged | Endpoint improvement | Branch wall (noop/shadow) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `1.53561` | `1.52901` | `1.51030` | `0.02531` | `253.29s / 257.03s` |
| 1 | `1.49854` | `1.49769` | `1.47577` | `0.02277` | `253.81s / 259.73s` |

This is a useful changed-distribution replication of the endpoint-quality
mechanism. It is still not a speedup: the live AdamW branch consumes the same
training tokens, and the merged weights are exposed only after the last
checkpoint. The seed-2 no-op parent completed, but its candidate process hit a
host-level `SIGILL`/`SIGSEGV` in the PyTorch/glibc stack while saving or copying
trajectory state; it is excluded rather than counted as a failed model branch.

The saved FineWeb-Edu seed-0 snapshots also answer the timing question. The
average of the latest two snapshots had validation losses `2.44547`, `2.02055`,
`1.78341`, `1.66406`, `1.59827`, `1.56854`, and `1.51030` at global steps
`768`, `1024`, `1280`, `1536`, `1792`, `2048`, and `2304`. It crossed the
matched no-op endpoint (`1.53561`) only at step `2304`.

Finally, a live correction at global step `1536`, followed by the same 768-step
continuation as a matched no-op, finished at `1.53933` versus `1.52830` for
no-op. Its immediate and recovery losses were also worse (`1.74164/1.60356`
versus `1.70629/1.58843`). The current action should therefore remain an
endpoint shadow baseline. The next driver work should learn when to expose or
reject such a correction, with a cost-to-target evaluator, rather than simply
making the correction more aggressive.

Post-hoc action-size and window scans on the same saved FineWeb-Edu trajectories
reinforce that boundary. With the latest two-window average as `alpha=0`, the
final losses for extrapolation alphas `0.0625`, `0.125`, `0.25`, `0.5`, `1.0`,
and `2.0` were respectively `1.51145`, `1.51360`, `1.52095`, `1.55005`,
`1.70077`, and `2.59727` on seed 0; seed 1 gave `1.47722`, `1.47941`,
`1.48612`, `1.50978`, `1.61432`, and `2.12693`. The latest two snapshots
were also the best fixed window: losses for windows 1/2/3/4 were
`1.52901/1.51030/1.51928/1.52902` on seed 0 and
`1.49769/1.47577/1.48266/1.48948` on seed 1. The initial action set can
therefore be reduced to no-op, window-2 shadow average, and a deliberately
bounded small extrapolation control; larger pulses have a clear, measurable
ceiling on this data.

The width holdout used the same FineWeb-Edu slice and protocol with an
85M-parameter decoder. All three seeds completed with zero branch failures:

| Seed | No-op final | Shadow raw | Shadow merged | Endpoint improvement | Branch wall (noop/shadow) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `1.46471` | `1.46657` | `1.44702` | `0.01769` | `175.69s / 178.41s` |
| 1 | `1.47994` | `1.47986` | `1.45824` | `0.02170` | `176.59s / 178.48s` |
| 2 | `1.47746` | `1.47947` | `1.45765` | `0.01981` | `176.61s / 178.61s` |

The mean endpoint improvement is `0.01973` at 85M versus `0.02404` across
the two completed 139M FineWeb-Edu seeds. This supports a width-robust
endpoint rectification mechanism, not a claim that overparameterization causes
the effect. Both widths still consume the same continuation tokens; the next
step is a learned timing/acceptance gate and an explicit cost-to-target test.

The 85M saved-snapshot cost-to-target check is negative in the stronger sense:
the window-2 average crossed its matched no-op endpoint only at global step
`2304` for all three seeds. At step `2048`, the merged losses were
`1.49618`, `1.50640`, and `1.50426`, while the corresponding full-budget no-op
endpoints were `1.46471`, `1.47994`, and `1.47746`. The endpoint merge is
therefore a consistent quality improvement at fixed work, not yet a measured
reduction in work to the same quality.

### First learned dynamics model

The first reusable driver component is an offline telemetry world model in
[`driver/trajectory_world_model.py`](../driver/trajectory_world_model.py). It
uses scale-free numerical features and compares a snapshot MLP with the same
MLP given the latest eight telemetry vectors. The target is the next observed
loss change; no action labels or imagined rollouts are used.

On a whole-run holdout of FineWeb-Edu 139M seed 1, training on the other
completed real runs gave next-loss RMSE `0.11757` for a train-mean baseline,
`0.09430` for snapshot-only features, and `0.08916` for the eight-step history
model. Correlations were `0.600` and `0.652`. A preliminary model trained only
on synthetic decoder campaigns catastrophically extrapolated to real-model
scales; log/relative normalization fixed that failure. This is the first
evidence that short history contains predictive information beyond the current
telemetry snapshot, and the first reason to test a small recurrent or
history-attention driver. It remains a passive prediction result, not a
steering or speedup result.

### Late nowcast control

A 139M FineWeb-Edu seed-0 branch applied a two-window secant nowcast with
`alpha=0.25` at global step `1536`, then continued for the same 768 steps as
the matched no-op. The no-op finished at `1.53032`; the nowcast finished at
`1.53726`, with immediate/recovery losses `1.71318/1.59894` versus
`1.70768/1.58575` and normalized displacement energy `0.293`. Together with
the earlier prefix and post-hoc extrapolation scans, this keeps secant
nowcasting as a negative control until a learned predictor beats it on a
held-out continuation.

### First paired action set for a future gate

A fresh 85M FineWeb-Edu seed-3 run collected four actions from one late parent
at global step `1536`, with a 768-step continuation: noop, live trajectory
average, live secant extrapolation (`alpha=0.25`), and endpoint-only shadow
average. All branches completed. Final losses were `1.48249`, `1.48931`,
`1.48696`, and `1.44416`, respectively; the shadow branch's raw endpoint was
`1.48203`. Live displacements were harmful after recovery, while the endpoint
merge improved the final score by `0.03834` without changing the live AdamW
trajectory. This is the cleanest current causal separation between a
trajectory maneuver and an endpoint rectifier.

An offline action-conditioned MLP built from the existing real branches has
only 20 candidate examples. It selected the shadow action on the fresh
FineWeb all-action holdout, but selected the wrong live action on a held-out
TinyStories trajectory group and produced badly calibrated absolute scores.
The gate is therefore not promoted. Collect more matched parent states and
calibrate a conservative action head before integrating it with the history
world model.

The same four-way late action set on fresh 85M FineWeb-Edu seed 4 reproduced
the split: noop `1.47555`, live average `1.48195`, live secant `1.47941`, and
shadow merged `1.44144` from a raw `1.47578`. The shadow improvement over noop
was `0.03410`; both live maneuvers were worse after the continuation. This
raises the paired action archive to two fresh all-action states, while the
selector remains deliberately unpromoted because its held-out sample is still
too small for calibration.

Fresh 85M FineWeb-Edu seed 5 supplied a third paired state: noop `1.47813`,
live average `1.48565`, live secant `1.48333`, and shadow merged `1.44783`
from raw `1.48182`. The shadow improvement was `0.03031`; both live actions
again lost after recovery. Three independent late all-action states now show
the same qualitative ranking, giving the next selector test a real safety
negative set instead of only endpoint-positive examples.

### Exact-parent early stopping

The saved late action states showed that the window-2 shadow average crossed
the matched no-op endpoint before the full 768-step continuation: seed 3 first
crossed at global step `2048`, while seeds 4 and 5 first crossed at `2176`.
To remove the small GPU-nondeterminism confound from regenerating a prefix, the
candidate branches were then run directly from the exact parent checkpoint
used by each original full continuation. The fixed candidates completed with
zero failures:

| Seed | Candidate endpoint | Matched full no-op | Improvement | Candidate branch wall | Full branch wall | Marginal compute reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3, stop 2048 | `1.47869` | `1.48249` | `0.00380` | `47.43s` | `66.85s` | `33.25%` |
| 4, stop 2176 | `1.46011` | `1.47555` | `0.01543` | `57.84s` | `66.90s` | `16.60%` |
| 5, stop 2176 | `1.46427` | `1.47813` | `0.01387` | `58.10s` | `67.12s` | `16.60%` |
| 6, stop 2176 | `1.45713` | `1.47738` | `0.02025` | `58.72s` | `67.01s` | `16.60%` |
| 7, stop 2176 | `1.45811` | `1.47488` | `0.01677` | `58.98s` | `66.89s` | `16.60%` |
| 8, stop 2176 | `1.45364` | `1.46851` | `0.01487` | `59.10s` | `67.23s` | `16.60%` |

The three fresh holdouts (seeds 6–8) all passed the locked rule. Their mean
endpoint improvement was `0.01730`, and their wall-time reductions on the
marginal branch were `12.10%`, `11.83%`, and `12.10%`. This is a real held-out
timing result for the fixed rule, not just another post-hoc scan.

This is the first exact-parent cost-to-target result: a shadow endpoint can
reach the matched full-run quality with less continuation work, rather than
only improving the endpoint at fixed work. The reported compute reduction is
for the branch after the shared 1536-step parent. If the common prefix is also
charged from initialization, the corresponding end-to-end reductions are only
`11.08%` for seed 3 and `5.53%` for seeds 4–8. The timing rule was selected
from seeds 3–5, so seeds 6–8 are held-out deployment results. The result is
now strong enough to promote the fixed shadow timing rule as a baseline, but
not as a learned driver: it is still an offline endpoint average with a
hand-locked stopping time. The next driver step is to learn the timing and
acceptance gate, with the fixed rule retained as its safety baseline.

Charging the shared prefix in wall time makes the practical saving smaller:
seed 3's 2048 case saves `9.75%` end to end, while the 2176 rule saves
`4.55%`, `4.50%`, `4.18%`, `4.00%`, and `4.08%` on seeds 4–8 (mean
`4.26%`). The branch-level compute reduction is therefore a mechanism signal,
not a claim of a 16.60% whole-training speedup.

### Driver architecture screen and first stop gate

The passive world-model screen now uses 52,448 telemetry examples from the
completed real runs. On a whole-seed holdout containing FineWeb-Edu seeds 6–8
(4,128 examples), the compact `history=8, hidden=64` predictor reached RMSE
`0.08190` and correlation `0.775`. The selected `history=16, hidden=128`
predictor reached RMSE `0.05173` and correlation `0.915`; the train target is
the next observed loss change. A single-seed capacity scan also preferred
history 16 over history 8 and did not show a penalty from width 128. This is
enough evidence to use a numerical history MLP as the first driver backbone;
it is not evidence that a recurrent or language-pretrained controller is
needed.

Using the saved shadow snapshots, 23 candidate stop labels were extracted at
global steps `1792`, `1920`, `2048`, and `2176` where available. A standardized
ridge predictor with a leave-one-seed-out split was used as the deliberately
cheap gate. With a low threshold it attempted an early `2048` stop and failed
on one fresh seed; with a conservative `0.005` predicted-gain threshold it
abstained to `2176` on all fresh seeds and did not beat the fixed rule. This
was a data ceiling rather than a reason to add a deeper policy, so the
next investment was matched action states and action-conditioned labels, not
imagined rollout.

### Action archive expansion

Two more fresh 85M FineWeb-Edu action sets at the original parent step 1536
again separated live maneuvers from endpoint rectification:

| Seed | No-op | Live average | Live extrapolate | Shadow raw | Shadow merged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | `1.47553` | `1.48238` | `1.47828` | `1.47553` | `1.44142` |
| 10 | `1.47445` | `1.48167` | `1.47585` | `1.47479` | `1.44142` |

All six branches succeeded. Relative to each no-op, the live average lost by
`0.00686` and `0.00722`, live extrapolation lost by `0.00276` and `0.00140`,
and shadow averaging gained `0.03411` and `0.03303`. A ridge action head using
the parent telemetry and action identity selected shadow on every leave-one-
seed-out split across seeds 3, 4, 5, 9, and 10, including held-out seed 10.
That is a useful safety baseline, but action identity alone can explain the
ranking; it is not yet evidence of state-dependent steering.

To force state variation, seed 11 used a later parent at global step 1792 and
a 512-step continuation to 2304. No-op ended at `1.47774`, live average at
`1.48241`, live extrapolation at `1.47950`, and shadow at `1.44375` from raw
`1.47745`. The same qualitative split held, so the shadow mechanism survives
a changed decision time. The next selector test should include this later
parent and report whether telemetry improves over the action-only prior.

The first such comparison used action-only versus action-plus-parent-telemetry
ridge heads. Both selected shadow on held-out seeds 9, 10, and 11. Action-only
prediction RMSEs were `0.00091`, `0.00127`, and `0.00138`; adding telemetry
gave `0.00207`, `0.00212`, and `0.00878`, respectively. With this archive,
the state features add variance but no decision value: the selector has learned
the action prior, not a state-dependent intervention rule.

A bounded secant scan over the saved shadow buffers also tested alphas
`0.0625`, `0.125`, and `0.25` at global steps 2048 and 2176 on seeds 6–10.
Every positive alpha was worse than alpha zero at both checkpoints. Keep
secant extrapolation as a negative control and retain plain window-2 averaging
as the only promoted endpoint action until a learned action-conditioned model
beats it on fresh states.

### 139M width transfer

A fresh 139M FineWeb-Edu four-way action set tested the same parent step and
action interface at width 1024 / 11 layers. All branches succeeded:

| Action | Final validation loss | Immediate | Recovery | Shadow raw |
| --- | ---: | ---: | ---: | ---: |
| No-op | `1.48691` | `1.63529` | `1.54183` | — |
| Live average | `1.48948` | `1.64814` | `1.54571` | — |
| Live extrapolate | `1.48859` | `1.63307` | `1.54537` | — |
| Shadow average | `1.45133` | `1.63492` | `1.54394` | `1.48732` |

The live maneuvers again lost after recovery, while endpoint averaging gained
`0.03558`. More importantly, an exact-parent shadow branch stopped at global
step `2048` and ended at `1.48279`, beating the matched full no-op endpoint
`1.48691`. Its marginal branch wall time was `68.47s` versus `96.54s`, and its
marginal compute reduction was `33.25%`; charging the shared 1536-step prefix
reduces the end-to-end compute saving to `11.08%`. This was the first
width-transfer replication of the timing mechanism.

The same exact-parent 2048 test on a fresh 139M seed 3 ended at `1.50839`
versus `1.51308` for its full no-op, with `69.33s` versus `96.61s` of
marginal branch wall time and zero failures. The 139M 2048 timing result has
therefore replicated across two seeds, although the end-to-end compute saving
remains `11.08%` after charging the shared prefix and the candidate is still a
fixed shadow rule rather than a learned driver. A third exact 139M seed 4
also passed: early `1.50529` versus no-op `1.50885`, with `68.79s` versus
`96.59s`. The fixed 2048 timing mechanism has now replicated across three
139M seeds. Charging each shared prefix in wall time gives end-to-end savings
of `9.81%`, `9.58%`, and `9.78%` (mean `9.72%`) on seeds 2–4.

### Earlier-parent timing failure

The 139M seed-7 earlier-parent test moved the common parent to global step
`1024`, then compared a shadow endpoint at `2048` with a full no-op to `2304`.
The exact-parent candidate ended at `1.50251` versus `1.50015` for no-op. It
saved `20.0%` of branch compute and `9.05%` end-to-end wall time, but failed
the same-quality requirement. This rules out moving the 2048 shadow rule
earlier as an open-loop shortcut; the useful timing depends on the trajectory
state and path, which is the remaining learned-driver problem.

### History-enabled action states

After adding the last 16 parent telemetry rows to each branch record, three
fresh 85M action sets supplied the first usable history-conditioned examples:

| Seed | No-op | Live average | Live extrapolate | Shadow merged |
| --- | ---: | ---: | ---: | ---: |
| 12 | `1.47867` | `1.48265` | `1.48028` | `1.44484` |
| 13 | `1.47553` | `1.48010` | `1.47444` | `1.44160` |
| 14 | `1.47515` | `1.48033` | `1.47783` | `1.43923` |
| 15 | `1.47614` | `1.48224` | `1.47804` | `1.44112` |
| 16 | `1.47779` | `1.48477` | `1.48187` | `1.44207` |
| 17, parent 1792 | `1.48632` | `1.49098` | `1.48603` | `1.45171` |
| 139M-5 | `1.50609` | `1.50701` | `1.50493` | `1.46615` |
| 139M-6 | `1.51096` | `1.51857` | `1.51665` | `1.47528` |

All branches succeeded and stored 16 parent rows. Shadow gains were
`0.03383`, `0.03393`, and `0.03592`. Live extrapolation was mixed: it lost by
`0.00162` and `0.00268` on seeds 12 and 14 but gained `0.00109` on seed 13.
A leave-one-seed-out ridge head over the history plus action identity still
selected shadow on all three splits, and its RMSE was no better than the
action-only prior. The history path is now ready for a larger archive, but it
has not earned online adaptation or a deeper policy yet.

Seeds 15 and 16 restored the usual split, with shadow gains `0.03502` and
`0.03572`. The later-parent seed 17 again made extrapolation slightly useful
(`0.00029` gain) while shadow gained `0.03461`. A selector trained on seeds
12–16 and tested on that later-parent state had RMSE `126.5` and support rate
zero, so it abstained. This stress-tests the support contract: the model must
not extrapolate from 1536-parent histories to a 1792-parent decision until
that regime is represented in training.

The 139M history roots also separated extrapolation: seed 5 gained `0.00117`
while seed 6 lost `0.00568`; shadow gained `0.03994` and `0.03568`. Training
on the five 85M roots plus 139M seed 5, then holding out 139M seed 6, gave
the action selector support rate `1.0`, prediction RMSE `0.0097`, and the
correct shadow choice. The actual selected gain was `0.03568`, equal to the
action-prior choice. This is the first safe cross-width selector transfer,
but still no evidence that telemetry beats the fixed action prior.

The first reusable implementation is
[`driver/trajectory_action_selector.py`](../driver/trajectory_action_selector.py).
On seeds 12–13 train / seed-14 holdout, its 128-wide history MLP was badly
out of support (test RMSE `0.860` and predicted shadow delta `+0.839` for true
loss deltas near `-0.03`). The leave-one-root support gate detected this
(`test_support_rate=0`) and abstained to noop. The action-only prior would have
selected shadow and gained `0.03592`, but it is not a state-dependent driver.
This is the intended safety failure: collect more roots before increasing
model capacity or enabling online updates. The selector now accepts an
explicit fixed-action fallback: default noop abstains, while a promoted
shadow fallback preserves the known `0.03592` seed-14 gain without treating
the unsupported MLP prediction as trusted.

The current timing policy is therefore architecture-conditioned but still
fixed: the 85M holdouts use global step `2176`, while three exact 139M seeds
use `2048`. This is a useful explicit input for the future driver and gives
the same action a measurable width-dependent timing signal; it is not yet a
learned architecture-transfer result.

### Changed-data transfer and timing boundary

The first TinyStories action set used the same 85M target, parent step
`1536`, four-way action set, and 16-row parent history as the FineWeb runs, but
changed both the training and validation corpus. The full `2304`-step no-op
ended at `0.85061`. Live trajectory averaging and extrapolation again hurt,
ending at `0.85546` and `0.85196`; the shadow average helped modestly at
`0.84538` (gain `0.00523`). All four branches succeeded.

The useful timing did not transfer. An exact shadow branch stopped at global
`2048` with loss `0.86998` and `47.69s` of marginal wall time, versus the
full no-op's `0.85061` and `66.82s`. Stopping at `2176` produced `0.88655` in
`58.57s`. These candidates save `33.25%` and `16.60%` of branch FLOPs,
respectively, but neither reaches the full no-op quality target. The full
`2304` shadow result is therefore an endpoint improvement, not a compute
shortcut on this corpus. This is a clean negative transfer result for the
current open-loop timing rule.

An offline selector trained on the FineWeb 85M/139M history archive had zero
support on the TinyStories parent. Its raw prediction was wrong (test RMSE
`0.185` and predicted a positive shadow delta), so the support gate abstained;
the explicit promoted-shadow fallback still selected shadow and realized the
`0.00523` gain. The fixed action prior from FineWeb expected a `0.03572` gain,
which makes the changed-data result a useful warning against treating action
identity as transferable knowledge. The next driver must learn timing and
calibrate its expected gain on held-out data regimes, not merely recognize the
shadow action.

The same action set at 139M on TinyStories separates width from data effects:

| Width | No-op | Live average | Live extrapolate | Shadow merged |
| ---: | ---: | ---: | ---: | ---: |
| 768 | `0.85061` | `0.85546` | `0.85196` | `0.84538` |
| 1024 | `0.84911` | `0.85325` | `0.84876` | `0.83929` |

At 139M, shadow gained `0.00982` and extrapolation gained `0.00035`; averaging
still lost `0.00415`. Thus width changes the magnitude and even the sign of
the live extrapolation effect, while shadow remains the most reliable action
in this small changed-data sample. A selector trained on the FineWeb roots
plus the TinyStories 85M root had full support on the TinyStories 139M root,
but its predicted shadow delta was still positive (`+0.00193`, RMSE
`0.00759`). The conservative risk gate therefore abstained and the explicit
shadow fallback realized the `0.00982` gain. Support alone is not calibration.

### Direct nowcasting negative control

To test an actual skip rather than an earlier stopping point, the 85M
TinyStories parent at global step `1536` was replaced immediately by a
two-window secant prediction and evaluated as if it were the `2304` endpoint.
The full no-op endpoint is `0.85061`; secant strengths `0.25`, `0.5`, `1`,
`2`, `3`, and `4` produced losses `0.91128`, `0.92722`, `1.00218`, `1.40364`,
`2.01555`, and `2.55956`. Relative displacement energy grew from `0.10` to
`23.72` across that scan.

This rejects the current two-window linear extrapolator as a skip mechanism:
it does not predict a useful future weight state, and larger jumps become
unstable quickly. It does not reject a learned nowcaster, but that model must
be trained against future state or decision-relevant outcomes and calibrated
before it is allowed to bypass real training. The existing live extrapolation
action remains a low-cost negative control, not a claimed speedup.

### Within-TinyStories replication

Two additional 85M TinyStories roots kept the same parent and horizon:

| Seed | No-op | Live average | Live extrapolate | Shadow merged |
| ---: | ---: | ---: | ---: | ---: |
| 3 | `0.85061` | `0.85546` | `0.85196` | `0.84538` |
| 4 | `0.84452` | `0.84715` | `0.84521` | `0.83785` |
| 5 | `0.84602` | `0.84939` | `0.84659` | `0.83947` |

Shadow gains were `0.00523`, `0.00667`, and `0.00655`; live averaging lost on
all three seeds, and live extrapolation lost on seeds 4 and 5. A selector
trained only on TinyStories seeds 3–4 predicted shadow on seed 5 with RMSE
`0.00064`, but its leave-one-root support gate abstained because two roots are
not enough to calibrate the state space. Adding the FineWeb roots gave support
`1.0`, RMSE `0.00049`, and the same correct shadow choice. Its mixed-archive
prior still overestimated the gain (`0.0291` expected versus `0.00655`
realized), so the result supports action ranking within a regime but not
cross-regime magnitude prediction.

The same exact-parent `2048` timing test was repeated on TinyStories seeds 4
and 5. Candidates ended at `0.86053` and `0.86252`, versus full no-op endpoints
`0.84452` and `0.84602`; both saved `33.25%` of branch FLOPs but failed the
quality target. Together with seed 3, this rules out the current early timing
shortcut across three independent roots at 85M.

The saved full-endpoint snapshots were also rescored with shadow windows of
one, two, and three recent windows:

| Seed | Window 1 | Window 2 | Window 3 |
| ---: | ---: | ---: | ---: |
| 3 | `0.85045` | `0.84538` | `0.84861` |
| 4 | `0.84452` | `0.83785` | `0.84063` |
| 5 | `0.84599` | `0.83947` | `0.84292` |

Window 2 wins all three roots, so it is a reasonable fixed baseline action;
the scan does not yet justify a learned window controller.

Finally, a 128-wide history MLP trained to predict next-step loss changes
within TinyStories used 9,024 examples and achieved RMSE `0.01689` with
correlation `0.922`, versus constant-baseline RMSE `0.04189`. Training on the
mixed FineWeb/Tiny archive improved the same TinyStories holdout over baseline
but only reached RMSE `0.03512` and correlation `0.590`. This supports the
history-conditioned representation as a useful passive dynamics model while
identifying regime calibration—not steerer capacity—as the next constraint.
