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
