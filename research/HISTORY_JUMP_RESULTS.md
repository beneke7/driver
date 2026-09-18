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
