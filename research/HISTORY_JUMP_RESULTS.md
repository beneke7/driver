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
