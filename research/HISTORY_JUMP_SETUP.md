# History-jump setup

## Working hypothesis

AdamW training contains locally smooth, state-dependent update patterns. A
driver that sees recent updates, optimizer state, tensor-role geometry, and
data position may predict a useful future displacement and skip redundant
ordinary steps. The hypothesis is not that an arbitrary extrapolation can
invent information absent from the data stream.

For a target state `theta_t`, recent AdamW updates `u_(t-j)`, and a tensor-role
basis `B_t`, the first action family is:

```text
delta_jump = blend * B_t @ coefficients(driver(history_t))
theta_next = theta_t + delta_jump
```

The basis is allowed to change with the run. A jump must also specify what
happens to AdamW's first and second moments and where the data cursor moves.
The initial recovery rule is ordinary AdamW continuation with a bounded
fallback, not an optimistic reset of optimizer state.

## Data contract

Each recorded decision needs enough information to replay or reject it without
future leakage:

- complete parent checkpoint: model, AdamW state, scheduler/scaler state,
  Python/CPU/CUDA RNG, data cursor, configuration hash, code revision, and data
  hash;
- decision-time telemetry sampled from the current and previous steps;
- proposed action, executed action, confidence, risk estimate, and selector
  metadata;
- actual steps/tokens, driver work, probes, checkpoint I/O, wall time, and
  estimated FLOPs;
- immediate, recovery, and longer-horizon loss/capability outcomes;
- failure status and cost of returning to a useful AdamW trajectory.

The compact history should include loss slope and curvature proxies,
gradient/update norms and alignment, parameter norms, per-role geometry,
AdamW moment norms, recent update projections, and data position. Keep original
scales as auxiliary values where they affect control; normalize only with
statistics available to the driver at that decision.

## First collection campaign

The first campaign should collect teacher trajectories before training a deep
driver:

1. Reuse the existing deterministic toy landscapes to validate per-step
   telemetry and storage.
2. Run 12–24 complete AdamW trajectories across the three existing decoder
   landscapes and seeds. Keep the current 85M decoder and objective fixed.
3. Record every 4 optimizer steps with a 16-step history. Keep full
   checkpoints only at parent, branch, jump, and recovery boundaries; store
   compact telemetry and role-level update bases between them.
4. Split by whole trajectory. Development trajectories may train the predictor;
   held-out trajectories must not influence online initialization or action
   selection before evaluation.
5. Include controlled unsuccessful probes. Passive AdamW data cannot identify
   whether a state caused an outcome or merely inherited the policy that reached
   it.

The first action grid is horizons 8, 16, and 32 steps with blend factors
`0.25`, `0.5`, and `1.0`. Compare each action with a matched AdamW/noop branch.
Do not add imagined rollouts, PPO, SAC, a learned data mixture, or a
pretrained language backbone to this collection campaign.

## Driver shape

Start with one shared numerical causal transformer, hidden size 512, eight
layers, and eight heads, plus a small per-run adapter. It predicts future loss,
update coefficients, uncertainty, and recovery risk. This is deliberately
large enough to test the “embedded landscape knowledge” idea but small enough
that driver cost can be measured on the 5090.

The adapter comparison is:

1. snapshot-only predictor;
2. history-conditioned predictor;
3. recurrent per-run memory;
4. small online adapter;
5. online core updates only if the adapter cannot track response changes.

Train the first world model with supervised next-outcome targets. The driver
must first predict intervention outcomes and rank actions on fresh branches.
Only then add short model-predictive comparisons. A predicted jump is not an
executed result.

## Evaluation and promotion

Compare against fixed AdamW/noop, an open-loop schedule with the same tuning
budget, a shallow state controller, recent-history retrieval, and a
curiosity-only selector as a failure control. Report:

- cost to several predeclared quality thresholds;
- immediate, recovery, and longer-horizon outcomes;
- jump acceptance rate, failure rate, and recovery cost;
- prediction error, action-ranking regret, calibration, and selected-action
  prediction-versus-reality gap;
- end-to-end time, target FLOPs, driver work, probes, rejected branches,
  evaluation, and amortized meta/search cost separately.

Promotion requires fresh seeds, complete-run holdouts, and at least one changed
data or architecture condition. A useful intermediate result is a reliable
action-ranking or prediction gain; it is not a speedup until it survives the
full cost account. If a jump improves immediate loss but worsens recovery or
the final target, keep the failure in the archive and reject the action.

## Planned file boundary

```text
configs/history_jump_adamw.json   # versioned setup contract
data/README.md                    # local data and manifest rules
data/trajectory/                  # ignored collected traces
driver/checkpoints.py             # existing matched-branch primitive
driver/core.py                    # existing archive and selector seam
driver/history_jump.py            # trace collector and cheap jump benchmark
runs/history-jump-*/              # ignored experiment artifacts
```

The trace collector and first momentum-jump control now exist in
`driver/history_jump.py`. Do not create a general driver framework before the
saved traces and matched branches show predictive signal worth modeling.
