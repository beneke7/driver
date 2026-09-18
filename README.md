# driver

Research harness for a landscape driver: a controller that observes training
dynamics, chooses bounded interventions, learns from their measured outcomes,
and is judged by end-to-end cost to a fixed capability.

This repository is deliberately at the contract stage. It contains no GPU
benchmark, learned optimizer, Dream-RSI clone, or claim of a training
speedup.

## First question

From identical checkpoints, can a state-dependent structured correction beat a
strong ordinary continuation after immediate, recovery, and longer horizons?

The first real comparison should keep the target architecture, data stream,
objective, evaluator, and seed policy fixed. Start with only two actions:

- `noop`: continue the baseline optimizer;
- `role_pulse`: apply a bounded, fixed-duration correction around the baseline
  update.

The driver must earn more actions, online learning, imagined rollouts, and
larger models by improving this comparison.

## Current seam

`driver/core.py` provides the smallest reusable boundary between a future
target-training loop and the research policy:

- validated observations, actions, outcomes, and causal transitions;
- an append-only JSONL archive with end-of-campaign causal validation;
- fixed, random, task-greedy, curiosity-only, and balanced selectors;
- a cheap per-run online action-statistics controller as the baseline that a
  learned numerical model must beat;
- an episode runner that records every intervention and its measured cost.

`driver/self_check.py` uses a deterministic toy target only to check that this
plumbing works. It is not a result and should never be used as evidence for
the research hypothesis.

Run it with:

```bash
python3 -m driver.self_check
```

## Autoresearch protocol

Each real run is an episode. Save the complete parent checkpoint before a
branch: model weights, optimizer and scheduler state, scaler state, random
generators, data position, configuration hash, code revision, and data hash.
Append one transition per decision to `runs/<run-id>/transitions.jsonl`.

The outer loop is intentionally small:

1. propose one bounded code or configuration change;
2. run baseline and candidate branches from matched checkpoints;
3. record immediate, recovery, and longer-horizon outcomes plus all costs;
4. replay alternative selectors only over recorded causal branches;
5. validate a survivor on fresh complete runs before promoting it.

Replay cannot reveal the outcome of an action that was never executed. A
learned simulator may propose counterfactuals later, but its selected actions
must be checked against real branches before entering the archive. This is the
useful Dream-RSI analogy without importing its code or making offline replay
claims it cannot support.

## Accounting

Report optimization, complete training-system, and knowledge-transfer tracks
separately. For quality target `Q*`, report

```text
S(Q*) = baseline_cost(Q*) / driver_cost(Q*)
```

Cost includes target work, driver inference and updates, probes, rejected
branches, evaluation, recovery, and failed runs. Keep one-time meta-training
and search costs separate, then state the deployment count used for any
amortized total. Use several quality thresholds, fresh seeds, whole-run
holdouts, and at least one changed data or architecture setting before making
a broad claim.

## Deferred until the seam earns it

The next implementation should be a fixed small decoder-LM loop with matched
checkpoint branches and the two-action comparison above. Defer deep numerical
models, PPO/SAC, imagined planning, weight nowcasting, data-mixture control,
pretrained language backbones, distributed execution, dashboards, and a
database until a simpler comparison identifies a bottleneck that needs them.

See [research/PLAN.md](research/PLAN.md) for the staged experiment contract.
