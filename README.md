# driver

Research harness for a landscape driver: a controller that observes training
dynamics, chooses bounded interventions, learns from their measured outcomes,
and is judged by end-to-end cost to a fixed capability.

The first contract-grade result is now in the optimization track: a bounded
curvature driver passes a preregistered 10× gate on five synthetic positive-
definite landscape families. That result is deliberately scoped; it is not a
claim about language-model pretraining.

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

`driver/quadratic_benchmark.py` is the first real mechanism benchmark. It
batch-runs development and held-out positive-definite landscapes on CUDA,
tunes AdamW on development cases, and compares it with a gradient-oracle
structured correction. The correction tries a diagonal secant candidate, two
Hessian-vector steps, and a full local probe only when the cheaper actions do
not reach the declared target. Every probe and solve is charged in the
estimated-work ledger.

Set up the project-local CUDA environment:

```bash
uv sync
```

Run it with that environment:

```bash
.venv/bin/python -m driver.quadratic_benchmark \
  --output runs/quadratic-first
```

Evidence directories are append-safe: use a fresh `--output` for every run;
the benchmark refuses to append duplicate transition IDs to an existing run.

The matched-checkpoint boundary is independently checked with:

```bash
.venv/bin/python -m driver.torch_self_check
```

The contract evaluator and serial timing path are:

```bash
.venv/bin/python -m driver.serial_quadratic \
  --source runs/contract-q-1e-4 \
  --output runs/contract-q-1e-4-serial

.venv/bin/python -m driver.quadratic_campaign \
  --source runs/contract-q-1e-4-serial \
  --source runs/contract-q-3e-5-serial \
  --source runs/contract-q-1e-5-cap50k-serial \
  --output runs/quadratic-10x-campaign.json

.venv/bin/python -m driver.contract \
  --input runs/quadratic-10x-campaign.json \
  --output runs/quadratic-10x-report.json
```

The measured campaign is eligible: five families, three held-out seeds per
family, three predeclared thresholds, dimension 8, and synchronized serial
wall timing. The wall-time geometric means after the declared one-time-cost
amortization are 26.36×, 29.52×, and 43.07×; clustered 95% lower bounds are
22.82×, 24.34×, and 28.34×. See [research/RESULTS.md](research/RESULTS.md)
for the exact scope and limitations.

The first target-training branch loop is:

```bash
.venv/bin/python -m driver.decoder_benchmark \
  --landscape delayed_copy --output runs/decoder-delayed-copy-0
```

It runs a common prefix, saves a complete parent checkpoint, restores matched
`noop` and `role_pulse` branches, and records immediate, recovery, and final
validation loss. The pulse is a deliberately fixed control action; it is not
yet a learned driver.

The first full decoder development campaign is:

```bash
.venv/bin/python -m driver.decoder_campaign \
  --output runs/decoder-development-33044e4
```

The default is an 85.35M-parameter decoder, three deterministic landscapes ×
three seeds, and six same-budget policies: AdamW/noop, role pulse, open-loop
schedule, shallow state controller, history retrieval, and online learning
rate control. It writes a contract-v1 manifest, immutable parent checkpoints,
`transitions.jsonl`, action-ranking/calibration data, and capability-cost
curves. The completed development run produced 54/54 branch records with zero
failures. It is calibration evidence, not the five-landscape promotion gate;
see [research/DECODER_CAMPAIGN.md](research/DECODER_CAMPAIGN.md).

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

The current 10× result is only an optimization-track result on synthetic
ill-conditioned quadratics. The matched-checkpoint decoder loop remains a
negative control: its fixed `role_pulse` has not produced a useful speedup.
Target-model training, changed architecture/data holdouts, learned online
adaptation, and Dream-RSI-style experiment selection remain open.

## Deferred until the seam earns it

The next implementation should be a fixed small decoder-LM loop with matched
checkpoint branches and the two-action comparison above. Defer deep numerical
models, PPO/SAC, imagined planning, weight nowcasting, data-mixture control,
pretrained language backbones, distributed execution, dashboards, and a
database until a simpler comparison identifies a bottleneck that needs them.

See [research/PLAN.md](research/PLAN.md) for the staged experiment contract.

## History-jump setup

The next route is an AdamW history-jump driver. It learns from per-step
trajectories, predicts a bounded displacement from recent AdamW updates, and
only spends a real branch when the predicted gain clears its confidence and
risk checks. The setup is deliberately a contract, not a runnable driver yet:

- [research/OPEN_SOURCE_LANDSCAPE.md](research/OPEN_SOURCE_LANDSCAPE.md) records
  the reusable open-source models, datasets, and reference implementations;
- [research/HISTORY_JUMP_SETUP.md](research/HISTORY_JUMP_SETUP.md) defines the
  first data collection and promotion experiment;
- [configs/history_jump_adamw.json](configs/history_jump_adamw.json) is the
  versioned initial configuration;
- [data/README.md](data/README.md) defines the local data layout. Raw data and
  trajectory artifacts stay ignored until they have a manifest and checksum.

The first runnable bridge is `driver/history_jump.py`:

```bash
.venv/bin/python -m driver.history_jump --mode self-check
.venv/bin/python -m driver.history_jump --mode collect --output runs/history-jump-traces
.venv/bin/python -m driver.history_jump --mode benchmark \
  --source runs/history-jump-traces --output runs/history-jump-results
```

Collection writes complete AdamW checkpoints at the configured cadence and
per-step JSONL telemetry. Benchmarking starts after the first initialized
AdamW state and compares matched `noop` branches with bounded momentum jumps.
The default target is the existing 85M decoder; use `--allow-small-target` for
fast smoke runs.

No external repository or dataset is a runtime dependency yet. The first
implementation should reuse the existing checkpoint/branch contract and add
only per-step telemetry plus a compact AdamW update history.

See [research/OPERATING_CONTRACT.md](research/OPERATING_CONTRACT.md) for the
ongoing experiment, parallelism, accounting, and commit/push rules.

The first screen is summarized in
[research/HISTORY_JUMP_RESULTS.md](research/HISTORY_JUMP_RESULTS.md). It
rejects a global momentum pulse and motivates a history-conditioned safe
selector.
