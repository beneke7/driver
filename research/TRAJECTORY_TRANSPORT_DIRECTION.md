# Trajectory transport and productive-region direction

Status: research-direction amendment for the decoder track, written 2026-09-19 after the timing-conditioned shadow screen at main commit edaf6c1.

This document preserves the existing contract and all prior negative results. It changes the next research question; it does not retroactively promote any pulse, shadow, Muon, or synthetic-quadratic result.

## Decision

The fixed pulse/catapult route is closed as a promotion route.

The current evidence shows:

- immediate loss improvement is not a safe proxy for long-horizon action value;
- adding absolute training position improves prediction calibration but has not produced held-out action ranking;
- the timing-conditioned shadow baseline is useful as a mechanism control at roughly 1.13x wall efficiency, not a learned driver or 10x result;
- the atlas selector still falls back to noop on unsupported or uncalibrated states;
- a larger policy, RL, imagined rollout, or pretrained language backbone is not justified yet.

The next route is learned trajectory transport with a cheap productive-region gate and few-shot initialization.

## Central hypothesis

A short calibration trajectory, together with architecture, data, optimizer, and history telemetry, may identify a mesoscopic training regime and generate a structured macro-action that moves the target model toward a future productive state.

The target is not merely a lower next-step loss. For state s, action a, and horizon H, the learned model should predict:

~~~text
response(s, a, H) =
    future latent state
    immediate/recovery/final capability
    cost and data exposure
    instability probability
    time-to-threshold
~~~

The contract-valid target remains the unchanged robust 10x gate: same locked capability at no more than one tenth of valid total deployment cost, including target work, driver work, probes, rejected/failed branches, recovery, evaluation, and shared prefix cost.

## What a transport action is

A transport action is not a scalar learning-rate pulse. It is a structured macro-action:

~~~text
a = (H, B, c, optimizer_state_transform, data_policy)

Delta_theta = B @ c
~~~

where:

- H is the proposed fast-forward horizon;
- B is a low-dimensional, role-wise, low-rank, or history-derived update basis;
- c contains the transport coefficients;
- optimizer_state_transform makes moments/scaler/scheduler state consistent;
- data_policy declares consumed, skipped, or reweighted data.

The first implementations must be structured and auditable. Arbitrary full-parameter teleportation is not admissible until a transport oracle and a state-consistency test show that it is meaningful.

## Productive-region classifier

The first deployable component may be a very cheap classifier rather than a deep policy.

Define action-conditioned labels from future real outcomes:

- productive: ordinary continuation reaches the target efficiently;
- jumpable: a declared transport action produces durable benefit;
- recoverable: an intervention is harmful briefly but safely recoverable;
- dangerous: an intervention causes a durable regression;
- stalled: ordinary continuation is inefficient or fails the target.

The classifier may use cheap current-state features:

- absolute step and consumed tokens;
- loss slope and curvature;
- train/validation gap;
- gradient and update norms;
- update alignment and noise estimates;
- layer-role norm balance;
- optimizer moment ratios;
- activation/attention summaries;
- recent action history;
- architecture and data metadata.

The fallback is always noop when support, calibration, or risk is insufficient.

## Data sources

Passive trajectory data may be used for representation pretraining and phase discovery:

- Pythia and PolyPythias for fixed-order, multi-seed, multi-scale trajectories;
- OLMo for openly released data, logs, and checkpoints;
- LLM360 for additional intermediate-checkpoint trajectories.

Passive data is not causal intervention evidence. It supplies state histories and future outcomes. All action claims require matched branches from immutable checkpoints in the driver harness.

Optimizer-state availability must be recorded per source. Public model checkpoints without optimizer state may be used for passive representation learning or oracle analysis, but not silently used as deployable jump states.

## Contract amendment for the next rung

The existing branch and promotion contract remains authoritative. The next campaign adds four explicitly separated evidence tracks:

1. Passive trajectory pretraining
   - trains state/phase encoders and future-outcome predictors;
   - never counts as a driver speedup.

2. Oracle ceiling
   - uses hindsight future checkpoints, hindsight productive-region labels, and data-utility oracles;
   - measures whether the action space could plausibly contain a 10x opportunity;
   - oracle results are diagnostic upper bounds, not deployment claims.

3. Learned transport
   - predicts a structured future-state displacement from current telemetry;
   - charges inference, update, recovery, data exposure, and failed attempts;
   - is evaluated on fresh complete roots.

4. Productive-region gating and few-shot initialization
   - uses only a short calibration prefix on a new run;
   - chooses noop, transport, data allocation, restart, or an explicitly declared initializer;
   - must beat fixed/action-only baselines before any planner is added.

Every branch still preserves parameters, optimizer/scheduler/scaler state, RNG states, data cursor, architecture, tokenizer, objective, code hash, and data hash. Immediate, recovery, and final outcomes remain mandatory.

The robust 10x claim remains eligible only when the existing preregistered requirements pass: complete matched coverage, at least five held-out landscapes with fresh seeds and thresholds or a declared equivalent, a changed-data or architecture holdout, zero catastrophic failures, every gated per-case ratio at least 10, landscape and seed medians at least 10, geometric mean at least 10, and the declared clustered lower bound at least 10.

## Experimental ladder

### Rung 1: passive state representation

Train a small numerical history model on public and local no-op trajectories. Predict phase, future loss/capability, and time-to-threshold. Hold out complete seeds and model widths.

### Rung 2: oracle ceiling

From matched 85M and 139M checkpoints, compare noop against:

- hindsight future-checkpoint transport;
- low-rank/role-wise transport approximation;
- trajectory-shadow endpoint;
- data-utility oracle;
- early-stop/nowcasting oracle.

If the best oracle cannot approach a large cost reduction, change the action hypothesis before training a larger model.

### Rung 3: learned transport

Predict future latent state and structured parameter/optimizer-state displacement for horizons 32, 128, and 512. Start with linear/low-rank and small MLP/GRU baselines. Do not begin with a large language backbone.

### Rung 4: productive-region gate

Train a calibrated classifier or conservative ensemble to decide whether to continue, transport, allocate data differently, or abstain. Evaluate action ranking and cost-to-threshold on locked fresh roots.

### Rung 5: few-shot initialization

Give the system only the first 1–5% of a new run. Allow it to select an initializer, trajectory anchor, optimizer-state warm start, or transport action. Charge all calibration and rejected attempts.

### Rung 6: planning and dreaming

Only after real selected actions remain calibrated on fresh branches may imagined rollouts, MPC, or model-based RL be added. Imagined transitions are proposals, never evidence.

## Decision rules

- If a hindsight transport oracle is weak, the current action space is wrong.
- If the oracle is strong but learned transport is weak, the representation or state-consistency model is the bottleneck.
- If data-utility oracles dominate, pivot toward data/work allocation.
- If early calibration predicts productive runs, combine seed/trajectory screening with transport.
- If no oracle route has large potential, do not scale the driver; reconsider the 10x mechanism at the initialization, data, or architecture level.

The project succeeds scientifically by discovering a transferable transport opportunity or by rigorously showing that the observed training states/actions do not contain one. Complexity must be earned by a failed cheaper rung.
