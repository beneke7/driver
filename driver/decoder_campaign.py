"""Contract-v1 development campaign for the matched-checkpoint decoder loop.

The campaign deliberately keeps the action set small.  Every strategy starts
from the same immutable parent checkpoint and consumes the same number of
training tokens.  No imagined transitions are used; the history policy only
sees branches completed before the current parent.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoints import Checkpoint, load_checkpoint, save_checkpoint
from .core import Action, Archive, Observation, Transition
from .decoder_benchmark import (
    Block,
    Config,
    DecoderLM,
    TokenStream,
    _flops,
    _hash_bytes,
    _hash_config,
    _hash_tensor,
    _role,
    evaluate,
    make_byte_stream,
    make_stream,
)


LANDSCAPES = ("delayed_copy", "phase_switch", "text_shard")
SEEDS = (0, 1, 2)
STRATEGIES = (
    "noop",
    "role_pulse",
    "open_loop",
    "shallow_controller",
    "history_retrieval",
    "online_lr_control",
)
TRAJECTORY_STRATEGIES = ("trajectory_average", "trajectory_extrapolate")
ALL_STRATEGIES = STRATEGIES + TRAJECTORY_STRATEGIES
QUALITY_FACTORS = (0.995, 0.99, 0.98)


@dataclass(frozen=True)
class CampaignConfig:
    width: int = 768
    layers: int = 12
    heads: int = 12
    vocab_size: int = 128
    context: int = 128
    batch_size: int = 128
    prefix_steps: int = 32
    immediate_steps: int = 4
    recovery_steps: int = 32
    final_steps: int = 128
    validation_batches: int = 8
    train_file: str | None = None
    validation_file: str | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    amp: bool = True
    pulse_attention: float = 1.05
    pulse_mlp: float = 0.95
    open_loop_start: float = 1.10
    open_loop_end: float = 0.95
    trajectory_interval: int = 32
    trajectory_window: int = 4
    trajectory_alpha: float = 1.0
    controller_up: float = 1.05
    controller_down: float = 0.80
    online_up: float = 1.05
    online_down: float = 0.70


@dataclass(frozen=True)
class Proposal:
    strategy: str
    selected_schedule: str
    predicted_gain: float
    predicted_final_loss: float
    uncertainty: float
    lr_multiplier: float = 1.0


@dataclass
class HistoryEntry:
    features: dict[str, float]
    selected_schedule: str
    gain: float
    final_loss: float
    seed: int


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _require_clean_repo(allow_dirty: bool) -> None:
    if allow_dirty:
        return
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify repository cleanliness") from exc
    if status.strip():
        raise RuntimeError(
            "decoder campaign requires a clean repository; commit changes or pass "
            "--allow-dirty for a smoke run"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _campaign_code_sha256() -> str:
    root = Path(__file__).resolve().parent
    files = (root / "decoder_benchmark.py", root / "decoder_campaign.py", root / "checkpoints.py", root / "core.py")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _target_config(campaign: CampaignConfig, landscape: str, seed: int) -> Config:
    return Config(
        landscape=landscape,
        seed=seed,
        vocab_size=campaign.vocab_size,
        context=campaign.context,
        batch_size=campaign.batch_size,
        width=campaign.width,
        layers=campaign.layers,
        heads=campaign.heads,
        prefix_steps=campaign.prefix_steps,
        immediate_steps=campaign.immediate_steps,
        recovery_steps=campaign.recovery_steps,
        final_steps=campaign.final_steps,
        learning_rate=campaign.learning_rate,
        weight_decay=campaign.weight_decay,
        pulse_attention=campaign.pulse_attention,
        pulse_mlp=campaign.pulse_mlp,
    )


def _model_and_optimizer(
    config: Config, device: torch.device
) -> tuple[DecoderLM, torch.optim.Optimizer]:
    model = DecoderLM(config).to(device)
    grouped: dict[str, list[torch.nn.Parameter]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        grouped[_role(name)].append(parameter)
    parameters = [
        {
            "params": grouped[role],
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "role": role,
        }
        for role in ("embedding", "attention", "mlp", "norm", "head")
    ]
    return model, torch.optim.AdamW(parameters)


def _set_group_lrs(
    optimizer: torch.optim.Optimizer,
    config: CampaignConfig,
    *,
    global_multiplier: float = 1.0,
    role_multipliers: dict[str, float] | None = None,
) -> None:
    role_multipliers = role_multipliers or {}
    for group in optimizer.param_groups:
        role = str(group["role"])
        group["lr"] = config.learning_rate * global_multiplier * role_multipliers.get(role, 1.0)


def _autocast(config: CampaignConfig, device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=config.amp and device.type == "cuda",
    )


def _safe_float(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError("non-finite decoder telemetry")
    return value


def _telemetry(
    model: DecoderLM,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
) -> dict[str, Any]:
    grad_sq = torch.zeros((), device=loss.device, dtype=torch.float64)
    update_sq = torch.zeros_like(grad_sq)
    alignment_dot = torch.zeros_like(grad_sq)
    momentum_sq = torch.zeros_like(grad_sq)
    variance_sq = torch.zeros_like(grad_sq)
    parameter_sq = torch.zeros_like(grad_sq)
    role_grad: dict[str, torch.Tensor] = {}
    role_update: dict[str, torch.Tensor] = {}
    role_momentum: dict[str, torch.Tensor] = {}
    role_variance: dict[str, torch.Tensor] = {}
    role_parameter: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        role = str(group["role"])
        role_grad[role] = torch.zeros_like(grad_sq)
        role_update[role] = torch.zeros_like(grad_sq)
        role_momentum[role] = torch.zeros_like(grad_sq)
        role_variance[role] = torch.zeros_like(grad_sq)
        role_parameter[role] = torch.zeros_like(grad_sq)
        lr = float(group["lr"])
        for parameter in group["params"]:
            parameter_value = parameter.detach().float()
            parameter_value_sq = parameter_value.square().sum(dtype=torch.float64)
            parameter_sq += parameter_value_sq
            role_parameter[role] += parameter_value_sq
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            gradient_sq = gradient.square().sum(dtype=torch.float64)
            grad_sq += gradient_sq
            role_grad[role] += gradient_sq
            state = optimizer.state.get(parameter, {})
            momentum = state.get("exp_avg")
            second = state.get("exp_avg_sq")
            if momentum is None or second is None:
                continue
            momentum = momentum.detach().float()
            second = second.detach().float()
            momentum_value_sq = momentum.square().sum(dtype=torch.float64)
            variance_value_sq = second.square().sum(dtype=torch.float64)
            momentum_sq += momentum_value_sq
            variance_sq += variance_value_sq
            role_momentum[role] += momentum_value_sq
            role_variance[role] += variance_value_sq
            step_value = state.get("step", 1.0)
            step = float(step_value.item() if torch.is_tensor(step_value) else step_value)
            beta1, beta2 = 0.9, 0.999
            normalized = (momentum / (1.0 - beta1**step)) / (
                (second / (1.0 - beta2**step)).sqrt() + 1e-8
            )
            update = lr * normalized
            update_value = update.square().sum(dtype=torch.float64)
            update_sq += update_value
            role_update[role] += update_value
            alignment_dot += (gradient * momentum).sum(dtype=torch.float64)
    grad_norm = grad_sq.sqrt()
    update_norm = update_sq.sqrt()
    alignment = alignment_dot / (grad_sq.sqrt() * momentum_sq.sqrt()).clamp_min(1e-12)
    return {
        "loss": _safe_float(loss.detach().item()),
        "gradient_norm": _safe_float(grad_norm.item()),
        "update_norm": _safe_float(update_norm.item()),
        "momentum_alignment": _safe_float(alignment.item()),
        "parameter_norm": _safe_float(parameter_sq.sqrt().item()),
        "adam_momentum_norm": _safe_float(momentum_sq.sqrt().item()),
        "adam_variance_norm": _safe_float(variance_sq.sqrt().item()),
        "role_gradient_norms": {
            role: _safe_float(value.sqrt().item()) for role, value in role_grad.items()
        },
        "role_update_norms": {
            role: _safe_float(value.sqrt().item()) for role, value in role_update.items()
        },
        "role_parameter_norms": {
            role: _safe_float(value.sqrt().item()) for role, value in role_parameter.items()
        },
        "adam_momentum_norms": {
            role: _safe_float(value.sqrt().item()) for role, value in role_momentum.items()
        },
        "adam_variance_norms": {
            role: _safe_float(value.sqrt().item()) for role, value in role_variance.items()
        },
    }


def _feature_vector(telemetry: dict[str, Any], loss_slope: float) -> dict[str, float]:
    return {
        "loss": float(telemetry["loss"]),
        "loss_slope": float(loss_slope),
        "gradient_norm": float(telemetry["gradient_norm"]),
        "update_norm": float(telemetry["update_norm"]),
        "momentum_alignment": float(telemetry["momentum_alignment"]),
        "parameter_norm": float(telemetry.get("parameter_norm", 0.0)),
        "adam_momentum_norm": float(telemetry.get("adam_momentum_norm", 0.0)),
        "adam_variance_norm": float(telemetry.get("adam_variance_norm", 0.0)),
    }


def _feature_distance(left: dict[str, float], right: dict[str, float]) -> float:
    scales = {"loss": 1.0, "loss_slope": 1e-2, "gradient_norm": 1.0, "update_norm": 1.0, "momentum_alignment": 1.0}
    return sum(
        ((left[name] - right[name]) / scales[name]) ** 2 for name in scales
    ) ** 0.5


def _nearest_history(
    features: dict[str, float], history: list[HistoryEntry], schedule: str | None = None
) -> HistoryEntry | None:
    candidates = [
        item
        for item in history
        if schedule is None or item.selected_schedule == schedule
    ]
    return min(candidates, key=lambda item: _feature_distance(features, item.features), default=None)


def _schedule_for_proposal(
    proposal: Proposal,
    *,
    parent_features: dict[str, float],
) -> str:
    if proposal.strategy == "shallow_controller":
        score = (
            -parent_features["loss_slope"]
            + 0.05 * parent_features["momentum_alignment"]
            - 0.01 * math.log1p(parent_features["update_norm"])
            + 0.001 * math.log1p(parent_features["gradient_norm"])
        )
        return "role_pulse" if score > 0.0 else "noop"
    return proposal.selected_schedule


def _predicted_gain(
    schedule: str,
    *,
    parent_features: dict[str, float],
    history: list[HistoryEntry],
    horizon: int,
) -> tuple[float, float]:
    base = max(0.0, -parent_features["loss_slope"] * max(1, horizon) * 0.5)
    nearest = _nearest_history(parent_features, history, schedule)
    if nearest is None:
        return base, max(base, parent_features["loss"] * 0.05)
    distance = _feature_distance(parent_features, nearest.features)
    uncertainty = max(base * 0.25, distance * 0.01)
    return max(-parent_features["loss"], nearest.gain), uncertainty


def _proposals(
    *,
    parent_features: dict[str, float],
    campaign: CampaignConfig,
    history: list[HistoryEntry],
) -> list[Proposal]:
    proposal_schedules = {
        "noop": "noop",
        "role_pulse": "role_pulse",
        "open_loop": "open_loop",
        "shallow_controller": "noop",
        "history_retrieval": "noop",
        "online_lr_control": "online_lr_control",
        "trajectory_average": "trajectory_average",
        "trajectory_extrapolate": "trajectory_extrapolate",
    }
    shallow_score = (
        -parent_features["loss_slope"]
        + 0.05 * parent_features["momentum_alignment"]
        - 0.01 * math.log1p(parent_features["update_norm"])
        + 0.001 * math.log1p(parent_features["gradient_norm"])
    )
    proposal_schedules["shallow_controller"] = (
        "role_pulse" if shallow_score > 0.0 else "noop"
    )
    nearest = _nearest_history(parent_features, history)
    if nearest is not None:
        proposal_schedules["history_retrieval"] = nearest.selected_schedule
    proposals: list[Proposal] = []
    for strategy in ALL_STRATEGIES:
        schedule = proposal_schedules[strategy]
        gain, uncertainty = _predicted_gain(
            schedule,
            parent_features=parent_features,
            history=history,
            horizon=campaign.final_steps,
        )
        if strategy == "online_lr_control":
            gain, uncertainty = _predicted_gain(
                "noop",
                parent_features=parent_features,
                history=history,
                horizon=campaign.final_steps,
            )
        proposals.append(
            Proposal(
                strategy=strategy,
                selected_schedule=schedule,
                predicted_gain=gain,
                predicted_final_loss=parent_features["loss"] - gain,
                uncertainty=uncertainty,
            )
        )
    return proposals


def _schedule_multipliers(
    schedule: str,
    step: int,
    *,
    campaign: CampaignConfig,
    runtime: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    if schedule == "role_pulse" and step < campaign.immediate_steps:
        return 1.0, {"attention": campaign.pulse_attention, "mlp": campaign.pulse_mlp}
    if schedule == "open_loop":
        span = max(1, campaign.immediate_steps)
        fraction = min(1.0, step / span)
        multiplier = campaign.open_loop_start + fraction * (
            campaign.open_loop_end - campaign.open_loop_start
        )
        return multiplier, {}
    if schedule in {"online_lr_control", "controller"}:
        return float(runtime.get("lr_multiplier", 1.0)), {}
    return 1.0, {}


def _train_step(
    model: DecoderLM,
    optimizer: torch.optim.Optimizer,
    stream: TokenStream,
    target_config: Config,
    campaign: CampaignConfig,
    device: torch.device,
    *,
    step: int,
    schedule: str,
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    global_multiplier, role_multipliers = _schedule_multipliers(
        schedule, step, campaign=campaign, runtime=runtime
    )
    _set_group_lrs(
        optimizer,
        campaign,
        global_multiplier=global_multiplier,
        role_multipliers=role_multipliers,
    )
    tokens, targets = stream.batch(
        batch_size=target_config.batch_size,
        context=target_config.context,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    with _autocast(campaign, device):
        loss = model(tokens, targets)
    loss.backward()
    telemetry = _telemetry(model, optimizer, loss)
    optimizer.step()
    telemetry["step"] = step + 1
    telemetry["global_lr_multiplier"] = global_multiplier
    telemetry["schedule"] = schedule
    return telemetry, target_config.batch_size * target_config.context


def _evaluate(
    model: DecoderLM,
    values: torch.Tensor,
    *,
    target_config: Config,
    campaign: CampaignConfig,
    device: torch.device,
) -> float:
    width = target_config.batch_size * (target_config.context + 1)
    required = width * campaign.validation_batches
    if values.numel() < required:
        raise ValueError("validation stream is shorter than validation_batches")
    losses = []
    for offset in range(0, required, width):
        chunk = values[offset : offset + width].view(
            target_config.batch_size, target_config.context + 1
        )
        tokens = chunk[:, :-1].to(device=device, dtype=torch.long)
        targets = chunk[:, 1:].to(device=device, dtype=torch.long)
        with torch.no_grad(), _autocast(campaign, device):
            losses.append(_safe_float(model(tokens, targets).item()))
    return sum(losses) / len(losses)


def _trajectory_snapshot_steps(campaign: CampaignConfig) -> tuple[int, ...]:
    if campaign.trajectory_interval <= 0:
        raise ValueError("trajectory_interval must be positive")
    if campaign.trajectory_window <= 0:
        raise ValueError("trajectory_window must be positive")
    count = 2 * campaign.trajectory_window
    first = campaign.prefix_steps - (count - 1) * campaign.trajectory_interval
    if first <= 0:
        raise ValueError(
            "prefix_steps must contain two trajectory windows; increase prefix_steps "
            "or reduce trajectory_interval/trajectory_window"
        )
    return tuple(
        first + index * campaign.trajectory_interval for index in range(count)
    )


def _save_model_snapshot(
    path: Path,
    *,
    model: DecoderLM,
    step: int,
    config_sha: str,
    data_sha: str,
) -> None:
    payload = {
        "metadata": {
            "config_sha256": config_sha,
            "data_sha256": data_sha,
            "step": step,
        },
        "model": {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _train_prefix(
    model: DecoderLM,
    optimizer: torch.optim.Optimizer,
    stream: TokenStream,
    target_config: Config,
    campaign: CampaignConfig,
    device: torch.device,
    *,
    snapshot_steps: tuple[int, ...] = (),
    snapshot_dir: Path | None = None,
    config_sha: str | None = None,
    data_sha: str | None = None,
) -> tuple[list[dict[str, Any]], float, int, list[Path]]:
    if snapshot_steps and snapshot_dir is None:
        raise ValueError("snapshot_dir is required when snapshot_steps are provided")
    if snapshot_steps and (config_sha is None or data_sha is None):
        raise ValueError("snapshot hashes are required when snapshots are provided")
    snapshot_lookup = set(snapshot_steps)
    snapshots: list[Path] = []
    started = time.perf_counter()
    telemetry: list[dict[str, Any]] = []
    tokens = 0
    for step in range(campaign.prefix_steps):
        current, consumed = _train_step(
            model,
            optimizer,
            stream,
            target_config,
            campaign,
            device,
            step=step,
            schedule="noop",
            runtime={},
        )
        telemetry.append(current)
        tokens += consumed
        step_number = step + 1
        if step_number in snapshot_lookup:
            path = snapshot_dir / f"step-{step_number:06d}.pt"
            _save_model_snapshot(
                path,
                model=model,
                step=step_number,
                config_sha=config_sha,
                data_sha=data_sha,
            )
            snapshots.append(path)
    _sync(device)
    return telemetry, time.perf_counter() - started, tokens, snapshots


def _loss_slope(history: list[dict[str, Any]]) -> float:
    if len(history) < 2:
        return 0.0
    return (history[-1]["loss"] - history[0]["loss"]) / max(1, len(history) - 1)


def _load_model_snapshot(
    path: Path,
    *,
    config_sha: str,
    data_sha: str,
) -> tuple[dict[str, torch.Tensor], int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"trajectory snapshot is missing metadata: {path}")
    if metadata.get("config_sha256") != config_sha or metadata.get("data_sha256") != data_sha:
        raise ValueError(f"trajectory snapshot provenance mismatch: {path}")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"trajectory snapshot is missing model state: {path}")
    return state, int(metadata["step"])


@torch.no_grad()
def _apply_trajectory_maneuver(
    model: DecoderLM,
    *,
    strategy: str,
    snapshots: tuple[Path, ...],
    window: int,
    alpha: float,
    config_sha: str,
    data_sha: str,
) -> dict[str, Any]:
    if strategy not in TRAJECTORY_STRATEGIES:
        raise ValueError(f"unknown trajectory strategy: {strategy}")
    if len(snapshots) != 2 * window:
        raise ValueError("trajectory snapshots do not contain two complete windows")
    if not math.isfinite(alpha):
        raise ValueError("trajectory_alpha must be finite")

    parameters = dict(model.named_parameters())
    past = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
    recent = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
    snapshot_steps: list[int] = []
    for index, path in enumerate(snapshots):
        state, snapshot_step = _load_model_snapshot(
            path, config_sha=config_sha, data_sha=data_sha
        )
        snapshot_steps.append(snapshot_step)
        target = past if index < window else recent
        for name, parameter in parameters.items():
            value = state.get(name)
            if value is None or value.shape != parameter.shape:
                raise ValueError(f"trajectory snapshot is missing parameter {name}")
            target[name].add_(value.to(device=parameter.device, dtype=parameter.dtype))
        del state

    energy = torch.zeros((), device=next(iter(parameters.values())).device, dtype=torch.float64)
    parameter_count = 0
    for name, parameter in parameters.items():
        past[name].div_(window)
        recent[name].div_(window)
        if strategy == "trajectory_average":
            candidate = recent[name]
        else:
            # Deliberate ceiling: a two-window secant is cheaper than full PCA over
            # merged checkpoints; upgrade only if this leaves measurable signal.
            candidate = recent[name] + alpha * (recent[name] - past[name])
        delta = candidate - parameter
        energy += delta.float().square().sum(dtype=torch.float64) / (
            parameter.float().square().sum(dtype=torch.float64) + 1e-12
        )
        parameter.copy_(candidate)
        parameter_count += parameter.numel()

    return {
        "kind": strategy,
        "snapshot_steps": snapshot_steps,
        "window": window,
        "alpha": 0.0 if strategy == "trajectory_average" else alpha,
        "energy": float(energy.item()),
        "optimizer_state": "preserved",
        "flops": 2.0 * parameter_count * len(snapshots),
    }


def _branch(
    *,
    proposal: Proposal,
    parent: Checkpoint,
    target_config: Config,
    campaign: CampaignConfig,
    train_values: torch.Tensor,
    validation_values: torch.Tensor,
    config_sha: str,
    data_sha: str,
    code_sha: str,
    before: Observation,
    device: torch.device,
    trajectory_snapshots: tuple[Path, ...] = (),
) -> tuple[Transition, dict[str, Any]]:
    model: DecoderLM | None = None
    optimizer: torch.optim.Optimizer | None = None
    started = time.perf_counter()
    try:
        model, optimizer = _model_and_optimizer(target_config, device)
        metadata = load_checkpoint(
            parent,
            model=model,
            optimizer=optimizer,
            config_sha256=config_sha,
            data_sha256=data_sha,
        )
        stream = TokenStream(
            train_values, cursor=int(metadata["data_state"]["cursor"])
        )
        parent_loss = before.loss
        selected_schedule = proposal.selected_schedule
        if proposal.strategy == "shallow_controller":
            selected_schedule = _schedule_for_proposal(
                proposal, parent_features=before.features
            )
        runtime: dict[str, Any] = {"lr_multiplier": 1.0, "adapter_updates": 0}
        maneuver: dict[str, Any] | None = None
        intervention_loss: float | None = None
        if proposal.strategy in TRAJECTORY_STRATEGIES:
            maneuver = _apply_trajectory_maneuver(
                model,
                strategy=proposal.strategy,
                snapshots=trajectory_snapshots,
                window=campaign.trajectory_window,
                alpha=campaign.trajectory_alpha,
                config_sha=config_sha,
                data_sha=data_sha,
            )
            intervention_loss = _evaluate(
                model,
                validation_values,
                target_config=target_config,
                campaign=campaign,
                device=device,
            )
        telemetry: list[dict[str, Any]] = []
        phase_metrics: list[dict[str, Any]] = [
            {
                "name": "parent",
                "loss": parent_loss,
                "tokens": before.tokens,
                "wall_seconds": 0.0,
            }
        ]
        consumed_tokens = 0
        phase_steps = (
            campaign.immediate_steps,
            campaign.recovery_steps,
            campaign.final_steps,
        )
        phase_names = ("immediate", "recovery", "final")
        previous = 0
        for phase_index, (boundary, phase_name) in enumerate(
            zip(phase_steps, phase_names)
        ):
            phase_started = time.perf_counter()
            for step in range(previous, boundary):
                current, consumed = _train_step(
                    model,
                    optimizer,
                    stream,
                    target_config,
                    campaign,
                    device,
                    step=step,
                    schedule=selected_schedule,
                    runtime=runtime,
                )
                telemetry.append(current)
                consumed_tokens += consumed
            immediate_loss: float | None = None
            if proposal.strategy == "online_lr_control" and phase_index == 0:
                immediate_loss = _evaluate(
                    model,
                    validation_values,
                    target_config=target_config,
                    campaign=campaign,
                    device=device,
                )
                runtime["lr_multiplier"] = (
                    campaign.online_up
                    if immediate_loss < parent_loss
                    else campaign.online_down
                )
                runtime["adapter_updates"] += 1
            loss = immediate_loss if immediate_loss is not None else _evaluate(
                model,
                validation_values,
                target_config=target_config,
                campaign=campaign,
                device=device,
            )
            _sync(device)
            phase_metrics.append(
                {
                    "name": phase_name,
                    "loss": loss,
                    "tokens": before.tokens + consumed_tokens,
                    "wall_seconds": time.perf_counter() - started,
                    "phase_wall_seconds": time.perf_counter() - phase_started,
                }
            )
            previous = boundary
        final_loss = phase_metrics[-1]["loss"]
        target_tokens = consumed_tokens
        target_flops = _flops(
            target_config,
            model,
            target_tokens,
            campaign.immediate_steps if selected_schedule == "role_pulse" else 0,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        evaluation_flops = 2.0 * parameter_count * target_config.batch_size * target_config.context * (
            3 + int(intervention_loss is not None)
        )
        driver_inference_flops = 64.0
        driver_update_flops = 1.0 if proposal.strategy == "online_lr_control" else 0.0
        maneuver_flops = 0.0 if maneuver is None else float(maneuver["flops"])
        recovery_flops = _flops(
            target_config,
            model,
            (campaign.recovery_steps - campaign.immediate_steps)
            * target_config.batch_size
            * target_config.context,
        )
        _sync(device)
        wall_seconds = time.perf_counter() - started
        after = Observation(
            step=target_config.prefix_steps + campaign.final_steps,
            tokens=before.tokens + target_tokens,
            loss=final_loss,
            quality=-final_loss,
            compute_flops=target_flops + evaluation_flops + maneuver_flops,
            features={
                "loss_slope": _loss_slope(telemetry),
                "gradient_norm": telemetry[-1]["gradient_norm"],
                "update_norm": telemetry[-1]["update_norm"],
                "momentum_alignment": telemetry[-1]["momentum_alignment"],
            },
        )
        schedule_codes = {
            "noop": 0.0,
            "role_pulse": 1.0,
            "open_loop": 2.0,
            "online_lr_control": 3.0,
            "trajectory_average": 4.0,
            "trajectory_extrapolate": 5.0,
        }
        transition = Transition(
            transition_id=f"{target_config.landscape}-{target_config.seed}:{proposal.strategy}",
            run_id=f"{target_config.landscape}-{target_config.seed}",
            parent_id=None,
            before=before,
            action=Action(
                proposal.strategy,
                strength=proposal.lr_multiplier,
                parameters={
                    "predicted_gain": proposal.predicted_gain,
                    "uncertainty": proposal.uncertainty,
                    "schedule_code": schedule_codes[selected_schedule],
                },
            ),
            after=after,
            compute_flops=(
                target_flops
                + evaluation_flops
                + maneuver_flops
                + driver_inference_flops
                + driver_update_flops
            ),
            wall_seconds=wall_seconds,
            reward_task=parent_loss - final_loss,
            learning_progress=parent_loss - final_loss,
            accepted=True,
            metadata={
                "strategy": proposal.strategy,
                "selected_schedule": selected_schedule,
                "prediction": {
                    "gain": proposal.predicted_gain,
                    "uncertainty": proposal.uncertainty,
                    "predicted_final_loss": parent_loss - proposal.predicted_gain,
                },
                "maneuver": maneuver,
                "intervention_loss": intervention_loss,
                "horizon_metrics": phase_metrics,
                "telemetry": telemetry,
                "target_tokens": target_tokens,
                "target_flops": target_flops,
                "driver_inference_flops": driver_inference_flops,
                "driver_update_flops": driver_update_flops,
                "maneuver_flops": maneuver_flops,
                "probe_flops": 0.0,
                "recovery_flops": recovery_flops,
                "evaluation_flops": evaluation_flops,
                "driver_cost": {
                    "inference_flops": driver_inference_flops,
                    "update_flops": driver_update_flops,
                    "maneuver_flops": maneuver_flops,
                    "probe_flops": 0.0,
                    "recovery_flops": recovery_flops,
                    "evaluation_flops": evaluation_flops,
                },
                "adapter_updates": runtime["adapter_updates"],
                "source_checkpoint_sha256": parent.sha256,
                "config_sha256": config_sha,
                "data_sha256": data_sha,
                "code_sha": code_sha,
            },
        )
        return transition, {
            "strategy": proposal.strategy,
            "selected_schedule": selected_schedule,
            "predicted_gain": proposal.predicted_gain,
            "predicted_final_loss": parent_loss - proposal.predicted_gain,
            "final_loss": final_loss,
            "gain": parent_loss - final_loss,
            "intervention_loss": intervention_loss,
            "maneuver": maneuver,
            "wall_seconds": wall_seconds,
            "target_flops": target_flops,
            "target_tokens": target_tokens,
            "phase_metrics": phase_metrics,
            "telemetry": telemetry,
            "failed": False,
        }
    except Exception as exc:
        _sync(device)
        wall_seconds = time.perf_counter() - started
        failure = f"{type(exc).__name__}: {exc}"
        transition = Transition(
            transition_id=f"{target_config.landscape}-{target_config.seed}:{proposal.strategy}",
            run_id=f"{target_config.landscape}-{target_config.seed}",
            parent_id=None,
            before=before,
            action=Action(proposal.strategy, strength=proposal.lr_multiplier),
            after=None,
            compute_flops=0.0,
            wall_seconds=max(0.0, wall_seconds),
            reward_task=-before.loss,
            learning_progress=0.0,
            accepted=False,
            failure=failure,
            metadata={
                "strategy": proposal.strategy,
                "prediction": asdict(proposal),
                "source_checkpoint_sha256": parent.sha256,
                "config_sha256": config_sha,
                "data_sha256": data_sha,
                "code_sha": code_sha,
            },
        )
        return transition, {
            "strategy": proposal.strategy,
            "selected_schedule": proposal.selected_schedule,
            "predicted_gain": proposal.predicted_gain,
            "predicted_final_loss": before.loss - proposal.predicted_gain,
            "final_loss": math.inf,
            "gain": -math.inf,
            "wall_seconds": max(0.0, wall_seconds),
            "target_flops": 0.0,
            "target_tokens": 0,
            "phase_metrics": [],
            "telemetry": [],
            "failed": True,
            "failure": failure,
        }
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _hash_json(value: Any) -> str:
    return _hash_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _parent_features(
    telemetry: list[dict[str, Any]], loss_slope: float
) -> dict[str, float]:
    if not telemetry:
        raise ValueError("prefix must produce telemetry")
    features = _feature_vector(telemetry[-1], loss_slope)
    for role, value in telemetry[-1]["role_gradient_norms"].items():
        features[f"gradient_{role}"] = float(value)
    for role, value in telemetry[-1]["role_update_norms"].items():
        features[f"update_{role}"] = float(value)
    return features


def _phase_point(
    phase: dict[str, Any],
    *,
    before: Observation,
    result: dict[str, Any],
) -> dict[str, Any]:
    target_tokens = max(1, result["target_tokens"])
    target_flops = result["target_flops"]
    branch_tokens = max(0, phase["tokens"] - before.tokens)
    return {
        "name": phase["name"],
        "step": before.step + (phase["tokens"] - before.tokens) // max(1, before.features.get("batch_context_tokens", 1)),
        "tokens": phase["tokens"],
        "wall_seconds": before.features.get("prefix_seconds", 0.0) + phase["wall_seconds"],
        "flops": before.compute_flops + target_flops * branch_tokens / target_tokens,
        "validation_loss": phase["loss"],
        "capability": -phase["loss"],
    }


def _threshold_result(
    points: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    for index, point in enumerate(points[1:], start=1):
        if point["validation_loss"] <= threshold and all(
            later["validation_loss"] <= threshold for later in points[index:]
        ):
            return {
                "reached": True,
                "wall_seconds": point["wall_seconds"],
                "tokens": point["tokens"],
                "flops": point["flops"],
            }
    return {"reached": False, "failure": "max_steps"}


def _ranking_report(
    ranking_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    finite = [row for row in ranking_rows if row["actual_gain"] is not None]
    if not finite:
        return {"rows": ranking_rows, "summary": {"count": 0}}
    errors = [row["predicted_gain"] - row["actual_gain"] for row in finite]
    regrets = [row["final_regret"] for row in finite if row["final_regret"] is not None]
    horizon_regret_means = {
        name: (
            sum(row[f"{name}_regret"] for row in finite if row[f"{name}_regret"] is not None)
            / len([row for row in finite if row[f"{name}_regret"] is not None])
            if any(row[f"{name}_regret"] is not None for row in finite)
            else None
        )
        for name in ("immediate", "recovery", "final")
    }
    bins: list[dict[str, Any]] = []
    ordered = sorted(finite, key=lambda row: row["predicted_gain"])
    bin_size = max(1, math.ceil(len(ordered) / 5))
    for start in range(0, len(ordered), bin_size):
        group = ordered[start : start + bin_size]
        bins.append(
            {
                "n": len(group),
                "mean_predicted_gain": sum(row["predicted_gain"] for row in group) / len(group),
                "mean_actual_gain": sum(row["actual_gain"] for row in group) / len(group),
            }
        )
    selected = [
        row
        for row in ranking_rows
        if row.get("strategy") == row.get("predicted_best")
    ]
    agreement = [row for row in selected if row.get("predicted_best") == row.get("actual_best")]
    return {
        "rows": ranking_rows,
        "summary": {
            "count": len(finite),
            "prediction_rmse": (sum(error * error for error in errors) / len(errors)) ** 0.5,
            "mean_regret": sum(regrets) / len(regrets) if regrets else None,
            "median_regret": sorted(regrets)[len(regrets) // 2] if regrets else None,
            "mean_horizon_regret": horizon_regret_means,
            "top1_agreement": len(agreement) / len(selected) if selected else None,
            "calibration_bins": bins,
        },
    }


def _run_case(
    *,
    campaign: CampaignConfig,
    landscape: str,
    seed: int,
    output: Path,
    archive: Archive,
    device: torch.device,
    code_revision: str,
    code_sha256: str,
    uv_lock_sha256: str,
    objective_sha256: str,
    history: list[HistoryEntry],
    strategies: tuple[str, ...],
) -> dict[str, Any]:
    target_config = _target_config(campaign, landscape, seed)
    train_tokens = (
        campaign.prefix_steps + campaign.final_steps + 2
    ) * campaign.batch_size * (campaign.context + 1)
    validation_tokens = (
        campaign.validation_batches * campaign.batch_size * (campaign.context + 1)
    )
    if campaign.train_file:
        if campaign.vocab_size < 256:
            raise ValueError("byte-corpus runs require vocab_size >= 256")
        train_values = make_byte_stream(
            campaign.train_file, seed=seed, length=train_tokens
        )
        validation_values = make_byte_stream(
            campaign.validation_file or campaign.train_file,
            seed=seed + 1_000_000,
            length=validation_tokens,
        )
    else:
        train_values = make_stream(
            landscape,
            seed=seed,
            length=train_tokens,
            vocab_size=campaign.vocab_size,
        )
        validation_values = make_stream(
            landscape,
            seed=seed + 1_000_000,
            length=validation_tokens,
            vocab_size=campaign.vocab_size,
        )
    data_sha = _hash_bytes(
        _hash_tensor(train_values).encode() + _hash_tensor(validation_values).encode()
    )
    config_sha = _hash_json({"target": asdict(target_config), "campaign": asdict(campaign)})
    case_id = f"{landscape}-{seed}"
    case_output = output / "checkpoints" / case_id
    if case_output.exists():
        raise FileExistsError(f"case output already exists: {case_output}")
    case_output.mkdir(parents=True, exist_ok=False)
    model, optimizer = _model_and_optimizer(target_config, device)
    stream = TokenStream(train_values)
    trajectory_enabled = any(strategy in TRAJECTORY_STRATEGIES for strategy in strategies)
    snapshot_steps = _trajectory_snapshot_steps(campaign) if trajectory_enabled else ()
    prefix_telemetry, prefix_seconds, prefix_tokens, trajectory_snapshots = _train_prefix(
        model,
        optimizer,
        stream,
        target_config,
        campaign,
        device,
        snapshot_steps=snapshot_steps,
        snapshot_dir=case_output / "trajectory",
        config_sha=config_sha,
        data_sha=data_sha,
    )
    prefix_loss = _evaluate(
        model,
        validation_values,
        target_config=target_config,
        campaign=campaign,
        device=device,
    )
    parent_stats = _parent_features(prefix_telemetry, _loss_slope(prefix_telemetry))
    parent_stats["batch_context_tokens"] = float(campaign.batch_size * campaign.context)
    parent_stats["prefix_seconds"] = float(prefix_seconds)
    parent_flops = _flops(target_config, model, prefix_tokens)
    before = Observation(
        step=campaign.prefix_steps,
        tokens=prefix_tokens,
        loss=prefix_loss,
        quality=-prefix_loss,
        compute_flops=parent_flops,
        features=parent_stats,
    )
    parent_path = case_output / "parent.pt"
    parent = save_checkpoint(
        parent_path,
        model=model,
        optimizer=optimizer,
        data_state=stream.state(),
        config_sha256=config_sha,
        data_sha256=data_sha,
        code_sha=code_revision,
        seed=seed,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    policy_history = [entry for entry in history if entry.seed != 2]
    proposals = _proposals(
        parent_features=parent_stats,
        campaign=campaign,
        history=policy_history,
    )
    proposals = [proposal for proposal in proposals if proposal.strategy in strategies]
    if not proposals:
        raise ValueError("strategy filter removed every proposal")
    branch_results: list[dict[str, Any]] = []
    for proposal in proposals:
        transition, result = _branch(
            proposal=proposal,
            parent=parent,
            target_config=target_config,
            campaign=campaign,
            train_values=train_values,
            validation_values=validation_values,
            config_sha=config_sha,
            data_sha=data_sha,
            code_sha=code_revision,
            before=before,
            device=device,
            trajectory_snapshots=tuple(trajectory_snapshots),
        )
        archive.append(transition)
        branch_results.append(result)
    actual = {result["strategy"]: result for result in branch_results}
    successful = [result for result in branch_results if not result["failed"]]
    actual_best = min(successful, key=lambda result: result["final_loss"]) if successful else None
    predicted_best = min(proposals, key=lambda proposal: proposal.predicted_final_loss)
    predicted_actual = actual[predicted_best.strategy]
    horizon_regrets: dict[str, float | None] = {}
    for phase_index, phase_name in enumerate(("immediate", "recovery", "final"), start=1):
        horizon_successes = [
            result
            for result in successful
            if len(result["phase_metrics"]) > phase_index
        ]
        if not horizon_successes or predicted_actual["failed"]:
            horizon_regrets[phase_name] = None
            continue
        best_loss = min(
            result["phase_metrics"][phase_index]["loss"]
            for result in horizon_successes
        )
        horizon_regrets[phase_name] = (
            predicted_actual["phase_metrics"][phase_index]["loss"] - best_loss
        )
    ranking_rows: list[dict[str, Any]] = []
    for proposal in proposals:
        result = actual[proposal.strategy]
        ranking_rows.append(
            {
                "case_id": case_id,
                "strategy": proposal.strategy,
                "selected_schedule": result["selected_schedule"],
                "predicted_gain": proposal.predicted_gain,
                "predicted_uncertainty": proposal.uncertainty,
                "predicted_final_loss": proposal.predicted_final_loss,
                "actual_gain": None if result["failed"] else result["gain"],
                "actual_immediate": None
                if result["failed"]
                else result["phase_metrics"][1]["loss"],
                "actual_recovery": None
                if result["failed"]
                else result["phase_metrics"][2]["loss"],
                "actual_final": None if result["failed"] else result["final_loss"],
                "predicted_best": predicted_best.strategy,
                "actual_best": actual_best["strategy"] if actual_best else None,
                "immediate_regret": horizon_regrets["immediate"]
                if proposal.strategy == predicted_best.strategy
                else None,
                "recovery_regret": horizon_regrets["recovery"]
                if proposal.strategy == predicted_best.strategy
                else None,
                "final_regret": horizon_regrets["final"]
                if proposal.strategy == predicted_best.strategy
                else None,
            }
        )
        if not result["failed"]:
            history.append(
                HistoryEntry(
                    features=parent_stats,
                    selected_schedule=result["selected_schedule"],
                    gain=result["gain"],
                    final_loss=result["final_loss"],
                    seed=seed,
                )
            )
    thresholds = {
        str(factor): prefix_loss * factor for factor in QUALITY_FACTORS
    }
    capability_rows: list[dict[str, Any]] = []
    for result in branch_results:
        if result["failed"]:
            capability_rows.append(
                {
                    "case_id": case_id,
                    "strategy": result["strategy"],
                    "points": [],
                    "thresholds": {
                        name: {"reached": False, "failure": result["failure"]}
                        for name in thresholds
                    },
                    "failure": result["failure"],
                }
            )
            continue
        points = [
            _phase_point(phase, before=before, result=result)
            for phase in result["phase_metrics"]
        ]
        capability_rows.append(
            {
                "case_id": case_id,
                "strategy": result["strategy"],
                "points": points,
                "thresholds": {
                    name: _threshold_result(points, threshold)
                    for name, threshold in thresholds.items()
                },
            }
        )
    return {
        "case_id": case_id,
        "landscape": landscape,
        "seed": seed,
        "target_parameters": parameter_count,
        "target_config": asdict(target_config),
        "data_sha256": data_sha,
        "config_sha256": config_sha,
        "parent_checkpoint_sha256": parent.sha256,
        "parent_checkpoint": str(parent.path),
        "trajectory_snapshots": [str(path) for path in trajectory_snapshots],
        "prefix_seconds": prefix_seconds,
        "prefix_tokens": prefix_tokens,
        "prefix_loss": prefix_loss,
        "quality_thresholds": thresholds,
        "branches": branch_results,
        "ranking": ranking_rows,
        "capability": capability_rows,
        "immutable": {
            "git_commit": code_revision,
            "code_sha256": code_sha256,
            "uv_lock_sha256": uv_lock_sha256,
            "objective_sha256": objective_sha256,
        },
    }


def _aggregate_curves(curves: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for curve in curves:
        grouped[curve["strategy"]].append(curve["points"])
    aggregate: dict[str, Any] = {}
    for strategy, traces in grouped.items():
        by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trace in traces:
            for point in trace:
                by_name[point["name"]].append(point)
        aggregate[strategy] = [
            {
                "name": name,
                "median_validation_loss": sorted(
                    point["validation_loss"] for point in points
                )[len(points) // 2],
                "median_capability": sorted(
                    point["capability"] for point in points
                )[len(points) // 2],
                "median_wall_seconds": sorted(
                    point["wall_seconds"] for point in points
                )[len(points) // 2],
                "median_tokens": sorted(point["tokens"] for point in points)[len(points) // 2],
                "n": len(points),
            }
            for name, points in by_name.items()
        ]
    return {"per_branch": curves, "aggregate": aggregate}


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_clean_repo(args.allow_dirty)
    campaign = CampaignConfig(
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        vocab_size=args.vocab_size,
        context=args.context,
        batch_size=args.batch_size,
        prefix_steps=args.prefix_steps,
        immediate_steps=args.immediate_steps,
        recovery_steps=args.recovery_steps,
        final_steps=args.final_steps,
        validation_batches=args.validation_batches,
        train_file=args.train_file,
        validation_file=args.validation_file,
        learning_rate=args.learning_rate,
        pulse_attention=args.pulse_attention,
        pulse_mlp=args.pulse_mlp,
        open_loop_start=args.open_loop_start,
        open_loop_end=args.open_loop_end,
        trajectory_interval=args.trajectory_interval,
        trajectory_window=args.trajectory_window,
        trajectory_alpha=args.trajectory_alpha,
        amp=not args.no_amp,
    )
    if campaign.width % campaign.heads:
        raise ValueError("width must be divisible by heads")
    if campaign.validation_batches <= 0:
        raise ValueError("validation_batches must be positive")
    if campaign.trajectory_interval <= 0 or campaign.trajectory_window <= 0:
        raise ValueError("trajectory interval and window must be positive")
    if not math.isfinite(campaign.trajectory_alpha):
        raise ValueError("trajectory_alpha must be finite")
    if not (
        0 < campaign.immediate_steps <= campaign.recovery_steps < campaign.final_steps
    ):
        raise ValueError("horizons must satisfy 0 < immediate <= recovery < final")
    landscapes = tuple(args.landscape) if args.landscape else LANDSCAPES
    seeds = tuple(args.seed) if args.seed else SEEDS
    strategies = tuple(args.strategy) if args.strategy else STRATEGIES
    if not strategies:
        raise ValueError("at least one strategy is required")
    if not landscapes or not seeds:
        raise ValueError("at least one landscape and seed are required")
    if any(item not in LANDSCAPES for item in landscapes):
        raise ValueError(f"landscapes must be selected from {LANDSCAPES}")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh --output; {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    probe_config = _target_config(campaign, landscapes[0], seeds[0])
    probe_model = DecoderLM(probe_config).to(device="meta")
    target_parameters = sum(parameter.numel() for parameter in probe_model.parameters())
    del probe_model
    if not args.allow_small_target and not 70_000_000 <= target_parameters <= 160_000_000:
        raise ValueError(
            f"target has {target_parameters} parameters; use a 70–160M configuration "
            "or pass --allow-small-target for a smoke run"
        )
    code_revision = _code_revision()
    code_sha256 = _campaign_code_sha256()
    repo_root = Path(__file__).resolve().parents[1]
    uv_lock = repo_root / "uv.lock"
    uv_lock_sha256 = _file_sha256(uv_lock) if uv_lock.exists() else "missing"
    objective_sha256 = _hash_json(
        {
            "quality": "negative_validation_cross_entropy",
            "thresholds": QUALITY_FACTORS,
            "utility": "validation_gain_minus_total_cost",
        }
    )
    archive = Archive(output / "transitions.jsonl")
    history: list[HistoryEntry] = []
    cases: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    capability_rows: list[dict[str, Any]] = []
    expected_branches = 0
    for landscape in landscapes:
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            case = _run_case(
                campaign=campaign,
                landscape=landscape,
                seed=seed,
                output=output,
                archive=archive,
                device=device,
                code_revision=code_revision,
                code_sha256=code_sha256,
                uv_lock_sha256=uv_lock_sha256,
                objective_sha256=objective_sha256,
                history=history,
                strategies=strategies,
            )
            cases.append(case)
            ranking_rows.extend(case["ranking"])
            capability_rows.extend(case["capability"])
            expected_branches += len(strategies)
    archive_records = archive.validate()
    if archive_records != expected_branches:
        raise RuntimeError(
            f"branch coverage mismatch: expected {expected_branches}, got {archive_records}"
        )
    ranking = _ranking_report(ranking_rows)
    curves = _aggregate_curves(capability_rows)
    failures = [
        {
            "case_id": case["case_id"],
            "strategy": branch["strategy"],
            "failure": branch.get("failure"),
        }
        for case in cases
        for branch in case["branches"]
        if branch["failed"]
    ]
    manifest = {
        "schema": "landscape-driver.decoder-campaign.v1",
        "contract_version": "v1",
        "stage": "development",
        "promotion_status": "not_evaluated",
        "contract": {
            "version": "v1",
            "preregistered": True,
            "primary_metric": "end_to_end_wall_seconds",
            "quality_definition": "negative_validation_cross_entropy",
            "quality_factors": list(QUALITY_FACTORS),
            "cost_components": [
                "target_training",
                "driver_inference",
                "driver_update",
                "probes",
                "recovery",
                "evaluation",
                "rejected_branches",
                "search",
                "meta_training",
            ],
        },
        "promotion_gate": {
            "required_heldout_landscapes": 5,
            "required_fresh_seeds": 3,
            "required_thresholds": 3,
            "required_changed_data_or_architecture": True,
            "status": "not_run_in_development_stage",
        },
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "landscapes": list(landscapes),
        "seeds": list(seeds),
        "strategies": list(strategies),
        "config": asdict(campaign),
        "immutable": {
            "git_commit": code_revision,
            "code_sha256": code_sha256,
            "uv_lock_sha256": uv_lock_sha256,
            "objective_sha256": objective_sha256,
        },
        "coverage": {
            "expected_cases": len(landscapes) * len(seeds),
            "observed_cases": len(cases),
            "expected_branches": expected_branches,
            "observed_branches": archive_records,
            "failed_branches": len(failures),
        },
        "quality_factors": list(QUALITY_FACTORS),
        "utility_definition": "predicted capability gain at equal token budget; total cost recorded separately",
        "cases": [
            {
                "case_id": case["case_id"],
                "landscape": case["landscape"],
                "seed": case["seed"],
                "data_sha256": case["data_sha256"],
                "config_sha256": case["config_sha256"],
                "parent_checkpoint_sha256": case["parent_checkpoint_sha256"],
                    "parent_checkpoint": case["parent_checkpoint"],
                    "trajectory_snapshots": case["trajectory_snapshots"],
                "target_parameters": case["target_parameters"],
            }
            for case in cases
        ],
        "files": {
            "transitions": "transitions.jsonl",
            "action_ranking": "action_ranking.json",
            "capability_curves": "capability_curves.json",
        },
        "failures": failures,
    }
    (output / "action_ranking.json").write_text(
        json.dumps(ranking, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "capability_curves.json").write_text(
        json.dumps(curves, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(output / "manifest.json"), **manifest["coverage"]}, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/decoder-development")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--landscape", action="append", choices=LANDSCAPES)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--strategy", action="append", choices=ALL_STRATEGIES)
    parser.add_argument("--width", type=int, default=CampaignConfig.width)
    parser.add_argument("--layers", type=int, default=CampaignConfig.layers)
    parser.add_argument("--heads", type=int, default=CampaignConfig.heads)
    parser.add_argument("--vocab-size", type=int, default=CampaignConfig.vocab_size)
    parser.add_argument("--context", type=int, default=CampaignConfig.context)
    parser.add_argument("--batch-size", type=int, default=CampaignConfig.batch_size)
    parser.add_argument("--prefix-steps", type=int, default=CampaignConfig.prefix_steps)
    parser.add_argument("--immediate-steps", type=int, default=CampaignConfig.immediate_steps)
    parser.add_argument("--recovery-steps", type=int, default=CampaignConfig.recovery_steps)
    parser.add_argument("--final-steps", type=int, default=CampaignConfig.final_steps)
    parser.add_argument(
        "--validation-batches", type=int, default=CampaignConfig.validation_batches
    )
    parser.add_argument("--train-file")
    parser.add_argument("--validation-file")
    parser.add_argument("--learning-rate", type=float, default=CampaignConfig.learning_rate)
    parser.add_argument("--pulse-attention", type=float, default=CampaignConfig.pulse_attention)
    parser.add_argument("--pulse-mlp", type=float, default=CampaignConfig.pulse_mlp)
    parser.add_argument("--open-loop-start", type=float, default=CampaignConfig.open_loop_start)
    parser.add_argument("--open-loop-end", type=float, default=CampaignConfig.open_loop_end)
    parser.add_argument(
        "--trajectory-interval", type=int, default=CampaignConfig.trajectory_interval
    )
    parser.add_argument(
        "--trajectory-window", type=int, default=CampaignConfig.trajectory_window
    )
    parser.add_argument(
        "--trajectory-alpha", type=float, default=CampaignConfig.trajectory_alpha
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-small-target", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
