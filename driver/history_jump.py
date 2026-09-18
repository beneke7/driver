"""Collect causal AdamW traces and test cheap multi-step jump baselines.

This module is intentionally a bridge, not a general driver framework. It
reuses the existing decoder campaign's model, checkpoint, and telemetry code.
The first useful question is whether a bounded jump has signal before a learned
controller is allowed to select one.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .checkpoints import load_checkpoint, save_checkpoint
from .decoder_campaign import (
    CampaignConfig,
    Config,
    DecoderLM,
    TokenStream,
    _code_revision,
    _evaluate,
    _hash_bytes,
    _hash_json,
    _hash_tensor,
    _model_and_optimizer,
    _safe_float,
    _sync,
    _train_step,
    make_stream,
)


MAX_BENCHMARK_HORIZON = 32


@dataclass(frozen=True)
class TraceConfig:
    width: int = 768
    layers: int = 12
    heads: int = 12
    vocab_size: int = 128
    context: int = 128
    batch_size: int = 128
    steps: int = 64
    checkpoint_every: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    amp: bool = True


def _target_config(config: TraceConfig, landscape: str, seed: int) -> Config:
    return Config(
        landscape=landscape,
        seed=seed,
        vocab_size=config.vocab_size,
        context=config.context,
        batch_size=config.batch_size,
        width=config.width,
        layers=config.layers,
        heads=config.heads,
        prefix_steps=0,
        immediate_steps=1,
        recovery_steps=2,
        final_steps=3,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _campaign_config(config: TraceConfig) -> CampaignConfig:
    return CampaignConfig(
        width=config.width,
        layers=config.layers,
        heads=config.heads,
        vocab_size=config.vocab_size,
        context=config.context,
        batch_size=config.batch_size,
        prefix_steps=0,
        immediate_steps=1,
        recovery_steps=2,
        final_steps=3,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        amp=config.amp,
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _validate_trace_config(config: TraceConfig) -> None:
    if config.width <= 0 or config.layers <= 0 or config.heads <= 0:
        raise ValueError("width, layers, and heads must be positive")
    if config.width % config.heads:
        raise ValueError("width must be divisible by heads")
    if config.context <= 0 or config.batch_size <= 0 or config.steps <= 0:
        raise ValueError("context, batch size, and steps must be positive")
    if not 0 < config.checkpoint_every <= config.steps:
        raise ValueError("checkpoint_every must be between 1 and steps")


def _case_data(
    config: TraceConfig, landscape: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor, str]:
    train_length = (
        config.steps + 6 * MAX_BENCHMARK_HORIZON + 1
    ) * config.batch_size * (config.context + 1)
    validation_length = config.batch_size * (config.context + 1)
    train_values = make_stream(
        landscape, seed=seed, length=train_length, vocab_size=config.vocab_size
    )
    validation_values = make_stream(
        landscape,
        seed=seed + 1_000_000,
        length=validation_length,
        vocab_size=config.vocab_size,
    )
    data_sha = _hash_bytes(
        _hash_tensor(train_values).encode() + _hash_tensor(validation_values).encode()
    )
    return train_values, validation_values, data_sha


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")


def _collect_case(
    *,
    config: TraceConfig,
    landscape: str,
    seed: int,
    output: Path,
    device: torch.device,
    code_revision: str,
) -> dict[str, Any]:
    target_config = _target_config(config, landscape, seed)
    campaign = _campaign_config(config)
    train_values, validation_values, data_sha = _case_data(config, landscape, seed)
    config_sha = _hash_json({"target": asdict(target_config), "trace": asdict(config)})
    case_id = f"{landscape}-{seed}"
    case_dir = output / "cases" / case_id
    checkpoint_dir = case_dir / "checkpoints"
    case_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir()
    trace_path = case_dir / "trace.jsonl"

    model, optimizer = _model_and_optimizer(target_config, device)
    stream = TokenStream(train_values)
    parent_loss = _evaluate(
        model,
        validation_values,
        target_config=target_config,
        campaign=campaign,
        device=device,
    )
    parent = save_checkpoint(
        checkpoint_dir / "step-000000.pt",
        model=model,
        optimizer=optimizer,
        data_state={"cursor": stream.cursor, "step": 0},
        config_sha256=config_sha,
        data_sha256=data_sha,
        code_sha=code_revision,
        seed=seed,
    )
    _append_jsonl(
        trace_path,
        {
            "case_id": case_id,
            "step": 0,
            "tokens": 0,
            "data_cursor": stream.cursor,
            "train_loss": None,
            "validation_loss": parent_loss,
            "loss_slope": 0.0,
            "loss_second_difference": 0.0,
            "checkpoint": str(parent.path.relative_to(output)),
            "checkpoint_sha256": parent.sha256,
        },
    )

    history: list[float] = []
    started = time.perf_counter()
    for step in range(config.steps):
        telemetry, consumed = _train_step(
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
        history.append(float(telemetry["loss"]))
        telemetry["case_id"] = case_id
        telemetry["tokens"] = stream.cursor
        telemetry["data_cursor"] = stream.cursor
        telemetry["consumed_tokens"] = consumed
        telemetry["loss_slope"] = (
            history[-1] - history[-2] if len(history) >= 2 else 0.0
        )
        telemetry["loss_second_difference"] = (
            history[-1] - 2.0 * history[-2] + history[-3]
            if len(history) >= 3
            else 0.0
        )
        if (step + 1) % config.checkpoint_every == 0 or step + 1 == config.steps:
            validation_loss = _evaluate(
                model,
                validation_values,
                target_config=target_config,
                campaign=campaign,
                device=device,
            )
            checkpoint = save_checkpoint(
                checkpoint_dir / f"step-{step + 1:06d}.pt",
                model=model,
                optimizer=optimizer,
                data_state={"cursor": stream.cursor, "step": step + 1},
                config_sha256=config_sha,
                data_sha256=data_sha,
                code_sha=code_revision,
                seed=seed,
            )
            telemetry["validation_loss"] = validation_loss
            telemetry["checkpoint"] = str(checkpoint.path.relative_to(output))
            telemetry["checkpoint_sha256"] = checkpoint.sha256
        else:
            telemetry["validation_loss"] = None
            telemetry["checkpoint"] = None
            telemetry["checkpoint_sha256"] = None
        _append_jsonl(trace_path, telemetry)
    _sync(device)
    elapsed = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "case_id": case_id,
        "landscape": landscape,
        "seed": seed,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
        "parameters": parameters,
        "trace": str(trace_path.relative_to(output)),
        "steps": config.steps,
        "checkpoint_every": config.checkpoint_every,
        "seconds": elapsed,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    config = TraceConfig(
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        vocab_size=args.vocab_size,
        context=args.context,
        batch_size=args.batch_size,
        steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        amp=not args.no_amp,
    )
    _validate_trace_config(config)
    landscapes = tuple(args.landscape or ("delayed_copy", "phase_switch", "text_shard"))
    seeds = tuple(args.seed or (0, 1))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh --output; {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    probe = DecoderLM(_target_config(config, landscapes[0], seeds[0])).to(device="meta")
    parameters = sum(parameter.numel() for parameter in probe.parameters())
    del probe
    if not args.allow_small_target and not 70_000_000 <= parameters <= 160_000_000:
        raise ValueError(
            f"target has {parameters} parameters; use 70–160M or pass --allow-small-target"
        )
    code_revision = _code_revision()
    cases: list[dict[str, Any]] = []
    for landscape in landscapes:
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            cases.append(
                _collect_case(
                    config=config,
                    landscape=landscape,
                    seed=seed,
                    output=output,
                    device=device,
                    code_revision=code_revision,
                )
            )
    manifest = {
        "schema": "landscape-driver.history-jump-traces.v1",
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": asdict(config),
        "landscapes": list(landscapes),
        "seeds": list(seeds),
        "cases": cases,
        "code_revision": code_revision,
    }
    _write_manifest(output, manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def _optimizer_direction(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> list[tuple[torch.nn.Parameter, float, float, torch.Tensor]]:
    directions: list[tuple[torch.nn.Parameter, float, float, torch.Tensor]] = []
    for group in optimizer.param_groups:
        lr = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        beta1, beta2 = (float(value) for value in group.get("betas", (0.9, 0.999)))
        eps = float(group.get("eps", 1e-8))
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg is None or exp_avg_sq is None:
                continue
            step_value = state.get("step", 1.0)
            step = float(step_value.item() if torch.is_tensor(step_value) else step_value)
            first = exp_avg / (1.0 - beta1**step)
            second = exp_avg_sq / (1.0 - beta2**step)
            normalized = first / (second.sqrt() + eps)
            directions.append((parameter, lr, weight_decay, normalized.detach()))
    return directions


@torch.no_grad()
def _apply_momentum_jump(
    model: nn.Module, optimizer: torch.optim.Optimizer, horizon: int, blend: float
) -> float:
    if horizon <= 0 or not math.isfinite(blend) or blend <= 0.0:
        raise ValueError("jump horizon must be positive and blend must be finite and positive")
    directions = _optimizer_direction(model, optimizer)
    parameters = 0
    multiplier = horizon * blend
    for parameter, lr, weight_decay, normalized in directions:
        parameter.add_(parameter, alpha=-lr * weight_decay * multiplier)
        parameter.add_(normalized, alpha=-lr * multiplier)
        parameters += parameter.numel()
    if not directions:
        raise RuntimeError("optimizer has no initialized AdamW state")
    return float(parameters)


def _advance_stream(stream: TokenStream, *, steps: int, target_config: Config) -> int:
    consumed = steps * target_config.batch_size * target_config.context
    cursor_increment = steps * target_config.batch_size * (target_config.context + 1)
    if stream.cursor + cursor_increment > stream.values.numel():
        raise RuntimeError("jump would exhaust the training stream")
    stream.cursor += cursor_increment
    return consumed


def _run_action(
    *,
    action: str,
    horizon: int,
    blend: float,
    parent: Path,
    target_config: Config,
    campaign: CampaignConfig,
    train_values: torch.Tensor,
    validation_values: torch.Tensor,
    config_sha: str,
    data_sha: str,
    device: torch.device,
) -> dict[str, Any]:
    model, optimizer = _model_and_optimizer(target_config, device)
    started = time.perf_counter()
    metadata = load_checkpoint(
        parent,
        model=model,
        optimizer=optimizer,
        config_sha256=config_sha,
        data_sha256=data_sha,
    )
    stream = TokenStream(train_values, cursor=int(metadata["data_state"]["cursor"]))
    batch_tokens = target_config.batch_size * target_config.context
    if action == "noop":
        for step in range(horizon):
            _train_step(
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
        jump_flops = 0.0
    elif action == "momentum_jump":
        parameter_count = _apply_momentum_jump(model, optimizer, horizon, blend)
        _advance_stream(stream, steps=horizon, target_config=target_config)
        jump_flops = 2.0 * parameter_count
    else:
        raise ValueError(f"unknown action: {action}")
    immediate = _evaluate(
        model,
        validation_values,
        target_config=target_config,
        campaign=campaign,
        device=device,
    )
    for step in range(horizon):
        _train_step(
            model,
            optimizer,
            stream,
            target_config,
            campaign,
            device,
            step=horizon + step,
            schedule="noop",
            runtime={},
        )
    recovery = _evaluate(
        model,
        validation_values,
        target_config=target_config,
        campaign=campaign,
        device=device,
    )
    for step in range(4 * horizon):
        _train_step(
            model,
            optimizer,
            stream,
            target_config,
            campaign,
            device,
            step=2 * horizon + step,
            schedule="noop",
            runtime={},
        )
    final = _evaluate(
        model,
        validation_values,
        target_config=target_config,
        campaign=campaign,
        device=device,
    )
    _sync(device)
    wall_seconds = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    consumed_tokens = 6 * batch_tokens * horizon
    target_tokens = (5 if action == "momentum_jump" else 6) * batch_tokens * horizon
    evaluation_flops = 2.0 * parameters * batch_tokens * 3
    target_flops = 6.0 * parameters * target_tokens
    flops = target_flops + evaluation_flops + jump_flops
    result = {
        "action": action,
        "horizon": horizon,
        "blend": blend,
        "immediate_loss": immediate,
        "recovery_loss": recovery,
        "final_loss": final,
        "wall_seconds": wall_seconds,
        "tokens_consumed": consumed_tokens,
        "tokens_trained": target_tokens,
        "estimated_flops": flops,
        "jump_flops": jump_flops,
        "failed": False,
    }
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def jump_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = TraceConfig(**manifest["config"])
    target_config_by_case: dict[str, Config] = {}
    rows: list[dict[str, Any]] = []
    device = torch.device(args.device)
    horizons = tuple(args.horizon or (8, 16, 32))
    blends = tuple(args.blend or (0.25, 0.5, 1.0))
    if any(horizon <= 0 or horizon > MAX_BENCHMARK_HORIZON for horizon in horizons):
        raise ValueError(
            f"horizons must be between 1 and {MAX_BENCHMARK_HORIZON} steps"
        )
    if any(not math.isfinite(blend) or blend <= 0.0 for blend in blends):
        raise ValueError("blends must be finite and positive")
    if args.max_parents is not None and args.max_parents <= 0:
        raise ValueError("max_parents must be positive")
    max_parents = args.max_parents
    for case in manifest["cases"]:
        if args.landscape and case["landscape"] not in args.landscape:
            continue
        if args.seed and case["seed"] not in args.seed:
            continue
        trace_path = source / case["trace"]
        trace_rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # The first checkpoint has no initialized AdamW moments. Keep it in the
        # trace for provenance, but begin jump comparisons after one update.
        parents = [
            row
            for row in trace_rows
            if row.get("checkpoint") and int(row["step"]) > 0
        ]
        if max_parents is not None:
            parents = parents[:max_parents]
        for parent_row in parents:
            target_config = target_config_by_case.setdefault(
                case["case_id"],
                _target_config(config, case["landscape"], case["seed"]),
            )
            campaign = _campaign_config(config)
            train_values, validation_values, data_sha = _case_data(
                config, case["landscape"], case["seed"]
            )
            parent_path = source / parent_row["checkpoint"]
            for horizon in horizons:
                for action, blend_values in (("noop", (0.0,)), ("momentum_jump", blends)):
                    for blend in blend_values:
                        try:
                            result = _run_action(
                                action=action,
                                horizon=horizon,
                                blend=blend,
                                parent=parent_path,
                                target_config=target_config,
                                campaign=campaign,
                                train_values=train_values,
                                validation_values=validation_values,
                                config_sha=case["config_sha256"],
                                data_sha=data_sha,
                                device=device,
                            )
                        except Exception as exc:
                            result = {
                                "action": action,
                                "horizon": horizon,
                                "blend": blend,
                                "failed": True,
                                "failure": f"{type(exc).__name__}: {exc}",
                            }
                        rows.append(
                            {
                                "case_id": case["case_id"],
                                "parent_step": parent_row["step"],
                                "parent_checkpoint": parent_row["checkpoint"],
                                **result,
                            }
                        )
    if not rows:
        raise ValueError("no parent checkpoints matched the benchmark filters")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh --output; {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "branches.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    result_manifest = {
        "schema": "landscape-driver.history-jump-results.v1",
        "source": str(source),
        "device": str(device),
        "torch": torch.__version__,
        "horizons": list(horizons),
        "blends": list(blends),
        "rows": len(rows),
        "failed": sum(1 for row in rows if row.get("failed")),
        "files": {"branches": "branches.jsonl"},
    }
    _write_manifest(output, result_manifest)
    print(json.dumps(result_manifest, indent=2))
    return result_manifest


def _self_check() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    count = _apply_momentum_jump(model, optimizer, horizon=2, blend=0.5)
    assert count == 20.0
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters())
    )
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    print("history-jump self-check passed: AdamW direction and bounded jump")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collect", "benchmark", "self-check"), default="collect")
    parser.add_argument("--output", default="runs/history-jump-traces")
    parser.add_argument("--source", help="trace directory for --mode benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--landscape", action="append", choices=("delayed_copy", "phase_switch", "text_shard"))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--width", type=int, default=TraceConfig.width)
    parser.add_argument("--layers", type=int, default=TraceConfig.layers)
    parser.add_argument("--heads", type=int, default=TraceConfig.heads)
    parser.add_argument("--vocab-size", type=int, default=TraceConfig.vocab_size)
    parser.add_argument("--context", type=int, default=TraceConfig.context)
    parser.add_argument("--batch-size", type=int, default=TraceConfig.batch_size)
    parser.add_argument("--steps", type=int, default=TraceConfig.steps)
    parser.add_argument("--checkpoint-every", type=int, default=TraceConfig.checkpoint_every)
    parser.add_argument("--learning-rate", type=float, default=TraceConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TraceConfig.weight_decay)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-small-target", action="store_true")
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument("--blend", type=float, action="append")
    parser.add_argument("--max-parents", type=int)
    return parser


def main(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.mode == "self-check":
        _self_check()
        return None
    if args.mode == "collect":
        return collect(args)
    if not args.source:
        raise ValueError("--source is required for --mode benchmark")
    return jump_benchmark(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
