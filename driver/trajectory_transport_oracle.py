"""Measure a hindsight future-checkpoint transport oracle.

This is an oracle-ceiling experiment, not a deployable driver.  It loads a
future model snapshot from an already recorded AdamW/no-op trajectory, applies
that state at an earlier immutable parent checkpoint, and measures a matched
short continuation.  The future snapshot is deliberately charged as a
hindsight intervention and is never evidence for a learned policy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_checkpoint
from .decoder_benchmark import (
    Config,
    TokenStream,
    _flops,
    _hash_bytes,
    _hash_tensor,
    _role,
    make_byte_stream,
    make_stream,
)
from .decoder_campaign import (
    CampaignConfig,
    _evaluate,
    _load_model_snapshot,
    _model_and_optimizer,
    _reset_adam_moments,
    _sync,
    _train_step,
)


TRANSPORT_ROLES = ("embedding", "attention", "mlp", "norm", "head")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = Path(__file__).resolve().parents[1] / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path)


def _campaign_and_config(
    manifest_config: dict[str, Any], case: dict[str, Any]
) -> tuple[CampaignConfig, Config]:
    optimizer = manifest_config.get("optimizer") or "adamw"
    common = {
        "width": int(manifest_config["width"]),
        "layers": int(manifest_config["layers"]),
        "heads": int(manifest_config["heads"]),
        "vocab_size": int(manifest_config["vocab_size"]),
        "context": int(manifest_config["context"]),
        "batch_size": int(manifest_config["batch_size"]),
        "prefix_steps": int(manifest_config["prefix_steps"]),
        "immediate_steps": int(manifest_config["immediate_steps"]),
        "recovery_steps": int(manifest_config["recovery_steps"]),
        "final_steps": int(manifest_config["final_steps"]),
        "validation_batches": int(manifest_config["validation_batches"]),
        "train_file": manifest_config.get("train_file"),
        "validation_file": manifest_config.get("validation_file"),
        "learning_rate": float(manifest_config["learning_rate"]),
        "weight_decay": float(manifest_config["weight_decay"]),
        "optimizer": optimizer,
        "amp": bool(manifest_config.get("amp", True)),
    }
    campaign = CampaignConfig(**common)
    target = Config(
        landscape=str(case["landscape"]),
        seed=int(case["seed"]),
        vocab_size=common["vocab_size"],
        context=common["context"],
        batch_size=common["batch_size"],
        width=common["width"],
        layers=common["layers"],
        heads=common["heads"],
        prefix_steps=common["prefix_steps"],
        immediate_steps=common["immediate_steps"],
        recovery_steps=common["recovery_steps"],
        final_steps=common["final_steps"],
        learning_rate=common["learning_rate"],
        weight_decay=common["weight_decay"],
        optimizer=optimizer,
    )
    return campaign, target


def _streams(
    campaign: CampaignConfig, *, landscape: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor, str]:
    train_tokens = (
        campaign.prefix_steps + campaign.final_steps + 2
    ) * campaign.batch_size * (campaign.context + 1)
    validation_tokens = campaign.validation_batches * campaign.batch_size * (
        campaign.context + 1
    )
    if campaign.train_file:
        train_values = make_byte_stream(
            _repo_path(campaign.train_file), seed=seed, length=train_tokens
        )
        validation_values = make_byte_stream(
            _repo_path(campaign.validation_file or campaign.train_file),
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
    return train_values, validation_values, data_sha


def _source_transition(source_manifest: Path, case_id: str) -> dict[str, Any]:
    transitions_path = source_manifest.parent / "transitions.jsonl"
    rows = [
        json.loads(line)
        for line in transitions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        row
        for row in rows
        if row.get("run_id") == case_id and row.get("action", {}).get("kind") == "noop"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one source noop transition for {case_id}")
    return matches[0]


def _future_snapshot(parent: Path, step: int) -> Path:
    candidates = sorted(
        parent.parent.glob(
            f"branches/trajectory_shadow_average/trajectory/step-{step:06d}.pt"
        )
    )
    if not candidates:
        candidates = sorted(
            parent.parent.glob(f"branches/*/trajectory/step-{step:06d}.pt")
        )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one raw future snapshot at step {step} beside {parent}"
        )
    return candidates[0]


def _snapshot_state(
    path: Path, *, config_sha: str, data_sha: str
) -> tuple[dict[str, torch.Tensor], int]:
    return _load_model_snapshot(path, config_sha=config_sha, data_sha=data_sha)


def _load_snapshot_loss(
    model: torch.nn.Module,
    path: Path,
    *,
    config_sha: str,
    data_sha: str,
    validation_values: torch.Tensor,
    target: Config,
    campaign: CampaignConfig,
    device: torch.device,
) -> tuple[float, int]:
    state, step = _snapshot_state(path, config_sha=config_sha, data_sha=data_sha)
    model.load_state_dict(state)
    del state
    return (
        _evaluate(
            model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        ),
        step,
    )


def _run_variant(
    *,
    parent: Path,
    future: Path,
    source_case: dict[str, Any],
    source_transition: dict[str, Any],
    campaign: CampaignConfig,
    target: Config,
    train_values: torch.Tensor,
    validation_values: torch.Tensor,
    config_sha: str,
    data_sha: str,
    future_step: int,
    recovery_after: int,
    cursor_policy: str,
    optimizer_state_policy: str,
    transport_roles: tuple[str, ...] | None,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = None
    optimizer = None
    try:
        model, optimizer = _model_and_optimizer(target, device)
        metadata = load_checkpoint(
            parent,
            model=model,
            optimizer=optimizer,
            config_sha256=config_sha,
            data_sha256=data_sha,
        )
        parent_cursor = int(metadata["data_state"]["cursor"])
        parent_step = int(target.prefix_steps)
        continuation_steps = int(target.prefix_steps + target.final_steps - future_step)
        if continuation_steps <= 0:
            raise ValueError("future snapshot must precede the source endpoint")
        if not 0 < recovery_after < continuation_steps:
            raise ValueError("recovery_after must lie inside the transport continuation")
        future_state, observed_step = _snapshot_state(
            future, config_sha=config_sha, data_sha=data_sha
        )
        if observed_step != future_step:
            raise ValueError(f"future snapshot step mismatch: {future}")
        if transport_roles is None:
            model.load_state_dict(future_state)
        else:
            for name, parameter in model.named_parameters():
                if _role(name) in transport_roles:
                    parameter.copy_(
                        future_state[name].to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    )
        del future_state
        if optimizer_state_policy == "zero_moments":
            reset_count = _reset_adam_moments(optimizer)
        else:
            reset_count = 0
        transport_loss = _evaluate(
            model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        cursor_delta_steps = future_step - parent_step
        skipped_tokens = (
            cursor_delta_steps * target.batch_size * target.context
            if cursor_policy == "skip"
            else 0
        )
        cursor = parent_cursor
        if cursor_policy == "skip":
            cursor += cursor_delta_steps * target.batch_size * (target.context + 1)
        stream = TokenStream(train_values, cursor=cursor)
        phase_metrics = [
            {
                "name": "transport",
                "global_step": future_step,
                "loss": transport_loss,
                "tokens": parent_cursor // (target.batch_size * (target.context + 1))
                * target.batch_size
                * target.context
                + skipped_tokens,
            }
        ]
        telemetry: list[dict[str, Any]] = []
        for local_step in range(continuation_steps):
            current, _ = _train_step(
                model,
                optimizer,
                stream,
                target,
                campaign,
                device,
                step=local_step,
                schedule="noop",
                runtime={},
            )
            telemetry.append(current)
            branch_step = local_step + 1
            if branch_step == recovery_after:
                phase_metrics.append(
                    {
                        "name": "recovery",
                        "global_step": future_step + branch_step,
                        "loss": _evaluate(
                            model,
                            validation_values,
                            target_config=target,
                            campaign=campaign,
                            device=device,
                        ),
                        "tokens": phase_metrics[0]["tokens"]
                        + branch_step * target.batch_size * target.context,
                    }
                )
        final_loss = _evaluate(
            model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        phase_metrics.append(
            {
                "name": "final",
                "global_step": future_step + continuation_steps,
                "loss": final_loss,
                "tokens": phase_metrics[0]["tokens"]
                + continuation_steps * target.batch_size * target.context,
            }
        )
        _sync(device)
        wall_seconds = time.perf_counter() - started
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        target_tokens = continuation_steps * target.batch_size * target.context
        transported_parameters = (
            parameter_count
            if transport_roles is None
            else sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if _role(name) in transport_roles
            )
        )
        transport_flops = 2.0 * transported_parameters
        evaluation_flops = (
            2.0
            * parameter_count
            * target.batch_size
            * target.context
            * 3
        )
        target_flops = _flops(target, model, target_tokens)
        actual_flops = target_flops + transport_flops + evaluation_flops
        conservative_flops = actual_flops + _flops(
            target, model, skipped_tokens
        )
        baseline_branch_flops = float(source_transition["compute_flops"])
        prefix_flops = float(source_transition["before"]["compute_flops"])
        baseline_total_flops = prefix_flops + baseline_branch_flops
        candidate_total_flops = prefix_flops + actual_flops
        baseline_wall = float(source_transition["before"]["features"].get("prefix_seconds", 0.0)) + float(
            source_transition["wall_seconds"]
        )
        candidate_wall = float(
            source_transition["before"]["features"].get("prefix_seconds", 0.0)
        ) + wall_seconds
        baseline_final = float(source_transition["after"]["loss"])
        return {
            "case_id": source_case["case_id"],
            "cursor_policy": cursor_policy,
            "optimizer_state_policy": optimizer_state_policy,
            "transport_roles": list(transport_roles or TRANSPORT_ROLES),
            "future_step": future_step,
            "continuation_steps": continuation_steps,
            "recovery_after": recovery_after,
            "parent_cursor": parent_cursor,
            "candidate_cursor": stream.cursor,
            "skipped_tokens": skipped_tokens,
            "data_policy": "declared_skip" if cursor_policy == "skip" else "replay_parent_cursor",
            "optimizer_state": (
                "preserved_parent" if optimizer_state_policy == "preserve" else "zero_moments"
            ),
            "reset_moment_values": reset_count,
            "phase_metrics": phase_metrics,
            "baseline_snapshot_losses": source_case["baseline_snapshot_losses"],
            "baseline_final_loss": baseline_final,
            "final_delta_vs_baseline": final_loss - baseline_final,
            "quality_pass": final_loss <= baseline_final,
            "wall_seconds": wall_seconds,
            "baseline_total_wall_seconds": baseline_wall,
            "candidate_total_wall_seconds": candidate_wall,
            "wall_speedup": baseline_wall / candidate_wall if candidate_wall > 0 else 0.0,
            "target_tokens": target_tokens,
            "target_flops": target_flops,
            "transport_flops": transport_flops,
            "evaluation_flops": evaluation_flops,
            "actual_branch_flops": actual_flops,
            "conservative_branch_flops_with_skipped_work": conservative_flops,
            "baseline_total_flops": baseline_total_flops,
            "candidate_total_flops": candidate_total_flops,
            "candidate_total_flops_with_skipped_work": prefix_flops + conservative_flops,
            "flop_speedup": baseline_total_flops / candidate_total_flops
            if candidate_total_flops > 0
            else 0.0,
            "conservative_flop_speedup": baseline_total_flops
            / (prefix_flops + conservative_flops)
            if prefix_flops + conservative_flops > 0
            else 0.0,
            "oracle_only": True,
            "oracle_violations": [
                "hindsight_future_model_state",
                "future_optimizer_state_unavailable",
            ],
            "telemetry_tail": telemetry[-1] if telemetry else {},
            "failed": False,
        }
    except Exception as exc:
        _sync(device)
        return {
            "case_id": source_case["case_id"],
            "cursor_policy": cursor_policy,
            "optimizer_state_policy": optimizer_state_policy,
            "transport_roles": list(transport_roles or TRANSPORT_ROLES),
            "failed": True,
            "failure": f"{type(exc).__name__}: {exc}",
            "oracle_only": True,
        }
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run_case(
    *,
    source_manifest: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    future_step: int,
    recovery_after: int,
    cursor_policies: tuple[str, ...],
    optimizer_state_policies: tuple[str, ...],
    transport_role_sets: tuple[tuple[str, ...] | None, ...],
    device: torch.device,
) -> dict[str, Any]:
    campaign, target = _campaign_and_config(manifest["config"], case)
    if future_step <= target.prefix_steps or future_step >= target.prefix_steps + target.final_steps:
        raise ValueError("future_step must be strictly inside the recorded continuation")
    train_values, validation_values, data_sha = _streams(
        campaign, landscape=target.landscape, seed=target.seed
    )
    if data_sha != case["data_sha256"]:
        raise ValueError(f"reconstructed data hash does not match {case['case_id']}")
    parent = _repo_path(case["parent_checkpoint"])
    config_sha = str(case["config_sha256"])
    future = _future_snapshot(parent, future_step)
    source_transition = _source_transition(source_manifest, case["case_id"])
    model, optimizer = _model_and_optimizer(target, device)
    metadata = load_checkpoint(
        parent,
        model=model,
        optimizer=optimizer,
        config_sha256=config_sha,
        data_sha256=data_sha,
    )
    parent_loss = _evaluate(
        model,
        validation_values,
        target_config=target,
        campaign=campaign,
        device=device,
    )
    baseline_snapshot_losses: dict[str, float] = {}
    source_paths: dict[str, str] = {}
    for step in (future_step, future_step + recovery_after, target.prefix_steps + target.final_steps):
        path = _future_snapshot(parent, step)
        loss, observed_step = _load_snapshot_loss(
            model,
            path,
            config_sha=config_sha,
            data_sha=data_sha,
            validation_values=validation_values,
            target=target,
            campaign=campaign,
            device=device,
        )
        if observed_step != step:
            raise ValueError(f"snapshot step mismatch: {path}")
        baseline_snapshot_losses[str(step)] = loss
        source_paths[str(step)] = str(path)
    _sync(device)
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    source_case = dict(case)
    source_case["baseline_snapshot_losses"] = baseline_snapshot_losses
    results = []
    for cursor_policy in cursor_policies:
        for optimizer_state_policy in optimizer_state_policies:
            for transport_roles in transport_role_sets:
                results.append(
                    _run_variant(
                        parent=parent,
                        future=future,
                        source_case=source_case,
                        source_transition=source_transition,
                        campaign=campaign,
                        target=target,
                        train_values=train_values,
                        validation_values=validation_values,
                        config_sha=config_sha,
                        data_sha=data_sha,
                        future_step=future_step,
                        recovery_after=recovery_after,
                        cursor_policy=cursor_policy,
                        optimizer_state_policy=optimizer_state_policy,
                        transport_roles=transport_roles,
                        device=device,
                    )
                )
    return {
        "case_id": case["case_id"],
        "landscape": case["landscape"],
        "seed": case["seed"],
        "target_parameters": case["target_parameters"],
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": case["parent_checkpoint_sha256"],
        "future_snapshot": str(future),
        "future_snapshot_sha256": _file_sha256(future),
        "source_data_sha256": data_sha,
        "source_config_sha256": config_sha,
        "parent_cursor": int(metadata["data_state"]["cursor"]),
        "parent_loss": parent_loss,
        "baseline_noop": {
            "final_loss": float(source_transition["after"]["loss"]),
            "wall_seconds": float(source_transition["wall_seconds"]),
            "total_flops": float(source_transition["before"]["compute_flops"])
            + float(source_transition["compute_flops"]),
        },
        "baseline_snapshot_losses": baseline_snapshot_losses,
        "baseline_snapshot_paths": source_paths,
        "results": results,
    }


def _self_check() -> None:
    assert 2048 - 1536 == 512
    assert 512 * 128 * 128 == 8_388_608
    assert _repo_path("research/PLAN.md").is_file()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        _self_check()
        print("trajectory transport oracle self-check: ok")
        return {"self_check": "ok"}
    if not args.source_manifest:
        raise ValueError("at least one --source-manifest is required")
    source_manifests = tuple(_repo_path(path) for path in args.source_manifest)
    cursor_policies = tuple(args.cursor_policy or ("preserve", "skip"))
    optimizer_state_policies = tuple(args.optimizer_state_policy or ("preserve",))
    transport_role_sets = (
        tuple((role,) for role in TRANSPORT_ROLES)
        if args.role_wise
        else (None,)
    )
    if args.future_step <= 0 or args.recovery_after <= 0:
        raise ValueError("future_step and recovery_after must be positive")
    device = torch.device(args.device)
    output = _repo_path(args.output) if Path(args.output).exists() else Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh --output; {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for source_manifest in source_manifests:
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            random.seed(int(case["seed"]))
            torch.manual_seed(int(case["seed"]))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(case["seed"]))
            cases.append(
                _run_case(
                    source_manifest=source_manifest,
                    manifest=manifest,
                    case=case,
                    future_step=args.future_step,
                    recovery_after=args.recovery_after,
                    cursor_policies=cursor_policies,
                    optimizer_state_policies=optimizer_state_policies,
                    transport_role_sets=transport_role_sets,
                    device=device,
                )
            )
    failures = [
        result
        for case in cases
        for result in case["results"]
        if result["failed"]
    ]
    repo_root = Path(__file__).resolve().parents[1]
    output_manifest = {
        "schema": "landscape-driver.trajectory-transport-oracle.v1",
        "stage": "oracle_ceiling",
        "promotion_status": "diagnostic_only",
        "oracle_only": True,
        "contract": {
            "baseline": "recorded AdamW/no-op branch from each source manifest",
            "quality": "candidate final validation loss <= matched no-op final loss",
            "required_horizons": ["transport", "recovery", "final"],
            "cost_includes": [
                "shared prefix",
                "candidate training",
                "transport copy",
                "evaluation",
                "declared skipped data",
            ],
        },
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "future_step": args.future_step,
        "recovery_after": args.recovery_after,
        "cursor_policies": list(cursor_policies),
        "optimizer_state_policies": list(optimizer_state_policies),
        "transport_role_sets": [
            list(roles or TRANSPORT_ROLES) for roles in transport_role_sets
        ],
        "source_manifests": [
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "git_commit": json.loads(path.read_text(encoding="utf-8"))["immutable"]["git_commit"],
            }
            for path in source_manifests
        ],
        "immutable": {
            "git_commit": _code_revision(),
            "code_sha256": _file_sha256(Path(__file__).resolve()),
        },
        "coverage": {
            "cases": len(cases),
            "variants": len(cases)
            * len(cursor_policies)
            * len(optimizer_state_policies)
            * len(transport_role_sets),
            "failed_variants": len(failures),
        },
        "cases": cases,
        "failures": failures,
    }
    (output / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(output / "manifest.json"), **output_manifest["coverage"]}, indent=2))
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", action="append")
    parser.add_argument("--output", default="runs/trajectory-transport-oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--future-step", type=int, default=2048)
    parser.add_argument("--recovery-after", type=int, default=128)
    parser.add_argument("--cursor-policy", action="append", choices=("preserve", "skip"))
    parser.add_argument(
        "--optimizer-state-policy",
        action="append",
        choices=("preserve", "zero_moments"),
    )
    parser.add_argument(
        "--role-wise",
        action="store_true",
        help="run one hindsight future-state branch per tensor role",
    )
    parser.add_argument("--self-check", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
