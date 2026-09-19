"""Measure a same-corpus trajectory anchor on disjoint adaptation data.

The target noop and anchor branches start from matched step-1536 parents but
consume the same explicitly sliced, byte-disjoint continuation interval.  The
anchor imports a complete source-seed state; source prefix cost is charged as
an amortized knowledge-transfer cost rather than treated as free transport.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoints import _restore_rng, load_checkpoint
from .decoder_benchmark import TokenStream, _flops, _hash_bytes, _hash_tensor
from .decoder_campaign import (
    _autocast,
    _evaluate,
    _model_and_optimizer,
    _sync,
)
from .trajectory_transport_oracle import (
    _campaign_and_config,
    _file_sha256,
    _repo_path,
    _source_transition,
    _streams,
)


HORIZONS = (128, 512, 768)
THRESHOLD_FACTORS = (0.995, 0.99, 0.98)
DEPLOYMENT_COUNTS = (1, 2, 4, 8, 16)
ARCHITECTURE_FIELDS = (
    "width",
    "layers",
    "heads",
    "vocab_size",
    "context",
    "batch_size",
)


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected finite numeric value")
    return result


def _architecture(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(config[name]) for name in ARCHITECTURE_FIELDS)


def _case_key(manifest_path: Path, case: dict[str, Any]) -> str:
    return f"{manifest_path.parent.name}-{case['case_id']}"


def _pool(values: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], int, str]] = set()
    for value in values:
        manifest_path = _repo_path(value)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["config"]
        for case in manifest["cases"]:
            key = (_architecture(config), int(case["seed"]), str(config.get("train_file")))
            if key in seen:
                raise ValueError(f"duplicate pool case: {key}")
            seen.add(key)
            entries.append(
                {
                    "manifest_path": manifest_path,
                    "manifest": manifest,
                    "config": config,
                    "case": case,
                    "case_key": _case_key(manifest_path, case),
                }
            )
    if not entries:
        raise ValueError("at least one source manifest is required")
    return entries


def _pairs(entries: list[dict[str, Any]], shift: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    regimes: dict[tuple[tuple[int, ...], str], list[dict[str, Any]]] = {}
    for entry in entries:
        key = (_architecture(entry["config"]), str(entry["config"].get("train_file")))
        regimes.setdefault(key, []).append(entry)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for items in regimes.values():
        ordered = sorted(items, key=lambda item: int(item["case"]["seed"]))
        if len(ordered) < 2:
            continue
        for index, target in enumerate(ordered):
            source = ordered[(index + shift) % len(ordered)]
            if source["case"]["seed"] == target["case"]["seed"]:
                raise ValueError("source and target seeds must differ")
            pairs.append((source, target))
    if not pairs:
        raise ValueError("no same-regime seed pair exists")
    return pairs


def _raw_interval(path: Path, *, offset: int, length: int) -> torch.Tensor:
    if offset < 0 or length <= 0:
        raise ValueError("adaptation offset and length must be positive")
    raw = path.read_bytes()
    end = offset + length
    if end > len(raw):
        raise ValueError(f"adaptation interval [{offset}, {end}) exceeds {path} ({len(raw)} bytes)")
    return torch.tensor(list(raw[offset:end]), dtype=torch.uint8)


def _reset_adam_state(optimizer: torch.optim.Optimizer) -> int:
    count = 0
    for state in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if value is not None:
                value.zero_()
                count += value.numel()
        step = state.get("step")
        if step is not None:
            if torch.is_tensor(step):
                step.zero_()
            else:
                state["step"] = 0
    return count


def _step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stream: TokenStream,
    target: Any,
    campaign: Any,
    device: torch.device,
) -> int:
    tokens, targets = stream.batch(
        batch_size=target.batch_size, context=target.context, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    with _autocast(campaign, device):
        loss = model(tokens, targets)
    loss.backward()
    optimizer.step()
    return target.batch_size * target.context


def _load_target_rng(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _restore_rng(payload["rng"])
    del payload


def _run_branch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stream: TokenStream,
    target: Any,
    campaign: Any,
    validation: torch.Tensor,
    device: torch.device,
    started: float,
) -> dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    eval_one_flops = 2.0 * parameter_count * target.batch_size * target.context
    metrics: list[dict[str, Any]] = []
    for step in range(HORIZONS[-1]):
        _step(model, optimizer, stream, target, campaign, device)
        if step + 1 in HORIZONS:
            _sync(device)
            metrics.append(
                {
                    "horizon": step + 1,
                    "loss": _evaluate(
                        model,
                        validation,
                        target_config=target,
                        campaign=campaign,
                        device=device,
                    ),
                    "target_tokens": (step + 1) * target.batch_size * target.context,
                    "target_flops": _flops(
                        target,
                        model,
                        (step + 1) * target.batch_size * target.context,
                    ),
                    "evaluation_flops": eval_one_flops * (len(metrics) + 1),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    return {
        "metrics": metrics,
        "target_tokens": stream.cursor,
        "target_flops": metrics[-1]["target_flops"],
        "evaluation_flops": metrics[-1]["evaluation_flops"],
        "wall_seconds": metrics[-1]["wall_seconds"],
        "end_cursor": stream.cursor,
    }


def _stable_horizon(metrics: list[dict[str, Any]], threshold: float) -> int | None:
    for index, metric in enumerate(metrics):
        if all(float(item["loss"]) <= threshold for item in metrics[index:]):
            return int(metric["horizon"])
    return None


def _run_pair(
    *,
    source: dict[str, Any],
    target_item: dict[str, Any],
    adaptation_offset: int,
    optimizer_state_policy: str,
    rng_policy: str,
    device: torch.device,
) -> dict[str, Any]:
    source_campaign, source_config = _campaign_and_config(source["config"], source["case"])
    target_campaign, target_config = _campaign_and_config(target_item["config"], target_item["case"])
    if source_campaign.train_file != target_campaign.train_file:
        raise ValueError("source and target must use the same training file")
    if optimizer_state_policy == "zero_moments" and target_config.optimizer != "adamw":
        raise ValueError("zero_moments control currently supports AdamW only")
    for name in ("vocab_size", "context", "batch_size", "width", "layers", "heads", "learning_rate", "weight_decay", "optimizer"):
        if getattr(source_config, name) != getattr(target_config, name):
            raise ValueError(f"source/target configuration mismatch: {name}")
    source_parent = _repo_path(source["case"]["parent_checkpoint"])
    target_parent = _repo_path(target_item["case"]["parent_checkpoint"])
    if _file_sha256(source_parent) != source["case"]["parent_checkpoint_sha256"]:
        raise ValueError("source parent checkpoint hash mismatch")
    if _file_sha256(target_parent) != target_item["case"]["parent_checkpoint_sha256"]:
        raise ValueError("target parent checkpoint hash mismatch")
    _, _, source_data_sha = _streams(source_campaign, landscape=source_config.landscape, seed=source_config.seed)
    _, target_validation, target_data_sha = _streams(target_campaign, landscape=target_config.landscape, seed=target_config.seed)
    if source_data_sha != source["case"]["data_sha256"] or target_data_sha != target_item["case"]["data_sha256"]:
        raise ValueError("source or target data hash mismatch")
    tokens_per_step = target_config.batch_size * (target_config.context + 1)
    interval_length = target_campaign.final_steps * tokens_per_step + 1
    train_path = _repo_path(str(target_campaign.train_file))
    raw_length = train_path.stat().st_size
    source_start = int(source_config.seed) % raw_length
    source_end = source_start + target_campaign.prefix_steps * tokens_per_step
    if adaptation_offset < source_end or adaptation_offset + interval_length > raw_length:
        raise ValueError(
            f"adaptation interval is not disjoint/in-bounds: source=[{source_start},{source_end}), "
            f"adaptation=[{adaptation_offset},{adaptation_offset + interval_length}), bytes={raw_length}"
        )
    adaptation_started = time.perf_counter()
    adaptation_train = _raw_interval(
        train_path, offset=adaptation_offset, length=interval_length
    )
    adaptation_load_seconds = time.perf_counter() - adaptation_started
    adaptation_data_sha = _hash_bytes(
        _hash_tensor(adaptation_train).encode() + _hash_tensor(target_validation).encode()
    )
    target_noop = _source_transition(target_item["manifest_path"], target_item["case"]["case_id"])
    target_parent_loss = _finite(target_noop["before"]["loss"])
    baseline_prefix_flops = _finite(target_noop["before"]["compute_flops"])
    baseline_prefix_wall = _finite(target_noop["before"].get("features", {}).get("prefix_seconds", 0.0))
    source_noop = _source_transition(source["manifest_path"], source["case"]["case_id"])
    source_prefix_flops = _finite(source_noop["before"]["compute_flops"])
    source_prefix_wall = _finite(source_noop["before"].get("features", {}).get("prefix_seconds", 0.0))
    model = optimizer = None
    baseline_model = baseline_optimizer = None
    try:
        baseline_model, baseline_optimizer = _model_and_optimizer(target_config, device)
        baseline_load_started = time.perf_counter()
        load_checkpoint(
            target_parent,
            model=baseline_model,
            optimizer=baseline_optimizer,
            config_sha256=str(target_item["case"]["config_sha256"]),
            data_sha256=target_data_sha,
        )
        baseline_checkpoint_load_seconds = time.perf_counter() - baseline_load_started
        baseline_started = time.perf_counter()
        baseline = _run_branch(
            model=baseline_model,
            optimizer=baseline_optimizer,
            stream=TokenStream(adaptation_train, cursor=0),
            target=target_config,
            campaign=target_campaign,
            validation=target_validation,
            device=device,
            started=baseline_started,
        )
        del baseline_model, baseline_optimizer
        baseline_model = baseline_optimizer = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model, optimizer = _model_and_optimizer(target_config, device)
        anchor_load_started = time.perf_counter()
        source_metadata = load_checkpoint(
            source_parent,
            model=model,
            optimizer=optimizer,
            config_sha256=str(source["case"]["config_sha256"]),
            data_sha256=source_data_sha,
        )
        anchor_checkpoint_load_seconds = time.perf_counter() - anchor_load_started
        reset_count = 0
        if optimizer_state_policy == "zero_moments":
            reset_count = _reset_adam_state(optimizer)
        if rng_policy == "target":
            _load_target_rng(target_parent)
        anchor_started = time.perf_counter()
        anchor = _run_branch(
            model=model,
            optimizer=optimizer,
            stream=TokenStream(adaptation_train, cursor=0),
            target=target_config,
            campaign=target_campaign,
            validation=target_validation,
            device=device,
            started=anchor_started,
        )
        baseline_final_flops = baseline_prefix_flops + baseline["target_flops"] + baseline["evaluation_flops"]
        baseline_final_wall = (
            baseline_prefix_wall
            + adaptation_load_seconds
            + baseline_checkpoint_load_seconds
            + baseline["wall_seconds"]
        )
        thresholds = {
            str(factor): target_parent_loss * factor for factor in THRESHOLD_FACTORS
        }
        threshold_results: dict[str, Any] = {}
        for factor, threshold in thresholds.items():
            threshold_results[factor] = {}
            for label, branch, prefix_flops, prefix_wall in (
                ("baseline", baseline, baseline_prefix_flops, baseline_prefix_wall),
                ("anchor", anchor, source_prefix_flops, source_prefix_wall),
            ):
                horizon = _stable_horizon(branch["metrics"], threshold)
                item: dict[str, Any] = {
                    "threshold": threshold,
                    "stable_horizon": horizon,
                }
                if horizon is not None:
                    metric = next(row for row in branch["metrics"] if row["horizon"] == horizon)
                    branch_flops = metric["target_flops"] + metric["evaluation_flops"]
                    if label == "anchor" and optimizer_state_policy == "zero_moments":
                        branch_flops += 2.0 * reset_count
                    item["ratios_by_deployment_count"] = {
                        str(count): {
                            "flop_speedup": baseline_final_flops
                            / max(1.0, prefix_flops / count + branch_flops),
                            "wall_speedup": baseline_final_wall
                            / max(
                                1e-9,
                                prefix_wall / count
                                + adaptation_load_seconds
                                + (
                                    anchor_checkpoint_load_seconds
                                    if label == "anchor"
                                    else baseline_checkpoint_load_seconds
                                )
                                + metric["wall_seconds"],
                            ),
                        }
                        for count in DEPLOYMENT_COUNTS
                    }
                threshold_results[factor][label] = item
        return {
            "source_case_key": source["case_key"],
            "target_case_key": target_item["case_key"],
            "source_seed": source["case"]["seed"],
            "target_seed": target_item["case"]["seed"],
            "landscape": target_item["case"]["landscape"],
            "target_parameters": target_item["case"]["target_parameters"],
            "optimizer_state_policy": optimizer_state_policy,
            "rng_policy": rng_policy,
            "source_checkpoint": str(source_parent),
            "target_parent_checkpoint": str(target_parent),
            "source_checkpoint_sha256": source["case"]["parent_checkpoint_sha256"],
            "target_checkpoint_sha256": target_item["case"]["parent_checkpoint_sha256"],
            "source_config_sha256": source["case"]["config_sha256"],
            "target_config_sha256": target_item["case"]["config_sha256"],
            "source_data_sha256": source_data_sha,
            "target_data_sha256": target_data_sha,
            "adaptation_data_sha256": adaptation_data_sha,
            "adaptation_offset": adaptation_offset,
            "adaptation_length": interval_length,
            "source_prefix_interval": [source_start, source_end],
            "adaptation_interval": [adaptation_offset, adaptation_offset + interval_length],
            "source_metadata": source_metadata,
            "target_parent_loss": target_parent_loss,
            "baseline": baseline,
            "anchor": anchor,
            "checkpoint_io_seconds": {
                "adaptation_data_load": adaptation_load_seconds,
                "baseline_parent_load": baseline_checkpoint_load_seconds,
                "anchor_source_parent_load": anchor_checkpoint_load_seconds,
                "source_prefix_checkpoint_write": None,
            },
            "wall_cost_notes": [
                "historical source prefix_seconds excludes its original checkpoint write; source cost is therefore a lower bound",
                "deployment ratios charge adaptation-data load and parent-checkpoint load",
            ],
            "thresholds": threshold_results,
            "final_delta_anchor_vs_baseline": anchor["metrics"][-1]["loss"] - baseline["metrics"][-1]["loss"],
            "reset_moment_values": reset_count,
            "failed": False,
            "oracle_only": True,
            "oracle_violations": [
                "source_checkpoint_reuse_requires_declared_amortization",
                "same_corpus_initialization_not_general_data_transfer",
            ],
        }
    except Exception as exc:
        _sync(device)
        return {
            "source_case_key": source["case_key"],
            "target_case_key": target_item["case_key"],
            "optimizer_state_policy": optimizer_state_policy,
            "rng_policy": rng_policy,
            "failed": True,
            "failure": f"{type(exc).__name__}: {exc}",
        }
    finally:
        del model, optimizer, baseline_model, baseline_optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _self_check() -> None:
    assert HORIZONS == (128, 512, 768)
    assert THRESHOLD_FACTORS == (0.995, 0.99, 0.98)
    assert DEPLOYMENT_COUNTS == (1, 2, 4, 8, 16)
    metrics = [{"horizon": 128, "loss": 1.0}, {"horizon": 512, "loss": 0.9}]
    assert _stable_horizon(metrics, 1.0) == 128
    assert _stable_horizon(metrics, 0.85) is None
    print("trajectory anchor adaptation self-check: ok")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        _self_check()
        return {"self_check": "ok"}
    entries = _pool(args.source_manifest)
    pairs = _pairs(entries, args.source_shift)
    if args.target_seed:
        target_seeds = set(args.target_seed)
        pairs = [pair for pair in pairs if int(pair[1]["case"]["seed"]) in target_seeds]
    if not pairs:
        raise ValueError("target seed filter removed all pairs")
    policies = tuple(args.optimizer_state_policy or ("preserve",))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results = [
        _run_pair(
            source=source,
            target_item=target,
            adaptation_offset=args.adaptation_offset,
            optimizer_state_policy=policy,
            rng_policy=args.rng_policy,
            device=device,
        )
        for source, target in pairs
        for policy in policies
    ]
    failures = [result for result in results if result["failed"]]
    manifest = {
        "schema": "landscape-driver.trajectory-anchor-adaptation.v1",
        "stage": "oracle_initialization",
        "promotion_status": "diagnostic_only",
        "oracle_only": True,
        "hypothesis": "a same-regime source state improves adaptation on a disjoint continuation interval",
        "horizons": list(HORIZONS),
        "threshold_factors": list(THRESHOLD_FACTORS),
        "deployment_counts": list(DEPLOYMENT_COUNTS),
        "adaptation_offset": args.adaptation_offset,
        "source_shift": args.source_shift,
        "optimizer_state_policies": list(policies),
        "rng_policy": args.rng_policy,
        "cost_definition": {
            "baseline": "target parent prefix plus matched target noop continuation on adaptation interval",
            "anchor": "source parent prefix divided by N plus target continuation and all state/evaluation work",
            "source_cost_is_free": False,
            "checkpoint_io": "adaptation-data load and deployment parent-checkpoint loads are charged in wall ratios; historical source-prefix checkpoint write time is unavailable and remains an explicit lower-bound limitation",
            "thresholds": "fixed factors times target parent loss; stable horizon requires all later observed horizons to pass",
        },
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_manifests": [
            {"path": str(_repo_path(value)), "sha256": _file_sha256(_repo_path(value))}
            for value in args.source_manifest
        ],
        "coverage": {
            "pairs": len(pairs),
            "variants": len(results),
            "failed_variants": len(failures),
            "final_anchor_better_count": sum(
                bool(not result["failed"] and result["final_delta_anchor_vs_baseline"] <= 0.0)
                for result in results
            ),
        },
        "results": results,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(output / "manifest.json"), **manifest["coverage"]}, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", action="append")
    parser.add_argument("--output", default="runs/trajectory-anchor-adaptation")
    parser.add_argument("--adaptation-offset", type=int, default=26_000_000)
    parser.add_argument("--source-shift", type=int, default=1)
    parser.add_argument("--target-seed", action="append", type=int)
    parser.add_argument(
        "--optimizer-state-policy",
        action="append",
        choices=("preserve", "zero_moments"),
    )
    parser.add_argument("--rng-policy", choices=("target", "source"), default="target")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--self-check", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if not arguments.self_check and not arguments.source_manifest:
        raise SystemExit("--source-manifest is required unless --self-check is used")
    run(arguments)
