"""Measure same-regime trajectory-anchor initialization.

The candidate starts from a complete step-1536 checkpoint produced on a
different seed, then consumes a fresh target seed's data from cursor zero.
This is a knowledge-transfer oracle, not free transport: source checkpoint
creation cost is reported and amortized only over explicit deployment counts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_checkpoint
from .decoder_benchmark import TokenStream, _flops
from .decoder_campaign import _evaluate, _model_and_optimizer, _reset_adam_moments, _sync
from .full_state_transport_sanity import _fast_noop_step
from .trajectory_transport_oracle import (
    _campaign_and_config,
    _file_sha256,
    _repo_path,
    _source_transition,
    _streams,
)


HORIZONS = (128, 512, 768)
ARCHITECTURE_FIELDS = (
    "width",
    "layers",
    "heads",
    "vocab_size",
    "context",
    "batch_size",
)
DEPLOYMENT_COUNTS = (1, 2, 4, 8, 16)


def _architecture(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(config[name]) for name in ARCHITECTURE_FIELDS)


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected a finite numeric value")
    return result


def _case_key(manifest_path: Path, case: dict[str, Any]) -> str:
    return f"{manifest_path.parent.name}-{case['case_id']}"


def _load_pool(paths: list[str]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], int, str]] = set()
    for value in paths:
        manifest_path = _repo_path(value)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["config"]
        for case in manifest["cases"]:
            key = (_architecture(config), int(case["seed"]), str(config.get("train_file")))
            if key in seen:
                raise ValueError(f"duplicate source/target case in pool: {key}")
            seen.add(key)
            pool.append(
                {
                    "manifest_path": manifest_path,
                    "manifest": manifest,
                    "case": case,
                    "config": config,
                    "case_key": _case_key(manifest_path, case),
                }
            )
    if not pool:
        raise ValueError("at least one manifest case is required")
    return pool


def _pair_cases(pool: list[dict[str, Any]], shift: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_regime: dict[tuple[int, ...] | tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in pool:
        regime = (_architecture(item["config"]), str(item["config"].get("train_file")))
        by_regime.setdefault(regime, []).append(item)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for regime, items in sorted(by_regime.items(), key=lambda pair: str(pair[0])):
        items = sorted(items, key=lambda item: int(item["case"]["seed"]))
        if len(items) < 2:
            continue
        for index, target in enumerate(items):
            source = items[(index + shift) % len(items)]
            if source["case"]["seed"] == target["case"]["seed"]:
                raise ValueError("anchor source and target seeds must differ")
            pairs.append((source, target))
    if not pairs:
        raise ValueError("no regime has at least two distinct seeds")
    return pairs


def _run_pair(
    *,
    source: dict[str, Any],
    target_item: dict[str, Any],
    optimizer_state_policy: str,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    source_campaign, source_config = _campaign_and_config(
        source["config"], source["case"]
    )
    target_campaign, target_config = _campaign_and_config(
        target_item["config"], target_item["case"]
    )
    if source_config != target_config:
        # Seed is intentionally allowed to differ; all model/data-shape fields
        # must match before a source optimizer state is transferred.
        if any(
            getattr(source_config, name) != getattr(target_config, name)
            for name in (
                "vocab_size",
                "context",
                "batch_size",
                "width",
                "layers",
                "heads",
                "learning_rate",
                "weight_decay",
                "optimizer",
            )
        ):
            raise ValueError("anchor source and target configurations are incompatible")
    if source_campaign.train_file != target_campaign.train_file:
        raise ValueError("anchor source and target must use the same training file")
    source_parent = _repo_path(source["case"]["parent_checkpoint"])
    target_parent = _repo_path(target_item["case"]["parent_checkpoint"])
    if _file_sha256(source_parent) != source["case"]["parent_checkpoint_sha256"]:
        raise ValueError("source parent checkpoint hash mismatch")
    if _file_sha256(target_parent) != target_item["case"]["parent_checkpoint_sha256"]:
        raise ValueError("target parent checkpoint hash mismatch")
    _, _, source_data_sha = _streams(
        source_campaign,
        landscape=source_config.landscape,
        seed=source_config.seed,
    )
    target_train, target_validation, target_data_sha = _streams(
        target_campaign,
        landscape=target_config.landscape,
        seed=target_config.seed,
    )
    if source_data_sha != source["case"]["data_sha256"]:
        raise ValueError("source data hash mismatch")
    if target_data_sha != target_item["case"]["data_sha256"]:
        raise ValueError("target data hash mismatch")
    target_noop = _source_transition(
        target_item["manifest_path"], target_item["case"]["case_id"]
    )
    baseline_final_loss = _finite(target_noop["after"]["loss"])
    baseline_prefix_flops = _finite(target_noop["before"]["compute_flops"])
    baseline_total_flops = baseline_prefix_flops + _finite(target_noop["compute_flops"])
    baseline_prefix_wall = _finite(
        target_noop["before"].get("features", {}).get("prefix_seconds", 0.0)
    )
    baseline_total_wall = baseline_prefix_wall + _finite(target_noop["wall_seconds"])
    source_noop = _source_transition(
        source["manifest_path"], source["case"]["case_id"]
    )
    source_prefix_flops = _finite(source_noop["before"]["compute_flops"])
    source_prefix_wall = _finite(
        source_noop["before"].get("features", {}).get("prefix_seconds", 0.0)
    )
    model = optimizer = None
    started = time.perf_counter()
    try:
        model, optimizer = _model_and_optimizer(target_config, device)
        source_metadata = load_checkpoint(
            source_parent,
            model=model,
            optimizer=optimizer,
            config_sha256=str(source["case"]["config_sha256"]),
            data_sha256=source_data_sha,
        )
        reset_count = 0
        if optimizer_state_policy == "zero_moments":
            reset_count = _reset_adam_moments(optimizer)
        stream = TokenStream(target_train, cursor=0)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        eval_one_flops = 2.0 * parameter_count * target_config.batch_size * target_config.context
        metrics: list[dict[str, Any]] = []
        for step in range(HORIZONS[-1]):
            _fast_noop_step(model, optimizer, stream, target_config, target_campaign, device)
            if step + 1 in HORIZONS:
                _sync(device)
                metrics.append(
                    {
                        "horizon": step + 1,
                        "global_target_step": step + 1,
                        "loss": _evaluate(
                            model,
                            target_validation,
                            target_config=target_config,
                            campaign=target_campaign,
                            device=device,
                        ),
                        "target_tokens": (step + 1) * target_config.batch_size * target_config.context,
                        "target_flops": _flops(
                            target_config,
                            model,
                            (step + 1) * target_config.batch_size * target_config.context,
                        ),
                        "evaluation_flops": eval_one_flops * len(metrics + [None]),
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
        final_metric = metrics[-1]
        losses = [float(metric["loss"]) for metric in metrics]
        first_stable_horizon = None
        for index, metric in enumerate(metrics):
            if all(loss <= baseline_final_loss for loss in losses[index:]):
                first_stable_horizon = metric["horizon"]
                break
        threshold_metrics: dict[str, Any] = {}
        for metric in metrics:
            horizon = int(metric["horizon"])
            candidate_flops = (
                metric["target_flops"]
                + metric["evaluation_flops"]
                + (2.0 * reset_count if optimizer_state_policy == "zero_moments" else 0.0)
            )
            threshold_metrics[str(horizon)] = {
                "candidate_loss": metric["loss"],
                "delta_vs_target_noop_final": metric["loss"] - baseline_final_loss,
                "candidate_target_flops": candidate_flops,
                "ratios_by_deployment_count": {
                    str(count): {
                        "flop_speedup": baseline_total_flops
                        / max(1.0, source_prefix_flops / count + candidate_flops),
                        "wall_speedup": baseline_total_wall
                        / max(
                            1e-9,
                            source_prefix_wall / count + metric["wall_seconds"],
                        ),
                    }
                    for count in DEPLOYMENT_COUNTS
                },
            }
        return {
            "source_case_key": source["case_key"],
            "target_case_key": target_item["case_key"],
            "source_case_id": source["case"]["case_id"],
            "target_case_id": target_item["case"]["case_id"],
            "source_seed": source["case"]["seed"],
            "target_seed": target_item["case"]["seed"],
            "landscape": target_item["case"]["landscape"],
            "target_parameters": target_item["case"]["target_parameters"],
            "optimizer_state_policy": optimizer_state_policy,
            "source_checkpoint": str(source_parent),
            "target_parent_checkpoint": str(target_parent),
            "source_checkpoint_sha256": source["case"]["parent_checkpoint_sha256"],
            "target_checkpoint_sha256": target_item["case"]["parent_checkpoint_sha256"],
            "source_config_sha256": source["case"]["config_sha256"],
            "target_config_sha256": target_item["case"]["config_sha256"],
            "source_data_sha256": source_data_sha,
            "target_data_sha256": target_data_sha,
            "source_metadata": source_metadata,
            "source_prefix_flops": source_prefix_flops,
            "source_prefix_wall_seconds": source_prefix_wall,
            "target_noop_baseline_final_loss": baseline_final_loss,
            "target_noop_total_flops": baseline_total_flops,
            "target_noop_total_wall_seconds": baseline_total_wall,
            "target_stream_start_cursor": 0,
            "target_stream_end_cursor": stream.cursor,
            "reset_moment_values": reset_count,
            "metrics": metrics,
            "first_stable_horizon": first_stable_horizon,
            "threshold_metrics": threshold_metrics,
            "final_delta_vs_target_noop": final_metric["loss"] - baseline_final_loss,
            "failed": False,
            "oracle_only": True,
            "oracle_violations": [
                "source_checkpoint_selection_is_hindsight_for_this_screen",
                "source_prefix_cost_requires_amortization",
            ],
        }
    except Exception as exc:
        _sync(device)
        return {
            "source_case_key": source["case_key"],
            "target_case_key": target_item["case_key"],
            "optimizer_state_policy": optimizer_state_policy,
            "failed": True,
            "failure": f"{type(exc).__name__}: {exc}",
        }
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _self_check() -> None:
    assert HORIZONS == (128, 512, 768)
    assert DEPLOYMENT_COUNTS == (1, 2, 4, 8, 16)
    assert _architecture({name: 1 for name in ARCHITECTURE_FIELDS}) == (1,) * len(ARCHITECTURE_FIELDS)
    print("trajectory anchor oracle self-check: ok")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        _self_check()
        return {"self_check": "ok"}
    pool = _load_pool(args.source_manifest)
    pairs = _pair_cases(pool, args.source_shift)
    if args.target_seed:
        requested = set(args.target_seed)
        pairs = [pair for pair in pairs if int(pair[1]["case"]["seed"]) in requested]
    if not pairs:
        raise ValueError("target seed filter removed every anchor pair")
    policies = tuple(args.optimizer_state_policy or ("preserve",))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results: list[dict[str, Any]] = []
    for source, target in pairs:
        for policy in policies:
            results.append(
                _run_pair(
                    source=source,
                    target_item=target,
                    optimizer_state_policy=policy,
                    device=device,
                    output=output,
                )
            )
    failures = [result for result in results if result["failed"]]
    output_manifest = {
        "schema": "landscape-driver.trajectory-anchor-oracle.v1",
        "stage": "oracle_initialization",
        "promotion_status": "diagnostic_only",
        "oracle_only": True,
        "hypothesis": "a same-regime complete trajectory checkpoint can initialize a fresh seed with fewer target steps",
        "horizons": list(HORIZONS),
        "deployment_counts": list(DEPLOYMENT_COUNTS),
        "source_shift": args.source_shift,
        "optimizer_state_policies": list(policies),
        "cost_definition": {
            "baseline": "target recorded prefix plus target AdamW/no-op branch",
            "candidate": "source prefix divided by N plus target continuation, evaluations, and state reset work",
            "source_cost_is_free": False,
            "threshold": "target noop final loss; first stable observed horizon is diagnostic",
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
            "final_passes": sum(
                bool(not result["failed"] and result.get("final_delta_vs_target_noop", 1.0) <= 0.0)
                for result in results
            ),
        },
        "results": results,
    }
    (output / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(output / "manifest.json"), **output_manifest["coverage"]}, indent=2))
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", action="append", required=False)
    parser.add_argument("--output", default="runs/trajectory-anchor-oracle")
    parser.add_argument("--source-shift", type=int, default=1)
    parser.add_argument("--target-seed", action="append", type=int)
    parser.add_argument(
        "--optimizer-state-policy",
        action="append",
        choices=("preserve", "zero_moments"),
    )
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
