"""Build a provenance-preserving passive trajectory corpus.

The corpus contains only observed no-op telemetry.  It is suitable for phase
and future-outcome representation learning, never for claiming an intervention
outcome or a training speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROLES = ("embedding", "attention", "mlp", "norm", "head")
DEFAULT_HORIZONS = (1, 8, 32, 128)


def _sha256(path: Path) -> str:
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


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _phase(local_step: int, config: dict[str, Any]) -> str:
    immediate = int(config["immediate_steps"])
    recovery = int(config["recovery_steps"])
    if local_step <= immediate:
        return "immediate"
    if local_step <= recovery:
        return "recovery"
    return "late"


def _features(
    telemetry: dict[str, Any], *, global_step: int, global_tokens: int, total_steps: int
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "global_step": global_step,
        "global_tokens": global_tokens,
        "step_fraction": global_step / max(1, total_steps),
    }
    for name in (
        "loss",
        "gradient_norm",
        "update_norm",
        "momentum_alignment",
        "parameter_norm",
        "adam_momentum_norm",
        "adam_variance_norm",
        "global_lr_multiplier",
    ):
        if name in telemetry:
            features[name] = _finite(telemetry[name], name)
    for group_name in (
        "role_gradient_norms",
        "role_update_norms",
        "role_parameter_norms",
        "adam_momentum_norms",
        "adam_variance_norms",
    ):
        groups = telemetry.get(group_name, {})
        if not isinstance(groups, dict):
            raise ValueError(f"{group_name} must be an object")
        features[group_name] = {
            role: _finite(groups.get(role, 0.0), f"{group_name}.{role}")
            for role in ROLES
        }
    return features


def _build_case(
    *,
    result: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    horizons: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transitions_path = result / "transitions.jsonl"
    transitions = _jsonl(transitions_path)
    case_id = str(case["case_id"])
    rows = [
        row
        for row in transitions
        if row.get("run_id") == case_id and row.get("action", {}).get("kind") == "noop"
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one noop transition for {case_id} in {result}")
    noop = rows[0]
    telemetry = noop.get("metadata", {}).get("telemetry")
    if not isinstance(telemetry, list) or not telemetry:
        raise ValueError(f"noop transition has no telemetry: {case_id}")
    before = noop.get("before")
    after = noop.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError(f"noop transition is incomplete: {case_id}")
    config = manifest["config"]
    parent_step = int(before["step"])
    batch_context_tokens = int(config["batch_size"]) * int(config["context"])
    total_steps = parent_step + int(config["final_steps"])
    records: list[dict[str, Any]] = []
    for index, current in enumerate(telemetry):
        local_step = int(current["step"])
        if local_step != index + 1:
            raise ValueError(f"telemetry step is not contiguous in {case_id}")
        global_step = parent_step + local_step
        global_tokens = int(before["tokens"]) + local_step * batch_context_tokens
        targets: dict[str, Any] = {}
        for horizon in horizons:
            target_index = index + horizon
            if target_index < len(telemetry):
                target = telemetry[target_index]
                targets[f"loss_delta_h{horizon}"] = _finite(
                    target["loss"], "future loss"
                ) - _finite(current["loss"], "current loss")
                targets[f"loss_h{horizon}"] = _finite(target["loss"], "future loss")
        if index:
            targets["loss_delta_h1_observed"] = _finite(
                current["loss"], "current loss"
            ) - _finite(telemetry[index - 1]["loss"], "previous loss")
        records.append(
            {
                "schema": "landscape-driver.passive-trajectory-row.v1",
                "root_id": f"{result.name}:{case_id}",
                "case_id": case_id,
                "source_result": str(result),
                "causal_status": "passive_noop_trajectory",
                "landscape": case["landscape"],
                "seed": int(case["seed"]),
                "architecture": {
                    key: config[key]
                    for key in ("width", "layers", "heads", "vocab_size", "context")
                },
                "global_step": global_step,
                "global_tokens": global_tokens,
                "phase": _phase(local_step, config),
                "features": _features(
                    current,
                    global_step=global_step,
                    global_tokens=global_tokens,
                    total_steps=total_steps,
                ),
                "targets": targets,
            }
        )
    root_metadata = {
        "root_id": f"{result.name}:{case_id}",
        "case_id": case_id,
        "source_result": str(result),
        "source_manifest_sha256": _sha256(result / "manifest.json"),
        "source_transitions_sha256": _sha256(transitions_path),
        "landscape": case["landscape"],
        "seed": int(case["seed"]),
        "architecture": {
            key: config[key]
            for key in ("width", "layers", "heads", "vocab_size", "context")
        },
        "train_file": config.get("train_file"),
        "validation_file": config.get("validation_file"),
        "optimizer": config.get("optimizer") or "adamw",
        "config_sha256": case["config_sha256"],
        "data_sha256": case["data_sha256"],
        "parent_checkpoint_sha256": case["parent_checkpoint_sha256"],
        "parent_checkpoint": case["parent_checkpoint"],
        "trajectory_snapshots": case.get("trajectory_snapshots", []),
        "validation_points": noop.get("metadata", {}).get("horizon_metrics", []),
        "final_loss": after["loss"],
        "record_count": len(records),
    }
    return records, root_metadata


def _self_check() -> None:
    config = {"immediate_steps": 2, "recovery_steps": 4}
    assert _phase(1, config) == "immediate"
    assert _phase(3, config) == "recovery"
    assert _phase(5, config) == "late"
    assert _finite(1, "test") == 1.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        _self_check()
        print("passive trajectory self-check: ok")
        return {"self_check": "ok"}
    if not args.result:
        raise ValueError("at least one --result is required")
    results = tuple(Path(value) for value in args.result)
    horizons = tuple(sorted(set(args.horizon or DEFAULT_HORIZONS)))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    for result in results:
        manifest_path = result / "manifest.json"
        if not manifest_path.is_file() or not (result / "transitions.jsonl").is_file():
            raise FileNotFoundError(f"result is missing manifest/transitions: {result}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            case_rows, root = _build_case(
                result=result,
                manifest=manifest,
                case=case,
                horizons=horizons,
            )
            rows.extend(case_rows)
            roots.append(root)
    corpus_manifest = {
        "schema": "landscape-driver.passive-trajectory-corpus.v1",
        "causal_status": "passive_only_no_intervention_evidence",
        "purpose": [
            "phase_representation",
            "future_training_loss_prediction",
            "support_and_transfer_diagnostics",
        ],
        "horizons": list(horizons),
        "source_results": [str(path) for path in results],
        "roots": roots,
        "coverage": {"roots": len(roots), "rows": len(rows)},
        "immutable": {
            "git_commit": _code_revision(),
            "collector_sha256": _sha256(Path(__file__).resolve()),
        },
        "limitations": [
            "no intervention counterfactuals",
            "no optimizer-state reconstruction beyond source telemetry",
            "future model-weight snapshots are provenance references only",
            "validation points are sparse campaign horizons",
        ],
    }
    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(output / "manifest.json"), **corpus_manifest["coverage"]}, indent=2))
    return corpus_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--horizon", action="append", type=int, default=[])
    parser.add_argument("--output", default="runs/passive-trajectory-corpus")
    parser.add_argument("--self-check", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
