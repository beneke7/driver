"""Build a causal, action-conditioned response atlas from decoder campaigns.

The campaign already records the expensive part of this dataset: matched
branches from immutable checkpoints.  This module only normalizes those
records into one row per action, adds deltas against the matched ``noop``, and
keeps the full provenance and cost ledger.  It does not invent counterfactuals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "landscape-driver.response-atlas.v1"
ROW_SCHEMA = "landscape-driver.response-atlas-row.v1"
HORIZONS = ("immediate", "recovery", "final")
FLOP_COMPONENTS = (
    "target_flops",
    "driver_inference_flops",
    "driver_update_flops",
    "probe_flops",
    "recovery_flops",
    "evaluation_flops",
    "rejected_branch_flops",
    "meta_training_flops",
    "search_flops",
    "other_flops",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite(value: Any, name: str, *, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = None
    if result is None or not math.isfinite(result):
        if default is not None:
            return default
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative(value: Any, name: str, *, default: float | None = None) -> float:
    result = _finite(value, name, default=default)
    if result is None or result < 0.0:
        if default is not None:
            return default
        raise ValueError(f"{name} must be non-negative")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing transition archive: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(dict(_mapping(json.loads(line), f"{path}:{line_number}")))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid transition at {path}:{line_number}") from exc
    return rows


def _case_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(manifest.get("cases", ())):
        case = _mapping(raw, f"manifest.cases[{index}]")
        case_id = str(case.get("case_id", "")).strip()
        if not case_id or case_id in result:
            raise ValueError(f"manifest has invalid or duplicate case_id: {case_id!r}")
        result[case_id] = case
    if not result:
        raise ValueError("campaign manifest has no cases")
    return result


def _horizon_metrics(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    metadata = _mapping(row.get("metadata", {}), "transition.metadata")
    result: dict[str, Mapping[str, Any]] = {}
    raw = metadata.get("horizon_metrics", ())
    if not isinstance(raw, list):
        return result
    for item in raw:
        value = _mapping(item, "transition.metadata.horizon_metrics item")
        name = str(value.get("name", ""))
        if name in HORIZONS:
            result[name] = value
    return result


def _group_key(row: Mapping[str, Any], case: Mapping[str, Any]) -> tuple[str, str, int, int]:
    before = _mapping(row.get("before"), "transition.before")
    metadata = _mapping(row.get("metadata", {}), "transition.metadata")
    run_id = str(row.get("run_id", "")).strip()
    source = str(
        metadata.get("source_checkpoint_sha256", case.get("parent_checkpoint_sha256", ""))
    ).strip()
    if not run_id or not source:
        raise ValueError("transition is missing run_id or source checkpoint provenance")
    step = int(before.get("step"))
    after = row.get("after")
    if isinstance(after, Mapping):
        horizon = int(after.get("step")) - step
    else:
        horizon = int(metadata.get("target_tokens", -1))
    return run_id, source, step, horizon


def _branch_cost(
    row: Mapping[str, Any],
    *,
    prefix_flops: float,
    prefix_tokens: int,
    prefix_wall: float,
) -> dict[str, Any]:
    metadata = _mapping(row.get("metadata", {}), "transition.metadata")
    driver_cost = _mapping(metadata.get("driver_cost", {}), "transition.metadata.driver_cost")
    branch_total = _non_negative(row.get("compute_flops"), "transition.compute_flops")
    recovery_flops = _non_negative(
        metadata.get("recovery_flops", 0.0), "recovery_flops", default=0.0
    )
    branch_target_flops = _non_negative(
        metadata.get("target_flops", 0.0), "target_flops"
    )
    branch_components = {
        # Recovery is a named additive component; remove it from target work
        # here because the campaign's target_flops includes all branch steps.
        "target_flops": max(0.0, branch_target_flops - recovery_flops),
        "driver_inference_flops": _non_negative(
            driver_cost.get("inference_flops", metadata.get("driver_inference_flops", 0.0)),
            "driver_inference_flops",
        ),
        "recovery_flops": recovery_flops,
        "driver_update_flops": _non_negative(
            driver_cost.get("update_flops", metadata.get("driver_update_flops", 0.0)),
            "driver_update_flops",
        ),
        "probe_flops": _non_negative(driver_cost.get("probe_flops", 0.0), "probe_flops"),
        "evaluation_flops": _non_negative(
            driver_cost.get("evaluation_flops", metadata.get("evaluation_flops", 0.0)),
            "evaluation_flops",
        ),
        "rejected_branch_flops": _non_negative(
            driver_cost.get("rejected_branch_flops", 0.0), "rejected_branch_flops"
        ),
        "meta_training_flops": _non_negative(
            driver_cost.get("meta_training_flops", 0.0), "meta_training_flops"
        ),
        "search_flops": _non_negative(
            driver_cost.get("search_flops", 0.0), "search_flops"
        ),
        "other_flops": 0.0,
    }
    known = sum(branch_components.values())
    accounting_error = branch_total - known
    if accounting_error >= 0.0:
        branch_components["other_flops"] = accounting_error
    else:
        # Keep the source ledger authoritative and make an over-report visible.
        accounting_error = abs(accounting_error)
    branch_tokens = int(metadata.get("target_tokens", 0))
    branch_wall = _non_negative(row.get("wall_seconds"), "transition.wall_seconds")
    components = dict(branch_components)
    components["target_flops"] += prefix_flops
    return {
        "wall_seconds": prefix_wall + branch_wall,
        "branch_wall_seconds": branch_wall,
        "tokens": prefix_tokens + branch_tokens,
        "branch_tokens": branch_tokens,
        "flops": prefix_flops + branch_total,
        "branch_flops": branch_total,
        "components": components,
        "reported_component_overrun_flops": accounting_error,
    }


def _outcomes(
    row: Mapping[str, Any], noop: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    action_metrics = _horizon_metrics(row)
    noop_metrics = _horizon_metrics(noop) if noop is not None else {}
    outcomes: dict[str, Any] = {}
    deltas: list[float] = []
    for horizon in HORIZONS:
        current = action_metrics.get(horizon)
        baseline = noop_metrics.get(horizon)
        if current is None or "loss" not in current:
            continue
        loss = _finite(current["loss"], f"{horizon}.loss")
        if loss is None:
            continue
        item: dict[str, Any] = {"loss": loss}
        for key in ("tokens", "wall_seconds", "phase_wall_seconds", "raw_loss"):
            if key in current:
                item[key] = current[key]
        if baseline is not None and "loss" in baseline:
            baseline_loss = _finite(baseline["loss"], f"noop.{horizon}.loss")
            if baseline_loss is not None:
                delta = loss - baseline_loss
                item["loss_delta_vs_noop"] = delta
                item["quality_gain_vs_noop"] = -delta
                deltas.append(delta)
        outcomes[horizon] = item
    response = {
        "final_loss_delta_vs_noop": outcomes.get("final", {}).get("loss_delta_vs_noop"),
        "final_quality_gain_vs_noop": outcomes.get("final", {}).get(
            "quality_gain_vs_noop"
        ),
        "safe_loss_delta_vs_noop": max(deltas) if deltas else None,
        "horizons_observed": [name for name in HORIZONS if name in outcomes],
    }
    return outcomes, response


def _record(
    row: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    manifest: Mapping[str, Any],
    noop: Mapping[str, Any] | None,
    group_id: str,
) -> dict[str, Any]:
    before = _mapping(row["before"], "transition.before")
    metadata = _mapping(row.get("metadata", {}), "transition.metadata")
    case_id = str(row["run_id"])
    immutable = _mapping(manifest.get("immutable", {}), "manifest.immutable")
    prefix_wall = _finite(
        _mapping(before.get("features", {}), "transition.before.features").get(
            "prefix_seconds", 0.0
        ),
        "prefix_seconds",
        default=0.0,
    ) or 0.0
    prefix_flops = _non_negative(before.get("compute_flops", 0.0), "prefix_flops")
    prefix_tokens = int(before.get("tokens", 0))
    cost = _branch_cost(
        row,
        prefix_flops=prefix_flops,
        prefix_tokens=prefix_tokens,
        prefix_wall=prefix_wall,
    )
    outcomes, response = _outcomes(row, noop)
    failed = row.get("after") is None or not bool(row.get("accepted", True))
    failure = row.get("failure")
    if failure is not None:
        failure = str(failure)
    action = _mapping(row.get("action"), "transition.action")
    risk = {
        "failed": failed,
        "failure": failure,
        "unsafe_horizon": (
            response["safe_loss_delta_vs_noop"] is not None
            and response["safe_loss_delta_vs_noop"] > 0.0
        ),
    }
    provenance = {
        "source_checkpoint_sha256": str(
            metadata.get("source_checkpoint_sha256", case.get("parent_checkpoint_sha256", ""))
        ),
        "parent_checkpoint": str(case.get("parent_checkpoint", "")),
        "data_sha256": str(metadata.get("data_sha256", case.get("data_sha256", ""))),
        "config_sha256": str(metadata.get("config_sha256", case.get("config_sha256", ""))),
        "objective_sha256": str(immutable.get("objective_sha256", "")),
        "code_sha": str(metadata.get("code_sha", immutable.get("git_commit", ""))),
        "code_sha256": str(immutable.get("code_sha256", "")),
        "git_commit": str(immutable.get("git_commit", "")),
    }
    return {
        "schema": ROW_SCHEMA,
        "atlas_id": f"{group_id}:{row.get('transition_id', '')}",
        "group_id": group_id,
        "case_id": case_id,
        "source_transition_id": str(row.get("transition_id", "")),
        "parent": {
            "step": int(before["step"]),
            "tokens": prefix_tokens,
            "loss": _finite(before["loss"], "parent.loss"),
            "features": dict(_mapping(before.get("features", {}), "parent.features")),
        },
        "action": {
            "kind": str(action.get("kind", "")),
            "strength": _finite(action.get("strength", 0.0), "action.strength"),
            "parameters": dict(_mapping(action.get("parameters", {}), "action.parameters")),
            "selected_schedule": metadata.get("selected_schedule"),
        },
        "outcomes": outcomes,
        "response": response,
        "cost": cost,
        "risk": risk,
        "matched_noop_transition_id": (
            None if noop is None else str(noop.get("transition_id", ""))
        ),
        "provenance": provenance,
    }


def collect(results: list[Path], *, require_matched: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not results:
        raise ValueError("at least one campaign result directory is required")
    groups: dict[tuple[str, str, int, int], list[tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    input_manifests: list[dict[str, Any]] = []
    for result in results:
        manifest_path = result / "manifest.json"
        manifest = _mapping(_read_json(manifest_path), str(manifest_path))
        if manifest.get("schema") != "landscape-driver.decoder-campaign.v1":
            raise ValueError(f"unsupported campaign schema in {manifest_path}")
        cases = _case_map(manifest)
        transitions = _read_jsonl(result / "transitions.jsonl")
        for row in transitions:
            case_id = str(row.get("run_id", ""))
            case = cases.get(case_id)
            if case is None:
                raise ValueError(f"transition {row.get('transition_id')} has unknown case {case_id}")
            groups[_group_key(row, case)].append((row, case, manifest))
        input_manifests.append(
            {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "schema": manifest.get("schema"),
                "git_commit": _mapping(manifest.get("immutable", {}), "manifest.immutable").get(
                    "git_commit"
                ),
            }
        )
    records: list[dict[str, Any]] = []
    unmatched = 0
    duplicate_actions = 0
    for key, entries in sorted(groups.items()):
        actions: dict[str, tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]] = {}
        for entry in entries:
            kind = str(_mapping(entry[0].get("action"), "transition.action").get("kind", ""))
            if kind in actions:
                duplicate_actions += 1
                raise ValueError(f"duplicate action {kind!r} in atlas group {key}")
            actions[kind] = entry
        noop = actions.get("noop")
        if noop is None:
            unmatched += sum(kind != "noop" for kind in actions)
        group_id = ":".join((key[0], key[1], str(key[2]), str(key[3])))
        for entry in entries:
            row, case, manifest = entry
            if require_matched and str(
                _mapping(row.get("action"), "transition.action").get("kind", "")
            ) != "noop" and noop is None:
                raise ValueError(f"unmatched action in atlas group {group_id}")
            records.append(
                _record(
                    row,
                    case=case,
                    manifest=manifest,
                    noop=None if noop is None else noop[0],
                    group_id=group_id,
                )
            )
    summary = {
        "schema": SCHEMA,
        "row_schema": ROW_SCHEMA,
        "inputs": input_manifests,
        "groups": len(groups),
        "rows": len(records),
        "matched_groups": sum("noop" in {str(_mapping(item[0].get("action"), "action").get("kind", "")) for item in entries} for entries in groups.values()),
        "unmatched_action_rows": unmatched,
        "actions": sorted({record["action"]["kind"] for record in records}),
        "failed_rows": sum(record["risk"]["failed"] for record in records),
        "duplicate_actions": duplicate_actions,
        "cost_definition": {
            "wall": "prefix plus branch end-to-end wall_seconds",
            "flops": "prefix plus branch compute_flops; additive components preserve reported driver work",
            "tokens": "prefix plus branch target_tokens",
            "recovery_flops": "named additive component removed from target_flops to avoid double counting",
        },
    }
    return records, summary


def write_atlas(records: list[dict[str, Any]], summary: Mapping[str, Any], output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "atlas.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    manifest = dict(summary)
    manifest["files"] = {"atlas": "atlas.jsonl"}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _self_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = root / "campaign"
        result.mkdir()
        manifest = {
            "schema": "landscape-driver.decoder-campaign.v1",
            "immutable": {
                "git_commit": "a" * 40,
                "code_sha256": "b" * 64,
                "objective_sha256": "c" * 64,
            },
            "cases": [
                {
                    "case_id": "text_shard-0",
                    "data_sha256": "d" * 64,
                    "config_sha256": "e" * 64,
                    "parent_checkpoint_sha256": "f" * 64,
                    "parent_checkpoint": "checkpoints/text_shard-0/parent.pt",
                }
            ],
        }
        (result / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        common = {
            "run_id": "text_shard-0",
            "before": {
                "step": 10,
                "tokens": 100,
                "loss": 2.0,
                "compute_flops": 100.0,
                "features": {"prefix_seconds": 1.0},
            },
            "accepted": True,
            "compute_flops": 30.0,
            "wall_seconds": 3.0,
            "metadata": {
                "source_checkpoint_sha256": "f" * 64,
                "target_flops": 20.0,
                "target_tokens": 10,
                "driver_cost": {"inference_flops": 1.0, "evaluation_flops": 9.0},
                "horizon_metrics": [
                    {"name": "immediate", "loss": 1.9},
                    {"name": "recovery", "loss": 1.8},
                    {"name": "final", "loss": 1.7},
                ],
            },
        }
        noop = dict(common)
        noop.update(
            {
                "transition_id": "text_shard-0:noop",
                "action": {"kind": "noop", "strength": 0.0, "parameters": {}},
            }
        )
        action = dict(common)
        action.update(
            {
                "transition_id": "text_shard-0:shadow",
                "action": {"kind": "trajectory_shadow_average", "strength": 0.0, "parameters": {}},
                "metadata": dict(common["metadata"], horizon_metrics=[
                    {"name": "immediate", "loss": 1.9},
                    {"name": "recovery", "loss": 1.75},
                    {"name": "final", "loss": 1.6},
                ]),
            }
        )
        (result / "transitions.jsonl").write_text(
            "\n".join(json.dumps(row) for row in (noop, action)) + "\n", encoding="utf-8"
        )
        records, summary = collect([result], require_matched=True)
        assert len(records) == 2 and summary["matched_groups"] == 1
        candidate = next(row for row in records if row["action"]["kind"] != "noop")
        assert abs(candidate["response"]["final_loss_delta_vs_noop"] + 0.1) < 1e-9
        assert abs(candidate["cost"]["flops"] - 130.0) < 1e-9
    print("response-atlas self-check passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", default=[])
    parser.add_argument("--output", default="runs/response-atlas")
    parser.add_argument("--require-matched", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser


def main(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.self_check:
        _self_check()
        return None
    results = [Path(value) for value in args.results]
    records, summary = collect(results, require_matched=args.require_matched)
    manifest = write_atlas(records, summary, Path(args.output))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main(build_parser().parse_args())
