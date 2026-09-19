"""Report matched fixed shadow-stop campaigns.

This is a mechanism screen, not the robust landscape-driver contract.  It
requires one immutable parent with exactly one ``noop`` and one
``trajectory_shadow_stop`` transition per case, then reports durable endpoint
outcomes and end-to-end cost ratios without inferring missing costs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


STRATEGIES = {"noop", "trajectory_shadow_stop"}
HORIZONS = ("immediate", "recovery", "final")
ALL_HORIZONS = ("parent", *HORIZONS)
PROVENANCE = ("source_checkpoint_sha256", "config_sha256", "data_sha256")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _horizon(row: dict[str, Any], name: str) -> dict[str, Any]:
    for item in row["metadata"]["horizon_metrics"]:
        if item["name"] == name:
            return item
    raise ValueError(f"missing {name} horizon in {row['transition_id']}")


def _costs(row: dict[str, Any]) -> dict[str, float]:
    before = row["before"]
    metadata = row["metadata"]
    return {
        "wall_seconds": _finite(
            before["features"].get("prefix_seconds", 0.0), "prefix_seconds"
        )
        + _finite(row["wall_seconds"], "wall_seconds"),
        "flops": _finite(before["compute_flops"], "prefix_flops")
        + _finite(row["compute_flops"], "branch_flops"),
        "tokens": _finite(before["tokens"], "prefix_tokens")
        + _finite(metadata["target_tokens"], "target_tokens"),
    }


def _strategy(row: dict[str, Any]) -> str:
    strategy = row["metadata"].get("strategy", row["action"]["kind"])
    if strategy not in STRATEGIES:
        raise ValueError(f"unexpected shadow-gate strategy: {strategy!r}")
    return strategy


def _curve_index(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    curves = _read_json(root / "capability_curves.json")["per_branch"]
    return {(row["case_id"], row["strategy"]): row for row in curves}


def _thresholds(
    curves: dict[tuple[str, str], dict[str, Any]], case_id: str, strategy: str
) -> dict[str, Any]:
    curve = curves.get((case_id, strategy))
    if curve is None:
        raise ValueError(f"missing capability curve for {case_id}/{strategy}")
    return curve["thresholds"]


def _case_report(
    root: Path,
    case_id: str,
    branches: dict[str, dict[str, Any]],
    curves: dict[tuple[str, str], dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if set(branches) != STRATEGIES:
        raise ValueError(f"{root}: {case_id} does not have exactly two paired branches")
    noop = branches["noop"]
    candidate = branches["trajectory_shadow_stop"]
    for field in PROVENANCE:
        if noop["metadata"].get(field) != candidate["metadata"].get(field):
            raise ValueError(f"{root}: {case_id} mismatched {field}")
    if noop["metadata"].get("code_sha") != candidate["metadata"].get("code_sha"):
        raise ValueError(f"{root}: {case_id} mismatched code_sha")

    noop_horizons = {name: _horizon(noop, name) for name in ALL_HORIZONS}
    candidate_horizons = {name: _horizon(candidate, name) for name in ALL_HORIZONS}
    noop_cost = _costs(noop)
    candidate_cost = _costs(candidate)
    endpoint_delta = _finite(
        candidate_horizons["final"]["loss"], "candidate final loss"
    ) - _finite(noop_horizons["final"]["loss"], "noop final loss")
    raw_final = candidate["metadata"].get("shadow_raw_final_loss")
    raw_delta = None if raw_final is None else _finite(raw_final, "raw final loss") - _finite(
        noop_horizons["final"]["loss"], "noop final loss"
    )
    failures = {
        strategy: branches[strategy].get("failure")
        for strategy in STRATEGIES
        if branches[strategy].get("failure")
    }
    threshold_report: dict[str, Any] = {}
    for name in _thresholds(curves, case_id, "noop"):
        baseline = _thresholds(curves, case_id, "noop")[name]
        candidate_threshold = _thresholds(curves, case_id, "trajectory_shadow_stop")[name]
        threshold_report[name] = {
            "noop": baseline,
            "trajectory_shadow_stop": candidate_threshold,
            "both_reached": bool(baseline.get("reached") and candidate_threshold.get("reached")),
            "same_first_point": (
                baseline.get("reached")
                and candidate_threshold.get("reached")
                and baseline.get("tokens") == candidate_threshold.get("tokens")
            ),
        }

    speedup = {
        metric: noop_cost[metric] / candidate_cost[metric]
        for metric in ("wall_seconds", "flops", "tokens")
    }
    planned_speedup = noop_cost["tokens"] / candidate_cost["tokens"]
    
    seed_by_case = {case["case_id"]: case["seed"] for case in manifest["cases"]}
    return {
        "case_id": case_id,
        "seed": seed_by_case.get(case_id),
        "width": manifest["config"]["width"],
        "train_file": manifest["config"]["train_file"],
        "parent_step": noop["before"]["step"],
        "noop_final_step": noop["after"]["step"],
        "candidate_final_step": candidate["after"]["step"],
        "endpoint_pass": not failures and endpoint_delta <= 0.0,
        "failures": failures,
        "immediate_delta": _finite(
            candidate_horizons["immediate"]["loss"], "candidate immediate loss"
        ) - _finite(noop_horizons["immediate"]["loss"], "noop immediate loss"),
        "recovery_delta": _finite(
            candidate_horizons["recovery"]["loss"], "candidate recovery loss"
        ) - _finite(noop_horizons["recovery"]["loss"], "noop recovery loss"),
        "final_delta": endpoint_delta,
        "raw_final_delta": raw_delta,
        "post_merge_delta": None if raw_delta is None else endpoint_delta - raw_delta,
        "losses": {
            strategy: {
                name: _finite(
                    horizons[name]["loss"], f"{strategy} {name} loss"
                )
                for name in ("parent", *HORIZONS)
            }
            for strategy, horizons in (
                ("noop", noop_horizons),
                ("trajectory_shadow_stop", candidate_horizons),
            )
        },
        "costs": {
            "noop": noop_cost,
            "trajectory_shadow_stop": candidate_cost,
            "speedup": speedup,
            "planned_speedup": planned_speedup,
            "wall_prediction_reality_gap": speedup["wall_seconds"] - planned_speedup,
        },
        "thresholds": threshold_report,
        "provenance": {
            field: noop["metadata"][field] for field in PROVENANCE
        },
    }


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    deltas = [row["final_delta"] for row in rows]
    raw_deltas = [
        row["raw_final_delta"]
        for row in rows
        if row.get("raw_final_delta") is not None
    ]
    speedups = {
        metric: [row["costs"]["speedup"][metric] for row in rows]
        for metric in ("wall_seconds", "flops", "tokens")
    }
    threshold_pairs = [
        threshold
        for row in rows
        for threshold in row["thresholds"].values()
    ]
    return {
        "n": len(rows),
        "endpoint_passes": sum(row["endpoint_pass"] for row in rows),
        "failures": sum(bool(row["failures"]) for row in rows),
        "final_delta": {
            "mean": statistics.fmean(deltas),
            "median": statistics.median(deltas),
            "min": min(deltas),
            "max": max(deltas),
        },
        "immediate_delta_mean": statistics.fmean(
            row["immediate_delta"] for row in rows
        ),
        "recovery_delta_mean": statistics.fmean(
            row["recovery_delta"] for row in rows
        ),
        "raw_final_delta": (
            {
                "mean": statistics.fmean(raw_deltas),
                "median": statistics.median(raw_deltas),
                "min": min(raw_deltas),
                "max": max(raw_deltas),
            }
            if raw_deltas
            else None
        ),
        "speedup": {
            metric: {
                "geometric_mean": _geomean(values),
                "median": statistics.median(values),
                "min": min(values),
            }
            for metric, values in speedups.items()
        },
        "wall_prediction_reality_gap_mean": statistics.fmean(
            row["costs"].get("wall_prediction_reality_gap", 0.0) for row in rows
        ),
        "thresholds": {
            "count": len(threshold_pairs),
            "both_reached": sum(item["both_reached"] for item in threshold_pairs),
            "same_first_point": sum(
                item["same_first_point"] for item in threshold_pairs
            ),
        },
    }


def report(results: list[Path]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    for root in results:
        manifest = _read_json(root / "manifest.json")
        transitions = [
            json.loads(line)
            for line in (root / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        curves = _curve_index(root)
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for row in transitions:
            grouped.setdefault(row["run_id"], {})[_strategy(row)] = row
        root_cases = []
        for case_id, branches in sorted(grouped.items()):
            root_cases.append(_case_report(root, case_id, branches, curves, manifest))
        expected = manifest["coverage"]["expected_cases"]
        if len(root_cases) != expected:
            raise ValueError(f"{root}: expected {expected} cases, found {len(root_cases)}")
        cases.extend(root_cases)
        roots.append(
            {
                "path": str(root),
                "git_commit": manifest["immutable"]["git_commit"],
                "coverage": manifest["coverage"],
                "config": manifest["config"],
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {"all": cases}
    for row in cases:
        data_name = "FineWeb-Edu" if "FineWeb" in row["train_file"] else "TinyStories"
        groups.setdefault(f"{data_name}-{row['width']}", []).append(row)
    return {
        "schema": "landscape-driver.shadow-gate-report.v1",
        "mechanism_screen": True,
        "results": [str(path) for path in results],
        "roots": roots,
        "overall": _summary(cases),
        "strata": {name: _summary(rows) for name, rows in sorted(groups.items())},
        "cases": cases,
        "interpretation": {
            "endpoint_definition": "trajectory_shadow_stop final validation loss <= matched noop final loss",
            "threshold_definition": "existing capability_curves first stable parent-relative threshold",
            "thresholds_non_discriminating": all(
                item["same_first_point"]
                for row in cases
                for item in row["thresholds"].values()
            ),
            "not_10x_claim": True,
        },
    }


def _self_check() -> None:
    assert math.isclose(_geomean([1.0, 4.0]), 2.0)
    assert _summary(
        [
            {
                "endpoint_pass": True,
                "failures": {},
                "final_delta": -1.0,
                "immediate_delta": 0.0,
                "recovery_delta": -0.1,
                "costs": {
                    "speedup": {"wall_seconds": 2.0, "flops": 3.0, "tokens": 4.0}
                },
                "thresholds": {"easy": {"both_reached": True, "same_first_point": True}},
            }
        ]
    )["endpoint_passes"] == 1
    print("shadow_gate_report self-check: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if not args.results or not args.output:
        parser.error("--results and --output are required unless --self-check is used")
    result = report(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
