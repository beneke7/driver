"""Assemble serial quadratic measurements into the fail-closed contract shape."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .contract import FLOP_COMPONENTS, evaluate_campaign


def _digest(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).hexdigest()


def _cost(
    *,
    wall_seconds: float,
    target_flops: float = 0.0,
    probe_flops: float = 0.0,
    search_flops: float = 0.0,
    scope: str,
    tokens: int = 0,
) -> dict[str, Any]:
    components = {name: 0.0 for name in FLOP_COMPONENTS}
    components["target_flops"] = target_flops
    components["probe_flops"] = probe_flops
    components["search_flops"] = search_flops
    return {
        "scope": scope,
        "wall_scope": "end_to_end" if scope == "deployment" else "one_time",
        "wall_seconds": wall_seconds,
        "tokens": tokens,
        "components": components,
    }


def _source(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    serial_path = Path(path) / "serial.json"
    serial = json.loads(serial_path.read_text(encoding="utf-8"))
    source = Path(serial["source"])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    return serial, manifest, summary


def build_campaign(sources: list[str], *, amortization_deployments: int) -> dict[str, Any]:
    if len(sources) < 2:
        raise ValueError("at least two threshold sources are required")
    loaded = [_source(path) for path in sources]
    first_serial, first_manifest, _ = loaded[0]
    heldout = [
        {
            "case_id": row["case_id"],
            "landscape": row["family"],
            "seed": row["seed"],
        }
        for row in first_serial["rows"]
    ]
    development = [
        {
            "case_id": row["case_id"],
            "landscape": row["family"],
            "seed": row["seed"],
        }
        for row in first_manifest["development_cases"]
    ]
    thresholds: list[str] = []
    rows: list[dict[str, Any]] = []
    baseline_search_flops = 0.0
    baseline_search_wall = 0.0
    driver_setup_wall = 0.0
    for serial, manifest, summary in loaded:
        threshold = format(float(serial["threshold"]), ".17g")
        thresholds.append(threshold)
        baseline_search_flops += float(serial.get("baseline_tuning_flops", 0.0))
        baseline_search_wall += float(serial.get("baseline_tuning_wall_seconds", 0.0))
        driver_setup_wall += float(serial.get("warmup_driver_wall_seconds", 0.0))
        if manifest["heldout_cases"] != first_manifest["heldout_cases"]:
            raise ValueError("threshold sources do not share an identical held-out roster")
        for measured in serial["rows"]:
            case_id = measured["case_id"]
            provenance_base = _digest(
                "quadratic-start",
                first_manifest["code_revision"],
                first_manifest["dimension"],
                first_manifest.get("condition_floor"),
                first_manifest.get("balanced_start"),
                case_id,
                measured["condition"],
            )
            provenance = {
                "source_checkpoint_sha256": _digest(provenance_base, "checkpoint"),
                "data_sha256": _digest(provenance_base, "data"),
                "objective_sha256": _digest(provenance_base, "objective"),
                "config_sha256": _digest(
                    "quadratic-config",
                    first_manifest["dimension"],
                    first_manifest.get("condition_floor"),
                    first_manifest.get("balanced_start"),
                ),
            }
            rows.append(
                {
                    "threshold": threshold,
                    "case_id": case_id,
                    "split": "heldout",
                    "landscape": measured["family"],
                    "seed": measured["seed"],
                    "match_id": f"{case_id}:parent",
                    "provenance": provenance,
                    "baseline": {
                        "reached": measured["baseline_reached"],
                        "failure": None
                        if measured["baseline_reached"]
                        else "threshold_not_reached",
                        "cost": _cost(
                            scope="deployment",
                            wall_seconds=measured["baseline_wall_seconds"],
                            target_flops=measured["baseline_flops"],
                        ),
                    },
                    "driver": {
                        "reached": measured["driver_reached"],
                        "failure": None
                        if measured["driver_reached"]
                        else "threshold_not_reached",
                        "cost": _cost(
                            scope="deployment",
                            wall_seconds=measured["driver_wall_seconds"],
                            probe_flops=measured["driver_flops"],
                        ),
                    },
                }
            )

    if len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold sources must have distinct thresholds")
    if amortization_deployments < 1:
        raise ValueError("amortization_deployments must be positive")
    baseline_one_time = _cost(
        scope="one_time",
        wall_seconds=baseline_search_wall,
        search_flops=baseline_search_flops,
    )
    driver_one_time = _cost(
        scope="one_time",
        wall_seconds=driver_setup_wall,
    )
    return {
        "schema": "landscape-driver.contract.v1",
        "contract": {
            "preregistered": True,
            "thresholds": thresholds,
            "target_speedup": 10.0,
            "primary_metric": "wall_seconds",
            "gated_metrics": ["wall_seconds"],
            "evidence_metrics": ["wall_seconds", "flops"],
            "min_landscapes": 5,
            "min_seeds": 3,
            "case_floor": 5.0,
            "confidence": 0.95,
            "bootstrap_samples": 4000,
            "random_seed": 0,
            "amortization_deployments": amortization_deployments,
        },
        "development_cases": development,
        "heldout_cases": heldout,
        "one_time_costs": {"baseline": baseline_one_time, "driver": driver_one_time},
        "rows": rows,
        "notes": {
            "track": "optimization",
            "target": "ill-conditioned positive-definite quadratic landscapes",
            "baseline": "development-tuned AdamW",
            "driver": "diagonal secant, two-probe CG, full probe fallback",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--amortization-deployments", type=int, default=45)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    campaign = build_campaign(
        args.source, amortization_deployments=args.amortization_deployments
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"choose a fresh --output; {output} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
    report = evaluate_campaign(campaign)
    print(json.dumps({"campaign": str(output), "eligible": report["eligible"]}, indent=2))
    if not report["eligible"]:
        print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run(build_parser().parse_args())
