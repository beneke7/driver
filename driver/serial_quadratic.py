"""Measure one held-out quadratic case at a time.

The main quadratic benchmark batches cases to make development sweeps cheap.
That is useful for search, but dividing a batch duration by its size is not a
deployment wall-clock measurement.  This evaluator replays a completed
benchmark configuration and records synchronized steady-state timings for
each case separately.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .quadratic_benchmark import (
    Case,
    QuadraticOracle,
    adam_batch,
    estimated_flops,
    make_batch,
    newton_probe_batch,
)


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fresh_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"choose a fresh --output; {path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)


def _median_wall(values: list[float]) -> float:
    return float(statistics.median(values))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    source = Path(args.source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    dimension = int(manifest["dimension"])
    threshold = float(manifest["threshold"])
    max_steps = int(manifest["max_steps"])
    condition_floor = manifest.get("condition_floor")
    balanced_start = bool(manifest.get("balanced_start", False))
    learning_rate = float(summary["baseline_learning_rate"])
    heldout = [Case(**payload) for payload in manifest["heldout_cases"]]
    matrix, initial = make_batch(
        heldout,
        dimension=dimension,
        device=device,
        dtype=torch.float32,
        balanced_start=balanced_start,
    )

    # Warm both code paths once.  Compilation/setup is reported separately;
    # the rows below measure repeatable deployment work after that fixed cost.
    warm_matrix = matrix[:1]
    warm_initial = initial[:1]
    warm_baseline_started = time.perf_counter()
    adam_batch(
        warm_matrix,
        warm_initial,
        learning_rate=learning_rate,
        threshold=threshold,
        max_steps=min(2, max_steps),
    )
    warm_baseline_wall = time.perf_counter() - warm_baseline_started
    warm_driver_started = time.perf_counter()
    newton_probe_batch(QuadraticOracle(warm_matrix), warm_initial, threshold=threshold)
    warm_driver_wall = time.perf_counter() - warm_driver_started

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(heldout):
        case_matrix = matrix[index : index + 1]
        case_initial = initial[index : index + 1]
        baseline_runs = [
            adam_batch(
                case_matrix,
                case_initial,
                learning_rate=learning_rate,
                threshold=threshold,
                max_steps=max_steps,
            )
            for _ in range(args.repeats)
        ]
        driver_runs = [
            newton_probe_batch(
                QuadraticOracle(case_matrix), case_initial, threshold=threshold
            )
            for _ in range(args.repeats)
        ]
        baseline_steps = int(baseline_runs[-1][0].item())
        driver_calls = int(driver_runs[-1][0].item())
        driver_reached = bool(driver_runs[-1][2].item())
        driver_full_solve = bool(driver_runs[-1][3].item())
        baseline_reached = baseline_steps > 0
        baseline_wall = _median_wall([result[2] for result in baseline_runs])
        driver_wall = _median_wall([float(result[4].item()) for result in driver_runs])
        baseline_flops = float(
            estimated_flops(
                "adamw",
                dimension,
                torch.tensor([baseline_steps], device=device),
            )[0].item()
        )
        driver_flops = float(
            estimated_flops(
                "hessian_probe",
                dimension,
                torch.tensor([driver_calls], device=device),
                full_solve=torch.tensor([driver_full_solve], device=device),
            )[0].item()
        )
        rows.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "family": case.family,
                "seed": case.seed,
                "condition": case.condition,
                "baseline_steps": baseline_steps,
                "driver_gradient_calls": driver_calls,
                "driver_full_solve": driver_full_solve,
                "baseline_reached": baseline_reached,
                "driver_reached": driver_reached,
                "baseline_flops": baseline_flops,
                "driver_flops": driver_flops,
                "baseline_wall_seconds": baseline_wall,
                "driver_wall_seconds": driver_wall,
                "wall_speedup": baseline_wall / driver_wall if driver_wall else 0.0,
            }
        )

    output = Path(args.output)
    _fresh_output(output)
    payload = {
        "source": str(source),
        "source_code_revision": manifest.get("code_revision"),
        "code_revision": _revision(),
        "device": str(device),
        "torch": torch.__version__,
        "dimension": dimension,
        "threshold": threshold,
        "max_steps": max_steps,
        "condition_floor": condition_floor,
        "balanced_start": balanced_start,
        "baseline_learning_rate": learning_rate,
        "repeats": args.repeats,
        "warmup_baseline_wall_seconds": warm_baseline_wall,
        "warmup_driver_wall_seconds": warm_driver_wall,
        "baseline_tuning_wall_seconds": summary.get("baseline_tuning_wall_seconds", 0.0),
        "baseline_tuning_flops": summary.get("baseline_tuning_flops", 0.0),
        "rows": rows,
        "aggregate_wall_speedup": sum(row["baseline_wall_seconds"] for row in rows)
        / sum(row["driver_wall_seconds"] for row in rows),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "serial.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed quadratic benchmark directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=3)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
